# Lint as: python3
"""Wrapper for fine-tuned HuggingFace models in LIT."""

import os
import re
from typing import Optional, Dict, List, Iterable

from absl import logging
import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
import numpy as np
import tensorflow as tf
import transformers

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec


def _from_pretrained(cls, *args, **kw):
  """Load a transformers model in TF2, with fallback to PyTorch weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning("Caught OSError loading model: %s", e)
    logging.warning(
        "Re-trying to convert from PyTorch checkpoint (from_pt=True)")
    return cls.from_pretrained(*args, from_pt=True, **kw)


@attr.s(auto_attribs=True, kw_only=True)
class LentaModelConfig(object):  #### EDIT
  """Config options for a GlueModel."""
  # Preprocessing options
  max_seq_length: int = 64 #### EDIT
  inference_batch_size: int = 16 #### EDIT
  # Input options
  text_a_name: str = "sentence1"
  text_b_name: Optional[str] = None  # set to None for single-segment #### EDIT
  label_name: str = "label"
  # Output options
  labels: Optional[List[str]] = ['0', '1', '2'] # set to None for regression #### EDIT
  null_label_idx: Optional[int] = None
  compute_grads: bool = True  # if True, compute and return gradients.


class LentaModelGeneral(lit_model.Model): #### EDIT
  """Benchmark model, using Keras/TF2 and Huggingface Transformers.

  This is a general-purpose classification or regression model. It works for
  one- or two-segment input, and predicts either a multiclass label or
  a regression score. See LentaModelConfig for available options.

  This implements the LIT API for inference (e.g. input_spec(), output_spec(),
  and predict()), but also provides a train() method to run fine-tuning.

  This is a full-featured implementation, which includes embeddings, attention,
  gradients, as well as support for the different input and output types above.
  For a more minimal example, see ../simple_tf2_demo.py.
  """

  @property
  def is_regression(self) -> bool:
    return self.config.labels is None

  # TODO(lit-dev): upgrade version of huggingface so we can just pass
  # output_hidden_states and output_attentions at inference time, rather than
  # as part of the model config. Then we don't need a special for_training mode.
  def __init__(self,
               model_name_or_path="DeepPavlov/rubert-base-cased",  #### EDIT
               for_training=False,
               **config_kw):
    self.config = LentaModelConfig(**config_kw)  #### EDIT
    self._load_model(model_name_or_path, for_training)

  def _load_model(self, model_name_or_path, for_training):
    """Load model. Can be overridden for testing."""
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=1 if self.is_regression else len(self.config.labels),
        output_hidden_states=(not for_training),
        output_attentions=(not for_training),
    )
    self.model = _from_pretrained(
        transformers.TFAutoModelForSequenceClassification,
        model_name_or_path,
        config=model_config)

  def _preprocess(self, inputs: Iterable[JsonDict]) -> Dict[str, tf.Tensor]:
    segments = [
        (ex[self.config.text_a_name],
         ex[self.config.text_b_name] if self.config.text_b_name else None)
        for ex in inputs
    ]
    encoded_input = self.tokenizer.batch_encode_plus(
        segments,
        return_tensors="tf",
        add_special_tokens=True,
        max_length=self.config.max_seq_length,
        pad_to_max_length=True)
    # Trim everything to the actual max length, to remove extra padding.
    max_tokens = tf.reduce_max(
        tf.reduce_sum(encoded_input["attention_mask"], axis=1))
    encoded_input = {k: v[:, :max_tokens] for k, v in encoded_input.items()}
    return encoded_input

  def _make_dataset(self, inputs: Iterable[JsonDict]) -> tf.data.Dataset:
    """Make a tf.data.Dataset from inputs in LIT format."""
    encoded_input = self._preprocess(inputs)
    if self.is_regression:
      labels = tf.constant([ex[self.config.label_name] for ex in inputs],
                           dtype=tf.float32)
    else:
      labels = tf.constant([
          self.config.labels.index(ex[self.config.label_name]) for ex in inputs
      ],
                           dtype=tf.int64)
    # encoded_input is actually a transformers.tokenization_utils.BatchEncoding
    # object, which tf.data.Dataset doesn't like. Convert to a regular dict.
    return tf.data.Dataset.from_tensor_slices((dict(encoded_input), labels))

  def train(self,
            train_inputs: List[JsonDict],
            validation_inputs: List[JsonDict],
            learning_rate=2e-5,
            batch_size=16,  #### EDIT
            num_epochs=2,  #### EDIT
            keras_callbacks=None):
    """Run fine-tuning."""
    train_dataset = self._make_dataset(train_inputs).shuffle(128).batch(
        batch_size).repeat(-1)
    # Use larger batch for validation since inference is about 1/2 memory usage
    # of backprop.
    eval_batch_size = 2 * batch_size
    validation_dataset = self._make_dataset(validation_inputs).batch(
        eval_batch_size)

    # Prepare model for training.
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    if self.is_regression:
      loss = tf.keras.losses.MeanSquaredError()
      metric = tf.keras.metrics.RootMeanSquaredError("rmse")
    else:
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
    self.model.compile(optimizer=opt, loss=loss, metrics=[metric])

    steps_per_epoch = len(train_inputs) // batch_size
    validation_steps = len(validation_inputs) // eval_batch_size
    history = self.model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=keras_callbacks,
        verbose=2)
    return history

  def save(self, path: str):
    """Save model weights and tokenizer info.

    To re-load, pass the path to the constructor instead of the name of a
    base model.

    Args:
      path: directory to save to. Will write several files here.
    """
    if not os.path.isdir(path):
      os.mkdir(path)
    self.tokenizer.save_pretrained(path)
    self.model.save_pretrained(path)

  def _segment_slicers(self, tokens: List[str]):
    """Slicers along the tokens dimension for each segment.

    For tokens ['[CLS]', a0, a1, ..., '[SEP]', b0, b1, ..., '[SEP]'],
    we want to get the slices [a0, a1, ...] and [b0, b1, ...]

    Args:
      tokens: <string>[num_tokens], including special tokens

    Returns:
      (slicer_a, slicer_b), slice objects
    """
    try:
      split_point = tokens.index(self.tokenizer.sep_token)
    except ValueError:
      split_point = len(tokens) - 1
    slicer_a = slice(1, split_point)  # start after [CLS]
    slicer_b = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
    return slicer_a, slicer_b

  def _postprocess(self, output: Dict[str, np.ndarray]):
    """Per-example postprocessing, on NumPy output."""
    ntok = output.pop("ntok")
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(
        output.pop("input_ids")[:ntok])

    # Tokens for each segment, individually.
    slicer_a, slicer_b = self._segment_slicers(output["tokens"])
    output["tokens_" + self.config.text_a_name] = output["tokens"][slicer_a]
    if self.config.text_b_name:
      output["tokens_" + self.config.text_b_name] = output["tokens"][slicer_b]

    # Embeddings for each segment, individually.
    output["input_embs_" + self.config.text_a_name] = (
        output["input_embs"][slicer_a])
    if self.config.text_b_name:
      output["input_embs_" + self.config.text_b_name] = (
          output["input_embs"][slicer_b])

    # Gradients for each segment, individually.
    if self.config.compute_grads:
      output["token_grad_" +
             self.config.text_a_name] = output["input_emb_grad"][slicer_a]
      if self.config.text_b_name:
        output["token_grad_" +
               self.config.text_b_name] = output["input_emb_grad"][slicer_b]
      if self.is_regression:
        output["grad_class"] = None
      else:
        # Return the label corresponding to the class index used for gradients.
        output["grad_class"] = self.config.labels[output["grad_class"]]

    # Process attention.
    for key in output:
      if not re.match(r"layer_(\d+)/attention", key):
        continue
      # Select only real tokens, since most of this matrix is padding.
      # <float32>[num_heads, max_seq_length, max_seq_length]
      # -> <float32>[num_heads, num_tokens, num_tokens]
      output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
      # Make a copy of this array to avoid memory leaks, since NumPy otherwise
      # keeps a pointer around that prevents the source array from being GCed.
      output[key] = output[key].copy()  # pytype: disable=attribute-error

    return output

  def _scatter_embs(self, passed_input_embs, input_embs, batch_indices,
                    offsets):
    """Scatters custom passed embeddings into the default model embeddings.

    Args:
      passed_input_embs: <tf.float32>[num_scatter_tokens], the custom passed
        embeddings to be scattered into the default model embeddings.
      input_embs: the default model embeddings.
      batch_indices: the indices of the embeddings to replace in the format
        (batch_index, sequence_index).
      offsets: the offset from which to scatter the custom embedding (number of
        tokens from the start of the sequence).

    Returns:
      The default model embeddings with scattered custom embeddings.
    """

    # <float32>[scatter_batch_size, num_tokens, emb_size]
    filtered_embs = [emb for emb in passed_input_embs if emb is not None]

    # Prepares update values that should be scattered in, i.e. one for each
    # of the (scatter_batch_size * num_tokens) word embeddings.
    # <np.float32>[scatter_batch_size * num_tokens, emb_size]
    updates = np.concatenate(filtered_embs)

    # Prepares indices in format (batch_index, sequence_index) for all
    # values that should be scattered in, i.e. one for each of the
    # (scatter_batch_size * num_tokens) word embeddings.
    scatter_indices = []
    for (batch_index, sentence_embs, offset) in zip(batch_indices,
                                                    filtered_embs, offsets):
      for (token_index, emb) in enumerate(sentence_embs):
        scatter_indices.append([batch_index, token_index + offset])

    # Scatters passed word embeddings into embeddings gathered from tokens.
    # <tf.float32>[batch_size, num_tokens + num_special_tokens, emb_size]
    return tf.tensor_scatter_nd_update(input_embs, scatter_indices, updates)

  def scatter_all_embeddings(self, inputs, input_embs):
    """Scatters custom passed embeddings for text segment inputs.

    Args:
      inputs: the model inputs, which contain any custom embeddings to scatter.
      input_embs: the default model embeddings.

    Returns:
      The default model embeddings with scattered custom embeddings.
    """
    # Gets batch indices of any word embeddings that were passed for text_a.
    passed_input_embs_a = [ex.get("input_embs_" + self.config.text_a_name)
                           for ex in inputs]
    batch_indices_a = [index for (index, emb) in enumerate(
        passed_input_embs_a) if emb is not None]

    # If word embeddings were passed in for text_a, scatter them into the
    # embeddings, gathered from the input ids. 1 is passed in as the offset
    # for each, since text_a starts at index 1, after the [CLS] token.
    if batch_indices_a:
      input_embs = self._scatter_embs(
          passed_input_embs_a, input_embs, batch_indices_a,
          offsets=np.ones(len(batch_indices_a), dtype=np.int64))

    if self.config.text_b_name:
      # Gets batch indices of any word embeddings that were passed for text_b.
      passed_input_embs_b = [ex.get("input_embs_" + self.config.text_b_name)
                             for ex in inputs]
      batch_indices_b = [
          index for (index, emb) in enumerate(passed_input_embs_b)
          if emb is not None]

      # If word embeddings were also passed in for text_b, scatter them into the
      # embeddings gathered from the input ids. The offsets are the [lengths
      # of the corresponding text_a embeddings] + 2, since text_b starts after
      # [CLS] [text_a tokens] [SEP]. (This assumes that text_b embeddings
      # will only be passed together with text_a embeddings.)
      if batch_indices_b:
        lengths = np.array([len(embed) for embed in passed_input_embs_a
                            if embed is not None])
        input_embs = self._scatter_embs(
            passed_input_embs_b, input_embs, batch_indices_b,
            offsets=(lengths + 2))
    return input_embs

  ##
  # LIT API implementation
  def max_minibatch_size(self):
    return self.config.inference_batch_size

  def predict_minibatch(self, inputs: Iterable[JsonDict]):
    # Use watch_accessed_variables to save memory by having the tape do nothing
    # if we don't need gradients.
    with tf.GradientTape(
        watch_accessed_variables=self.config.compute_grads) as tape:
      encoded_input = self._preprocess(inputs)

      # Gathers word embeddings from BERT model embedding layer using input ids
      # of the tokens.
      input_ids = encoded_input["input_ids"]
      word_embeddings = self.model.bert.embeddings.word_embeddings
      # <tf.float32>[batch_size, num_tokens, emb_size]
      input_embs = tf.gather(word_embeddings, input_ids)

      # Scatter in any passed in embeddings.
      # <tf.float32>[batch_size, num_tokens, emb_size]
      input_embs = self.scatter_all_embeddings(inputs, input_embs)

      tape.watch(input_embs)  # Watch input_embs for gradient calculation.

      model_inputs = encoded_input.copy()
      model_inputs["input_ids"] = None
      logits, embs, attentions = self.model(model_inputs,
                                            inputs_embeds=input_embs,
                                            training=False)

      batched_outputs = {
          "input_ids": encoded_input["input_ids"],
          "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1),
          "cls_emb": embs[-1][:, 0],  # last layer, first token
          "input_embs": input_embs
      }
      assert len(attentions) == self.model.config.num_hidden_layers
      for i, layer_attention in enumerate(attentions):
        batched_outputs[f"layer_{i}/attention"] = layer_attention

      if self.is_regression:
        # <tf.float32>[batch_size]
        batched_outputs["score"] = tf.squeeze(logits, axis=-1)
        scalar_pred_for_gradients = batched_outputs["score"]
      else:
        # <tf.float32>[batch_size, num_labels]
        batched_outputs["probas"] = tf.nn.softmax(logits, axis=-1)

        # If a class for the gradients has been specified in the input,
        # calculate gradients for that class. Otherwise, calculate gradients for
        # the arg_max class.
        arg_max = tf.math.argmax(batched_outputs["probas"], axis=-1).numpy()
        grad_classes = [ex.get("grad_class", arg_max[i]) for (i, ex) in
                        enumerate(inputs)]
        # Convert the class names to indices if needed.
        grad_classes = [self.config.labels.index(label)
                        if isinstance(label, str) else label
                        for label in grad_classes]

        gather_indices = list(enumerate(grad_classes))
        # <tf.float32>[batch_size]
        scalar_pred_for_gradients = tf.gather_nd(batched_outputs["probas"],
                                                 gather_indices)
        if self.config.compute_grads:
          batched_outputs["grad_class"] = tf.convert_to_tensor(grad_classes)

    # Request gradients after the tape is run.
    # Note: embs[0] includes position and segment encodings, as well as subword
    # embeddings.
    if self.config.compute_grads:
      # <tf.float32>[batch_size, num_tokens, emb_dim]
      batched_outputs["input_emb_grad"] = tape.gradient(
          scalar_pred_for_gradients, input_embs)

    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Sequence of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self) -> Spec:
    ret = {}
    ret[self.config.text_a_name] = lit_types.TextSegment()
    if self.config.text_b_name:
      ret[self.config.text_b_name] = lit_types.TextSegment()
    if self.is_regression:
      ret[self.config.label_name] = lit_types.RegressionScore(required=False)
    else:
      ret[self.config.label_name] = lit_types.CategoryLabel(
          required=False, vocab=self.config.labels)
    # The input_embs_ and grad_class fields are used for Integrated Gradients.
    ret["input_embs_" + self.config.text_a_name] = lit_types.TokenEmbeddings(
        align="tokens", required=False)
    if self.config.text_b_name:
      ret["input_embs_" + self.config.text_b_name] = lit_types.TokenEmbeddings(
          align="tokens", required=False)
    ret["grad_class"] = lit_types.CategoryLabel(required=False,
                                                vocab=self.config.labels)
    return ret

  def output_spec(self) -> Spec:
    ret = {"tokens": lit_types.Tokens()}
    ret["tokens_" + self.config.text_a_name] = lit_types.Tokens()
    if self.config.text_b_name:
      ret["tokens_" + self.config.text_b_name] = lit_types.Tokens()
    if self.is_regression:
      ret["score"] = lit_types.RegressionScore(parent=self.config.label_name)
    else:
      ret["probas"] = lit_types.MulticlassPreds(
          parent=self.config.label_name,
          vocab=self.config.labels,
          null_idx=self.config.null_label_idx)
    ret["cls_emb"] = lit_types.Embeddings()

    # The input_embs_ and grad_class fields are used for Integrated Gradients.
    ret["input_embs_" + self.config.text_a_name] = lit_types.TokenEmbeddings(
        align="tokens_" + self.config.text_a_name)
    if self.config.text_b_name:
      ret["input_embs_" + self.config.text_b_name] = lit_types.TokenEmbeddings(
          align="tokens_" + self.config.text_b_name)

    # Gradients, if requested.
    if self.config.compute_grads:
      ret["grad_class"] = lit_types.CategoryLabel(required=False,
                                                  vocab=self.config.labels)
      ret["token_grad_" + self.config.text_a_name] = lit_types.TokenGradients(
          align="tokens_" + self.config.text_a_name,
          grad_for="input_embs_" + self.config.text_a_name,
          grad_target="grad_class")
      if self.config.text_b_name:
        ret["token_grad_" + self.config.text_b_name] = lit_types.TokenGradients(
            align="tokens_" + self.config.text_b_name,
            grad_for="input_embs_" + self.config.text_b_name,
            grad_target="grad_class")

    # Attention heads, one field for each layer.
    for i in range(self.model.config.num_hidden_layers):
      ret[f"layer_{i}/attention"] = lit_types.AttentionHeads(
          align=("tokens", "tokens"))
    return ret


class LentaModel(LentaModelGeneral):
  """Classification model on Lenta Data"""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name="sentence",
        text_b_name=None,
        labels=['????????????????', '?????????? ?? ??????????????', '??????????????????'],  #### EDIT
        null_label_idx=0,
        **kw)
