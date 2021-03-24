# Lint as: python3
r"""Quick-start demo for a sentence classification model.

This demo fine-tunes a Transformer (RuBERT) on the Lenta dataset,
and starts a LIT server.

To run locally:
  python -m lit_nlp.examples.quickstart_lenta --port=5433 (check if the port not used!!!)

Training should take less than 30 minutes on a single GPU 8Gb. Once you see the
ASCII-art LIT logo, navigate to localhost:5433 to access the demo UI.
"""
import tempfile

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import lenta  #### EDIT
from lit_nlp.examples.models import lenta_models  #### EDIT

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "encoder_name", "DeepPavlov/rubert-base-cased",  #### EDIT
    "Encoder name to use for fine-tuning. See https://huggingface.co/DeepPavlov/rubert-base-cased.")

flags.DEFINE_string("model_path", None, "Path to save trained model.")


def run_finetuning(train_path):
  """Fine-tune a transformer model."""
  train_data = lenta.LentaData("train")  #### EDIT
  val_data = lenta.LentaData("validation")  #### EDIT
  model = lenta_models.LentaModel(FLAGS.encoder_name, for_training=True)  #### EDIT
  model.train(train_data.examples, validation_inputs=val_data.examples)
  model.save(train_path)


def main(_):
  model_path = FLAGS.model_path or tempfile.mkdtemp()
  logging.info("Working directory: %s", model_path)
  run_finetuning(model_path)

  # Load our trained model.
  models = {"lenta_model": lenta_models.LentaModel(model_path)}  #### EDIT
  datasets = {"lenta_data": lenta.LentaData("validation")}  #### EDIT

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
