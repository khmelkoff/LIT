# Lint as: python3
"""GLUE benchmark datasets, using TFDS.

See https://gluebenchmark.com/ and
https://www.tensorflow.org/datasets/catalog/glue

Note that this requires the TensorFlow Datasets package, but the resulting LIT
datasets just contain regular Python/NumPy data.
"""
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import pandas as pd

def load_data(data_path, split='train'):
    '''Data Loader from pandas dataset
       Return list of dicts: {'idx': (int), 'label': (str), 'sentence': (str)}
    '''

    df = pd.read_csv(data_path, sep=';')
    # max validation size for lit 10k points
    margin = int(df.shape[0] * 0.95)

    if split == 'train':
        df = df.iloc[:margin, :]

    else:
        df = df.iloc[margin:, :]

    return [{'idx': row[0], 'label': row[1]['topic'], 'sentence': row[1]['text']}
            for row in df.iterrows()]



class LentaData(lit_dataset.Dataset):  #### EDIT
  """Lenta news loader
  """

  LABELS = ['Культура', 'Наука и техника', 'Экономика']  #### EDIT

  def __init__(self, split: str):
    self._examples = []
    for ex in load_data('D:/news.csv', split=split):  #### EDIT
      self._examples.append({
          'sentence': ex['sentence'],
          'label': ex['label'],
      })

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }