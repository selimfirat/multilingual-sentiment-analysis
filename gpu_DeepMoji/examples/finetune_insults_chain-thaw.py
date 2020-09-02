"""Finetuning example.

Trains the DeepMoji model on the kaggle insults dataset, using the 'chain-thaw'
finetuning method and the accuracy metric. See the blog post at
https://medium.com/@bjarkefelbo/what-can-we-learn-from-emojis-6beb165a5ea0
for more information. Note that results may differ a bit due to slight
changes in preprocessing and train/val/test split.

The 'chain-thaw' method does the following:
0) Load all weights except for the softmax layer. Extend the embedding layer if
   necessary, initialising the new weights with random values.
1) Freeze every layer except the last (softmax) layer and train it.
2) Freeze every layer except the first layer and train it.
3) Freeze every layer except the second etc., until the second last layer.
4) Unfreeze all layers and train entire model.
"""

from __future__ import print_function
import example_helper
import json
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH
from deepmoji.finetuning import (
    load_benchmark,
    finetune)

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

DATASET_PATH = '../data/kaggle-insults/raw.pickle'
nb_classes = 2

with open('/home/turing/SSD_Main/batu/sentiment-dml/DeepMoji/model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

# Load dataset. Extend the existing vocabulary with up to 10000 tokens from
# the training dataset.
data = load_benchmark(DATASET_PATH, vocab, extend_with=10000)

# Set up model and finetune. Note that we have to extend the embedding layer
# with the number of tokens added to the vocabulary.
model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH,
                          extend_embedding=data['added'])
model.summary()
model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='chain-thaw')
print('Acc: {}'.format(acc))
