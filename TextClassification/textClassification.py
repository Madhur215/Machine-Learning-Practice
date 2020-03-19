# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:17:55 2020

@author: hp
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words= 10000)



word_ind = data.get_word_index()

word_ind = {k:(v+3) for k, v in word_ind.items()}
word_ind["<PAD>"] = 0
word_ind["<START>"] = 1
word_ind["<UNK>"] = 2
word_ind["<UNUSED>"] = 3

reverse_ind = dict([(value, key) for (key, value)])



