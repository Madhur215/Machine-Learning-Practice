# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:23:05 2020

@author: Madhur Vashistha
"""

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Flatten


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_dim=128, activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

