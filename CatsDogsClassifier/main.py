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
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(output_dim=128, activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

X_train = train_data_gen.flow_from_directory('training_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

X_test = train_data_gen.flow_from_directory('test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

model.fit_generator(X_train,
                    samples_per_epoch=8000,
                    nb_epoch=25,
                    validation_data=X_test,
                    nb_val_samples=2000)
