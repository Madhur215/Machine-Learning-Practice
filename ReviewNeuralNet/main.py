# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:46:28 2020

@author: hp
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

"""
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
Y = data.iloc[:,1].values

data['Review'] = data.Review.map(lambda x: x.lower())
data['Review'] = data.Review.str.replace('[^\w\s]', '')
data['Review'] = data.Review.apply(nltk.word_tokenize)

data['Review'] = data.Review.apply(lambda b:[stemmer.stem(a) for a in b])
data['Review'] = data.Review.apply(lambda b: ''.join(b))


cv = CountVectorizer(max_features=15000)
X = cv.fit_transform(data['Review']).toarray()
"""

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# print(data)
Y = data.iloc[:, 1].values
data['Review'] = data.Review.map(lambda x: x.lower())
# data['Review'] = map(lambda x: x.lower(), data['Review'])
data['Review'] = data.Review.str.replace('[^\w\s]', '')
data['Review'] = data.Review.apply(nltk.word_tokenize)
stemmer = PorterStemmer()
data['Review'] = data['Review'].apply(lambda b: [stemmer.stem(a) for a in b])
data['Review'] = data['Review'].apply(lambda b: ' '.join(b))
countVect = CountVectorizer(max_features=1500)
X = countVect.fit_transform(data['Review']).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state= 99)

model = Sequential()
model.add(Dense(input_dim=len(X[0]), output_dim= 128, init= 'uniform', activation= 'relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(64, init='uniform', activation='relu'))
model.add(Dense(32, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs= 100)

y_pred = model.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        
matrix = confusion_matrix(Y_test, y_pred)
print(matrix)