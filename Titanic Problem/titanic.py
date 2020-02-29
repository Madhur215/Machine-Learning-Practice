# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:26:23 2020

@author: hp
"""
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
X_train = data.iloc[:, 2:].values
Y_train = data.iloc[:, 1].values
X_train = np.delete(X_train, [1, 6, 8], 1)

testData = pd.read_csv('test.csv')
X_test = testData.iloc[:, 1:].values
X_test = np.delete(X_test, [1, 6, 8], 1)

# FILLING THE MISSING VALUES
from sklearn.preprocessing import Imputer

imputer = Imputer()
X_train[:, 2:3] = imputer.fit_transform(X_train[:, 2:3])

testImputer = Imputer()
X_test[:, 2:3] = imputer.fit_transform(X_test[:, 2:3])

# CONVERTING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder = LabelEncoder()
X_train[:, 1] = labelEncoder.fit_transform(X_train[:, 1])
X_train[:, 6] = labelEncoder.fit_transform(X_train[:, 6])
hotEncoder = OneHotEncoder(categorical_features=[1, 6])
X_train = hotEncoder.fit_transform(X_train).toarray()

testLabelEncoder = LabelEncoder()
X_test[:, 1] = testLabelEncoder.fit_transform(X_test[:, 1])
X_test[:, 6] = testLabelEncoder.fit_transform(X_test[:, 6])
testHotEncoder = OneHotEncoder(categorical_features=[1, 6])
X_test = testHotEncoder.fit_transform(X_test).toarray()

"""
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
sc_new = mm_scaler.fit_transform(X_train)

mm_sc = MinMaxScaler()
sc_new_y = mm_sc.fit_transform(X_test)
"""

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)

# IMPLEMENTING LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=90)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier

clsf = KNeighborsClassifier()
clsf.fit(X_train, Y_train)

y_pred2 = clsf.predict(X_test)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)

y_pred4 = nb_clf.predict(X_test)

# SVM
from sklearn.svm import SVC

svc_clf = SVC(kernel="linear")
svc_clf.fit(X_train, Y_train)

y_pred3 = svc_clf.predict(X_test)
