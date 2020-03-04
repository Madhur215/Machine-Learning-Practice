# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:08:19 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
x_train ,x_test ,  y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)


from sklearn.externals import joblib
knn = joblib.load('saved_model.pkl')


y_pred = knn.predict(x_test)