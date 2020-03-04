# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 01:28:07 2020

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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

from sklearn.externals import joblib
joblib.dump(regressor, 'saved_model.pkl')


y_pred = regressor.predict(x_test)

plt.scatter(x_train , y_train , color = 'blue')
plt.plot(x_train , regressor.predict(x_train) , color = "red")
plt.title("Regression line")
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test , y_test , color = "red")
plt.plot(x_train , regressor.predict(x_train) , color = "blue")
plt.title("Test sets")
plt.show()

