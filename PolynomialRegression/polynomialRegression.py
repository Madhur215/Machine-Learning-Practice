# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:35:58 2020

@author: hp
"""

import numpy as mp
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[: , 1:2].values
Y = data.iloc[: , -1].values


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
pw = PolynomialFeatures(4)
X_poly = pw.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly , Y)


plt.scatter(X , Y , color = 'green')
plt.plot(X , regressor.predict(pw.fit_transform(X)) , color = 'blue')
plt.title('Polynomial Regression'   )
plt.show()