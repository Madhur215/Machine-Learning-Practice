# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:43:31 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
X = data.iloc[: , :-1].values
Y = data.iloc[: , -1].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelEncoder = LabelEncoder()
X[: , 3] = labelEncoder.fit_transform(X[: , 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2
                                                                  , random_state = 90) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

y_pred = regressor.predict(X_test)

acc = regressor.score(X_test , Y_test)
print(acc)



import statsmodels.formula.api as sm
X_new = np.append( arr = np.ones((len(X), 1)).astype(int) , values = X , axis = 1)
X_new = X[: , 1:]
Xt = X[: , [0 , 1 , 2 , 3 , 4 ]]
reg = sm.OLS(Y , Xt).fit()
reg.summary()

"""
xt = X

xt = np.delete(xt , 2 , 1)
xt = np.delete(xt , 1 , 1)
xt = np.delete(xt , 2 , 1)

from sklearn.model_selection import train_test_split
xt_train , xt_test , yt_train , yt_test = train_test_split(xt , Y , test_size = 0.2 , random_state = 99)





rg = LinearRegression()
rg.fit(xt_train , yt_train)

print(rg.score(xt_test , yt_test))

"""

import statsmodels.formula.api as sm



SL = 0.05
X_opt = X[: , [0 , 1 , 2 , 3 , 4 ]]



x_2 = X_opt
vr = len(X[0])
for i in range(0 , vr):
    reg = sm.OLS(Y , x_2).fit()
    mx = max(reg.pvalues).astype(float)
    if mx > 0.05:
        for j in range(0 , vr-i):
            if(reg.pvalues[j].astype(float) == mx):
                x_2 = np.delete(x_2 , j , 1)
reg.summary()
    
                

