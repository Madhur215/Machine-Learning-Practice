# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:01:23 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run():
    data = pd.read_csv('finalData.csv')
    testData = pd.read_csv('Data.csv')
  #  print(data)
 #   print(np.ndim(data))
    a = testData.iloc[: , :-1].values
    b = testData.iloc[: , 1].values
     
    x = data.iloc[: , :-1].values
    y = data.iloc[: , 2].values
    
    x_transpose = x.transpose()
    X = np.dot(x_transpose , x)
    nxt = np.linalg.inv(X)
    Y = np.dot(x_transpose , y)
    
    optimal_values = np.dot(nxt , Y)
    m = optimal_values[1]
    n = optimal_values[0]
    
    from sklearn.model_selection import train_test_split
    X_train , X_test ,Y_train, Y_test = train_test_split(a , b , test_size = 0.2 , 
                                                           random_state = 12)
    
    y_pred = [0 for i in range(len(X_test))]
    
    for i in range(0,len(X_test)):
        y_pred[i] = m * X_test[i] + n
    
    plt.scatter(X_test , Y_test , color = 'red')
    plt.plot(X_test , y_pred , color = 'blue')
    plt.title("Normal Equation")
    plt.show()
    

run()
