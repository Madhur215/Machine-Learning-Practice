# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:03:43 2020

@author: hp
"""

import numpy as np
import pandas as pd 


data = pd.read_csv('Social_network_Ads.csv')
X = data.iloc[: , [2,3]].values
Y = data.iloc[: , -1].values


from sklearn.model_selection import train_test_split
X_test , X_train , Y_test , Y_train = train_test_split(X , Y , test_size = 0.9 , 
                                                       random_state = 99)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 90)
classifier.fit(X_train , Y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(Y_test , y_pred)


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

X_set , Y_set = X_train , Y_train
x1 , x2 = np.meshgrid(np.arrange(start = X_set[:,0].min() - 1 , stop = X_set[:,0].max() + 1
                                 , step = 0.01) , 
                        np.arrange(start = X_set[:,1].min() - 1 , stop = X_set[:,1].max() + 1
                                 , step = 0.01)
#plt.contourf(x1 , x2 , classifier.predict(np.array().T).reshape(x1.shape),
#             alpha = 0.75 , cmap = ListedColormap(('red' , 'green')))   

plt.contourf(x1 , x2 , classifier.predict(np.array([x1.ravel() , x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red' , 'green' )))


plt.xlim(x1.min() , x1.max())
plt.ylim(x2.min() , x2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j , 0] , X_set[Y_set == j , 1],
                c = ListedColormap(('red' , 'green'))(i) , label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()































