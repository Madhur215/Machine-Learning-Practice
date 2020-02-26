from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Social_Network_Ads.csv')
df = df.drop(['User ID'] , axis=1)
# print(df)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

label = LabelEncoder()
X[:, 0] = label.fit_transform(X[:, 0])
hotEncoder = OneHotEncoder(categorical_features=[0])
X = hotEncoder.fit_transform(X).toarray()

sc_x = MinMaxScaler()
X = sc_x.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=99)


classifier = GaussianNB()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

print(y_pred)


matrix = confusion_matrix(Y_test, y_pred)
print(matrix)
print(56/60)






