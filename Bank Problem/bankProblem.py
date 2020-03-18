from tensorflow import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13].values
Y = data.iloc[:, 13].values

label_encoder_1 = LabelEncoder()
X[:, 1] = label_encoder_1.fit_transform(X[:, 1])
label_encoder_2 = LabelEncoder()
X[:, 2] = label_encoder_2.fit_transform(X[:, 2])

hotEncoder = OneHotEncoder(categorical_features=[1])
X = hotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=99)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# model = keras.Sequential([
#         keras.layers.Dense(11, activation='relu'),
#         keras.layers.Dense(6, activation='relu'),
#         keras.layers.Dense(1, activation='softmax')
#     ])

model = Sequential()
model.add(Dense(input_dim=11, output_dim=6, init= 'uniform', activation='relu'))
model.add(Dense(output_dim=6, init= 'uniform', activation='relu'))
model.add(Dense(output_dim=1, init= 'uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs= 25)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


cm = confusion_matrix(Y_test, y_pred)
print(cm)