import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
y_train = data.iloc[:, 0].values
x_train = data.iloc[:, 1:].values

test_data = pd.read_csv('test.csv')
x_test = test_data.iloc[:, :].values

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for tr in range(len(x_train)):
	for j in range(len(x_train[0])):
		if x_train[tr][j] != 0:
			x_train[tr][j] = 1




# print(x_train[0])
# img = x_train[0]
# img = np.array(img)
# img = img.reshape(28,28)
# plt.imshow(img)

model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])

model.fit(x_train, y_train, epochs=5)

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred[1])

pred = []

for i in range(len(y_pred)):
	pred.append(np.argmax(y_pred[i]))

for i in range(5):
	print(pred[i])


imageId = []

for i in range(1, 28001):
	imageId.append(i)

submission = pd.DataFrame({
        "ImageId": imageId,
        "Label": pred
})

submission.to_csv('solution.csv', index=False)

