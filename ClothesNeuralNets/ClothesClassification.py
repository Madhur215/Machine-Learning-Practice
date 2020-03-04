import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#plt.imshow(train_images[7])
# print(train_images)
#print(train_labels)

names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
        ])

model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# loss, acc = model.evaluate(test_images, test_labels)
# print(acc)

from sklearn.externals import joblib

joblib.dump(model, 'saved_model.pkl')


pred = model.predict(test_images)

for i in range(7):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: " + names[test_labels[i]])
    plt.title("Predicted: " + names[np.argmax(pred[i])])
    plt.show()
    

