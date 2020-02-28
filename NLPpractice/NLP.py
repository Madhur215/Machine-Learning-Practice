import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  confusion_matrix
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import GaussianNB
import nltk


data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# print(data)
Y = data.iloc[:, 1].values
data['Review'] = data.Review.map(lambda x: x.lower())
# data['Review'] = map(lambda x: x.lower(), data['Review'])
data['Review'] = data.Review.str.replace('[^\w\s]', '')
data['Review'] = data.Review.apply(nltk.word_tokenize)
stemmer = PorterStemmer()
data['Review'] = data['Review'].apply(lambda b: [stemmer.stem(a) for a in b])
data['Review'] = data['Review'].apply(lambda b: ' '.join(b))
countVect = CountVectorizer(max_features=1500)
X = countVect.fit_transform(data['Review']).toarray()

sc_x = MinMaxScaler()
X = sc_x.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.10, random_state=99)

nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred = nb.predict(X_test)
print(Y_pred)

print("Confusion Matrix: ")
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
print((26+46)/(26+46+7+21))


