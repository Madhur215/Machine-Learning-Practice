import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv('Wine.csv')
X = data.iloc[:, 0:13].values
Y = data.iloc[:, 13].values

sc = StandardScaler()
X = sc.fit_transform(X)
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_tran = X.T
# covariance_matrix = np.matmul(X_tran, X)
# print(covariance_matrix.shape)
# 
# e_values, e_vectors = eigh(covariance_matrix, eigvals=(11, 12))
# print(e_vectors.shape)
# print(e_values.shape)

pca = PCA(n_components=2)
X = pca.fit_transform(X)
variance = pca.explained_variance_ratio_
print(variance)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=98)

classifier = LogisticRegression(random_state=12)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
print(cm)
