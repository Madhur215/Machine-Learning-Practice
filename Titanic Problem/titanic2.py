import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')


for family in data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
for family in testData:
    testData['FamilySize'] = testData['SibSp'] + testData['Parch'] + 1
    


data['Title'] = data.Name.str.extract('([A-Za-z]+)\.' , expand = False)
testData['Title'] = testData.Name.str.extract('([A-Za-z]+)\.', expand = False)



#for dt1 in data:
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')


testData['Title'] = testData['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
testData['Title'] = testData['Title'].replace('Mlle', 'Miss')
testData['Title'] = testData['Title'].replace('Ms', 'Miss')
testData['Title'] = testData['Title'].replace('Mme', 'Mrs')


# data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived')

data['Title'] = data['Title'].map({"Mr" : 1, "Mrs": 2, "Master": 3, "Miss": 4, "Rare": 5})
testData['Title'] = testData['Title'].map({"Mr" : 1, "Mrs": 2, "Master": 3, "Miss": 4, "Rare": 5})


data["Embarked"].fillna('S', inplace=True)
testData["Embarked"].fillna('S', inplace=True)


data['Embarked'] = data['Embarked'].map({'S': 1, 'Q': 2, 'C': 3}).astype(int)
testData['Embarked'] = testData['Embarked'].map({'S': 1, 'Q': 2, 'C': 3}).astype(int)


data['FamilyType'] = 0
data.loc[data['FamilySize'] == 1, 'FamilyType'] = 1
data.loc[data['FamilySize'] > 5, 'FamilyType'] = 2
    
testData['FamilyType'] = 0
testData.loc[testData['FamilySize'] == 1, 'FamilyType'] = 1
testData.loc[testData['FamilySize'] > 5, 'FamilyType'] = 2
    
    
data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
testData['Sex'] = testData['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


data.loc[data['Age'] <= 14 , 'Age'] = 0
data.loc[(data['Age'] > 14) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[data['Age'] > 64, 'Age' ] = 4

        
testData.loc[data['Age'] <= 14 , 'Age'] = 0
testData.loc[(data['Age'] > 14) & (data['Age'] <= 32), 'Age'] = 1
testData.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
testData.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
testData.loc[data['Age'] > 64, 'Age' ] = 4


data.loc[data['Fare'] <= 10, 'Fare'] = 0
data.loc[(data['Fare'] > 10) & (data['Fare'] <= 30), 'Fare'] = 1
data.loc[(data['Fare'] > 30) & (data['Fare'] < 60), 'Fare'] = 2
data.loc[data['Fare'] > 60, 'Fare'] = 3


testData.loc[testData['Fare'] <= 10, 'Fare'] = 0
testData.loc[(testData['Fare'] > 10) & (data['Fare'] <= 30), 'Fare'] = 1
testData.loc[(testData['Fare'] > 30) & (data['Fare'] < 60), 'Fare'] = 2
testData.loc[testData['Fare'] > 60, 'Fare'] = 3

del data['PassengerId']
#del testData['PassengerId']

del data['Name']
del testData['Name']

del data['Parch']
del testData['Parch']

del data['SibSp']
del testData['SibSp']

del data['Ticket']
del testData['Ticket']

del data['Cabin']
del testData['Cabin']

del data['FamilySize']
del testData['FamilySize']

X_train = data.iloc[:, 1:].values
Y_train = data.iloc[:, 0].values
X_test = testData.iloc[:, 1:].values


from sklearn.preprocessing import Imputer
imputer = Imputer()
X_train[:, 2:3] = imputer.fit_transform(X_train[:, 2:3])
testImputer = Imputer()
X_test[:, 2:3] = imputer.fit_transform(X_test[:, 2:3])


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 90)
classifier.fit(X_train , Y_train)


from sklearn.neighbors import KNeighborsClassifier

clsf = KNeighborsClassifier()
clsf.fit(X_train, Y_train)

y_pred2 = clsf.predict(X_test)



from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)

y_pred4 = nb_clf.predict(X_test)

"""

from sklearn.svm import SVC

svc_clf = SVC(kernel="linear")
svc_clf.fit(X_train, Y_train)

y_pred3 = svc_clf.predict(X_test)

#y_pred = classifier.predict(X_test)

"""
submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": y_pred3
}) 
"""
result_train = svc_clf.score(X_train, Y_train)
# submission.to_csv('titanic_new_SVM.csv', index=False)










