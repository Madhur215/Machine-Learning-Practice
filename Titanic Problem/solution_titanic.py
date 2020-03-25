import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import random

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
Y_train = data.iloc[:, 1].values

data = data.drop(['Cabin', 'Ticket', 'PassengerId', 'Survived'], axis=1)
test_data = test_data.drop(['Cabin', 'Ticket'], axis=1)

# Mapping 'Sex' as 0 for male and 1 for female
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1}).astype(int)

# Removing the brackets and their content in 'Name'
for i in range(len(data['Name'])):
    name = ""
    for j in data['Name'][i]:
        if j != '(':
            name += j
        else:
            break
    data['Name'][i] = name

# Filling the missing values in 'Embarked'
vals = ['S', 'Q', 'C']
data['Embarked'].fillna(random.choice(vals), inplace=True)
test_data['Embarked'].fillna(random.choice(vals), inplace=True)

# Mapping S as 1 , Q as 2 and C as 3 in 'Embarked'
data['Embarked'] = data['Embarked'].map({'S': 1, 'Q': 2, 'C': 3}).astype(int)
test_data['Embarked'] = test_data['Embarked'].map({'S': 1, 'Q': 2, 'C': 3}).astype(int)

# Filling the missing values in 'Age'
data['Age'].fillna(data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# Calculating total family size from 'SibSp' and 'Parch'
for i in range(len(data['SibSp'])):
    data['FamilySize'] = data['SibSp'] + data['Parch']

for i in range(len(test_data['SibSp'])):
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

data = data.drop(['SibSp', 'Parch'], axis=1)
test_data = test_data.drop(['SibSp', 'Parch'], axis=1)

# Getting title from 'Name'
data['Title'] = data.Name.str.extract('([Ma-z]+)\.', expand=False)
test_data['Title'] = test_data.Name.str.extract('([Ma-z]+)\.', expand=False)

data = data.drop(['Name'], axis=1)
test_data = test_data.drop(['Name'], axis=1)

data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

data['Title'] = data['Title'].map({"Mr": 1, "Mrs": 2, "Master": 3, "Miss": 4, "Rare": 5})
test_data['Title'] = test_data['Title'].map({"Mr": 1, "Mrs": 2, "Master": 3, "Miss": 4, "Rare": 5})

data['Title'].fillna(random.choice([1, 2, 3, 4, 5]), inplace=True)
test_data['Title'].fillna(random.choice([1, 2, 3, 4, 5]), inplace=True)

X_train = data.iloc[:, :].values
X_test = test_data.iloc[:, 1:].values


scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

svc = SVC(kernel="linear")
svc.fit(X_train, Y_train)
y_pred2 = svc.predict(X_test)

acc = cross_val_score(X=X_train, y=Y_train, cv=10)
print(acc)

# gradient = GradientBoostingClassifier()
# gradient.fit(X_train, Y_train)
# y_pred = gradient.predict(X_test)
#
# print(y_pred)
#
# submission = pd.DataFrame({
#         "PassengerId": test_data["PassengerId"],
#         "Survived": y_pred
# })
#
# submission.to_csv('solution.csv', index=False)
