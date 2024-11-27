from Random_Forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


data = datasets.load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

clf = RandomForest()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)
