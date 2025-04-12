import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from scipy.stats import mode

from AdaBoost import AdaBoost
from Decision_Tree import DecisionTree
from Naive_Bayes import NaiveBayes
from K_Means_Clustering import KMeansClustering
from Linear_Regression import LinearRegression
from Random_Forest import RandomForest


def generate_data():

    iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    iris_data = pd.read_csv(iris_url, names=attributes)

    y = iris_data["class"].tolist()
    y = preprocessing.LabelEncoder().fit_transform(y)
    X = iris_data.drop(columns="class")
    X = np.array(X.values.tolist())

    return train_test_split(X, y, test_size=0.33, random_state=42)


def calculate_accuracy(prediction, y):
    return np.sum(prediction == y) / len(prediction)


import numpy as np
from scipy.stats import mode


def map_clusters_to_labels(kmeans_labels, true_labels):
    true_labels = np.array(true_labels)
    label_mapping = {}

    for cluster in np.unique(kmeans_labels):
        mask = (kmeans_labels == cluster)
        cluster_true_labels = np.atleast_1d(true_labels[mask])

        m = mode(cluster_true_labels)
        if np.isscalar(m.mode):
            most_common = m.mode
        else:
            most_common = m.mode[0]
        label_mapping[cluster] = most_common

    mapped_labels = np.array([label_mapping[label] for label in kmeans_labels])
    return mapped_labels


def calculate_kmeans_accuracy():
    k_means_model = KMeansClustering(3, 150)
    prediction = np.array(k_means_model.predict(X_test))
    prediction = map_clusters_to_labels(prediction, y_test)
    acc = calculate_accuracy(prediction, y_test)
    return acc


def calculate_naive_bayes_accuracy():
    naive_bayes_model = NaiveBayes()
    naive_bayes_model.fit(X_train, y_train)
    prediction = naive_bayes_model.predict(X_test)
    return calculate_accuracy(prediction, y_test)


def calculate_decision_tree_accuracy():
    model = DecisionTree()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return calculate_accuracy(prediction, y_test)


def calculate_random_forest_accuracy():
    model = RandomForest()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return calculate_accuracy(prediction, y_test)


X_train, X_test, y_train, y_test = generate_data()

kmeans_accuracy = calculate_kmeans_accuracy()
naive_bayes_accuracy = calculate_naive_bayes_accuracy()
decision_tree_accuracy = calculate_decision_tree_accuracy()
random_forest_accuracy = calculate_random_forest_accuracy()

print(f"kmeans_accuracy: {kmeans_accuracy}")
print(f"naive_bayes_accuracy: {naive_bayes_accuracy}")
print(f"decision_tree_accuracy: {decision_tree_accuracy}")
print(f"random_forest_accuracy: {random_forest_accuracy}")
