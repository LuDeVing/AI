import numpy as np


class LinearRegression:

    def __init__(self, iters=1000, learning_rate=0.001):
        self.iters = iters
        self.lr = learning_rate

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.random.rand(features)
        self.bias = np.random.rand()

        for _ in range(self.iters):
            prediction = np.dot(X, self.weights) + self.bias

            dw = (2 / samples) * np.dot(X.T, (prediction - y))
            db = (2 / samples) * np.sum(prediction - y)

            print(dw)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias