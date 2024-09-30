import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iters):

            linear_prediction = np.dot(X, self.weights) + self.bias
            logistic_prediction = 1 / (1 + np.exp(-linear_prediction))

            dw = (1 / samples) * np.dot(X.T, logistic_prediction - y)
            db = (1 / samples) * np.sum(logistic_prediction - y)

            self.weights -= dw * self.lr
            self.bias -= db + self.lr

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        logistic_prediction = 1 / (1 + np.exp(-linear_prediction))
        return np.where(logistic_prediction >= 0.5, 1, 0)
