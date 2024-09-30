import numpy as np


class Perceptron:

    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    @staticmethod
    def activation_function(z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iters):

            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = [self.activation_function(z) for z in linear_model]

            self.weights += self.lr * np.dot(X.T, (y - y_predicted))
            self.bias += self.lr * np.sum(y - y_predicted)

    def predict(self, X):
        return [self.activation_function(z) for z in np.dot(X, self.weights) + self.bias]
