import numpy as np

class SVM:

    def __init__(self, lr=0.001, iters=1000, lambda_param=0.01):
        self.lr = lr
        self.iters = iters
        self.lambda_param = lambda_param

        self.w = None
        self.b = None

    def fit(self, X, y):

        samples, features = X.shape
        y_normalized = np.where(y <= 0, -1, 1)

        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iters):
            for i in range(samples):
                condition = y_normalized[i] * (np.dot(X[i], self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X[i], y_normalized[i]))
                    self.b -= self.lr * y_normalized[i]


    def predict(self, X):
        predicted = np.dot(X, self.w) + self.b
        return np.sign(predicted)
