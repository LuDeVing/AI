import numpy as np


class NaiveBayes:

    def __init__(self):
        self.classes = None
        self.priors = None
        self.var = None
        self.mean = None
        self.class_to_index = None

    def fit(self, X, y):
        samples, features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.mean = np.zeros((num_classes, features), dtype=np.float64)
        self.var = np.zeros((num_classes, features), dtype=np.float64)
        self.priors = np.zeros(num_classes, dtype=np.float64)
        self.class_to_index = {c: idx for idx, c in enumerate(self.classes)}

        for c in self.classes:
            idx = self.class_to_index[c]
            X_c = X[y == c]

            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)

            self.priors[idx] = X_c.shape[0] / float(samples)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        log_probs = []

        for idx in range(len(self.classes)):
            prior = np.log(self.priors[idx] + 1e-10)
            likelihood = np.sum(np.log(self.gaussian_pdf(x, self.mean[idx], self.var[idx]) + 1e-10))
            log_probs.append(prior + likelihood)

        return self.classes[np.argmax(log_probs)]

    @staticmethod
    def gaussian_pdf(x, mean, var):
        var = np.where(var == 0, 1e-10, var)
        coefficient = 1 / (np.sqrt(2 * np.pi * var))
        exponent = -((x - mean) ** 2) / (2 * var)
        return coefficient * np.exp(exponent)
