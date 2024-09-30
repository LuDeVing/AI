import numpy as np


class MultinomialNaiveBayes:

    def __init__(self):
        self.classes = None
        self.priors = None
        self.feature_probs = None
        self.class_to_index = None

    def fit(self, X, y):
        samples, features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.feature_probs = np.zeros((num_classes, features), dtype=np.float64)
        self.priors = np.zeros(num_classes, dtype=np.float64)
        self.class_to_index = {c: idx for idx, c in enumerate(self.classes)}

        for c in self.classes:
            idx = self.class_to_index[c]
            X_c = X[y == c]

            features_count = np.sum(X_c, axis=0) + 1
            total_count = np.sum(features_count)
            self.feature_probs[idx] = features_count / float(total_count)

            self.priors[idx] = X_c.shape[0] / float(samples)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        log_probs = []

        for idx in range(len(self.classes)):
            prior = np.log(self.priors[idx] + 1e-10)
            likelihood = np.sum(np.log((self.feature_probs[idx]) * x + 1e-10))
            log_probs.append(prior + likelihood)

        return self.classes[np.argmax(log_probs)]
