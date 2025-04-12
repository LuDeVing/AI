import numpy as np


class AdaBoost:

    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clf_list = []
        self.alpha_list = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_clf):

            clf = self.create_clf(X, y, w)
            pred = clf["pred"]

            error = np.sum(w * (pred != y))
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            w = w * np.exp(-alpha * y * pred) / sum(w)

            self.clf_list.append(clf)
            self.alpha_list.append(alpha)

    def create_clf(self, X, y, w):

        clf = {"feature": None, "threshold": None, "polarity": None, "pred": None}
        n_samples, n_features = X.shape

        min_error = float("inf")

        for feature in range(n_features):
            x = X[:, feature]
            thresholds = np.unique(x)

            for threshold in thresholds:

                for polarity in [-1, 1]:

                    pred = np.ones(n_samples)
                    pred[polarity * x < polarity * threshold] = -1

                    error = np.sum(w * (pred != y))

                    if error < min_error:
                        min_error = error

                        clf["feature"] = feature
                        clf["threshold"] = threshold
                        clf["polarity"] = polarity
                        clf["pred"] = pred

        return clf

    def predict(self, X):

        final_pred = np.zeros(X.shape[0])

        for clf, alpha in zip(self.clf_list, self.alpha_list):

            pred = np.ones(X.shape[0])
            x = X[:, clf['feature']]
            pred[clf["polarity"] * x < clf["polarity"] * clf["threshold"]] = -1

            final_pred += pred * alpha

        return np.sign(final_pred)