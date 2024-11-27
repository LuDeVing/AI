from collections import Counter

import numpy as np
from Decision_Tree import DecisionTree


class RandomForest:

    def __init__(self, n_trees=10, max_depth=100):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = None

    def fit(self, X, y):

        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):

            tree = DecisionTree(self.max_depth)
            ids = np.random.choice(n_samples, size=n_samples, replace=True)

            X_bootstrap = X[ids]
            y_bootstrap = y[ids]

            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = predictions.T
        most_common_predictions = [Counter(prediction).most_common(1)[0][0] for prediction in predictions]
        return np.array(most_common_predictions)
