import numpy as np


class Node:

    def __init__(self):
        self.l = None
        self.r = None
        self.predicate = None
        self.is_leaf = False
        self.output = None


class DecisionTree:

    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def gini_impurity(y):
        class_counts = np.bincount(y)
        proportions = class_counts / len(y)
        return 1 - np.sum(proportions ** 2)

    def fit(self, X, y):

        self.root = self.build_tree(X, y, 0)

    def build_tree(self, X, y, depth):

        impurity = self.gini_impurity(y)

        if depth > self.max_depth or impurity == 0:
            node = Node()
            node.is_leaf = True
            node.output = np.bincount(y).argmax()
            return node

        designator_gini = None
        designator_idx = None
        designator_number = None

        for idx, x in enumerate(X.T):

            if len(set(x)) == 1:
                continue

            sorted_pairs = sorted(zip(x, y), key=lambda pair: pair[0])
            sorted_x, sorted_y = zip(*sorted_pairs)

            sorted_x = list(sorted_x)
            sorted_y = list(sorted_y)

            best_gini = None
            best_des = None

            for i in range(1, len(sorted(sorted_x))):

                if i != 1 and sorted_x[i] == sorted_x[i - 1]:
                    continue

                left_y = sorted_y[:i]
                right_y = sorted_y[i:]

                left_weight = len(left_y) / len(y)
                right_weight = len(right_y) / len(y)

                current_gini = left_weight * self.gini_impurity(left_y) + right_weight * self.gini_impurity(right_y)

                if best_gini is None or current_gini < best_gini:
                    best_gini = current_gini
                    best_des = sorted_x[i]

            if designator_gini is None or best_gini < designator_gini:
                designator_gini = best_gini
                designator_idx = idx
                designator_number = best_des

        X1, y1, X2, y2 = self.split_X_y_by_column(X, y, designator_idx, designator_number)

        node = Node()

        node.predicate = (lambda input: input[designator_idx] < designator_number)

        node.l = self.build_tree(X1, y1, depth + 1)
        node.r = self.build_tree(X2, y2, depth + 1)

        return node

    @staticmethod
    def find_value(x, node):

        if node.is_leaf:
            return node.output

        if node.predicate(x):
            return DecisionTree.find_value(x, node.l)
        else:
            return DecisionTree.find_value(x, node.r)

    def predict(self, X):
        return np.array([self.find_value(x, self.root) for x in X])

    @staticmethod
    def split_X_y_by_column(X, y, idx, best_des):
        X1, y1 = zip(*[(row, y[i]) for i, row in enumerate(X) if row[idx] < best_des])
        X2, y2 = zip(*[(row, y[i]) for i, row in enumerate(X) if row[idx] >= best_des])

        X1 = np.array(X1)
        y1 = np.array(y1)
        X2 = np.array(X2)
        y2 = np.array(y2)

        return X1, y1, X2, y2
