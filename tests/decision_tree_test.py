# Step 1: Import necessary libraries
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Decision_Tree import DecisionTree


class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.clf = DecisionTree()

    def test_decision_tree_accuracy(self):
        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(y_pred)
        print(self.y_test)
        self.assertGreater(accuracy, 0.8, "Accuracy should be greater than 80%")
        print(accuracy)


if __name__ == '__main__':
    unittest.main()
