# Step 1: Import necessary libraries
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Decision_Tree import DecisionTree

# Step 2: Create a test class inheriting from unittest.TestCase
class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        # Generate a synthetic dataset using sklearn's make_classification
        self.X, self.y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=42)
        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Initialize the DecisionTreeClassifier
        self.clf = DecisionTree()

    def test_decision_tree_accuracy(self):
        # Step 3: Train the model on the training set
        self.clf.fit(self.X_train, self.y_train)
        # Make predictions on the test set
        y_pred = self.clf.predict(self.X_test)
        # Step 4: Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(y_pred)
        print(self.y_test)
        # Check if accuracy is greater than 80%
        self.assertGreater(accuracy, 0.8, "Accuracy should be greater than 80%")
        print(accuracy)

# Step 5: Run the test
if __name__ == '__main__':
    unittest.main()
