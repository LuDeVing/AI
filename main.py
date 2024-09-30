from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from SVM_Kernel import SVM_Kernel

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0,
                           random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVM_Kernel(C=5.0, model='rbf', gamma=2.0)
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
print(predictions)
print(y_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")


def plot_decision_boundary(svm, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', label='Train')

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


plot_decision_boundary(svm, X, y)
