import numpy as np


class PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):

        self.mean = np.mean(X, axis=0)
        X_transformed = X - self.mean

        eigen_values, eigen_vectors = np.linalg.eig(np.cov(X_transformed.T))

        sorted_eigen_values_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_vectors = eigen_vectors[:, sorted_eigen_values_indices]

        if self.n_components is not None:
            sorted_eigen_vectors = sorted_eigen_vectors[:, :self.n_components]

        self.components = sorted_eigen_vectors

    def predict(self, X):
        X_transformed = X - self.mean
        return np.dot(X_transformed, self.components)
