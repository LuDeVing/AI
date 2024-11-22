import numpy as np


class SVM_Kernel:

    def __init__(self, C=100.0, tol=1e-3, max_iter=1000, model="linear", gamma=1):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.w = None
        self.b = None
        self.model = model
        self.gamma = gamma

    def decision_function(self, x):
        if self.model == "linear":
            return np.dot(x, self.w) + self.b
        elif self.model == "rbf":
            return (self.alpha * self.y) @ self.kernel(self.X, x) + self.b

    def kernel(self, x1, x2):
        if self.model == "linear":
            return np.dot(x1, x2.T)
        if self.model == "rbf":
            if np.ndim(x1) == 1 and np.ndim(x2) == 1:
                return np.exp(-(np.linalg.norm(x1 - x2, 2)) ** 2 / (2 * self.gamma ** 2))
            elif (np.ndim(x1) > 1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2) > 1):
                return np.exp(-(np.linalg.norm(x1 - x2, 2, axis=1) ** 2) / (2 * self.gamma ** 2))
            elif np.ndim(x1) > 1 and np.ndim(x2) > 1:
                return np.exp(-(np.linalg.norm(x1[:, np.newaxis]
                                               - x2[np.newaxis, :], 2, axis=2) ** 2) / (2 * self.gamma ** 2))

    def calculate_bounds(self, alpha_j, alpha_i, y_j, y_i):
        if y_i != y_j:
            return max(0, alpha_j - alpha_i), min(self.C, self.C + alpha_j - alpha_i)
        else:
            return max(0, alpha_j + alpha_i - self.C), min(self.C, alpha_j + alpha_i)

    def fit(self, X, y):
        samples, features = X.shape
        self.alpha = np.zeros(samples)
        self.w = np.zeros(features)
        self.b = 0

        iterations = 0

        self.X = X
        self.y = y

        while iterations < self.max_iter:
            alpha_prev = np.copy(self.alpha)
            for i in range(samples):
                error_i = self.decision_function(X[i]) - y[i]

                if (error_i * y[i] < -self.tol and self.alpha[i] < self.C) or \
                   (error_i * y[i] > self.tol and self.alpha[i] > 0):

                    j = np.random.randint(0, samples)
                    while i == j:
                        j = np.random.randint(0, samples)

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    error_j = self.decision_function(X[j]) - y[j]

                    L, H = self.calculate_bounds(self.alpha[j], self.alpha[i], y[j], y[i])

                    if L == H:
                        continue

                    eta = 2 * self.kernel(X[i], X[j]) - self.kernel(X[i], X[i]) - self.kernel(X[j], X[j])
                    if eta >= 0:
                        continue

                    self.alpha[j] = alpha_j_old - y[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - error_i - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i], X[i]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[i], X[j])

                    b2 = self.b - error_j - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i], X[j]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[j], X[j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
            iterations += 1

        self.w = np.dot((self.alpha * y).T, X)

    def predict(self, X):
        return np.sign(self.decision_function(X))
