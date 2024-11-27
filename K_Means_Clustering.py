import numpy as np


class KMeansClustering:
    def __init__(self, K, iters=1000):
        self.K = K
        self.iters = iters
        self.centers = None

    def predict(self, X):

        n_samples, n_features = X.shape

        rand_indices = np.random.choice(n_samples, self.K, replace=False)
        self.centers = X[rand_indices]

        for _ in range(self.iters):

            clusters = self.create_clusters(X)

            previous_centers = self.centers
            self.centers = self.get_new_centers(X, clusters)

            c1 = np.sort(self.centers, axis=0)
            c2 = np.sort(previous_centers, axis=0)

            if np.array_equal(c1, c2):
                break

        return self.get_cluster_labels(X)

    def create_clusters(self, X):

        clusters = [[] for _ in range(self.K)]

        for idx, x in enumerate(X):
            clusters[self.get_closest_cluster(x)].append(idx)

        return clusters

    def get_closest_cluster(self, x):

        min_dist = None
        cluster_idx = None

        for idx, cluster in enumerate(self.centers):
            diff = x - cluster
            dist_squared = np.dot(diff, diff.T)

            if min_dist is None or min_dist > dist_squared:
                min_dist = dist_squared
                cluster_idx = idx

        return int(cluster_idx)

    def get_new_centers(self, X, clusters):

        n_features = X.shape[1]

        centers = np.zeros((self.K, n_features))

        for idx in range(self.K):
            centers[idx] = np.mean(X[clusters[idx]], axis=0)

        return centers

    def get_cluster_labels(self, X):
        labels = np.zeros(X.shape[0])

        for idx, x in enumerate(X):
            labels[idx] = self.get_closest_cluster(x)

        return np.array(labels)

