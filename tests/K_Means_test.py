from sklearn.datasets import make_blobs
from K_Means_Clustering import KMeansClustering
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(centers=10, n_samples=20000, n_features=2, shuffle=True, random_state=40)
print(X.shape)

n_clusters = len(np.unique(y))
print(n_clusters)

kmeans = KMeansClustering(K=n_clusters, iters=150)

y_pred = kmeans.predict(X)

fig, ax = plt.subplots(figsize=(12, 8))

cmap = plt.cm.get_cmap("viridis", n_clusters)  # You can choose other colormaps like "plasma", "inferno", etc.
colors = [cmap(i) for i in range(n_clusters)]

for idx in range(n_clusters):
    points_in_cluster = X[y_pred == idx]
    ax.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], s=50, c=colors[idx], label=f'Cluster {idx + 1}')

ax.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=300, c='yellow', marker='*', edgecolor='black', label='Centers')

ax.set_title('KMeans Clustering Results')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

plt.show()
