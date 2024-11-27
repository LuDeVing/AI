from sklearn.datasets import make_blobs
from K_Means_Clustering import KMeansClustering
import matplotlib.pyplot as plt
import numpy as np

# Create a synthetic dataset with 3 clusters
X, y = make_blobs(centers=10, n_samples=20000, n_features=2, shuffle=True, random_state=40)
print(X.shape)

# Number of clusters in the generated dataset
n_clusters = len(np.unique(y))
print(n_clusters)

# Create an instance of your KMeansClustering with the detected number of clusters
kmeans = KMeansClustering(K=n_clusters, iters=150)

# Predict the cluster labels for each data point in X
y_pred = kmeans.predict(X)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))

cmap = plt.cm.get_cmap("viridis", n_clusters)  # You can choose other colormaps like "plasma", "inferno", etc.
colors = [cmap(i) for i in range(n_clusters)]

# Plot each cluster with a different color
for idx in range(n_clusters):
    # Points in the cluster
    points_in_cluster = X[y_pred == idx]
    ax.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], s=50, c=colors[idx], label=f'Cluster {idx + 1}')

# Plot the final cluster centers
ax.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=300, c='yellow', marker='*', edgecolor='black', label='Centers')

# Add labels and a legend
ax.set_title('KMeans Clustering Results')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

# Display the plot
plt.show()
