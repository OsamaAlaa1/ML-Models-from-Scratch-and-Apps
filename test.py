import numpy as np

class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, data):
        # Randomly initialize K centroids
        self.centroids = data[np.random.choice(range(len(data)), self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Assign data points to the nearest centroid
            labels = self._assign_labels(data)

            # Update centroids
            new_centroids = self._update_centroids(data, labels)

            # Check convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return labels, self.centroids

    def _assign_labels(self, data):
        # Calculate Euclidean distance between data points and centroids
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))

        # Assign labels based on the nearest centroid
        labels = np.argmin(distances, axis=0)

        return labels

    def _update_centroids(self, data, labels):
        centroids = []
        for i in range(self.k):
            # Calculate mean of data points assigned to the cluster
            cluster_points = data[labels == i]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

        return np.array(centroids)


# Example usage
from sklearn.datasets import make_blobs

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

kmeans = KMeans(k=clusters, max_iterations=150)

labels, centroids = kmeans.fit(X)

print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)
