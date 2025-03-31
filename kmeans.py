import numpy as np

class KMeans:
  def __init__(self, n_clusters=3, max_iters=300, tol=1e-4):
    self.n_clusters = n_clusters
    self.max_iters = max_iters
    self.tol = tol
    self.centroids = None


  def euclidean_distance(self, c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))


  def _assign_clusters(self, X):
    distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
    return np.argmin(distances, axis=1) # assign closest centroid


  def fit(self, X_train):
    n, p = X_train.shape

    # Step 1: Randomly initialize n_clusters centroids
    random_indices = np.random.choice(n, self.n_clusters, replace=False)
    self.centroids = X_train[random_indices]

    for _ in range(self.max_iters):
      # Step 2: Assign each point to the closest centroid
      labels = self._assign_clusters(X_train)

      # Step 3: Compute new centroids
      new_centroids = np.array([
          X_train[labels == i].mean(axis=0) for i in range(self.n_clusters)])

      # Step 4: Check for convergence (if centroids do not change significantly)
      if np.linalg.norm(self.centroids - new_centroids) < self.tol:
        break

      self.centroids = new_centroids
    return labels


  def predict(self, X_test):
    return self._assign_clusters(X_test)
