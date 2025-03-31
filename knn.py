from collections import Counter
import numpy as np

class KNearestNeighbors:
  def __init__(self, k=3):
    self.k = k

  def euclidean_distance(self, x, y):
    return np.sqrt(np.sum((x-y) ** 2))

  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

  def get_distances(self, x):
    distances = []
    for i in range(self.X_train.shape[0]):
      distances.append(self.euclidean_distance(X[i], x))
    return distances

  def predict(self, X_test):
    return np.array([self._predict(x) for x in X_test])

  def _predict(self, x):
    # compute distances from x to all training points
    distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

    # get indices of k nearest neighbors
    k_indices = np.argsort(distances)[:self.k]

    # retrieve labels of the k nearest neighbors
    k_nearest_labels = [self.y_train[i] for i in k_indices]

    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]
    