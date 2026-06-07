import numpy as np
from collections import Counter

class KNN:
  def __init__(self, k=3, distance_metric='euclidean', type='classification'):
    self.k = k
    self.distance_metric = distance_metric
    self.type = type
  
  def _euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
  
  def _manhattan_distance(self, x1, x2):
    return np.sum(np.abs(x1 - x2))

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
    
  def predict(self, X):
    predicted_labels = [self._predict(x) for x in X]
    return np.array(predicted_labels)
    
  def _predict(self, x):
    # Compute distances between x and all examples in the training set
    if self.distance_metric == 'euclidean':
      distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
    elif self.distance_metric == 'manhattan':
      distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
    
    # Sort by distance and return indices of the first k neighbors
    k_indices = np.argsort(distances)[:self.k]
    
    # Extract the labels of the k nearest neighbor training samples
    k_nearest_neighbors = [self.y_train[i] for i in k_indices]
    
    # For regression, return the mean of the k nearest neighbors
    if self.type == 'regression':
      return np.mean(k_nearest_neighbors)
    
    # For classification, return the most common class label among the neighbors
    if self.type == 'classification':
      most_common = Counter(k_nearest_neighbors).most_common(1)
      return most_common[0][0]