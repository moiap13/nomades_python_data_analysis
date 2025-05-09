import numpy as np
import matplotlib.pyplot as plt

class KMeans:
  def __init__(self, k:int=3, max_iters:int=100):
    self.k = k
    self.max_iters = max_iters
    self.centroids = None
    self.labels = None
  
  def __distance(self, a:np.ndarray, b:np.ndarray):
    """
    The distance method compute the euclidean distance between two points.

    Parameters:
    a : np.ndarray
        The first point.
    b : np.ndarray
        The second point.
    
    Returns:
    float
        The euclidean distance between the two points.
    """
    return np.sqrt(np.sum((a - b) ** 2))
  
  def fit(self, X:np.ndarray, track_history:bool=False):
    """
    The fit method compute the k-means clustering algorithm.

    Steps:
    1. Randomly initialize the centroids.
      We'll randomly set the initial centroids from the data points.
      We'll bound the centroids to each data dimension.
      Each centroids should be in the range (min, max) of each datapoint dimension.
      we'll use the `np.random.uniform` function to generate random values.
    2. Assign each data point to the nearest centroid.
    3. Update the centroids by computing the mean of the assigned data points.
    4. Repeat steps 2 and 3 until convergence (the centroids stopped moving) or max iterations reached.

    Parameters:
    X : np.ndarray
        The input data, shape (n_samples, n_features).
    
    Returns:
    Y: labels
    """
    # Step 1: Randomly initialize the centroids of shape (k, d)
    # use np.random.uniform
    self.centroids = np.random.uniform(
      np.min(X, axis=0),
      np.max(X, axis=0),
      (self.k, X.shape[1])
    )
    
    for _ in range(self.max_iters):
      y: list[int] = []

      # Step 2: Assign each data point to the nearest centroid
      for data_point in X:
        # Compute the distance between the data point and each centroid
        distances: list[float] = []
        for centroid in self.centroids:
          distances.append(self.__distance(data_point, centroid))
        # Find the index of the closest centroid
        cluster_number: int = np.argmin(distances)
        # Append the cluster number to the list
        y.append(cluster_number)

      y = np.array(y)

      # Step 3: Update the centroids (reposition the centroids based on the mean of the assigned data points)
      # compute the indices that belong to each cluster
      cluster_indices: list[np.ndarray] = [] # list of list, where the first dimmension is the cluster number and the second dimension is the indeces of the data points that belong to that cluster

      for i in range(self.k):
        cluster_indices.append(np.argwhere(y == i))
      
      cluster_center: list[np.ndarray] = []
      for i, indices in enumerate(cluster_indices):
        if len(indices) == 0:
          cluster_center.append(self.centroids[i])
          continue
        
        cluster_center.append(X[indices].mean(axis=0)[0])

      # Step 4: Check for convergence (if the centroids stopped moving)
      # use break statement to stop the loop
      if np.max(np.abs(self.centroids-cluster_center)) < 1e-4:
        break

      # Update the centroids
      self.centroids = np.array(cluster_center)

    # Store the labels
    self.labels = y
    return self.labels
  