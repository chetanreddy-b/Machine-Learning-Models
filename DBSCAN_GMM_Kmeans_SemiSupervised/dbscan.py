import numpy as np
from kmeans import pairwise_dist

class DBSCAN:
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

    def fit(self):
        num_points = len(self.dataset)
        cluster_labels = np.full(num_points, -1) 
        visited = set()
        current_cluster = 0
        
        for idx in range(num_points):
            if idx not in visited:
                visited.add(idx)
                neighbors = self.regionQuery(idx)
                
                if len(neighbors) < self.minPts:
                    cluster_labels[idx] = -1  
                else:
                    self.expandCluster(idx, neighbors, current_cluster, cluster_labels, visited)
                    current_cluster += 1 
                    
        return cluster_labels

    def expandCluster(self, idx, neighbors, current_cluster, cluster_labels, visited):
  
        cluster_labels[idx] = current_cluster
        while len(neighbors) > 0:
            curr_idx = neighbors[0]
            neighbors = np.delete(neighbors, 0)
            
            if curr_idx not in visited:
                visited.add(curr_idx)
                new_neighbors = self.regionQuery(curr_idx)
                
                if len(new_neighbors) >= self.minPts:
                    neighbors = np.unique(np.concatenate((neighbors, new_neighbors)))
            
            if cluster_labels[curr_idx] == -1:
                cluster_labels[curr_idx] = current_cluster

    def regionQuery(self, idx):
        distances = pairwise_dist(self.dataset[idx].reshape(1, -1), self.dataset).flatten()
        return np.argwhere(distances <= self.eps).flatten()
