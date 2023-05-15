import numpy as np 

class KMeans: 

    def __init__(self,k,max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self,data):

        # initialize the centroids - choose random data points from the dataset and set them as centroids 
        self.centroids = data[np.random.choice(range(len(data)),self.k,replace=False)]

        # iterate number of iterations 
        for _ in range(self.max_iterations):
            
            # 1. Assign data points to the nearest centroid
            labels = self._assign_labels(data)

            # 2. Update centroids 
            new_centroids = self._update_centroids(data, labels)

            # check convergence 
            if np.all(self.centroids == new_centroids):
                break

            # if above condition not true then update the centroids
            self.centroids = new_centroids

            return labels , self.centroids
    
    # Assign data points to the nearest centroid function 
    def _assign_labels(self,X):
        
        #calulate the Euclidean distance between data points and centroids
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

        # get label by get the minimum value per data point according to centroids values
        labels = np.argmin(distances,axis=0)

        return labels

    
    def _update_centroids(self,data, lables):
        centroids = []

        # Calculate mean of data points assigned to the cluster
        for i in range(self.k):

            # get all data points for each cluster 
            cluster_points = data[lables == i]

            # calculate the mean 
            centroids.append(np.mean(cluster_points ,axis = 0))
        
        return np.array(centroids)
