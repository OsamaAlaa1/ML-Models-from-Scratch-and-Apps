from sklearn.datasets import make_blobs
from K_means import np, KMeans
from sklearn.cluster import KMeans as KM
from timeit import default_timer as timer 
# prepare the dataset
X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)

# get the number of clusters 
clusters = len(np.unique(y))

# scratch Section 

start = timer()
# call scratch kmeans 
kmeans = KMeans(k=clusters, max_iterations=150)
labels, centroids = kmeans.fit(X)

end = timer()

# count time token
scratch_time = end - start 
print (f"Time token by Buildin Kmeans :{scratch_time}")


print("Scratch Cluster Labels:", labels)
print("Scratch Cluster Centroids:", centroids)


# build in section 

start = timer()
# Create KMeans model
kmeans = KM(n_clusters=3, random_state=42)
# Fit the model to the data
kmeans.fit(X)

end = timer()

# count time token
buildin_time = end - start 
print (f"Time token by Buildin Kmeans :{buildin_time}")

# Get predicted labels

print("buildin Cluster Labels:", kmeans.labels_)
print("buildin Cluster Centroids:", kmeans.cluster_centers_)


