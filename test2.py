import numpy as np 
data = np.array([[3,4],[1,2],[4,5]])

centroids = data[np.random.choice(range(data.shape[0]),2,replace=False)]

distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
print(distances)
labels = np.argmin(distances,axis=0)
print (labels)

x = data[labels==0]

print (np.mean(x , axis = 0))
