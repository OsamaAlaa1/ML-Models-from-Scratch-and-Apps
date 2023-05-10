
import numpy as np
from collections import Counter

# build Euclidean Distance claculator that takes 2 lists and return distance 
def euclidean_distance(a,b):
    '''
    a,b -> arrays 
    return: Euclidean distance  
    '''
    distance = np.sqrt(np.sum((a-b)**2)) 
    return distance

# build class for KNN 
class KNN: 
    def __init__(self, K = 3 ):
        self.K = K

    def fit(self,X,y):
        
        self.X_train = X
        self.y_train = y 

    def predict(self,X):
        
        predictions = [self.get_label(x) for x in X ]
        return predictions
        

    
    def get_label(self,x):
        
        # compute the distances 
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        
        # get the closest K
        k_indeces = np.argsort(distances)[:self.K]
        k_nearest_label = [self.y_train[i] for i in k_indeces ]

        # get the most common label: return list of tuple [(1-> label,2-> frequency)]
        return Counter(k_nearest_label).most_common()[0][0]


# Main Section 

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN_buildin

# load iris dataset
iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)


# calc the time taken for each classifier - scratch and buildin models
from timeit import default_timer as timer


start = timer()

# define the classifer with K= 5 
clf = KNN(5)

# fit the data 
clf.fit(X_train,y_train)

# predict the output
predictions = clf.predict(X_test)

# calc the accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print ( accuracy)
stop = timer()

# print the time taken by scratch model
print (round((stop - start),5))

#--------------------------------- build in KNN clf ------------------
start = timer()

# define the classifer with K = 5 
clf = KNN_buildin(5)

# fit the data 
clf.fit(X_train,y_train)

# predict the output
predictions = clf.predict(X_test)

# calc the accuracy

accuracy = np.sum(predictions == y_test) / len(y_test)
print (accuracy)
stop = timer()

# print the time taken by Bulid in KNN model
print (round((stop - start),5))

