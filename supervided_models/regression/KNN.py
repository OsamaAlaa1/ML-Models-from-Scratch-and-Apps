
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

        # return the average between 3 values 
        return np.average(k_nearest_label)



