import numpy as np

# create a sigmoid function 
def sigmoid (x):
    
    return 1/(1 + np.exp(-x))


# clss for logistic regression 
class LogisticRegression:
    
    # constructor : define weights and bias  
    # initialize the learning rate and nuber of iterations

    def __init__(self,lr = 0.001, n_iters = 1000):
        
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 

    # function to fit the data 
    def fit (self,X,y):

        n_samples, n_features = X.shape

        # initialize the weights and bias 
        self.weights = np.zeros(n_features)
        self.bias = 0

        # loop to update weights and bias 
        for _ in range(self.n_iters):

            y_pred = np.dot(X,self.weights) + self.bias
            predictions = sigmoid(y_pred)

            dw = (1/n_samples) * np.dot(X.T,(predictions- y))
            db = (1/n_samples) * np.sum(predictions- y)

            # update here
            self.weights -= (self.lr * dw)
            self.bias -= (self.lr * db)

    def predict(self,X):
            
            y_pred= np.dot(X,self.weights) + self.bias
            probs = sigmoid(y_pred)  
            return [0  if y <=0.5 else 1 for y in probs]  

