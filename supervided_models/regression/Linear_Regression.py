import numpy as np 

class LinearRegression:
    
    def __init__(self,lr = 0.01, n_iter = 1000):
        
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        
        # create the model equation y =  bias*x0 + w1*x1+ W2*x2 ....etc
        n_samples, n_features = X.shape
        
        # initialize the model weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            
            # start the model equation
            y_pred = np.dot(X,self.weights) + self.bias

            # derivative of weights and bias 
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum((y_pred-y))

            # update weights and bias 
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
