"""
PCA

Principal Component Analysis, or PCA, is an unsupervised learning method
that is often used to reduce the dimensionality of the dataset by
transforming a large set into a lower dimensional set that still contains most
of the information of the large set.

"""

import numpy as np 

class PCA: 

    def __init__(self, n_components):

        # n_components are number of dimentions we wanna get after transformation 
        self.n_components = n_components
        self.components = None
        self.mean = None 

    def fit (self, X):
        
        # mean centering 
        self.mean = np.mean(X,axis= 0)
        X-=self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvecotors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov) 
        
        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors. T
        
        #sort eigenvectors
        idxs = np.argsort (eigenvalues) [::-1]
        eigenvalues = eigenvalues [idxs]
        eigenvectors = eigenvectors [idxs]
        self.components = eigenvectors [:self.n_components]


    def transform (self,X):
        X -= self.mean
        return np.dot(X, self.components.T)
