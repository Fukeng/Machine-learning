import numpy as np
from sklearn import datasets

class DBSCAN:
    def __init__(self, n_neighbour, radius, random_seed=None):
        self.n_neighbour = n_neighbour
        self.radius = radius
        self.random_seed = random_seed
        
    @staticmethod
    def Euclidean(center, data):
        return np.sqrt(((center - data) ** 2).sum(axis=1))
    
    def fit(self, X):
        pass
    
    def label(self, X_new):
        pass
    
    def fit_label(self, X):
        self.fit(X)
        return self.label(X)

def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    dbscan = DBSCAN(3, 0.1)
    dbscan.fit(X)
    print(dbscan.label(X))

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    