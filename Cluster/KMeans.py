import numpy as np
from sklearn import datasets

class KMeans:
    def __init__(self, n_cluster, max_iter=1000, eps=1e-12, random_seed=None):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.eps = eps
        self.random_seed = random_seed
        self.labels = None
        self.center = None
    
    @staticmethod
    def Euclidean(center, data):
        return np.sqrt(((center - data) ** 2).sum(axis=1))
    
    def fit(self, X):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        X = np.array(X)
        X_max = X.max(axis=0)
        X_min = X.min(axis=0)
        
        center = np.array([[X_min[j] + np.random.rand() * (X_max[j] - X_min[j]) for j in range(len(X_max))] for i in range(self.n_cluster)])
        
        for k in range(self.max_iter):
            labels = np.stack([(KMeans.Euclidean(item, X)) for item in center], axis=0).argmin(axis=0)
            center_ = [(X[labels == i]).mean(axis=0) for i in range(self.n_cluster)]
            last_center = center
            center = np.stack(center_, axis=0)
            if (center - last_center).max() < self.eps:
                break
            
        self.labels = labels
        self.center = center
        
    def cluster(self, X_new):
        if self.center is None:
            raise ValueError('The cluster model is not fitted.')
        labels = np.stack([(KMeans.Euclidean(item, X_new)) for item in self.center], axis=0).argmin(axis=0)
        return labels
    
    def fit_cluster(self, X):
        self.fit(X)
        return self.cluster(X)
    
def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    kmeans = KMeans(3)
    kmeans.fit(X)
    
    print(kmeans.cluster(X))
    print(kmeans.fit_cluster(X))
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
