import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import cvxpy as cp
from cvxpy.reductions.solvers import defines as slv_def

class SVC:
    def __init__(self, C=1.0, method=None, random_seed=None):
        self.C = C
        if method is None:
            method = 'ECOS'
        if method not in slv_def.installed_solvers():
            raise ValueError('Optimization methods error.')
        self.method = method
        self.random_seed = random_seed
        self.beta = None
        self.v = None
    
    def fit(self, X, y):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        m, n = X.shape
        beta = cp.Variable((n, 1))
        v = cp.Variable()
        loss = cp.sum(cp.pos(1 - cp.multiply(y, X * beta + v)))
        reg = cp.norm(beta, 1)
        lambd = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Minimize(loss / m + lambd * reg))
            
        lambd.value = self.C
        prob.solve(self.method)
        self.beta = beta.value
        self.v = v
    
    def predict(self, X_new):
        if self.beta is None or self.v is None:
            raise ValueError('The SVM classifier is not fitted.')
        return np.sign(X_new.dot(self.beta) + self.v.value).reshape(-1)
    
    
def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    X = X[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    
    y[y == 0] = -1 
    svc = SVC(C=1)
    svc.fit(X, y) 
    print('acc:', np.mean(svc.predict(X) == y))
    
if __name__ == '__main__':
    main()