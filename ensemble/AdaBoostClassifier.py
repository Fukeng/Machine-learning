import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC

class AdaBoostClassifier:
    def __init__(self, BaseEstimator, T=10, random_seed=None):
        self.BaseEstimator = BaseEstimator
        self.EstimatorClass = BaseEstimator.__class__
        self.T = T
        self.random_seed = random_seed
        self.Boosting = None
        self.alpha = None
        
        if not isinstance(self.T, int):
            raise ValueError('The value T is not correct.')
        
    def fit(self, X, y):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        w = np.ones(X.shape[0])
        w = w / sum(w)
        G_list = []
        alpha_list = []
        for k in range(self.T):
            Gk = self.EstimatorClass()
            Gk.fit(X, y, w)
            G_list.append(Gk)
            ek = sum(w * (y != Gk.predict(X)))
            if ek <= 0:
                G_list = [Gk]
                alpha_list = [1.0]
                break
            alpha_k = np.log((1 - ek) / ek) / 2
            w = w * np.exp(- alpha_k * y.reshape(-1) * Gk.predict(X).reshape(-1))
            w = w / sum(w)
        self.Boosting = G_list
        self.alpha = np.array(alpha_list)
        self.alpha = self.alpha / sum(self.alpha)
            
    def predict(self, X_new):
        result = []
        for clf in self.Boosting:
            pre = clf.predict(X_new)
            result.append(pre)
            
        result_mat = self.alpha.reshape(1, -1).dot(np.stack(result, axis=0))
        return np.sign(result_mat.reshape(-1))
        

def main():
    X, y = datasets.make_classification(n_samples=100, n_features=50)
    y[y == 0] = -1
    
    ada = AdaBoostClassifier(BaseEstimator=DecisionTreeClassifier())
    ada.fit(X, y)
    print(np.mean(ada.predict(X) == y))
    
    
if __name__ == '__main__':
    main()













