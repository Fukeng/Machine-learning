import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

class Blending:
    def __init__(self, BaseEstimators, shuffle=True, random_seed=None, 
                 task='Classification'):
        
        if task not in {'Classification', 'Regression'}:
            raise ValueError(r'The task should be one of {Classification, Regression}')
        self.BaseEstimators = BaseEstimators
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.task = task
        self.type_num = None
        
    def fit(self, X, y, sample_weight=None):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        X = np.array(X)
        y = np.array(y)
        if self.task == 'Classification':
            self.type_num = len(np.unique(y))
        if self.shuffle:
            ind = np.arange(len(X))
            np.random.shuffle(ind)
            X, y = X[ind], y[ind]
        
        n = len(self.BaseEstimators)
        sample_num = X.shape[0]
        for i, estimator in enumerate(self.BaseEstimators):
            estimator.fit(X[int(i * sample_num / n) : int((i + 1) * sample_num / n)], 
                            y[int(i * sample_num / n) : int((i + 1) * sample_num / n)],
                            sample_weight=sample_weight)
        return
    
    def predict(self, X_new):
        if self.task == 'Classification':
            result = np.zeros([X_new.shape[0], self.type_num, len(self.BaseEstimators) + 1])
        elif self.task == 'Regression':
            result = np.zeros([X_new.shape[0], 1, len(self.BaseEstimators) + 1])
        
        for i, estimator in enumerate(self.BaseEstimators):
            if self.task == 'Classification':
                y_pre = estimator.predict_proba(X_new)
            elif self.task == 'Regression':
                y_pre = estimator.predict(X_new).reshape(-1, 1)
            else:
                raise ValueError(r'The task should be one of {Classification, Regression}')
            result[:, :, i] = y_pre
        
        if self.task == 'Classification':
            return result.mean(axis=2).argmax(axis=1)   
        else:            
            return result.mean(axis=2).reshape(-1)

    
def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    dt = DecisionTreeClassifier()
    et = ExtraTreeClassifier()
    
    clf = [rf, gb, dt, et]
    blending = Blending(clf, shuffle=True, task='Classification')
    
    blending.fit(X, y)
    print(np.mean(blending.predict(X) == y))

if __name__ == '__main__':
    main()
        

