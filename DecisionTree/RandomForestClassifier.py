import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DecisionTreeClassifier import DecisionTreeClassifier
from BaggingClassifier import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from sklearn import datasets

class RandomForestClassifier(BaggingClassifier):
    def __init__(self, max_depth=None, n_estimators=10, max_features='auto', 
                 criterion='entropy', random_seed=None):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.decision_trees = []
        self.max_features = max_features
        self.random_seed = random_seed
        self.criterion = criterion
        
    def fit(self, X, y, sample_weight=None):
        boosting_list = [RandomForestClassifier.Bootstrap(X, y) for i in range(self.n_estimators)]
        
        for b_x, b_y in boosting_list:
            dt = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features,
                                        random_seed=self.random_seed, criterion=self.criterion)
            dt.fit(b_x, b_y, sample_weight=sample_weight)
            self.decision_trees.append(dt)

def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    rf = RandomForestClassifier(max_features='log2', criterion='gini')
    rf.fit(X, y)
    
    pre = rf.predict(X)
    print('acc:', np.mean(pre == y))
    
    sk_rf = sklearn_RandomForestClassifier()
    sk_rf.fit(X, y)
    
    pre = sk_rf.predict(X)
    print('sklearn random forest acc:', np.mean(pre == y))

if __name__ == '__main__':
    main()

