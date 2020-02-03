import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier as sklearn_BaggingClassifier
from sklearn import datasets

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=None, 
                 criterion='entropy', random_seed=None):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.decision_trees = []
        self.random_seed = random_seed
        self.criterion = criterion
    
    @staticmethod
    def Bootstrap(X, y):
        boosting_size = X.shape[0]
        boosting_index = np.random.choice(range(len(X)), size=boosting_size, replace=True)
        boosting_X = X[boosting_index]
        boosting_y = y[boosting_index]
        return boosting_X, boosting_y
        
    def fit(self, X, y, sample_weight=None):
        boosting_list = [BaggingClassifier.Bootstrap(X, y) for i in range(self.n_estimators)]
        
        for b_x, b_y in boosting_list:
            dt = DecisionTreeClassifier(max_depth=self.max_depth, random_seed=self.random_seed,
                                        max_features=None, criterion=self.criterion)
            dt.fit(b_x, b_y, sample_weight=sample_weight)
            self.decision_trees.append(dt)
            
    def predict(self, X_new):
        if len(self.decision_trees) == 0:
            raise ValueError('The Bagging is not fitted.')
        result = []
        for dt in self.decision_trees:
            result.append(dt.predict(X_new))
        result = np.stack(result, axis=1)
        return np.array([pd.Series(item).value_counts().idxmax() for item in result])

def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    bg = BaggingClassifier(max_depth=1)
    bg.fit(X, y)
    
    pre = bg.predict(X)
    print('acc:', np.mean(pre == y))
    
    sk_bg = sklearn_BaggingClassifier()
    sk_bg.fit(X, y)
    
    pre = sk_bg.predict(X)
    print('sklearn bagging acc:', np.mean(pre == y))

if __name__ == '__main__':
    main()

