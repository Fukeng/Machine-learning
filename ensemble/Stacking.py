import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

class Stacking:
    def __init__(self, BaseEstimators, StackEstimator, random_seed=None):
        self.BaseEstimators = BaseEstimators
        self.StackEstimator = StackEstimator
        self.random_seed = random_seed
    
    def fit(self, X, y, sample_weight=None):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        pre = []
        for clf in self.BaseEstimators:
            clf.fit(X, y, sample_weight)
            pre.append(clf.predict(X))
        pre_mat = np.stack(pre, axis=1)
        self.StackEstimator.fit(pre_mat, y)
        return
    
    def predict(self, X_new):
        pre = []
        for clf in self.BaseEstimators:
            pre.append(clf.predict(X_new))
        pre_mat = np.stack(pre, axis=1)
        return self.StackEstimator.predict(pre_mat)

def main():
    X, y = datasets.make_classification(n_samples=1000, n_features=50)
    
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    dt = DecisionTreeClassifier()
    et = ExtraTreeClassifier()
    
    clfs = [rf, gb, et]
    blending = Stacking(clfs, dt)
    
    blending.fit(X, y)
    print(np.mean(blending.predict(X) == y))

if __name__ == '__main__':
    main()