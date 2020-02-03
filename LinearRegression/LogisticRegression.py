import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression import LinearRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold
from sklearn import datasets

class LogisticRegression(LinearRegression):
    def __init__(self, eps=1e-12, lr=1e-4, random_seed=None, penalty=None, penalty_lambda=0):
        super(LogisticRegression, self).__init__(eps, lr, random_seed, penalty, penalty_lambda)

    @staticmethod
    def sigmoid(x):
        proba = np.exp(x) / (1 + np.exp(x))
        return proba
    
    def gradient(self, X, y, sample_weight):
        W = np.diag(sample_weight)
        return ((LogisticRegression.sigmoid(X.dot(self.beta)) - y).T.dot(W).dot(X)).T

    def predict_proba(self, X_new):
        X_new = np.hstack([np.ones([X_new.shape[0], 1]), X_new])
        return LogisticRegression.sigmoid(X_new.dot(self.beta))
    
    def predict(self, X_new):
        prob = self.predict_proba(X_new).reshape(-1)
        return np.array([int(item) for item in (prob > 0.5)])

def main():    
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    X = X[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    
    acc_list = []
    for train_index, test_index in kfold.split(X, y):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        logisticReg = LogisticRegression(lr=0.0001, eps=0.00001)
        logisticReg.fit(train_x, train_y, sample_weight=np.random.rand(len(train_x)))
        acc_list.append(np.mean(logisticReg.predict(test_x) == test_y))
    print(acc_list)
    

if __name__ == '__main__':
    main()



