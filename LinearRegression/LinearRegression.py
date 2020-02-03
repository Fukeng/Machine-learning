import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

class LinearRegression:
    def __init__(self, eps=1e-12, lr=1e-4, random_seed=None, penalty=None, 
                 penalty_lambda=0):
        self.eps = eps
        self.lr = lr
        self.beta = None
        self.random_seed = random_seed
            
        if penalty is not None and penalty not in {'l1', 'l2'}:
            raise NotImplementedError
        self.penalty = penalty
        self.penalty_lambda = penalty_lambda
        self.params = None
                 
    def regression_assert(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)       
        if X.shape[0] != y.shape[0]:
            raise ValueError('The number of rows of X and Y is not equal.')
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        return X, y
    
    def gradient(self, X, y, sample_weight):
        W = np.diag(sample_weight)
        return 2 * (X.T.dot(W).dot(X).dot(self.beta) - X.T.dot(W).dot(y))
    
    def fit(self, X, y, sample_weight=None):
    	if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        gradient_fun = self.gradient
        X, y = self.regression_assert(X, y)
        self.beta = np.random.rand(X.shape[1], 1)
        
        beta_gradient = None
        while beta_gradient is None or np.abs(max(beta_gradient) * self.lr) > self.eps:
            beta_gradient = gradient_fun(X, y, sample_weight)
            if self.penalty == 'l1':
                beta_gradient += self.penalty_lambda * np.sign(self.beta)
            elif self.penalty == 'l2':
                beta_gradient += 2 * self.penalty_lambda * self.beta
            self.beta -= beta_gradient * self.lr
        
        d = {}
        beta = self.beta.reshape(-1)
        for i in range(beta.shape[0]):
            d['beta_' + str(i)] = beta[i]
        self.params = d
            
    def get_params(self, deep=False):
        if deep:
            return self.params.copy()
        else:
            return self.params
    
    def predict(self, X_new):
        X_new = np.hstack([np.ones([X_new.shape[0], 1]), X_new])
        return X_new.dot(self.beta)
        
        
def main():
    LR = LinearRegression()
    lasso = LinearRegression(penalty='l1', penalty_lambda=0.1)
    ridge = LinearRegression(penalty='l2', penalty_lambda=0.1)
    
    X = np.random.rand(90, 3)
    y = 0.1 * X[:, 0] + 0.6 * X[:, 1] + 0.3 * X[:, 2] + 0.01 * np.random.rand(X.shape[0])
    
    LR.fit(X, y)
    lasso.fit(X, y)
    ridge.fit(X, y)
    
    print(LR.get_params())
    print(lasso.get_params())
    print(ridge.get_params())
    
    y_pre = LR.predict(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.regplot(y.reshape(-1), y_pre.reshape(-1), line_kws={'color':'r', 'linestyle':'dashed'})
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted Value')
    plt.title('OLS')
    plt.show()
    
    y_pre = lasso.predict(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.regplot(y.reshape(-1), y_pre.reshape(-1), line_kws={'color':'r', 'linestyle':'dashed'})
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted Value')
    plt.title('Lasso Regression')
    plt.show()
    
    y_pre = ridge.predict(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.regplot(y.reshape(-1), y_pre.reshape(-1), line_kws={'color':'r', 'linestyle':'dashed'})
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted Value')
    plt.title('Ridge Regression')
    plt.show()
    
        
if __name__ == '__main__':
    main()     
    
    
    
    