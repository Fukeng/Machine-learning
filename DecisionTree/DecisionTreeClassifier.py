import numpy as np
import pandas as pd
from sklearn import datasets

class TreeNode:
    def __init__(self, var, threshold, mode, entropy, 
                 left=None, right=None):
        self.var = var
        self.threshold = threshold
        self.mode = mode
        self.entropy = entropy
        self.left = left
        self.right = right

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, max_features=None, criterion='entropy',
                 random_seed=None):
        if max_depth is None:
            self.max_depth = float('inf')
        elif isinstance(max_depth, int) and max_depth > 0:
            self.max_depth = max_depth
        else:
            raise ValueError('Maximum depth Error.')
        self.root = None
        self.random_seed = random_seed
        self.max_features = max_features
        self.criterion = criterion
        self.entropy_fun = None
        
    def get_feature_num(self, X):
        feature_num = X.shape[1]
        if self.max_features is None:
            return feature_num
        elif isinstance(self.max_features, float):
            return int(feature_num * self.max_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features in {'auto', 'sqrt'}:
            return int(np.sqrt(feature_num))
        elif self.max_features == 'log2':
            return int(np.log2(feature_num))
        else:
            raise ValueError('The maximum features is not correct.')
        
    @staticmethod
    def entropy(y):
        y_counts = pd.Series(y).value_counts()
        y_counts = y_counts / sum(y_counts)  
        info_entropy = - sum(y_counts * np.log(y_counts))
        return info_entropy
    
    @staticmethod
    def gini(y):
        y_counts = pd.Series(y).value_counts()
        y_counts = y_counts / sum(y_counts)  
        gini_impurity = 1 - sum(y_counts ** 2)
        return gini_impurity
    
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weight = np.array(sample_weight) / sum(sample_weight)
        self._feature_num = self.get_feature_num(X)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        X = np.array(X)
        y = np.array(y)
        self.fit_(X, y, None, 0, sample_weight)
    
    def fit_(self, X, y, root, depth, sample_weight):
        if depth > self.max_depth:
            return
        
        if root is None:
            if self.criterion == 'entropy':
                self.entropy_fun = DecisionTreeClassifier.entropy
            elif self.criterion == 'gini':
                self.entropy_fun = DecisionTreeClassifier.gini
            else:
                raise ValueError('The criterion is not correct.')
                
            entropy = self.entropy_fun(y)
            mode = pd.Series(y).value_counts().idxmax()
            root = TreeNode(var=None, threshold=None, mode=mode, entropy=entropy)
            self.root = root
        
        if len(X) == 0 or len(y) == 0 or root.entropy == 0:
            return
        X = np.array(X)
        min_entropy = None
        left_entropy = None
        right_entropy = None
        min_j = None
        min_level = None
        min_bool_left = None
        min_bool_right = None
        min_mode_left = None
        min_mode_right = None
        
        random_cols = np.random.choice(range(X.shape[1]), self._feature_num, replace=False)
        for j in random_cols:
            X_j = X[:, j]
            levels = np.sort(np.unique(X_j))
            
            for level in levels[:-1]:
                left_bool = X_j <= level
                right_bool = X_j > level
                
                y_level = y[left_bool]
                info_entropy = self.entropy_fun(y_level)
                
                y_level2 = y[right_bool]        
                info_entropy2 = self.entropy_fun(y_level2)
                
                info_sum = info_entropy * np.mean(
                        left_bool * sample_weight) + info_entropy2 * np.mean(
                        right_bool * sample_weight)  

                if min_entropy is None or info_sum < min_entropy:
                    min_bool_left = left_bool
                    min_bool_right = right_bool
                    min_entropy = info_sum
                    min_j = j
                    min_level = level
                    min_mode_left = pd.Series(y_level).value_counts().idxmax()
                    min_mode_right = pd.Series(y_level2).value_counts().idxmax()
                    left_entropy = info_entropy
                    right_entropy = info_entropy2
                            
        leftNode = TreeNode(var=min_j, threshold=min_level, mode=min_mode_left, entropy=left_entropy)
        rightNode = TreeNode(var=min_j, threshold=min_level, mode=min_mode_right, entropy=right_entropy)
        
        root.left = leftNode
        root.right = rightNode

        self.fit_(X[min_bool_left], y[min_bool_left], root.left, depth + 1, sample_weight[min_bool_left])
        self.fit_(X[min_bool_right], y[min_bool_right], root.right, depth + 1, sample_weight[min_bool_right])
        return
        
    def predict(self, X_new):
        if self.root is None:
            raise ValueError('The decision tree is not fitted.')
        root = self.root
        result = np.array([None for i in range(len(X_new))])
        response_bool = np.array([True for i in range(len(X_new))])
        
        self.predict_(X_new, root, result, response_bool)
        return result
        
    def predict_(self, X_new, node, result, response_bool):
        # print(node.mode)
        if node.left is None or node.right is None:
            result[response_bool] = node.mode
            return
        self.predict_(X_new, node.left, result, response_bool & (X_new[:, node.left.var] <= node.left.threshold))
        self.predict_(X_new, node.right, result, response_bool & (X_new[:, node.right.var] > node.right.threshold))
        return
        
def main():
    iris = datasets.load_iris()
    n_samples, n_features = iris.data.shape
    X, y = iris.data, iris.target
    
    dt = DecisionTreeClassifier(max_depth=1, max_features='log2',
                                criterion='gini')
    dt.fit(X, y, sample_weight=np.random.rand(len(X)))
    
    print('acc:', np.mean(dt.predict(X) == y))

if __name__ == '__main__':
    main()

