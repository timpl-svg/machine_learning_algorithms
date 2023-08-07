import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradBoostOnDT:
    def __init__(self, k, nu, tree_max_depth):
        self.k = k
        self.nu = nu
        self.tree_max_depth = tree_max_depth
        self.trees = []
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.pred = np.array([np.mean(y) for i in range(len(y))])
        
    def train(self):
        for i in range(self.k):
            residual = self.y - self.pred
            tree = DecisionTreeRegressor(max_depth=self.tree_max_depth)
            tree.fit(self.X, residual)
            self.pred += self.nu * tree.predict(self.X)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([np.mean(self.y) for i in range(X.shape[0])])
        for tree in self.trees:
            predictions += self.nu * tree.predict(X)
        return predictions