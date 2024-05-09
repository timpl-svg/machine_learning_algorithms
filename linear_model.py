import numpy as np

class LinearRegression:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X = X 
        self.y = y
        
    def train_exact(self):
        return np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T) @ y
    
    def train_close(self):
        weights = np.random.normal(1, 1, X.shape[1])
        
    
    def predict(self, X, option='close'):
        predictions = []
        if option == 'close':
            close_weights = self.train_close()
            predictions = [np.dot(X[i], close_weights) for i in range(X.shape[0])]
        elif option == 'exact':
            exact_weights = self.train_exact()
            predictions = [np.dot(X[i], exact_weights) for i in range(X.shape[0])]
        
        return predictions