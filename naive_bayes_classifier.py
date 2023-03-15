import numpy as np

class NaiveBayesGD:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X = X 
        self.y = y
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
    
    def calculate_mean_values(self):
        mean_values = np.zeros([self.n_classes, self.n_features])
        for i in range(self.n_classes):
                mean_values[i] = np.mean(self.X[self.y == np.unique(self.y)[i]], axis=0)
        
        return mean_values
    
    def calculate_variances(self):
        variances = np.zeros([self.n_classes, self.n_features])
        for i in range(self.n_classes):
                variances[i] = np.var(self.X[self.y == np.unique(self.y)[i]], axis=0)
        
        return variances
    
    def calculate_frequencies(self):
        frequencies = np.zeros(self.n_classes)
        unique, counts = np.unique(self.y, return_counts=True)
        for i in range(self.n_classes):
            frequencies[i] = counts[i] / len(self.y)
            
        return frequencies
    
    def calculate_probabilities(self, X):
        means = self.calculate_mean_values()
        variances = self.calculate_variances()
        frequencies = self.calculate_frequencies()
        probabilities = np.zeros([X.shape[0], self.n_classes])
        for i in range(X.shape[0]):
            for j in range(self.n_classes):
                probabilities[i, j] = np.log(frequencies[j])
                for k in range(self.n_features):
                    probabilities[i, j] -= (X[i, k] - means[j, k]) ** 2 / (2 * variances[j, k])
                
        return probabilities
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        probabilities = self.calculate_probabilities(X)
        for i in range(X.shape[0]):
            y_pred[i] = np.argmax(probabilities[i])
        
        return y_pred