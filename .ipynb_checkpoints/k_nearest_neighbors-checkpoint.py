import numpy as np
"""
Based on the Stanford CS231n course material and mipt ML course.
Source link: http://cs231n.github.io/assignments2019/assignment1/
"""

class KNearestNeighbor:
    def __init__(self):
        pass

    def fit(self, X, y):
        # X - training points, y - training labels
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        # X - test points
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        # distances between test points X and training points self.X_train in two loops
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.linalg.norm(X[i] - self.X_train[j])

        return dists

    def compute_distances_one_loop(self, X):
        # distances between test points X and training points self.X_train in one loop
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.linalg.norm(self.X_train - X[i,:], axis = 1)

        return dists

    def compute_distances_no_loops(self, X):
        # distances between test points X and training points self.X_train without any loops
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T))

        return dists

    def predict_labels(self, dists, k=1):
        # predict labels for test points
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        closest_y = np.zeros((num_test, k), dtype=int)
        for i in range(num_test):
            closest_y[i] = np.argsort(dists[i])[:k]
            y_pred[i] = np.bincount(self.y_train[closest_y[i].tolist()]).argmax()
            
        return y_pred
