import numpy as np

class DistanceCalculator:
    def __init__(self, metric='euclidean'):
        assert metric in ['euclidean', 'manhattan', 'minkowski'], \
            "距离测度可选：'euclidean', 'manhattan', 或 'minkowski'"
        self.metric = metric

    def calculate(self, X, Y):
        if self.metric == 'euclidean':
            return self.euclidean(X, Y)
        elif self.metric == 'manhattan':
            return self.manhattan(X, Y)
        elif self.metric == 'minkowski':
            return self.minkowski(X, Y)

    def euclidean(self, X, Y):
        return np.sqrt(np.sum((X[:, None] - Y) ** 2, axis=-1))

    def manhattan(self, X, Y):
        return np.sum(np.abs(X[:, None] - Y), axis=-1)

    def minkowski(self, X, Y, p=3):
        return np.sum(np.abs(X[:, None] - Y) ** p, axis=-1) ** (1 / p)
