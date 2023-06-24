import numpy as np
from model.DistanceCalculator import DistanceCalculator

class FCM:
    def __init__(self, n_clusters, m=2, max_iter=1000, error=0.005, distance_metric='euclidean'):
        assert m > 1
        self.U = None
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.distance_calculator = DistanceCalculator(distance_metric)

    def fit(self, X):
        self.initialize(X)  # 初始化隶属度矩阵

        for iteration in range(self.max_iter):
            U_old = self.U.copy()

            self.update_centroids(X)    # 更新聚类中心
            self.update_U(X)     # 更新隶属度矩阵

            if self.norm(U_old - self.U) < self.error:  # 检查是否满足停止条件
                break

        return self

    def predict(self, X):
        return np.argmin(self.distance_calculator.calculate(X, self.centroids), axis=-1)

    def initialize(self, X):
        n_samples = len(X)
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)  # 使用狄利克雷分布随机生成初始隶属度矩阵

    # 更新聚类中心
    def update_centroids(self, X):
        um = self.U**self.m
        self.centroids = (X.T @ um / np.sum(um, axis=0)).T

    # 更新隶属度矩阵
    def update_U(self, X):
        self.U = self._predict(X)

    def _predict(self, X):
        power = float(2 / (self.m - 1))
        temp = self.distance_calculator.calculate(X, self.centroids)
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = np.power(temp[:, :, np.newaxis] / denominator_.transpose((0, 2, 1)), power)

        return 1 / denominator_.sum(2)

    def norm(self, X):
        return np.sum(X ** 2)