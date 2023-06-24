# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 23:07
# @Author  : 宋楚嘉
# @FileName: KMeans.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

from model.FCM import FCM
import numpy as np
import time
from model.DistanceCalculator import DistanceCalculator

class KMeans:
    def __init__(self, n_clusters, max_iter=1000, distance_metric='euclidean'):
        self.n_clusters = n_clusters  # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数
        self.centroids = None  # 聚类中心
        self.distance_calculator = DistanceCalculator(distance_metric)  # 距离计算方式

    def fit(self, X):
        start_time = time.time()  # 开始时间
        # 用FCM算法初始化聚类中心
        fcm = FCM(self.n_clusters, distance_metric=self.distance_calculator.metric)
        fcm.fit(X)
        self.centroids = fcm.centroids

        for _ in range(self.max_iter):
            # 为每个样本指定最近的聚类中心
            labels = self.predict(X)

            # 计算新的聚类中心
            new_centroids = []
            for i in range(self.n_clusters):
                if X[labels == i].size != 0:
                    new_centroids.append(X[labels == i].mean(axis=0))  # 对指定类别的数据求均值
                else:
                    new_centroids.append(self.centroids[i])

            new_centroids = np.array(new_centroids)

            # 检查聚类中心是否改变
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids

        end_time = time.time()  # 结束时间
        print(f"训练时间: {end_time - start_time:.2f} seconds")  # 打印训练时间
        return self

    def predict(self, X):
        # 对于给定的数据点，找到距离最近的聚类中心
        return np.argmin(self.distance_calculator.calculate(X, self.centroids), axis=1)
