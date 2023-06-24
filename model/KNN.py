# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 23:07
# @Author  : 宋楚嘉
# @FileName: KNN.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from collections import Counter
import time

class KNN:
    def __init__(self, k=3, dist_measure="euclidean", use_kdtree=True):
        self.k = k
        self.dist_measure = dist_measure
        self.use_kdtree = use_kdtree

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        if self.use_kdtree and self.dist_measure !="minkowski" :
            self.tree = KDTree(X_train, metric=self.dist_measure)
        elif self.use_kdtree:
            self.tree = KDTree(X_train, metric=self.dist_measure, p=3)

    def predict(self, X_test):
        start_time = time.time()
        predicted_labels = [self._predict(x) for x in X_test]
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.2f} seconds")
        return np.array(predicted_labels)

    def _predict(self, x):
        if self.use_kdtree:
            if self.dist_measure == "euclidean" or self.dist_measure == "manhattan" or self.dist_measure == "minkowski":
                _, ind = self.tree.query([x], k=self.k)
            else:
                raise ValueError("您输入的距离测度不存在，请从以下方法选择：'euclidean', 'manhattan', 'minkowski'.")
            k_nearest_labels = [self.y_train[i] for i in ind[0]]
        else:
            if self.dist_measure == "euclidean":
                distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
            elif self.dist_measure == "manhattan":
                distances = [distance.cityblock(x, x_train) for x_train in self.X_train]
            elif self.dist_measure == "minkowski":
                distances = [distance.minkowski(x, x_train) for x_train in self.X_train]
            else:
                raise ValueError("您输入的距离测度不存在，请从以下方法选择：'euclidean', 'manhattan', 'minkowski'.")
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 少数服从多数的投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]