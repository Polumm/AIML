# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 23:07
# @Author  : 宋楚嘉
# @FileName: LinearRegression.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

class LinearRegression:

    def __init__(self, alpha=0.03, n_iterations=1500):
        # alpha为学习率，n_iterations为迭代次数
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 获取训练数据的样本数和特征数
        num_samples, num_features = X.shape
        # 初始化权重和偏置
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0.0

        # 梯度下降
        for iteration in range(self.n_iterations):
            # 使用当前的权重和偏置进行预测
            predicted = self.predict(X)
            # 计算偏置的梯度
            db = sum([(y_i - predicted_i) for y_i, predicted_i in zip(y, predicted)]) / num_samples
            # 初始化权重梯度列表
            dw = [0.0 for _ in range(num_features)]
            # 计算每个权重的梯度
            for i in range(num_samples):
                for j in range(num_features):
                    dw[j] += -2 * X[i][j] * (y[i] - predicted[i]) / num_samples
            # 更新权重和偏置
            self.weights = [w - self.alpha * dw_i for w, dw_i in zip(self.weights, dw)]
            self.bias -= self.alpha * db

            # 每100次迭代，打印损失值
            if (iteration+1) % 100 == 0:
                loss = sum([(y_i - predicted_i)**2 for y_i, predicted_i in zip(y, predicted)] ) / (2*num_samples)
                print("Iteration: {}, Loss: {}".format(iteration+1, loss))

    def predict(self, X):
        # 使用权重和偏置进行预测
        return [sum(w_j * x_j for w_j, x_j in zip(self.weights, x)) + self.bias for x in X]
