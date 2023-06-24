import numpy as np
import time

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None  # 初始化类别
        self.mean_std = None  # 初始化均值和标准差

    def mean(self, X):
        return np.mean(X, axis=0)  # 计算特征的均值

    def std(self, X):
        return np.std(X, axis=0)  # 计算特征的标准差

    def gaussian_probability(self, x, mean, std):
        # 计算高斯概率密度
        exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def fit(self, X, y):
        self.classes = np.unique(y)  # 获取所有的类别
        self.mean_std = {}  # 初始化均值和标准差的存储字典
        for c in self.classes:
            X_c = X[y == c]
            self.mean_std[c] = {
                'mean': self.mean(X_c),  # 计算并存储每个类别的特征均值
                'std': self.std(X_c)  # 计算并存储每个类别的特征标准差
            }

    def predict_single(self, x):
        probabilities = {}  # 初始化类别概率字典
        for c in self.classes:
            mean = self.mean_std[c]['mean']
            std = self.mean_std[c]['std']
            # 计算高斯概率
            gaussian_prob = self.gaussian_probability(x, mean, std)
            # 计算并存储每个类别的概率
            probabilities[c] = np.prod(gaussian_prob) * (1 / len(self.classes))
        # 返回概率最大的类别
        return max(probabilities, key=probabilities.get)

    def predict(self, X):
        start_time = time.time()  # 开始计时
        result = np.array([self.predict_single(x) for x in X])  # 对所有样本进行预测
        end_time = time.time()  # 结束计时
        # 打印预测耗时
        print(f"Prediction time: {end_time - start_time:.2f} seconds")
        return result  # 返回预测结果
