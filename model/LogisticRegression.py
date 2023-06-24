import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 迭代次数
        self.weights = None  # 权重
        self.bias = None  # 偏差项

    # 定义sigmoid函数
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 模型训练函数
    def fit(self, X, y):
        n_samples, n_features = X.shape  # 获取样本数和特征数
        self.weights = np.zeros(n_features)  # 初始化权重
        self.bias = 0  # 初始化偏差项

        # 梯度下降
        for i in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias  # 计算线性模型预测值
            y_predicted = self._sigmoid(linear_model)  # 计算预测概率

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # 计算权重的梯度
            db = (1 / n_samples) * np.sum(y_predicted - y)  # 计算偏差项的梯度

            self.weights -= self.learning_rate * dw  # 更新权重
            self.bias -= self.learning_rate * db  # 更新偏差项

            # 每迭代100次，打印一次损失值
            if i % 100 == 0:
                loss = -y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted)
                print(f"Iteration {i}, Loss: {np.mean(loss)}")

    # 计算预测概率
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    # 预测类别
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

    # 计算准确率
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
