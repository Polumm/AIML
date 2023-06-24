# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 23:07
# @Author  : 宋楚嘉
# @FileName: run.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# 在其他文件夹下实现
from model.KNN import KNN
from model.NaiveBayes import GaussianNaiveBayes
from model.MultinomialNaiveBayes import MultinomialNaiveBayes
from model.Kmeans import KMeans
from model.LinearRegression import LinearRegression
from model.LogisticRegression import LogisticRegression
from model.DecisionTree import DecisionTreeClassifier
from model.LeNet import LeNet
# 数据预处理、效果可视化与模型评估
from utils.utils import draw
from utils.utils import valid
from utils.utils import prepHSI # 高光谱数据预处理
from utils.utils import pre_boston_house # 波士顿房价预处理
from utils.utils import mean_squared_error
from utils.utils import preMudou
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import roc_auc_score
# from utils.utils import silhouette_score, calinski_harabasz_score
# from utils.utils import valid_kmeans
# from utils.utils import count_elements
# from utils.utils import roc_auc_score

# 深度学习
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 训练函数
def train(expName, datasetName, modelName, test_size=0.8, use_kdtree=True, k=9, dist_measure='euclidean', n_components='3', n_clusters=3, max_iter=1000):
    ## 读取数据
    if datasetName == 'Iris':
        # 读取数据集
        data = pd.read_csv('dataset/Iris/iris.data.txt', header=None)
        data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        # 将类别转换为数值
        le = LabelEncoder()
        data['class'] = le.fit_transform(data['class'])
        # 分割数据
        X = data.drop('class', axis=1).values
        y = data['class'].values
    elif datasetName == 'IndianPine':
        data = loadmat(r'dataset/IndianPine/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = loadmat(r'dataset/IndianPine/Indian_pines_gt.mat')['indian_pines_gt']
        X, y, X_all = prepHSI(data,gt,n_components)
    elif datasetName == 'PaviaU':
        data = loadmat(r'dataset/Pavia/paviaU.mat')['paviaU']
        gt = loadmat(r'dataset/Pavia/paviaU_gt.mat')['Data_gt']
        X, y, X_all = prepHSI(data,gt,n_components)
    elif datasetName == 'Bouston_house':
        # 读取训练集和测试集文件
        train_data = pd.read_csv("dataset/Bouston/boston_house.train", header=None)
        test_data = pd.read_csv("dataset/Bouston/boston_house.test", header=None)
        unif_trainX, train_Y, unif_testX, test_Y = pre_boston_house(train_data, test_data)
        # 模型训练
        model = LinearRegression(alpha=0.01, n_iterations=2000)
        model.fit(unif_trainX, train_Y)
        # 预测
        test_pred = model.predict(unif_testX)
        # 计算误差
        test_pred_error = mean_squared_error(test_Y, test_pred) / 2
        print("Test error is %f" % (test_pred_error))
        return
    elif datasetName == 'Mudou':
        # 数据集文件地址
        train_file_path = "dataset/Mudou/mudou_spam.train"
        test_file_path = "dataset/Mudou/mudou_spam.test"

        train_X, train_label, test_X, test_label = preMudou(train_file_path, test_file_path)

        # 训练模型
        model = LogisticRegression()
        model.fit(train_X, train_label)
        pred = model.predict(test_X)

        # 模型评估
        train_acc = model.score(train_X, train_label)
        print("Train accuracy: ", train_acc)
        test_acc = model.score(test_X, test_label)
        print("Test accuracy: ", test_acc)

        pred_prob_y = model.predict_proba(test_X)
        test_auc = roc_auc_score(test_label, pred_prob_y)
        print("Test AUC: ", test_auc)
        return
    elif datasetName == 'Minist':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        train_data = pd.DataFrame(pd.read_csv("D:\APythonWorkSpace\LeNet_learning/mnist_train.csv"))
        test_data = pd.DataFrame(pd.read_csv("D:\APythonWorkSpace\LeNet_learning/mnist_test.csv"))
        model = LeNet()
        print(model)
        loss_fc = nn.CrossEntropyLoss()  # 定义损失函数
        optimizer = optim.SGD(params=model.parameters(), lr=0.001)  # 采用随机梯度下降SGD
        loss_list = []  # 记录每次的损失值
        accuracy_list = []  # 记录每次的精度
        x = []  # 记录训练次数
        for i in range(1000):
            batch_data = train_data.sample(n=30, replace=False)  # 每次随机读取30条数据
            batch_y = torch.from_numpy(batch_data.iloc[:, 0].values).long()  # 标签值
            batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
            # 图片信息，一条数据784维将其转化为通道数为1，大小28*28的图片。

            prediction = model.forward(batch_x)  # 前向传播
            loss = loss_fc(prediction, batch_y)  # 计算损失值

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            print("第%d次训练，loss为%.3f" % (i, loss))
            loss_list.append(loss.detach().numpy())
            x.append(i)

            with torch.no_grad():  # 测试不需要反向传播
                batch_data = test_data.sample(n=50, replace=False)
                batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
                batch_y = batch_data.iloc[:, 0].values
                prediction = np.argmax(model(batch_x).numpy(), axis=1)
                accuracy = np.mean(prediction == batch_y)
                accuracy_list.append(accuracy)
                print("第%d组测试集，准确率为%.3f" % (i, accuracy))

        torch.save(model.state_dict(), "model/LeNet.pkl")  # 保存模型参数
        plt.ion()  # 交互模式，避免后续阻塞进程
        fig, ax1 = plt.subplots()
        ax1.plot(x, loss_list, "r-", label="Loss")  # 可以将损失值进行绘制
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss", color="r")
        ax2 = ax1.twinx()
        ax2.plot(x, accuracy_list, "b-", label="Accuracy")  # 可以将精度进行绘制
        ax2.set_ylabel("Accuracy", color="b")
        fig.tight_layout()
        plt.savefig('figures/' + str(expName) + datasetName + '_' + modelName + '.png', format='png',dpi=600)  # 将图保存为SVG矢量图
        plt.show()
        plt.pause(2)
        plt.close()
        return

    if modelName == 'KNN':
        model = KNN(k=k, dist_measure=dist_measure, use_kdtree=use_kdtree)
    elif modelName == 'GaussianNaiveBayes':
        model = GaussianNaiveBayes()
    elif modelName == 'MultinomialNaiveBayes':
        model = MultinomialNaiveBayes()
    elif modelName == 'KMeans':
        model = KMeans(n_clusters=n_clusters,max_iter=max_iter,distance_metric=dist_measure)
    elif modelName == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=3, random_state=1)

    print('\n')
    print('datasetName: ', datasetName)
    print('modelName: ', modelName)
    print('test_size: ', test_size)
    print('use_kdtree: ', use_kdtree)
    print('k: ', k)
    print('dist_measure: ', dist_measure)
    print('n_components: ', n_components)
    print('n_clusters: ', n_clusters)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = test_size, random_state = 0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.5, random_state=0)


    if modelName != 'KMeans':
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        valid(y_test, pred_test)
    else:
        model.fit(X_test) # 找到样本的聚类中心
        # valid_kmeans(model, X_test)
        labels = model.predict(X_test) # 分配标签
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X_test, labels)
        print("The average silhouette_score is :", silhouette_avg)

        # 计算Calinski-Harabasz Index
        calinski_harabasz_avg = calinski_harabasz_score(X_test, labels)
        print("The average calinski_harabasz_score is :", calinski_harabasz_avg)

    # 分类效果可视化
    if datasetName == 'Iris':
        draw('', datasetName, modelName, expName, X_test, y_test, model)
        return
    if modelName == 'KMeans':
        model.fit(X_all)
        predAll = model.predict(X_all)
        draw(predAll, datasetName, modelName, expName, X_test, y_test, model)
        # draw('', datasetName, modelName, expName, X_test, y_test, model)
    else:
        predAll = model.predict(X_all)
        draw(predAll, datasetName, modelName, expName)


# 按照指定编号执行不同的实验
def expsContent(expNo):
    if expNo==0:
        train(expNo, datasetName='Iris',modelName='GaussianNaiveBayes',test_size=0.6)
    elif expNo==1:
        train(expNo, datasetName="PaviaU", modelName="KNN", test_size=0.8, use_kdtree=True, k=3, dist_measure="euclidean",n_components=4)
    elif expNo == 2:
        train(expNo, datasetName="PaviaU", modelName="KNN", test_size=0.8, use_kdtree=True, k=9, dist_measure="euclidean",n_components=4)
    elif expNo == 3:
        train(expNo, datasetName="PaviaU", modelName="KNN", test_size=0.8, use_kdtree=True, k=27, dist_measure="euclidean",n_components=4)
    elif expNo == 4:
        train(expNo, datasetName="PaviaU", modelName="KNN", test_size=0.8, use_kdtree=True, k=9, dist_measure="euclidean",n_components=4)
    elif expNo == 5:
        train(expNo, datasetName="PaviaU", modelName="KNN", test_size=0.8, use_kdtree=True, k=9, dist_measure="manhattan",n_components=4)
    elif expNo == 6:
        train(expNo, datasetName="PaviaU", modelName="KNN", test_size=0.8, use_kdtree=True, k=9, dist_measure="minkowski",n_components=4)
    elif expNo == 7:
        train(expNo, datasetName="PaviaU", modelName="GaussianNaiveBayes", test_size=0.8,n_components=4)
    elif expNo == 8:
        train(expNo, datasetName="PaviaU", modelName="GaussianNaiveBayes", test_size=0.5,n_components=4)
    elif expNo == 9:
        train(expNo, datasetName="PaviaU", modelName="GaussianNaiveBayes", test_size=0.2,n_components=4)
    elif expNo == 10:
        train(expNo, datasetName="IndianPine", modelName="GaussianNaiveBayes", test_size=0.8, n_components=4)
    elif expNo == 11:
        train(expNo, datasetName="IndianPine", modelName="KNN", test_size=0.8, use_kdtree=True, k=9, dist_measure="euclidean",n_components=4)
    elif expNo == 12:
        train(expNo, datasetName="IndianPine", modelName="KNN", test_size=0.8, use_kdtree=False, k=9, dist_measure="euclidean",n_components=4)
    elif expNo == 13:
        train(expNo, datasetName="IndianPine", modelName="KMeans", test_size=0.8, n_components=4, n_clusters=16, dist_measure='euclidean')
    elif expNo == 14:
        train(expNo, datasetName="PaviaU", modelName="KMeans", test_size=0.8, n_components=4, n_clusters=9, dist_measure='euclidean')
    elif expNo == 15:
        train(expNo, datasetName="PaviaU", modelName="KMeans", test_size=0.8, n_components=4, n_clusters=9,dist_measure='manhattan')
    elif expNo == 16:
        train(expNo, datasetName="PaviaU", modelName="KMeans", test_size=0.8, n_components=4, n_clusters=9, dist_measure='minkowski')
    elif expNo == 17:
        train(expNo, datasetName="Iris", modelName="KMeans", test_size=0.8, n_clusters=3, dist_measure='euclidean')
    elif expNo == 18:
        train(expNo, datasetName="Iris", modelName="KMeans", test_size=0.8, n_clusters=3, dist_measure='manhattan')
    elif expNo == 19:
        train(expNo, datasetName="Iris", modelName="KMeans", test_size=0.8, n_clusters=3, dist_measure='minkowski')
    elif expNo == 20:
        train(expNo, datasetName="Bouston_house", modelName="LinearRegression")
    elif expNo == 21:
        train(expNo, datasetName="Mudou", modelName="LogisticRegression")
    elif expNo == 22:
        train(expNo, datasetName="Iris", modelName="DecisionTree")
    elif expNo == 23:
        train(expNo, datasetName="PaviaU", modelName="DecisionTree", n_components=4)
    elif expNo == 24:
        train(expNo, datasetName='Minist', modelName='LeNet')
if __name__ == "__main__":
    # # 方法一：训练指定参数的模型
    # train("", datasetName='Iris', modelName='KNN', test_size=0.6)

    #方法二：完成指定编号的实验
    expsContent(24)

    # # 方法三：批量实验
    # for expNo in range(0,25):
    #     print(f"正在进行实验 {expNo}...")
    #     expsContent(expNo)
    #     print(f"实验 {expNo} 已完成.\n")