# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 23:07
# @Author  : 宋楚嘉
# @FileName: utils.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 获取所有不同的元素及它们在原始数组中出现的次数
def count_elements(array):
    unique_elements, counts = np.unique(array, return_counts=True)
    # 输出所有不同的元素及它们在原始数组中出现的次数
    for element, count in zip(unique_elements, counts):
        print('Element:', element, 'Count:', count)
    # 返回结果
    return unique_elements, counts

def valid(y_test, pred_test):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, pred_test)

    # 计算 TP, FP, TN, FN
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    epsilon = 1e-10

    # 计算 Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN + epsilon)
    print(f'Accuracy: {np.mean(accuracy):.4f}')

    # 计算 Precision
    precision = TP / (TP + FP + epsilon)
    print(f'precision: {np.mean(precision):.4f}')
    # 计算 Recall
    recall = TP / (TP + FN + epsilon)
    print(f'recall: {np.mean(recall):.4f}')
    # 计算 F1 score
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + epsilon) , 0)
    print(f'f1: {np.mean(f1):.4f}')
    # 计算Specificity
    specificity = TN / (TN + FP + epsilon)
    print(f'Specificity: {np.mean(specificity):.4f}')

    # 计算每个类别的权重
    class_weights = np.sum(cm, axis=1) / (np.sum(cm) + epsilon)

    # 宏平均 - 每个类别的指标都有相同的权重
    macro_f1 = np.mean(f1)
    print(f'macro_f1: {macro_f1:.4f}')

    # 微平均 - 总体表现，所有类别的样本都有相同的权重
    micro_precision = TP.sum() / (TP.sum() + FP.sum() + epsilon)
    micro_recall = TP.sum() / (TP.sum() + FN.sum() + epsilon)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + epsilon)
    print(f'micro_f1: {micro_f1:.4f}')

    # 加权平均 - 每个类别的指标根据该类别在样本中的出现频率进行加权
    weighted_f1 = np.sum(f1 * class_weights)
    print(f'weighted_f1: {weighted_f1:.4f}')

    # 计算Kappa指数
    total_samples = np.sum(cm)
    Po = np.sum(np.diag(cm)) / total_samples

    # 计算每一行和每一列的和
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)

    # 计算Pe
    Pe = np.sum((row_sum * col_sum)) / (total_samples ** 2 + epsilon)

    kappa = (Po - Pe) / (1 - Pe + epsilon)
    print(f'kappa: {kappa:.4f}')

# HSI数据预处理
def prepHSI(data, gt, n_components):
    # 三维展平为（像素，特征）的二维张量
    data_reshaped = data.reshape(data.shape[0]*data.shape[1],-1)
    df = pd.DataFrame(data_reshaped)
    print(f'Data Shape: {data.shape[:-1]}\nNumber of Bands: {data.shape[-1]}')

    # 对全部像素进行标准化
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.iloc[:,:])  # 只对特征进行缩放，不对'class'列进行缩放

    # 对全部像素执行PCA降维
    pca = PCA(n_components = n_components)  # 你想要的维度数量
    data_pca = pca.fit_transform(df_scaled)

    # 根据类别标签剔除无实地类别的像元；
    df_pca_all = pd.DataFrame(data=data_pca)
    df_pca_all['class'] = gt.ravel()
    df_pca = df_pca_all[df_pca_all['class']!=0]

    # 得到标签向量
    y = df_pca['class']  # 这是标签的DataFrame

    # 移除class列，得到训练数据的特征矩阵
    X = df_pca.drop(columns=['class'])  # 这是训练数据的DataFrame
    return X.values, y.values, df_pca_all.drop(columns=['class']).values #转化为纯净的numpy，避免引入DataFrame的索引

# 绘制HSI分类效果图
def draw(predAll,datasetName,modelName,expName="",X_test="",y_pred="",model=""):
    plt.ion() #交互模式，避免后续阻塞进程
    if modelName == "KMeans":
        # 可视化结果
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
        plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='x')
        plt.title('KMeans Clustering on '+ datasetName +' Dataset')
        plt.savefig('figures/' + str(expName) + datasetName + '_' + modelName + '_cluster.png', dpi=1200, bbox_inches='tight',pad_inches=0)
        plt.show()
        plt.pause(2)
        plt.close()
        # return
    if datasetName == "Iris":
        return
    if datasetName == "PaviaU":
        forshow = predAll.reshape(610, 340)
        colors = [  # '#000000',
            '#c7c9cb',
            '#6eb145',
            '#69bcc8',
            '#3a7a43',
            '#9d5796',
            '#945332',
            '#752d79',
            '#da332c',
            '#e0db54',
        ]
    if datasetName == "IndianPine":
        forshow = predAll.reshape(145, 145)
        colors = [  # '#000000',
            '#53ab48',
            '#89ba43',
            '#42845b',
            '#3c8345',
            '#905236',
            '#69bcc8',
            '#ffffff',
            '#c7b0c9',
            '#da332c',
            '#772324',
            '#57585a',
            '#e0db54',
            '#d98e34',
            '#54307e',
            '#e3775b',
            '#9d5796',
        ]
    cmap = ListedColormap(colors)
    plt.figure(figsize=(forshow.shape[1] / 100, forshow.shape[0] / 100))
    plt.imshow(forshow, cmap=cmap, extent=[0, forshow.shape[1], forshow.shape[0], 0], interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()  # 自动调整子图或坐标轴的位置，避免重叠或裁剪
    plt.savefig('figures/' +str(expName) + datasetName+ '_' + modelName + '.png', dpi=1200, bbox_inches='tight', pad_inches=0)
    # 展示两秒后关闭，避免阻塞线程
    plt.show()
    plt.pause(2)
    plt.close()

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

# 计算轮廓系数
def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    silhouette_vals = []
    for label in unique_labels:
        cluster_i = X[labels == label]
        other_clusters = X[labels != label]
        if len(cluster_i) < 2:
            # 如果簇内的数据少于2，不计算轮廓系数
            continue
        # 计算样本i到同簇其他样本的平均距离a(i)
        a = np.mean([euclidean_distance(x_i, cluster_i) for x_i in cluster_i])
        # 计算样本i到其他某簇所有样本的平均距离b(i)
        b = np.min([np.mean([euclidean_distance(x_i, other_cluster) for x_i in other_clusters]) for other_cluster in other_clusters])
        silhouette = (b - a) / max(a, b)
        silhouette_vals.append(silhouette)
    return np.mean(silhouette_vals)

# 计算Calinski-Harabasz Index
def calinski_harabasz_score(X, labels):
    n_clusters = len(np.unique(labels))
    cluster_means = [np.mean(X[labels == label], axis=0) for label in np.unique(labels)]
    grand_mean = np.mean(X, axis=0)

    # 计算簇间距离
    between_cluster_dispersion = np.sum([len(X[labels == label]) * euclidean_distance(cluster_mean, grand_mean)**2 for label, cluster_mean in enumerate(cluster_means)])
    # 计算簇内距离
    within_cluster_dispersion = np.sum([np.sum(euclidean_distance(X[labels == label], cluster_mean)**2) for label, cluster_mean in enumerate(cluster_means)])

    score = (between_cluster_dispersion / within_cluster_dispersion) * ((len(X) - n_clusters) / (n_clusters - 1.0))
    return score

def valid_kmeans(model, X):
    labels = model.predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print("轮廓系数为:", silhouette_avg)

    calinski_harabasz_avg = calinski_harabasz_score(X, labels)
    print("Calinski-Harabasz分数为:", calinski_harabasz_avg)


def pre_boston_house(train_data, test_data):
    # 对训练集和测试集进行X，Y分离
    train_X, train_Y = feature_label_split(train_data)
    test_X, test_Y = feature_label_split(test_data)

    # 对X（包括train_X, test_X）进行归一化处理，方便后续操作
    unif_trainX, X_max, X_min = uniform_norm(train_X)
    unif_testX = (test_X - X_min) / (X_max - X_min)

    return unif_trainX, train_Y, unif_testX, test_Y


def feature_label_split(pd_data):
    # 行数、列数
    row_cnt = pd_data.shape[0]
    column_cnt = len(pd_data.iloc[0, 0].split())
    # 生成新的X、Y矩阵
    X = np.empty([row_cnt, column_cnt - 1])  # 生成两个随机未初始化的矩阵
    Y = np.empty([row_cnt, 1])
    for i in range(0, row_cnt):
        row_array = pd_data.iloc[i, 0].split()
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y


def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in - X_min) / (X_max - X_min)
    return X, X_max, X_min

def mean_squared_error(y_true, y_pred):
    sum_error = 0.0
    for i in range(len(y_true)):
        prediction_error = y_pred[i] - y_true[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(y_true))
    return mean_error


# 读取输入数据，并返回相应的分词后的特征集和标识集
def get_data(data_file_path):
    lines = []
    labels = []

    for line in open(data_file_path, "r", encoding="utf-8"):
        # 对每一个行样本信息按照制表符进行分割得到list
        # 第一行为标签，第二行为原短信内容，第三行为分词处理之后的短信内容
        arr = line.rstrip().split("\t")
        # 如果样本信息不完整（不足三行）就丢弃
        if len(arr) < 3:
            continue

        # 读取标签，并当负标签为-1时，对其进行转化为0
        if int(arr[0]) == 1:
            label = 1
        elif int(arr[0]) == 0 or int(arr[0]) == -1:
            label = 0
        else:
            continue
        labels.append(label)

        # 读取分词之后的句子
        line = arr[2].split()
        lines.append(line)

    return lines, labels


# 创建词袋字典
def create_vocab_dict(data_lines):
    vocab_dict = {}
    for data_line in data_lines:
        for word in data_line:
            if word in vocab_dict:
                vocab_dict[word] += 1
            else:
                vocab_dict[word] = 1
    return vocab_dict


# 得到输入分词后的样本的BOW特征
def BOW_feature(vocab_list, input_line):
    return_vec = np.zeros(len(vocab_list), )
    for word in input_line:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def preMudou(train_file_path, test_file_path):
    # 得到分离后的特征和标记
    train_data, train_label = get_data(train_file_path)
    test_data, test_label = get_data(test_file_path)

    # 构建文本的BOW词袋特征
    # BOW字典
    vocab_dict = create_vocab_dict(train_data)
    # 对字典按照value进行排序
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda d: d[1], reverse=True)
    # 筛出字典中value小于min_freq的键值对，并生成相应的键队列，得到BOW特征
    min_freq = 5
    vocab_list = [v[0] for v in sorted_vocab_list if int(v[1]) > min_freq]

    # 生成文本的BOW特征
    train_X = []
    for one_msg in train_data:
        train_X.append(BOW_feature(vocab_list, one_msg))  # 这里应使用 one_msg 而非 train_data

    test_X = []
    for one_msg in test_data:
        test_X.append(BOW_feature(vocab_list, one_msg))  # 这里应使用 one_msg 而非 test_data

    # 将数据格式转化为 numpy.array 格式
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_X, train_label, test_X, test_label

def roc_auc_score(y_true, y_score):
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    sort_inds = np.argsort(y_score)
    y_true_sorted = y_true[sort_inds]
    fp = np.cumsum(y_true_sorted == 0)  # false positive
    tp = np.cumsum(y_true_sorted == 1)  # true positive
    fp_rate = fp / n_neg
    tp_rate = tp / n_pos
    auc = np.sum(tp_rate[y_true_sorted == 0]) / n_neg / n_pos
    return auc, fp_rate, tp_rate
