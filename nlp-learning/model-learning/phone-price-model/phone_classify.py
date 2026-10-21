import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


# 构建数据集
def create_dataset():
    data = pd.read_csv("data/手机价格预测.csv")

    # 提取特征值和目标值
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # 获取输入特征维度与分类类别数
    input_dim = x.shape[1]
    num_classes = y.nunique()  

    # 特征用 float32，标签用 int64（适配 PyTorch 分类损失函数）
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 划分数据集（分层抽样）
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)


    #【核心归一化步骤】
    # 实例化标准化器：让每个特征的均值为 0，方差为 1
    scaler = StandardScaler()

    # 训练集：fit（计算训练集的均值、标准差） + transform（应用变换）
    x_train = scaler.fit_transform(x_train)

    # 验证集：只能 transform（使用训练集的均值、标准差，防止数据穿越/数据泄露）
    x_valid = scaler.transform(x_valid)

    # 构建 pytorch 数据集
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), 
        torch.from_numpy(y_train.to_numpy()).long()
    )

    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(), 
        torch.from_numpy(y_valid.to_numpy()).long()
    )

    # 返回数据集，其中 x_train.shape[1] 返回训练数据的维度（列）,len(np.unique(y) 标注对应多少种标签
    return train_dataset, valid_dataset, input_dim, num_classes

