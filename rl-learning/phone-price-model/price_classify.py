# 案例-价格分类
# 学习目标
# 掌握构建分类模型流程
# 小明创办了一家手机公司，他不知道如何估算手机产品的价格。为了解决这个问题，他收集了多家公司的手机销售数据。
# 我们需要帮助小明找出手机的功能（例如：RAM等）与其售价之间的某种关系。我们可以使用机器学习的方法来解决这个问题，也可以构建一个全连接的网络。
# 需要注意的是：在这个问题中，我们不需要预测实际价格，而是一个价格范围，它的范围使用0、1、2、3来表示，所以该问题也是一个分类问题。
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


# 1.构建数据集
def create_dataset():
    data  = pd.read_csv("data/手机价格预测.csv")

    # 特征值和目标值
    x, y = data.iloc[:,:-1], data.iloc[:, -1]
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)

    # 做标准化
    # 在 train_test_split 划分之后，对特征进行标准化：
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    # 构建数据集
    # 修正点：x 不需要 .values，y 需要 .values (或者用 torch.from_numpy)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)), 
        torch.tensor(y_train.values, dtype=torch.int64)
    )
    
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid.astype(np.float32)), 
        torch.tensor(y_valid.values, dtype=torch.int64)
    )

    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))



