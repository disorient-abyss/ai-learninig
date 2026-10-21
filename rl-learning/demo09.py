import torch
import torch.nn as nn

# 设置随机数种子
torch.manual_seed(0)

def caculate_gradient(x, w):

    y = x @ w
    y = y.sum()
    y.backward()
    print('Gradient:', w.grad.reshape(1, -1).squeeze().numpy())

def test01():

    # 初始化权重
    w = torch.randn(15, 1, requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0, 10, size=[5, 15]).float()
    # 计算梯度
    caculate_gradient(x, w)

def test02():

    # 初始化权重
    w = torch.randn(15, 1, requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0, 10, size=[5, 15]).float()
    # 初始化丢弃层
    dropout = nn.Dropout(p=0.8)
    x = dropout(x)
    # 计算梯度
    caculate_gradient(x, w)

if __name__ == '__main__':
    test01()
    print('-' * 70)
    test02()