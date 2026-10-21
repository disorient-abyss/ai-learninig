import torch
from torch import nn


def test():
    # 初始化丢弃层
    dropout = nn.Dropout(p=0.8)
    # 初始化输入数据
    inputs = torch.randint(0, 10, size=[5, 8]).float()
    print(inputs)
    print('-' * 50)

    outputs = dropout(inputs)
    print(outputs)

if __name__ == '__main__':
    test()