import numpy as np
import torch


def test01():
    data = torch.tensor([[10, 20, 30], [40, 50, 60]])
    print("data shape:", data.size())

    # 1. 使用 view 函数修改形状
    new_data = data.view(3, 2)
    print("new_data shape:", new_data.shape)

    # 2. 判断张量是否使用整块内存
    print("data:", data.is_contiguous())  # True

    # 3. 使用 transpose 函数修改形状
    new_data = torch.transpose(data, 0, 1)
    print("new_data.shape:", new_data.shape)
    print(new_data)
    print("new_data:", new_data.is_contiguous())  # False
    print("#" * 50)

    # new_data = new_data.view(2, 3)  # RuntimeError
    new_data = new_data.reshape(2, 3)
    print("new_data.shape:", new_data.shape)
    print(new_data.is_contiguous())

    new_data = new_data.reshape(1, 6)
    print("new_data:", new_data)
    print(new_data.shape)
    print(new_data.is_contiguous())
    temp_data1 = new_data.view(6, 1)
    print("temp_data1.shape:", temp_data1.shape)

    # new_data = torch.transpose(data, 0, 1)
    # print('new_data.shape:', new_data.shape)
    # print('new_data:', new_data.is_contiguous())

    # 需要先使用 contiguous 函数转换为整块内存的张量，再使用 view 函数
    # print(new_data.contiguous().is_contiguous())
    # new_data = new_data.contiguous().view(2, 3)
    # new_data = new_data.reshape(2, 3)
    # print('new_data shape:', new_data.shape)


def test02():
    data = torch.tensor(np.random.randint(0, 10, [1, 3, 1, 5]))
    print("data shape:", data.size())
    print(data)

    # 1. 去掉值为1的维度
    new_data = data.squeeze()
    print("new_data shape:", new_data.size())
    print(new_data)

    # 2. 去掉指定位置为1的维度，注意：如果指定位置不是1，则不删除
    new_data = data.squeeze(1)
    print("new_data shape:", new_data.size())

    # 3. 在2维度增加一个维度
    new_data = data.unsqueeze(-1)
    print("new_data shape:", new_data.size())


def test03():
    data = torch.tensor(np.random.randint(0, 10, [2, 4]))
    print("data:", data)

    data1 = data.reshape(4, 2)
    print(data1.shape)
    print(data1)

    data2 = torch.transpose(data, 0, 1)
    print(data2.shape)
    print(data2)


if __name__ == "__main__":
    test01()
    test02()
    test03()
