import numpy as np
import torch


# 1. 将张量转换为 numpy 数组
def test00():

    data_tensor = torch.tensor([2, 3, 4])
    # 使用张量对象中的 numpy 函数进行转换
    data_numpy = data_tensor.numpy()

    print(type(data_tensor))
    print(type(data_numpy))

    # 注意: data_tensor 和 data_numpy 共享内存
    # 修改其中的一个，另外一个也会发生改变
    # data_tensor[0] = 100
    data_numpy[0] = 100

    print(data_tensor)
    print(data_numpy)

# 2. 使用from_numpy函数
def test01():
    data_numpy = np.array([2, 3, 4])
    # 将numpy数组转换为张量类型
    # 1. from_numpy
    # 2. torch.tensor(ndarray)

    # 浅拷贝
    data_tensor = torch.from_numpy(data_numpy)

    # numpy和tensor共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100

    print(data_tensor)
    print(data_numpy)

# 3. 使用 torch.tensor 函数
def test02():
    data_numpy = np.array([2, 3, 4])

    data_tensor = torch.tensor(data_numpy)

    # numpy 和 tensor 不共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100

    print(data_tensor)
    print(data_numpy)

# 4. 标量张量和数字的转换
def test03():
    # 当张量只包含一个元素时，可以通过 item 函数提取出该值
    data = torch.tensor([30,])
    print(type(data))
    print(data.item())

    data = torch.tensor(30)
    print(type(data))
    print(data.item())

    # data1 = torch.tensor([1.55, 2.55])
    data1 = torch.tensor([1.55])
    print(data1.item())
    print(data1.item() == 1.55)

if __name__ == "__main__":
    # test00()
    # test01()
    # test02()
    test03()
