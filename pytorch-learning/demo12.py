import torch


def test():

    data = torch.tensor([[10, 20, 30], [40, 50, 60]])

    # 1. 使用 shape 属性或者 size 方法都可以获得张量的形状
    print(data.shape, data.shape[0], data.shape[1])
    print(data.size(), data.size(0), data.size(1))

    # 2. 使用 reshape 函数修改张量形状
    new_data = data.reshape(1, 6)
    print(new_data.shape)


if __name__ == "__main__":
    test()
