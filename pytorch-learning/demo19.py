import torch


# 1. detach 函数用法
def test01():
    x = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)

    # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # print(x.numpy())        # 错误
    print(x.detach().numpy())  # 正确


# 2. detach 前后张量共享内存
def test02():
    x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)

    # x2 作为叶子结点
    x2 = x1.detach()

    x2.data = torch.tensor([100, 200])
    print(x1)
    print(x2)

    # x2 不会自动计算梯度: False
    print(x2.requires_grad)


if __name__ == "__main__":
    test01()
    test02()
