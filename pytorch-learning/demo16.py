import torch


# 1. 单标量梯度的计算
# y = x**2 + 20
def test01():
    # 定义需要求导的张量
    # 张量的值类型必须是浮点类型
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)

    # 变量经过中间运算
    f = x**2 + 20

    # 自动微分
    f.backward()

    # 打印 x 变量的梯度
    # backward 函数计算的梯度值会存储在张量的 grad 变量中
    print(x.grad)
    print(x.data)

# 2. 单向量梯度的计算
# y = x**2 + 20
def test02():
    # 定义需要求导的张量
    x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float64)
    # 变量经过中间计算
    f1 = x**2 + 20

    # 注意：
    # 由于求导的结果必须是标量
    # 而 f 的结果是：tensor([120., 420.])
    # 所以，不能直接自动微分
    # 需要将结果计算为标量才能进行计算
    f2 = f1.mean()  # f2 = 1/2 * x

    # 自动微分
    f2.backward()

    # 打印 x 变量的梯度
    print(x.grad)


if __name__ == "__main__":
    test01()
    test02()
