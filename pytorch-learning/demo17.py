import torch


# 1. 控制不计算梯度
def test01():
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64, device="cuda")
    print(x.requires_grad)

    # 第一种方式：对代码进行装饰
    with torch.no_grad():
        y = x**2
        print(y.requires_grad)

    # 第二种方式：对函数进行装饰
    @torch.no_grad()
    def my_func(x):
        return x**2

    print(my_func(x).requires_grad)

    # 第三种方式
    torch.set_grad_enabled(False)
    y = x**2
    print(y.requires_grad)


# 2. 注意：累计梯度
def test02():
    # 定义需要求导张量
    x = torch.tensor(
        [10, 20, 30, 40], requires_grad=True, dtype=torch.float64, device="cuda"
    )

    for _ in range(3):
        f1 = x**2 + 20
        f2 = f1.mean()

        # 默认张量的 grad 属性会累计历史梯度值
        # 所以，需要我们每次手动清理上次的梯度
        # 注意：一开始梯度不存在，需要做判断
        if x.grad is not None:
            x.grad.data.zero_()

        f2.backward()
        print(x.grad)
        print("-" * 50)


# 3. 梯度下降优化最优解
def test03():
    # y = x**2
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64, device="cuda")
    count = 0
    for _ in range(50000):
        # 正向计算
        f = x**2

        # 梯度清零
        if x.grad is not None:
            x.grad.data.zero_()

        # 反向传播计算梯度
        f.backward()

        # 更新参数
        if x.grad is not None:
            x.data = x.data - 0.001 * x.grad

            count += 1
            if count % 100 == 0:
                print("%.10f" % x.data)


if __name__ == "__main__":
    # test01()
    # print('----------------------------')
    # test02()
    # print('----------------------------')
    test03()
