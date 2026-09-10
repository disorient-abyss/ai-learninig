import torch


def test():
    data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
    print(data)
    print("-" * 50)

    # 1. 计算均值
    # 注意：tensor 必须为 Float 或者 Double 类型
    print(data.mean())
    print(data.mean(dim=0))  # 按列计算均值
    print(data.mean(dim=1))  # 按行计算均值
    print("-" * 50)

    # 2. 计算总和
    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))
    print("-" * 50)

    # 3. 计算平方
    print(data.pow(2))
    print("-" * 50)

    # 4. 计算平方根
    print(data.sqrt())
    print("-" * 50)

    # 5. 指数计算, e^n 次方
    print(data.exp())
    print("-" * 50)

    # 6. 对数计算
    print(data.log())  # 以 e 为底
    print(data.log2())
    print(data.log10())

if __name__ == "__main__":
    test()
