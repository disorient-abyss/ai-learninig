import torch

# ==========================================
# 核心：配置 GPU 设备
# ==========================================
# 如果有英伟达显卡，device 会变成 "cuda"，否则是 "cpu"
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
# 核心：设置全局默认设备
torch.set_default_device(device)
print(f"当前全局默认设备是: {device}")

# 矩阵乘法
def test01():
    print( "test01()" + "-" * 50)
    data1 = torch.randint(0, 10, (3, 2))
    print("data1:\n", data1)
    data2 = torch.randint(0, 10, (2, 2))
    print("data2:\n", data2)

    # 第一种方法
    data  = data1 @ data2
    print(data)
    print('-' * 50)

    # 第二种方法  备注：这是推荐用法
    data = torch.matmul(data1, data2)
    print(data)
    print('-' * 50)

    # 第三种方法
    data = torch.mm(data1, data2)
    print(data)
    print('-' * 50)

# 2. torch.mm 和 torch.matmul 的区别
def test02():
    print( "\ntest02()" + "-" * 50)
    # matmul可以用于两个不同维度矩阵相乘，
    # 第一个张量：torch.randint(0, 10, (3, 4, 5))
    # 第二个张量：torch.randint(0, 10, (5, 4))
    # 而mm只能用于两个矩阵相乘且这两个矩阵是二维矩阵

    print(torch.matmul(torch.randn(3, 4, 5), torch.randn(5, 4)).shape)
    print(torch.matmul(torch.randn(5, 4), torch.randn(2, 3, 4, 5)).shape)

    # torch.mm() 只针对二维矩阵运算，超过二维张量会直接报错，所以建议使用 torch.matmul() 进行运算

    # print(torch.mm(torch.randn(2, 3, 4), torch.randn(4, 5)).shape)
    # print(torch.mm(torch.randn(5, 4), torch.randn(2, 3, 4, 5)).shape)
    print(torch.matmul(torch.randn(2, 3, 4), torch.randn(4, 5)).shape)
    print(torch.matmul(torch.randn(5, 4), torch.randn(2, 3, 4, 5)).shape)

# 3. torch.mm函数的用法
def test03():
    print( "\ntest03()" + "-" * 50)
    # 批量点积运算
    # 第一个维度为batch_size
    # 矩阵的二，三维要满足矩阵乘法规则

    # 使用 torch.bmm() 严格控制为 3 维张量，Batch 维度必须相等（第一个维度 B 必须完全一致），并且不支持广播机制
    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(3, 5, 8)

    data = torch.bmm(data1, data2)
    print(data.shape)

# 4. 简单展示 matmul 的张量的自动广播机制
def test04():
    print( "\ntest04()" + "-" * 50)
    data_1 = torch.randn(2, 3, 4, 5)
    data_2 = torch.randn(2, 1, 5, 8)
    data_3 = torch.matmul(data_1, data_2)
    print(f"data_3的形状为：{data_3.shape}")
# 以上面例子为例，就是把 dim = 1 的维度通过复制 2 份 dim = 2, 3 维度的数据进行填充，使 torch.Size(2, 1, 5, 8) -> torch.Size(2, 3, 5, 8)
# 为什么只有 1 时才能广播 --- 原因是 dim = x 的内容为 1 时，dim > x 的其它内容的数据才能当作唯一一份进行广播，如果内容大于 1 则系统无法选择需要进行广播的内容 

if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    test04()