import torch

data = torch.randint(0, 10, [4, 5])
print(data)
print("-" * 50)

# 1. 简单行、列索引
def test01():

	print(data[0])
	print(data[:,0])
	print('-' * 50)

# 2. 列表索引
def test02():
    # 返回 (0, 1), (1, 2) 两个位置的元素
    print(data[[0, 1], [1, 2]])
    print('-' * 50)

    # 返回 0, 1 行的 1, 2 列共4个元素
    print(data[[[0], [1]], [1, 2]])
	
# 3. 范围索引
def test03():
    # 前3行的前2列数据
    print(data[:3, :2])

    # 第2行到最后的前2列数据
    print(data[2:, :2])
	
# 布尔索引
def test04():
    # 第2列大于5的行数据
    print(data[:, 2] > 5)
    print(data[data[:, 2] > 5])

    # 第1行大于5的列数据
    print(data[1] > 5)
    print(data[:, data[1] > 5])

# 多维索引
def test05():
    data = torch.randint(0, 10, [3, 4, 5])
    print(data)
    print('-' * 50)

    print(data[0, :, :])
    print(data[:, 0, :])
    print(data[:, :, 0])
    print(data.shape)

if __name__=='__main__':
	test01()
	test02()
	test03()
	test04()
	test05()