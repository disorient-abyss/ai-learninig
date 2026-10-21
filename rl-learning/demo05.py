import torch.nn as nn

# 1. 均匀分布随机初始化
def test01():
    linear = nn.Linear(5, 3)
    # 从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    print(linear.weight.data)
	
# 2. 固定初始化
def test02():
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 5)
    print(linear.weight.data)
	
# 3. 全0初始化
def test03():
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    print(linear.weight.data)

# 4. 全1初始化
def test04():
    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight.data)

# 5. 正态分布随机初始化
def test05():
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(linear.weight.data)

# 6. kaiming 初始化
def test06():
    # kaiming 正态分布初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight.data)
    # kaiming 均匀分布初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight)
    
# 7. xavier 初始化
def test07():

    # xavier 正态分布初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight.data)

    # xavier 均匀分布初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)

if __name__ == '__main__':
    test01()
    test02()
    test03()
    test04()
    test05()
    test06()
    test07()