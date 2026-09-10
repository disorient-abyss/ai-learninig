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

# 1. 创建全0张量
def test01():
    print("test01------------------------")
    # 1.1  创建指定形状的全0张量
    data = torch.zeros(2, 3)
    print(data)

    # 1.2  根据张量形状创建全0张量
    data = torch.zeros_like(data)
    print(data)

# 2. 创建全1张量
def test02():
    print("\ntest02------------------------")
    # 2.1 创建指定形状的全1张量
    data = torch.ones(2, 3)
    print(data)

    # 2.2  根据张量形状创建全1张量
    data = torch.ones_like(data)
    print(data)

# 3. 创建全为指定值的张量
def test03():
    print("\ntest03------------------------")
    # 3.1  创建指定形状指定值的张量
    data = torch.full([2, 3], 10)
    print(data)
    # 3.2  根据张量形状创建全指定值张量
    data = torch.full_like(data, 20)
    print(data)

if __name__ == "__main__":
    test01()
    test02()
    test03()