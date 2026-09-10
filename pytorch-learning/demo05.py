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

def test():
    data = torch.randint(0, 10, [2, 3])
    print(data)
    print("-" * 50)

    # 1. 不修改原数据
    new_data = data.add(10)  # 等价 new_data = data + 10
    print(new_data)
    print("-" * 50)

    # 2. 直接修改原数据
    # 注意：带下划线的函数为修改原数据本身
    print(data)
    data.add_(10)  # 等价 data.add(data.add(10)),其中"_"的意思是表示之前的表达式
    print(data)

    # 3. 其他函数
    print("-" * 50)
    print(data.sub(100))
    print(data.mul(100))
    print(data.div(100))
    print(data.neg())
    print(data.abs())


if __name__ == "__main__":
    test()
