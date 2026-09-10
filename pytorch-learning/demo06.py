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
    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    # 第一种方式
    data = torch.mul(data1, data2)
    print(data)
    print("-" * 50)

    # 第二种方法
    data = data1 * data2
    print(data)
    print("-" * 50)


if __name__ == "__main__":
    test()
