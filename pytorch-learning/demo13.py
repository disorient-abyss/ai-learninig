import numpy as np
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
    data = torch.tensor(np.random.randint(0, 10, [3, 4, 5]))
    print("data shape:", data.size())

    # 1. 交换1和2维度
    new_data = torch.transpose(data, 1, 2)
    print("data shape:", new_data.size())

    # 2. 将 data 的形状修改为 (4, 5, 3)
    new_data = torch.transpose(data, 0, 1)
    new_data = torch.transpose(new_data, 1, 2)
    print("new_data shape:", new_data.size())

    # 3. 使用 permute 函数将形状修改为 (4, 5, 3)
    new_data = torch.permute(data, [1, 2, 0])
    print("new_data shape:", new_data.size())


if __name__ == "__main__":
    test()
