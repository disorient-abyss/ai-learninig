import torch
from torch import nn

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


# 假设我们有一个模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建一个与保存时相同结构的模型
model = SimpleModel()

# 保存模型的参数
torch.save(model.state_dict(), "model_weights_20260910.pth")
print(model)
print("------------------")
print(model.state_dict())

# 加载模型的参数
model.load_state_dict(torch.load("model_weights_20260910.pth"))
print(model)
print("------------------")
print(model.state_dict())

# 保存完整的模型 --备注：官方建议大家只保存状态字典
# torch.save(model, 'model_weights_1.pth')
# print(model)

# model = torch.load('model_weights_1.pth')
# print(model)
