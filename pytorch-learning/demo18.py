import torch
import torch.nn as nn

# ==========================================
# 核心：配置 GPU 设备
# ==========================================
# 如果有英伟达显卡，device 会变成 "cuda"，否则是 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的训练设备是: {device}")


# 假设我们有一个模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建一个与保存时相同结构的模型
model = SimpleModel()

# ==========================================
# 核心：将模型搬移到 GPU
# ==========================================
model = model.to(device)

# 保存模型的参数
torch.save(model.state_dict(), "model_weights_20260629.pth")
print(model)
print("------------------")
print(model.state_dict())

# 加载模型的参数
model.load_state_dict(torch.load("model_weights_20260629.pth"))
print(model)
print("------------------")
print(model.state_dict())

# 保存完整的模型 --备注：官方建议大家只保存状态字典
# torch.save(model, 'model_weights_1.pth')
# print(model)

# model = torch.load('model_weights_1.pth')
# print(model)
