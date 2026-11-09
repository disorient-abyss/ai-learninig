import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

# 定义⼀个简单的⽹络，但要将nn.Linear替换为bnb.nn.Linear4bit以模拟量化

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = bnb.nn.Linear4bit(10, 20)
        # 4bit量化线性层
        self.relu = nn.ReLU()
        self.fc2 = bnb.nn.Linear4bit(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 检查 GPU 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
  # 实例化模型
  net1 = SimpleNet()
  print("=== 原始量化模型结构 ===")
  print(net1)

  # 配置 LoRA：针对 fc1 层应用适配器
  config = LoraConfig(
      r=8,
      lora_alpha=16,
      target_modules=["fc1"],
      lora_dropout=0.05,
      bias="none",
  )

  # 包装为 PEFT 模型 (QLoRA)
  net2 = get_peft_model(net1, config)

  print("\n--------------------")
  print("=== 应用 LoRA 后的模型结构 ===")
  print(net2)

  print("\n=== 可训练参数统计 ===")
  net2.print_trainable_parameters()
else:
  print("bitsandbytes 4-bit 线性层需要 CUDA 环境支持，请在配有 GPU 的环境中运行。")