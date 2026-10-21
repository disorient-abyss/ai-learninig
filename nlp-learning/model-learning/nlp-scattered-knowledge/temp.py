import numpy as np
import torch


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