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

# 1. 创建线性空间的张量
def test01():
   
   # 1. 在指定区间按照步长生成元素([start, end), step)
   data = torch.arange(0, 10, 2)
   print(data)
   
# 2. 在指定区间按照元素个数生成 10 个值，包括 0 和 11
   data = torch.linspace(0, 11, 10)
   print(data)

# 2. 创建随机张量
def test02():
   
   # 1. 创建随机张量
   data = torch.randn(2, 3) # 创建2行3列张量
   print(data)

   # 2. 随机数种子设置
   print("随机数种子：", torch.initial_seed())
   torch.manual_seed(100)
   print("随机数种子：", torch.initial_seed())

# 3. 随机种子的复现作用测试
def test03():
   # 设置随机种子
   torch.manual_seed(100)

   #创建一个随机初始化的张量
   tensor_1 = torch.randn(3, 3)
   print("Tensor1:\n", tensor_1)

   # 再次设置相同的随机种子
   # torch.manual_seed(100)

   #再次创建一个随机初始化的张量
   tensor_2 = torch.randn(3, 3)
   print("Tensor2:\n", tensor_2)

   #检查两个张量是否相同
   print("Are tensors equal?\n", torch.equal(tensor_1, tensor_2))

# 4. 对比封装numpy和pytorch生成的张量
def test04():
   tensor1 = torch.tensor(np.random.randn(2, 3))
   print(tensor1)

   tensor2 = torch.randn(2, 3)
   print(tensor2)
#numpy默认引用float64类型的数据，pytorch默认是float32类型的数据。
   
if __name__ == "__main__":
   # test01()
   # test02()
   # test03()
   test04()