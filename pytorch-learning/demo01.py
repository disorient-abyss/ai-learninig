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

# 1. 根据已有数据创建张量
def test01():

   # 1.1  创建张量标量
   x = 10
   print(f"x={x}")
   print({type(x)})
   
   data = torch.tensor(10)
   print(data)
   print(type(data))
   
   # 1.2  numpy 数组，由于 data 为 float64，下面代码也使用该类型
   data = np.random.randn(2, 3)
   data = torch.tensor(data)
   print(data)

   # 1.3  列表，下面代码使用默认元素类型 float32
   data = ([[10.,20.,30.],[40.,50.,60.]])
   data = torch.tensor(data)
   print(data)
   print(data.dtype)

# 2. 创建指定形状的张量
def test02():
   
   # 2.1  创建2行3列的张量，默认dtype为float32
   data = torch.Tensor(2, 3)
   print(data)
   
   # 2.2  注意：如果传递列表，则创建包含指定元素的张量
   data = torch.Tensor([10])
   print(data)

   data = torch.Tensor([10, 20])
   print(data)
   
# 3. 使用具体类型的张量
def test03():
   # 3.1  创建2行3列，dtype为int32的张量
   data = torch.IntTensor(2, 3)	
   print(data)

   # 3.2  注意：如果传递元素类型不正确，则会进行数据类型转换
   data = torch.IntTensor([2.9, 3.3])	
   print(data)

   # 3.3  其他的类型
   data = torch.ShortTensor([2, 3])    # int16
   print(data)
   data = torch.LongTensor([2, 3])     #int 64
   print(data)
   data = torch.FloatTensor([2.5, 3.3])    #float 32
   print(data)
   data = torch.DoubleTensor([-2.5, -3.3])   #float 64
   print(data)
   
   # 3.4 自动判断,默认为float 32
   data = torch.Tensor([2.5, 3.3])   #float 32
   print(data)
   print(data.dtype)
   
if __name__ == '__main__':
   print("--------------")
   print("test01:")
   test01()
   print("--------------")
   print("test02:")
   test02()
   print("--------------")
   print("test03:")
   test03()
   print("--------------")
