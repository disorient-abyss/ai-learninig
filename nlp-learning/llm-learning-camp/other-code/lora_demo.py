from peft import LoraConfig, get_peft_model
from torch import nn

net1 = nn.Sequential(nn.Linear(10, 20),
                     nn.ReLU(),
                     nn.Linear(20, 2))

print(net1)

config = LoraConfig(target_modules=['0'])
net2 = get_peft_model(net1, config)

print('-' * 50)
print(net2)