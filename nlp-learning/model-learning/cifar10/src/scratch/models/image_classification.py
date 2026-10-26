import torch
import torch.nn.functional as F
from torch import nn


class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积块 1
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(9)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块 2
        self.conv2 = nn.Conv2d(9, 27, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(27)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块 3: 64 -> 128 (8x8 -> 4x4)
        self.conv3 = nn.Conv2d(27, 81, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(81)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接分类头
        self.linear1 = nn.Linear(81 * 4 * 4, 324)
        self.dropout = nn.Dropout(p=0.15)
        self.linear2 = nn.Linear(324, 81)
        self.linear3 = nn.Linear(81, 27)
        self.out = nn.Linear(27, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.out(x)