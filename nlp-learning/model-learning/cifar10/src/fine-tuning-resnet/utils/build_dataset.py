# coding: UTF-8
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def build_dataset(config, only_test=False, dev_split=0.1):
    # 1. 验证/测试图像预处理（无数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # 2. 如果仅测试，直接构建并返回测试集（保持 run_test.py 的兼容性）
    if only_test:
        test_dataset = datasets.CIFAR10(
            root=str(config.data_path),
            train=False,
            download=False,
            transform=test_transform,
        )
        return None, test_dataset

    # 3. 训练集增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # 4. 加载数据集底层对象
    base_train = datasets.CIFAR10(
        root=str(config.data_path),
        train=True,
        download=False,
        transform=train_transform,
    )
    base_dev = datasets.CIFAR10(
        root=str(config.data_path),
        train=True,
        download=False,
        transform=test_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(config.data_path),
        train=False,
        download=False,
        transform=test_transform,
    )

    # 5. 固定随机种子划分为 Train 和 Dev
    num_train = len(base_train)
    dev_size = int(num_train * dev_split)
    train_size = num_train - dev_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_train, generator=generator).tolist()
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:]

    train_dataset = Subset(base_train, train_indices)
    dev_dataset = Subset(base_dev, dev_indices)

    return train_dataset, dev_dataset, test_dataset