import os
import sys

import torch
from models.image_classification import ImageClassification
from torch import nn
from torch.utils.data import DataLoader
from utils.build_dataset import build_dataset
from utils.train_eval import train

# 1. 路径解析（传统清晰写法）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(current_dir)

class Config:
    def __init__(self):
        # 数据集根目录定位到 cifar10/data
        self.data_path = os.path.join(project_root, "data")
        self.batch_size = 64
        self.epochs = 50
        self.lr = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型权重保存至 cifar10/out_models_bin
        self.save_path = os.path.join(project_root, "out_models_bin", "image_classification.pt")


def main():
    config = Config()
    print(f"Device: {config.device}")

    # 1. 载入数据集
    train_dataset, test_dataset = build_dataset(config, only_test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. 构建模型与优化器
    model = ImageClassification().to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 3. 开始训练
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        epochs=config.epochs,
        save_path=config.save_path,
    )


if __name__ == "__main__":
    main()