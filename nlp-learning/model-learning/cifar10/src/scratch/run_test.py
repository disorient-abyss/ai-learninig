import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

# 1. 路径解析（传统 os.path 风格）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(current_dir)

from models.image_classification import ImageClassification
from utils.build_dataset import build_dataset
from utils.train_eval import evaluate


class Config:
    def __init__(self):
        # 数据集根目录定位到 cifar10/data
        self.data_path = os.path.join(project_root, "data")
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型权重路径定位到 cifar10/out_models_bin/image_classification.pt
        self.model_path = os.path.join(project_root, "out_models_bin", "image_classification.pt")

        # CIFAR-10 标准类别名称
        self.class_list = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]


def main():
    config = Config()
    print(f"Device: {config.device}")

    # 1. 仅加载测试集
    _, test_dataset = build_dataset(config, only_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. 载入模型与权重
    model = ImageClassification().to(config.device)
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"未找到已训练的权重文件: {config.model_path}，请先运行 run_train.py")

    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    criterion = nn.CrossEntropyLoss()

    # 3. 运行评估（传入 class_names，直接输出 Loss、Acc、F1 报告和混淆矩阵）
    evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=config.device,
        class_names=config.class_list
    )


if __name__ == "__main__":
    main()