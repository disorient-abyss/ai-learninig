# coding: UTF-8
import argparse
import importlib
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

# 确保脚本识别同级模块引用
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from utils.build_dataset import build_dataset
from utils.train_eval import test


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classification - Test")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_34",
        choices=["resnet_34", "resnet_18", "resnet-34", "resnet-18"],
        help="指定评估的模型架构 (默认: resnet_34)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 统一命名格式：将中划线替换为下划线以匹配文件名 (如 resnet-18 -> resnet_18)
    model_name = args.model.replace("-", "_")

    # 1. 动态载入指定模型的 Config 与 Model
    try:
        model_module = importlib.import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"未在 models 目录下找到对应的模型定义文件: models/{model_name}.py"
        )

    Config = model_module.Config
    Model = model_module.Model

    config = Config()

    # 2. 检查待评测的模型权重是否存在
    if not os.path.exists(config.save_path):
        raise FileNotFoundError(
            f"未找到已保存的权重文件: {config.save_path}，请先执行 run_train.py 进行训练！"
        )

    print(f"Loaded model architecture: {model_name}")
    print(f"Using device: {config.device}")
    print(f"Loading weights from: {config.save_path}")

    # 3. 仅构建 CIFAR-10 测试集（避免构建训练集的额外开销）
    _, test_dataset = build_dataset(config, only_test=True)

    # 4. 构造测试数据迭代器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # 5. 实例化模型并迁移至对应计算设备
    model = Model(config).to(config.device)

    # 6. 执行测试评估（test 函数内部会自动载入 config.save_path 权重并打印指标）
    print("\nRunning evaluation on test set...")
    test(config, model, test_loader)


if __name__ == "__main__":
    main()