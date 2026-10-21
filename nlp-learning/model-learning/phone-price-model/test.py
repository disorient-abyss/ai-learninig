import numpy as np
import torch
from phone_class_model import PhoneClassModel
from phone_classify import create_dataset
from torch.utils.data import DataLoader
from train import Trainer

# 全局超参数与路径配置
MODEL_SAVE_PATH = "data/best_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)


def run_train():
    """纯训练流程：训练模型并保存最优权重"""
    print(">>> 启动训练任务...")
    
    # 1. 读取并构建数据集
    train_dataset, valid_dataset, input_dim, num_classes = create_dataset()

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)

    # 2. 实例化空模型
    model = PhoneClassModel(input_dim=input_dim, num_classes=num_classes)

    # 3. 实例化 Trainer 并开始训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=2e-3,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH
    )
    trainer.train(num_epochs=50)


def run_evaluate():
    """纯评估/测试流程：脱离训练，直接加载本地已有的权重进行评估"""
    print(">>> 启动独立评估任务...")

    # 1. 独立获取测试/验证数据及维度信息
    _, valid_dataset, input_dim, num_classes = create_dataset()
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 必须先实例化同样结构的空模型
    model = PhoneClassModel(input_dim=input_dim, num_classes=num_classes)

    # 3. 借助 Trainer 加载权重并评估（无需传入 train_loader）
    trainer = Trainer(
        model=model,
        train_loader=None,      # 仅评估不需要训练集
        valid_loader=valid_loader,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH
    )

    # 4. 加载权重并执行闭卷评估
    trainer.load_best_model()
    test_loss, test_acc = trainer.evaluate(valid_loader)

    print("\n" + "=" * 35)
    print("         独立评估最终报告         ")
    print("=" * 35)
    print(f"评估数据集 Loss  : {test_loss:.4f}")
    print(f"评估数据集 准确率: {test_acc * 100:.2f}%")
    print("=" * 35)


if __name__ == "__main__":
    # 1. 固定全局随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # ===================================================
    # 控制模式开关：可填 'train'（只训练）或 'eval'（只测试）
    # ===================================================
    run_train()
    run_evaluate()

   