import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def get_time_dif(start_time):
    """计算耗时并格式化输出"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=round(time_dif))


# 1. 训练主循环函数
def train(config, model, train_loader):
    start_time = time.time()

    # 针对视觉卷积网络配置权重衰减
    no_decay = ["bias", "bn", "downsample.1", "LayerNorm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # 1：余弦退火学习率调度器（平滑衰减至 1e-6）
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    use_amp = config.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()

    for epoch in range(config.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}] | LR: {current_lr:.6f}")
        epoch_loss = 0.0
        correct_count = 0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 改进点 2：修复多余前向传播，并加入 label_smoothing 提升泛化能力
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels, label_smoothing=0.1)

            scaler.scale(loss).backward()

            # 改进点 3：梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # 统计指标
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            preds = torch.argmax(outputs.detach(), dim=1)
            correct_count += (preds == labels).sum().item()
            total_samples += batch_size

        # 每个 Epoch 结束更新学习率
        scheduler.step()

        avg_train_loss = epoch_loss / total_samples
        avg_train_acc = correct_count / total_samples
        time_dif = get_time_dif(start_time)

        msg = (
            f"Epoch [{epoch + 1:>2}/{config.num_epochs:>2}] "
            f"Train Loss: {avg_train_loss:>5.4f}, Train Acc: {avg_train_acc:>6.2%}, "
            f"Time Elapsed: {time_dif}"
        )
        print(msg)

    # 训练完成，保存权重
    torch.save(model.state_dict(), config.save_path)
    print(f"\nTraining Complete. Model saved to: {config.save_path}")


# 2. 评估函数
def evaluate(config, model, data_loader):
    model.eval()
    loss_total = 0.0
    total_samples = 0
    predict_list = []
    labels_list = []
    use_amp = config.device.type == "cuda"

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            total_samples += batch_size

            labels_list.append(labels.detach().cpu().numpy())
            predict_list.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())

    labels_all = np.concatenate(labels_list, axis=0)
    predict_all = np.concatenate(predict_list, axis=0)

    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / total_samples

    class_list = getattr(
        config,
        "class_list",
        ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    )
    report = metrics.classification_report(
        labels_all, predict_all, target_names=class_list, digits=4
    )
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    return acc, avg_loss, report, confusion


# 3. 最终测试函数
def test(config, model, test_loader):
    if os.path.exists(config.save_path):
        model.load_state_dict(torch.load(config.save_path, map_location=config.device, weights_only=True))
    model.eval()
    start_time = time.time()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_loader)

    print(f"\nFinal Test Loss: {test_loss:>5.4f}, Test Acc: {test_acc:>6.2%}")
    print("\nPrecision, Recall and F1-Score:")
    print(test_report)
    print("Confusion Matrix:")
    print(test_confusion)
    print("Time usage:", get_time_dif(start_time))