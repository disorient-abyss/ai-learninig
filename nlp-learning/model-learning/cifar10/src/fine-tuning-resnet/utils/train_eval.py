# coding: UTF-8
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


def evaluate(config, model, data_loader, test=False):
    """评估函数：test=False 时为轻量模式，test=True 时生成详细分类报告与混淆矩阵"""
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

    if test:
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

    return acc, avg_loss


def train(config, model, train_loader, dev_loader, test_loader=None):
    """训练主循环函数"""
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
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    use_amp = config.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    total_batch = 0
    dev_best_loss = float("inf")

    for epoch in range(config.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}] | LR: {current_lr:.6f}")

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels, label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_batch += 1

            # 每隔 117 个 batch 输出在训练集和验证集上的效果
            if total_batch % 117 == 0:
                true = labels.detach().cpu().numpy()
                predic = torch.argmax(outputs.detach(), dim=1).cpu().numpy()
                train_acc = metrics.accuracy_score(true, predic)

                dev_acc, dev_loss = evaluate(config, model, dev_loader, test=False)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = (
                    f"Iter: {total_batch:>6}, Train Loss: {loss.item():>5.2f}, "
                    f"Train Acc: {train_acc:>6.2%}, Val Loss: {dev_loss:>5.2f}, "
                    f"Val Acc: {dev_acc:>6.2%}, Time: {time_dif} {improve}"
                )
                print(msg)

                # 评估完成后将模型重新置于训练模式
                model.train()

        # 每个 Epoch 结束更新学习率
        scheduler.step()

    print(f"\nTraining Complete. Best model saved to: {config.save_path}")

    # 若传入了 test_loader，直接加载最佳权重执行最终测试
    if test_loader is not None:
        print("\nLoading best weights for final test evaluation...")
        test(config, model, test_loader)


def test(config, model, test_loader):
    """最终测试函数"""
    if os.path.exists(config.save_path):
        model.load_state_dict(
            torch.load(config.save_path, map_location=config.device, weights_only=True)
        )
        print(f"Loaded weights from: {config.save_path}")

    model.eval()
    start_time = time.time()

    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_loader, test=True
    )

    print(f"\nFinal Test Loss: {test_loss:>5.4f}, Test Acc: {test_acc:>6.2%}")
    print("\nPrecision, Recall and F1-Score:")
    print(test_report)
    print("Confusion Matrix:")
    print(test_confusion)
    print("Time usage:", get_time_dif(start_time))