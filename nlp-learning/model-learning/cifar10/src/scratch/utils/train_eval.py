import os
import time

from sklearn import metrics
import torch


def evaluate(model, test_loader, criterion, device, class_names=None):
    """
    模型评估函数：
    - 在训练阶段调用：快速计算并返回 (avg_loss, acc)
    - 在独立测试阶段调用：如果传入 class_names，额外打印分类报告与混淆矩阵
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    start_time = time.time()
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)

            if criterion is not None:
                loss = criterion(outputs, y_val)
                total_loss += loss.item() * len(y_val)

            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y_val).sum().item()
            total_samples += len(y_val)

            # 仅在需要生成完整评测报告时收集预测值与真实值，节省显存/内存开销
            if class_names is not None:
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_val.cpu().numpy())

    elapsed = time.time() - start_time
    avg_loss = (total_loss / total_samples) if (criterion and total_samples > 0) else 0.0
    acc = correct / total_samples if total_samples > 0 else 0.0

    # 如果传入了类别名称，打印完整评测报告（对应 run_test.py）
    if class_names is not None:
        print(f"Final Test Loss: {avg_loss:.4f}, Test Acc: {acc * 100:.2f}%\n")

        print("Precision, Recall and F1-Score:")
        report = metrics.classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            digits=4
        )
        print(report)

        print("Confusion Matrix:")
        cm = metrics.confusion_matrix(all_targets, all_preds)
        print(cm)

        # 格式化耗时
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"Time usage: {time_str}")

    return avg_loss, acc


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, save_path):
    """
    模型训练主循环：支持每个 Epoch 评估与保存最优模型
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0

    for epoch_idx in range(epochs):
        model.train()
        start_time = time.time()

        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=-1)
            train_correct += (preds == y).sum().item()
            train_loss_total += loss.item() * len(y)
            train_samples += len(y)

        train_loss = train_loss_total / train_samples
        train_acc = train_correct / train_samples

        # 评估阶段（不传 class_names，只获取数值用于监控和保存最优检查点）
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # 记录最佳模型权重
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

        flag = "★ BEST" if is_best else ""
        print(
            f"Epoch: [{epoch_idx + 1:2d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}% | "
            f"Time: {elapsed:.2f}s {flag}"
        )

    print(f"\n训练结束！最佳测试集准确率: {best_acc * 100:.2f}%，模型已保存至: {save_path}")