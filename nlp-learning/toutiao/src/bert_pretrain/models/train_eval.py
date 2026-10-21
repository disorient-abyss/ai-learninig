import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.optim import AdamW
from tqdm import tqdm
from utils.get_time_dif import get_time_dif


def _to_device(data, device):
    """辅助函数：递归将张量或张量元组/列表迁移到指定设备并启用异步传输"""
    if isinstance(data, (tuple, list)):
        return tuple(_to_device(x, device) for x in data)
    return data.to(device, non_blocking=True)


# 第2步: 编写训练函数
def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    
    # 区分需要权重衰减和不需要权重衰减的参数
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_batch = 0                # 记录进行到多少 batch（跨 epoch 持续累加）初始化置 0
    dev_best_loss = float("inf")   # 记录验证集最佳 loss，初始设置为无穷大
    last_improve = 0               # 记录上一次验证集指标刷新纪录时的 total_batch 数值
    early_stop = False             # 早停标志位（布尔开关），验证是否过了 1000 batch 没有提升

    model.train()
    for epoch in range(config.num_epochs):
        print(f"Epoch [{epoch + 1}/{config.num_epochs}]")
        # 1. 分批驱动循环：按 batch_size 逐批遍历训练迭代器 train_iter，避免整个数据集一次性涌入显存导致崩溃。
        # 2. 解包输入数据：将迭代器每次产出的二元组 ((x, seq_len, mask), y) 拆开，把特征部分赋给 trains（供模型做前向传播），把标签部分赋给 labels（供损失函数对比误差）。
        # 3. 渲染动态进度：通过 tqdm 实时追踪处理进度，并在最左侧标明当前的训练轮次（如 Training Epoch 1），直观展示百分比、批次计数与剩余时间。
        for trains, labels in tqdm(train_iter, desc=f"Training Epoch {epoch + 1}"):
            # 数据迁入 GPU (支持 non_blocking 异步重叠)
            trains = _to_device(trains, config.device)
            labels = labels.to(config.device, non_blocking=True)

            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)

            # set_to_none=True 比默认赋零能略微提升性能并节省显存
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_batch += 1

            # 每隔 200 个 batch 输出在训练集和验证集上的效果
            if total_batch % 200 == 0:
                true = labels.detach().cpu().numpy()
                predic = torch.argmax(outputs.detach(), dim=1).cpu().numpy()
                train_acc = metrics.accuracy_score(true, predic)

                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_batch
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

            # 验证集 loss 超过设定阈值仍未下降，提前结束训练
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                early_stop = True
                break

        if early_stop:
            break


# 第3步: 编写测试函数
def test(config, model, test_iter):
    # 如需加载最佳权重进行测试，取消下一行注释（量化推理时保持关闭）
    # model.load_state_dict(torch.load(config.save_path, weights_only=True))
    model.eval()
    start_time = time.time()
    
    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_iter, test=True
    )
    
    print(f"Test Loss: {test_loss:>5.2f}, Test Acc: {test_acc:>6.2%}")
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


# 第4步: 编写验证函数
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0.0
    predict_list = []
    labels_list = []

    with torch.no_grad():
        for texts, labels in data_iter:
            # 同样迁移验证集/测试集数据到 GPU
            texts = _to_device(texts, config.device)
            labels = labels.to(config.device, non_blocking=True)

            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()

            labels_np = labels.detach().cpu().numpy()
            predic_np = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            labels_list.append(labels_np)
            predict_list.append(predic_np)

    # 一次性拼接所有 batch 的结果，避免循环内频繁拷贝内存
    labels_all = np.concatenate(labels_list, axis=0)
    predict_all = np.concatenate(predict_list, axis=0)
    
    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / len(data_iter)

    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_list, digits=4
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, avg_loss, report, confusion

    return acc, avg_loss