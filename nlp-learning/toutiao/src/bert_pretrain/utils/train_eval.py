# coding: UTF-8
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from utils.get_time_dif import get_time_dif


def _to_device(data, device):
    """辅助函数：递归将张量或张量元组/列表迁移到指定设备"""
    if isinstance(data, (tuple, list)):
        return tuple(_to_device(x, device) for x in data)
    return data.to(device, non_blocking=True)


# 1. 蒸馏损失函数定义
criterion_kldiv = nn.KLDivLoss(reduction="batchmean")


def loss_fn_kd(student_outputs, labels, teacher_outputs, alpha=0.8, T=2.0):
    """知识蒸馏组合损失函数

    Args:
        outputs: 学生模型 TextCNN 的未归一化 logits
        labels: 真实分类硬标签
        teacher_outputs: 教师模型 BERT 的未归一化 logits
        alpha: 软标签蒸馏损失权重 (0~1)
        T: 蒸馏温度系数
    """
    student_log_probs_t = F.log_softmax(student_outputs / T, dim=1)
    teacher_probs_t = F.softmax(teacher_outputs / T, dim=1)
    distill_loss = criterion_kldiv(student_log_probs_t, teacher_probs_t)

    hard_loss = F.cross_entropy(student_log_probs_t, labels)

    kd_loss = distill_loss * (alpha * T * T) + hard_loss * (1.0 - alpha)
    return kd_loss


# 2. 单模型训练函数 (适用于单独微调 BERT 教师模型或基线训练)
def train(config, model, train_iter, dev_iter, test_iter=None):
    start_time = time.time()

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    early_stop = False

    model.train()
    for epoch in range(config.num_epochs):
        print(f"Epoch [{epoch + 1}/{config.num_epochs}]")
        for trains, labels in tqdm(
            train_iter, desc=f"Training Epoch {epoch + 1}"
        ):
            trains = _to_device(trains, config.device)
            labels = labels.to(config.device, non_blocking=True)

            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_batch += 1

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
                model.train()

            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                early_stop = True
                break

        if early_stop:
            break

    # 保底机制：若数据集较小（总 batch < 200）从未触发保存，则在结束时保存一次
    if not os.path.exists(config.save_path):
        torch.save(model.state_dict(), config.save_path)

    if test_iter is not None:
        test(config, model, test_iter, load_from_save_path=True)


# 3. 知识蒸馏专属训练函数 (BERT 教师指导 TextCNN 学生)
def train_kd(
    cnn_config,
    bert_model,
    cnn_model,
    train_iter,
    dev_iter,
    test_iter=None,
    alpha=0.8,
    T=2.0,
):
    """Args:

    train_iter: 吐出 ((t_ids, t_mask), (s_ids, s_len)), labels 的双输入迭代器
    dev_iter / test_iter: 单输入迭代器，只包含学生端特征 (s_ids, s_len), labels
    """
    start_time = time.time()

    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False

    cnn_model.train()
    optimizer = AdamW(cnn_model.parameters(), lr=cnn_config.learning_rate)

    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    early_stop = False

    for epoch in range(cnn_config.num_epochs):
        print(f"KD Epoch [{epoch + 1}/{cnn_config.num_epochs}]")

        for inputs, labels in tqdm(
            train_iter, desc=f"Distilling Epoch {epoch + 1}"
        ):
            teacher_inputs, student_inputs = inputs

            teacher_inputs = _to_device(teacher_inputs, cnn_config.device)
            student_inputs = _to_device(student_inputs, cnn_config.device)
            labels = labels.to(cnn_config.device, non_blocking=True)

            with torch.no_grad():
                teacher_outputs = bert_model(teacher_inputs)

            student_outputs = cnn_model(student_inputs)

            loss = loss_fn_kd(
                student_outputs, labels, teacher_outputs, alpha=alpha, T=T
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_batch += 1

            if total_batch % 200 == 0:
                true = labels.detach().cpu().numpy()
                predic = (
                    torch.argmax(student_outputs.detach(), dim=1).cpu().numpy()
                )
                train_acc = metrics.accuracy_score(true, predic)

                dev_acc, dev_loss = evaluate(cnn_config, cnn_model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(cnn_model.state_dict(), cnn_config.save_path)
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
                cnn_model.train()

            if total_batch - last_improve > cnn_config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                early_stop = True
                break

        if early_stop:
            break

    if not os.path.exists(cnn_config.save_path):
        torch.save(cnn_model.state_dict(), cnn_config.save_path)

    if test_iter is not None:
        test(cnn_config, cnn_model, test_iter, load_from_save_path=True)


# 4. 评估函数 (纯针对单模型推理，复用性高)
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0.0
    total_samples = 0
    predict_list = []
    labels_list = []

    with torch.no_grad():
        for texts, labels in data_iter:
            texts = _to_device(texts, config.device)
            labels = labels.to(config.device, non_blocking=True)

            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)

            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            total_samples += batch_size

            labels_np = labels.detach().cpu().numpy()
            predic_np = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            labels_list.append(labels_np)
            predict_list.append(predic_np)

    labels_all = np.concatenate(labels_list, axis=0)
    predict_all = np.concatenate(predict_list, axis=0)

    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / max(total_samples, 1)

    if test:
        report = metrics.classification_report(
            labels_all,
            predict_all,
            target_names=config.class_list,
            digits=4,
            zero_division=0,
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, avg_loss, report, confusion

    return acc, avg_loss


# 5. 测试函数
def test(config, model, test_iter, load_from_save_path: bool = True):
    # 若在 train/train_kd 结束后直接调用，则加载验证集最优权重；
    # 若在 run.py 的 run_test 中调用（外部已加载好浮点或量化权重），则跳过重复加载
    if load_from_save_path:
        model.load_state_dict(
            torch.load(
                config.save_path,
                map_location=config.device,
                weights_only=True,
            )
        )
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