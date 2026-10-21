import os

import torch
from torch import nn, optim


class Trainer:
    def __init__(self, model, train_loader, valid_loader=None, lr=2e-3, device=None, save_path="data/best_model.pth"):
        # 自动选择最优计算硬件
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = device

        # 初始化参数列表
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 设置优化方法
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.save_path = save_path
        self.best_val_acc = 0.0

    def _train_one_epoch(self):
        """单轮训练：所有统计在 GPU 上完成，不在 batch 循环中调用 .item() 阻塞显卡"""
        self.model.train()
        # 用张量累加这一整个 Epoch 所有批次的 Loss 总和，避免 gpu 与 cpu 之间频繁切换
        total_loss_tensor = torch.tensor(0.0, device=self.device)
        # 创建一个整数张量，用来累加这一轮中模型猜对标签的样本总数
        total_correct_tensor = torch.tensor(0, device=self.device)
        # 统计样本总数
        total_samples = 0

        for x, y in self.train_loader:
            # 开启 non_blocking 异步数据拷贝，使得 CPU 和 GPU 并行工作
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播
            output = self.model(x)
            # 计算损失
            loss = self.criterion(output, y)
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()

            # 指标全部留在 GPU 上运算
            # 累积当前 Batch 所有样本的 Loss 总和，并同通过 detach() 保留数值，防爆显存
            total_loss_tensor += loss.detach() * len(y)
            # 获取当前 Batch 的分类结果 
            pred = output.argmax(dim=1)
            # 累加当前 Batch 的总正确数
            total_correct_tensor += (pred == y).sum()
            # 记录到目前为止一共跑了多少个样本
            total_samples += len(y)

        # 整个 epoch 结束，只向显卡拉取一次数据
        train_loss = (total_loss_tensor / total_samples).item()
        train_acc = (total_correct_tensor.float() / total_samples).item()
        return train_loss, train_acc

    def train(self, num_epochs=50):
        """执行完整训练流程并保存最优模型"""
        print(f"[Trainer] 训练引擎启动 | 设备: {self.device} | 轮数: {num_epochs}")

        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_one_epoch()

            if self.valid_loader is not None:
                val_loss, val_acc = self.evaluate(self.valid_loader)
                print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

                # 保存历史最优权重
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.save_path)
            else:
                print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        print(f"[Trainer] 训练结束！最高验证准确率: {self.best_val_acc*100:.2f}%\n")

    def evaluate(self, dataloader=None):
        """模型评估：可用于验证集或独立测试集"""
        loader = dataloader if dataloader is not None else self.valid_loader
        if loader is None:
            raise ValueError("未提供用于评估的 DataLoader！")

        self.model.eval()
        total_loss_tensor = torch.tensor(0.0, device=self.device)
        total_correct_tensor = torch.tensor(0, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                output = self.model(x)
                loss = self.criterion(output, y)

                total_loss_tensor += loss.detach() * len(y)
                pred = output.argmax(dim=1)
                total_correct_tensor += (pred == y).sum()
                total_samples += len(y)

        eval_loss = (total_loss_tensor / total_samples).item()
        eval_acc = (total_correct_tensor.float() / total_samples).item()
        return eval_loss, eval_acc

    def load_best_model(self):
        """加载保存的最优权重"""
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device, weights_only=True))
            print(f"[Trainer] 成功载入最优模型参数: {self.save_path}")
        else:
            print(f"[Trainer] 警告：未找到权重文件 {self.save_path}")
   
