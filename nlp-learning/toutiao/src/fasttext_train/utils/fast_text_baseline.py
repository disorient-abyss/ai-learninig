import os
import time

import fasttext


class FastTextTrainer:
    def __init__(
        self,
        train_path: str,
        dev_path: str | None = None,
        test_path: str | None = None,
        num_threads: int | None = None,
    ):
        """
        初始化 FastText 训练器。

        :param train_path: 训练集文件路径
        :param dev_path: 验证集文件路径（用于超参数自动调优）
        :param test_path: 测试集文件路径（用于最终性能评估）
        :param num_threads: 线程数，默认自动获取 CPU 核心数
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.num_threads = num_threads or (os.cpu_count() or 1)
        self.model = None

    def train_with_autotune(
        self,
        duration: int = 600,
        word_ngrams: int = 2,
        verbose: int = 3,
        **extra_kwargs,
    ):
        """
        使用验证集进行超参数随机搜索调优并训练模型。
        """
        if not self.dev_path or not os.path.exists(self.dev_path):
            raise FileNotFoundError(f"未找到验证集文件: {self.dev_path}，无法执行 autotune。")

        print(f"开始 FastText 调优训练，线程数: {self.num_threads}，搜索时长: {duration}s...")

        self.model = fasttext.train_supervised(
            input=self.train_path,
            autotuneValidationFile=self.dev_path,
            autotuneDuration=duration,
            wordNgrams=word_ngrams,
            thread=self.num_threads,
            verbose=verbose,
            **extra_kwargs,
        )
        print("模型训练及超参数调优完成。")
        return self.model

    def evaluate(self, test_path: str | None = None) -> tuple[int, float, float]:
        """
        评估模型在测试集上的表现。
        返回元组: (样本数, Precision@1, Recall@1)
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练或加载，无法评估。")

        eval_target = test_path or self.test_path
        if not eval_target or not os.path.exists(eval_target):
            raise FileNotFoundError(f"未找到有效的测试集路径: {eval_target}")

        n_samples, precision, recall = self.model.test(eval_target)
        print(f"评估完成 - 样本量: {n_samples}, Precision@1: {precision:.4f}, Recall@1: {recall:.4f}")
        return n_samples, precision, recall

    def save(self, output_dir: str = "./model_bin", prefix: str = "toutiao_fasttext") -> str:
        """
        将训练完成的模型持久化保存至磁盘。
        """
        if self.model is None:
            raise RuntimeError("当前无可用模型可保存。")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        save_path = os.path.join(output_dir, f"{prefix}_{timestamp}.bin")

        self.model.save_model(save_path)
        print(f"模型成功保存至: {save_path}")
        return save_path
