import os
from collections.abc import Callable
from typing import Literal

import jieba


class FastTextPreprocessor:
    """FastText 统一数据预处理工具。

    支持词粒度 (jieba)、字粒度 (char) 以及用户自定义的分词处理方式。
    """

    def __init__(
        self,
        class_file: str,
        tokenize_mode: Literal["word", "char"] = "word",
        custom_tokenizer: Callable[[str], list[str]] | None = None,
    ):
        """初始化预处理器。

        :param class_file: 类别文件路径
        :param tokenize_mode: 分词模式，可选 "word" (默认 jieba 分词) 或 "char" (单字切分)
        :param custom_tokenizer: 自定义分词函数 (输入 str，返回 list[str])，优先级高于 tokenize_mode
        """
        self.id_to_label = self._load_classes(class_file)
        self.tokenize_mode = tokenize_mode
        self.custom_tokenizer = custom_tokenizer

        # 预热 jieba，避免首行切分阻塞
        if custom_tokenizer is None and tokenize_mode == "word":
            jieba.initialize()

        print(
            f"成功加载 {len(self.id_to_label)} 个类别映射，当前切分模式: "
            f"{'custom' if custom_tokenizer else tokenize_mode}。"
        )

    @staticmethod
    def _load_classes(filepath: str) -> dict[int, str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"类别文件未找到: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return dict(enumerate(line.strip() for line in f if line.strip()))

    def _tokenize(self, text: str) -> list[str]:
        """统一分词接口，剔除空白字符。"""
        if self.custom_tokenizer is not None:
            return [t for t in self.custom_tokenizer(text) if not t.isspace()]

        if self.tokenize_mode == "word":
            return [w for w in jieba.cut(text) if not w.isspace()]
        elif self.tokenize_mode == "char":
            return [c for c in text if not c.isspace()]
        else:
            raise ValueError(f"不支持的 tokenize_mode: {self.tokenize_mode}")

    def run(self, input_path: str, output_path: str):
        count = 0
        skipped = 0

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with (
            open(input_path, "r", encoding="utf-8") as f_in,
            open(
                output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024
            ) as f_out,
        ):
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    skipped += 1
                    continue

                sentence, label_str = parts

                # 1. 类别解析
                try:
                    label_name = self.id_to_label[int(label_str)]
                except (ValueError, KeyError):
                    skipped += 1
                    continue

                # 2. 统一分词
                tokens = self._tokenize(sentence)
                if not tokens:
                    skipped += 1
                    continue

                sent_tokenized = " ".join(tokens)

                # 3. 拼接 FastText 格式输出
                new_sentence = f"__label__{label_name} {sent_tokenized}\n"
                f_out.write(new_sentence)

                count += 1
                if count % 10000 == 0:
                    print(f"[{self.tokenize_mode}] 已处理: {count} 行")

        print(
            f"预处理完毕 [{self.tokenize_mode}]!\n"
            f"- 成功输出: {count} 条\n"
            f"- 跳过脏数据: {skipped} 条\n"
            f"- 结果已保存至: {output_path}\n"
        )

