import os

from tqdm import tqdm


def build_dataset(config, only_test=False):
    # 直接在内部提取特殊 token，支持默认兜底
    cls_token = getattr(config.tokenizer, "cls_token", "[CLS]")
    sep_token = getattr(config.tokenizer, "sep_token", "[SEP]")
    pad_token_id = getattr(config.tokenizer, "pad_token_id", 0)
    def load_dataset(
            path: str, 
            pad_size: int = 32, 
        ) -> list[tuple[list[int], int, int, list[int]]]:
        contents = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到数据集文件: {path}")

        file_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {file_name}"):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue  # 跳过格式不合规的脏数据行
                content, label = parts[0], parts[1]

                # 以下内容可以用 Hugging Face 一步搞定截断、补齐、添加 [CLS]/[SEP] 以及生成 attention_mask

                # 1  分词
                tokens =  config.tokenizer.tokenize(content)

                # 2  截断留出 2 个位置给 [CLS] 和 [SEP]
                if pad_size and len(tokens) > pad_size - 2:
                    tokens = tokens[: pad_size -2]

                # 3  规范拼接 BERT 首尾标记
                tokens = [cls_token] + tokens + [sep_token]
                # 记录真实有效文本的长度（此时包含首尾特殊标记，但不包含后续填充的 0）
                seq_len = len(tokens)
                # 查词表，将汉字/单词列表转换成一串唯一的数字 ID
                token_ids = config.tokenizer.convert_tokens_to_ids(tokens)

                # 4  补齐（Padding）与生成 Attention Mask
                # 计算当前句子距离目标固定长度 pad_size 还差多少位。
                pad_len = pad_size - seq_len

                # 统一生成 mask 和补齐 token_ids（pad_len 为 0 时自动不补，结果完全正确）
                mask = [1] * seq_len + [0] * pad_len
                token_ids += [pad_token_id] * pad_len

                # # Hugging Face 一步搞定截断、补齐、添加 [CLS]/[SEP] 以及生成 attention_mask
                # encoded = config.tokenizer.encode_plus(
                #     content,
                #     max_length=pad_size,
                #     padding="max_length",
                #     truncation=True,
                #     return_attention_mask=True,
                # )

                # token_ids = encoded["input_ids"]
                # mask = encoded["attention_mask"]
                
                # 实际有效 token 长度（不含 padding）
                # seq_len = sum(mask)

                contents.append((token_ids, int(label), seq_len, mask))

        return contents
    
    if only_test is True:
        test = load_dataset(config.test_path, config.pad_size)
        return None, None, test
    
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test