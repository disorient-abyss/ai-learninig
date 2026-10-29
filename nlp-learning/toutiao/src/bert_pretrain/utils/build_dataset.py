# coding: UTF-8
import os
import pickle as pkl

from tqdm import tqdm

try:
    from utils.build_vocab import build_vocab_CNN
except ImportError:
    try:
        from .build_vocab import build_vocab_CNN
    except ImportError:
        from build_vocab import build_vocab_CNN


def build_dataset(config, only_test=False):
    """单模型（如 BERT）数据加载函数"""
    cls_token = getattr(config.tokenizer, "cls_token", "[CLS]")
    sep_token = getattr(config.tokenizer, "sep_token", "[SEP]")
    pad_token_id = getattr(config.tokenizer, "pad_token_id", 0)

    def load_dataset(path: str, pad_size: int = 32):
        contents = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到数据集文件: {path}")

        file_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {file_name}"):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                content, label = parts[0], parts[1]

                # 1. 分词与截断
                tokens = config.tokenizer.tokenize(content)
                if pad_size and len(tokens) > pad_size - 2:
                    tokens = tokens[: pad_size - 2]

                # 2. 拼接 BERT 首尾标记
                tokens = [cls_token] + tokens + [sep_token]
                seq_len = len(tokens)
                token_ids = config.tokenizer.convert_tokens_to_ids(tokens)

                # 3. 补齐与生成 Attention Mask
                pad_len = pad_size - seq_len
                mask = [1] * seq_len + [0] * pad_len
                token_ids += [pad_token_id] * pad_len

                contents.append((token_ids, int(label), seq_len, mask))

        return contents

    if only_test:
        test = load_dataset(config.test_path, config.pad_size)
        return None, None, test

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


def build_dataset_CNN(config, only_test: bool = False):
    """支持知识蒸馏双输入（BERT+CNN）与纯 TextCNN 单输入的数据加载函数"""
    pad_size = getattr(config, "pad_size", 32)
    batch_size = 2048

    # 1. 构建/加载学生模型专属词表
    student_tokenizer = getattr(config, "student_tokenizer", lambda x: list(x))
    vocab_path = getattr(config, "vocab_path", "./data/text_cnn/vocab.pkl")

    os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)

    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pkl.load(f)
    else:
        max_vocab_size = getattr(config, "max_vocab_size", 10000)
        min_freq = getattr(config, "min_freq", 1)
        vocab = build_vocab_CNN(
            file_path=config.train_path,
            tokenizer=student_tokenizer,
            max_size=max_vocab_size,
            min_freq=min_freq,
            pad_token="<PAD>",
            unk_token="<UNK>",
        )
        with open(vocab_path, "wb") as f:
            pkl.dump(vocab, f)

    print(f"Student Vocab size: {len(vocab)}")

    pad_id = vocab.get("<PAD>", 0)
    unk_id = vocab.get("<UNK>", 1)
    teacher_tokenizer = getattr(config, "tokenizer", None)

    def load_dataset(path: str, is_train: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到数据集文件: {path}")

        file_name = os.path.basename(path)
        raw_texts, labels = [], []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                raw_texts.append(parts[0])
                labels.append(int(parts[1]))

        dataset = []
        total_batches = (len(raw_texts) + batch_size - 1) // batch_size

        for i in tqdm(range(total_batches), desc=f"Processing {file_name}"):
            start_idx = i * batch_size
            batch_texts = raw_texts[start_idx : start_idx + batch_size]
            batch_labels = labels[start_idx : start_idx + batch_size]

            has_teacher = (teacher_tokenizer is not None) and is_train
            if has_teacher:
                teacher_encoded = teacher_tokenizer(
                    batch_texts,
                    padding="max_length",
                    max_length=pad_size,
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                t_ids = teacher_encoded["input_ids"]
                t_masks = teacher_encoded["attention_mask"]
            else:
                t_ids = [None] * len(batch_texts)
                t_masks = [None] * len(batch_texts)

            for text, t_id, t_mask, label in zip(
                batch_texts, t_ids, t_masks, batch_labels
            ):
                tokens = student_tokenizer(text)
                if len(tokens) > pad_size:
                    tokens = tokens[:pad_size]
                seq_len = len(tokens)

                s_ids = [vocab.get(token, unk_id) for token in tokens]
                if seq_len < pad_size:
                    s_ids += [pad_id] * (pad_size - seq_len)

                if has_teacher:
                    dataset.append(((t_id, t_mask), (s_ids, seq_len), label))
                else:
                    dataset.append(((s_ids, seq_len), label))

        return dataset

    if only_test:
        test = load_dataset(config.test_path, is_train=False)
        return vocab, None, None, test

    train = load_dataset(config.train_path, is_train=True)
    dev = load_dataset(config.dev_path, is_train=False)
    test = load_dataset(config.test_path, is_train=False)

    return vocab, train, dev, test