from collections import Counter
from collections.abc import Callable

from tqdm import tqdm


def build_vocab_CNN(
        file_path: str, 
        tokenizer: Callable[[str], list[str]], 
        max_size: int=10000, 
        min_freq: int=1, 
        pad_token: str="<PAD>", 
        unk_token: str="<UNK>"
    ) -> dict[str, int]:

    counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting vocab"):
            line = line.strip()
            if not line:
                continue
            
            # 仅取第一列内容（指定 maxsplit=1 避免整行拆分的额外开销）
            content = line.split("\t", 1)[0]
            counter.update(tokenizer(content))

    # 先加入特殊 token（通常 PAD=0, UNK=1）
    special_tokens = [pad_token, unk_token]
    vocab_dic = {token: idx for idx, token in enumerate(special_tokens)}
    
    # 计算扣除特殊 token 后的最大保留词数
    max_vocab_items = max_size - len(special_tokens) if max_size is not None else None

    # 按词频排序，过滤低于 min_freq 的词，并截断
    sorted_words = [
        word for word, count in counter.most_common()
        if count >= min_freq
    ]
    
    if max_vocab_items is not None:
        sorted_words = sorted_words[:max_vocab_items]

    # 将高频词追加到词表
    for word in sorted_words:
        vocab_dic[word] = len(vocab_dic)

    return vocab_dic