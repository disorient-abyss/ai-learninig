import os

from toutiao.src.fasttext_train.utils.fast_text_baseline import FastTextTrainer

if __name__ == "__main__":
    # 推荐：动态获取当前机器的 CPU 逻辑核心数（避免写死导致在其他机器上报资源不足或浪费算力）
    cpu_threads = os.cpu_count() or 1

    ft_model = FastTextTrainer(
        train_path="./data/train_fast_jieba.txt",
        dev_path="./data/dev_fast_jieba.txt",
        test_path="./data/test_fast_jieba.txt",
        num_threads=cpu_threads
    )

    ft_model.train_with_autotune(duration=600, word_ngrams=2, verbose=3)
    ft_model.evaluate()

    # 模型保存
    ft_model.save(output_dir="./model_bin", prefix="toutiao_fasttext_jieba")

