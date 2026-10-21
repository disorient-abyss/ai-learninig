from toutiao.src.fasttext_train.utils.preprocess import FastTextPreprocessor

if __name__ == "__main__":
    class_path = "./data/class.txt"
    splits = ["train", "dev", "test"]
    preprocessor_word = FastTextPreprocessor(class_path, tokenize_mode="word")

    # 按词划分
    for split in splits:
        preprocessor_word.run(
            input_path = f"./data/{split}.txt",
            output_path = f"./data/{split}_fast_jieba.txt"
        )

