# 第⼀步:安装rouge: pip install rouge
from rouge import Rouge

if __name__ == "__main__":
    # ⽣成⽂本
    generated_text = "This is some generated text."

    # 计算 ROUGE 指标
    reference_texts = ["This is a reference text.", "This is another generated reference text."]
    rouge = Rouge()
    scores = rouge.get_scores(generated_text, reference_texts[1])
    
    # 打印结果
    print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
    print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
    print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])