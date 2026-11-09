from nltk.translate.bleu_score import sentence_bleu


def coumulative_bleu(reference, candidate):
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1,0,0,0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram

if __name__ == "__main__":
    generated_text = "This is some generated text."
    # 计算BLEU指标
    reference_texts = ["This is a reference text.", "This is another reference text."]
    c_bleu = coumulative_bleu(reference_texts, generated_text)
    print("The Bleu score's [1_gram, 2_gram, 3_gram, 4_gram] are:", c_bleu)