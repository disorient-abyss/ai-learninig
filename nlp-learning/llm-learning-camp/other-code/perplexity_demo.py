import math

if __name__ == "__main__":
    sentences = [
        ['He', 'has', 'a', 'book'],
        ['She', 'has', 'a', 'cat'],
        ['I', 'have', 'a', 'pen'],
    ]

    unigram = {
        'I': 1/11, 'have': 1/11, 'a': 3/11, 'pen': 1/11,
        'He': 1/11, 'has': 2/11, 'cat': 1/11, 'book': 1/11,
        'She': 1/11,
    }

    log_prob_sum = 0
    total_words = 0

    for sentence in sentences:
        for word in sentence:
            log_prob_sum += math.log2(unigram[word])
            total_words += 1

    # 公式: 2 ** (- (1/N) * sum(log2(p)))
    perplexity = 2 ** (-log_prob_sum / total_words)

    print('困惑度为:', perplexity)