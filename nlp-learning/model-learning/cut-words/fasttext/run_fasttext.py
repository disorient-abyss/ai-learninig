import fasttext

# 使用fasttext的train_unsupervised(无监督训练方法)进行词向量的训练
# 它的参数是数据集的持久化文件路径'data/fil9'

# 在训练词向量过程中，我们可以设定很多常用超参数来调节我们的模型效果，如：
# 无监督训练模式： 'skipgram' 或者 'cbow'，默认为'skipgram'，在实践中，skipgram 模式在利用子词方面比 cbow 更好。
# 词嵌入维度dim：默认为100，但随着语料库的增大，词嵌入的维度往往也要更大。
# 数据循环次数epoch：默认为5，但当你的数据集足够大，可能不需要那么多次。
# 学习率lr：默认为0.05，根据经验，建议选择[0.01，1]范围内。
# 使用的线程数thread：默认为12个线程，一般建议和你的cpu核数相同。

def cbow_train():
    model = fasttext.train_unsupervised('data/fil9', "cbow", dim=300, epoch=5, lr=0.05, thread=16)
    return model

def skipgram_train():
    model = fasttext.train_unsupervised('data/fil9', "skipgram", dim=300, epoch=5, lr=0.05, thread=16)
    return model

if __name__ == "__main__":
    # 手动切换 word2vec 的训练方式
    # model = cbow_train()
    # model = skipgram_train()

    # 保存模型
    # model.save_model("data/fil9.bin") 

    # 加载已保存的模型验证
    loaded_model = fasttext.load_model("data/fil9.bin")
    
    # 测试一下获取词向量或者查看近义词
    word = "你好"
    results = loaded_model.get_nearest_neighbors(word, k=5)
    print(f"与 {word} 最相近的词: {results}", )

    print(f"与 '{word}' 最相关的词/词项：")
    for score, neighbor in results:
        print(f"  {neighbor}: {score:.4f}")