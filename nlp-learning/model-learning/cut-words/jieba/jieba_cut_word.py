import jieba
import jieba.posseg as pseg


def test01():
    content = "无线电法国别研究"
    # 默认全模式
    print(jieba.cut(content, cut_all=False)) 

    # 若直接返回列表内容，使用 jieba.lcut 即可
    print(jieba.lcut(content, cut_all=False)) # False 精确模式
    print(jieba.lcut(content, cut_all=True)) # True 全词模式

# 搜索引擎模式
def test02():
    content = "无线电法国别研究"

    print(jieba.lcut_for_search(content))

# 导入 userdict.txt 词表
def test03():
    jieba.load_userdict("data/userdict.txt")

    # content = "烦恼即是菩提，我暂且不提"
    # print(jieba.lcut(content, cut_all=True))

    # print(jieba.lcut("念念相续，不离般若"))

    print(jieba.lcut("八一双鹿更名为八一南昌篮球队！"))
    # print(jieba.lcut("我一把把车把把住了。"))

# 引入带词性的标注分词 import jieba.posseg as pseg
def test04():
    print(pseg.lcut("我爱北京天安门"))

if __name__ == "__main__":
    test01()
    test02()
    test03()
    test04()