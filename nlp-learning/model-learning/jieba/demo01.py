import jieba.posseg as pseg

# content = "无线电法国别研究"
# print(jieba.cut(content, cut_all=False)) # cut_all默认为False

# 若需直接返回列表内容，使用jieba.lcut即可
# print(jieba.lcut(content, cut_all=False))

# 若需直接返回列表内容，使用jieba.lcut即可
# print(jieba.lcut(content, cut_all=True))

# print(jieba.lcut(content))

# print(jieba.lcut_for_search(content))

# 若需直接返回列表内容，使用jieba.lcut_for_search即可
# print(jieba.lcut_for_search(content))

# content = "煩惱即是菩提，我暫且不提"
# print(jieba.lcut(content, cut_all=True))

# print(jieba.lcut("念念相续，不离般若"))

# print(jieba.lcut("我一把把车把把住了"))

# print(jieba.lcut("八一双鹿更名为八一南昌 篮球队！"))

# jieba.load_userdict("data/userdict.txt")

# print(jieba.lcut("八一双鹿更名为八一南昌篮球队，小猪老师去看了比赛！"))

print(pseg.lcut("我爱北京天安门，天安门上太阳升"))