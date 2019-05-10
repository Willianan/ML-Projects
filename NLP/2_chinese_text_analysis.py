# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/28 11:20
@Project:NLP
@Filename:2_chinese_text_analysis.py
"""

import jieba.analyse as analyse
import pandas as pd

# 基于TF-IDF算法的关键词抽取
df = pd.read_csv("/Data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()
content = "".join(lines)
print(" ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))

df = pd.read_csv("Data/military_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()
content = "".join(lines)
print(" ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))

# 基于TextRank算法的关键词抽取
df = pd.read_csv("Data/military_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()
content = "".join(lines)
print(" ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
print("==================================================================")
print(" ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n'))))

"""
LDA主题模型
"""
from gensim import corpora, models, similarities
import gensim

stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values

import jieba
import pandas as pd

df = pd.read_csv("Data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()

sentences = []
for line in lines:
	try:
		segs = jieba.lcut(line)
		segs = filter(lambda x: len(x) > 1, segs)
		segs = filter(lambda x: x not in stopwords, segs)
		sentences.append(segs)
	except Exception as e:
		print(line)
		continue

for word in sentences[5]:
	print(word)

# 词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

# LDA建模
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(3, topn=5))

for topic in lda.print_topics(num_topics=20, num_words=8):
	print(topic[1])

lda.get_document_topics(bow)
