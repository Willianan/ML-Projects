# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/5/10 10:35
@Project:NLP
@Filename:4_facebook_FastTest.py
"""

"""
	生成文本格式
"""

import jieba
import pandas as pd
import random

cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}

df_technology = pd.read_csv("Data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

df_car = pd.read_csv("Data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

df_entertainment = pd.read_csv("Data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("Data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

df_sports = pd.read_csv("Data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values


def preprocess_text(content_lines, sentences, category):
	for line in content_lines:
		try:
			segs = jieba.lcut(line)
			segs = filter(lambda x: len(x) > 1, segs)
			segs = filter(lambda x: x not in stopwords, segs)
			sentences.append("__label__" + str(category) + " , " + " ".join(segs))
		except Exception as e:
			print(line)
			continue


# 生成训练数据
sentences = []

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

random.shuffle(sentences)
print("writing data to fasttext format...")
out = open('train_data.txt', 'w')
for sentence in sentences:
	out.write(sentence.encode('utf8') + "\n")
print("done!")

# 使用fastTest训练生成模型
import fasttext

classifier = fasttext.supervised('train_data.txt', 'classifier.model', label_prefix='__label__')

# 对模型效果进行评估
result = classifier.text('train_data.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

# 实际预测
label_to_cate = {1: 'technology', 2: 'car', 3: 'entertainment', 4: 'military', 5: 'sports'}

texts = ['中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟 主场 作战 中国队 率先 打破 场上 僵局 利用 角球 机会 大宝 前点 攻门 得手 中国队 领先']
labels = classifier.predict(texts)
print(labels)
print(label_to_cate[int(labels[0][0])])

labels = classifier.predict_proba(texts)
print(labels)

# Top K 个预测结果
labels = classifier.predict(texts, k=3)
print(labels)
labels = classifier.predict_proba(texts, k=3)
print(labels)


# fastText文本无监督学习

def preprocess_text_unsupervised(content_lines, sentences, category):
	for line in content_lines:
		try:
			segs = jieba.lcut(line)
			segs = filter(lambda x: len(x) > 1, segs)
			segs = filter(lambda x: x not in stopwords, segs)
			sentences.append(" ".join(segs))
		except Exception as e:
			print(line)
			continue


# 生成无监督训练数据
sentences = []

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

print("writing data to fasttext unsupervised learning format...")
out = open('unsupervised_train_data.txt', 'w')
for sentence in sentences:
	out.write(sentence.encode('utf8') + "\n")
print("done!")

import fasttext

# Skipgram model
model = fasttext.skipgram('unsupervised_train_data.txt', 'model')
print(model.words)  # list of words in dictionary

# CBOW model
model = fasttext.cbow('unsupervised_train_data.txt', 'model')
print(model.words)  # list of words in dictionary


# 对比gensim的word2vec
def preprocess_text_unsupervised(content_lines, sentences, category):
	for line in content_lines:
		try:
			segs = jieba.lcut(line)
			segs = filter(lambda x: len(x) > 1, segs)
			segs = filter(lambda x: x not in stopwords, segs)
			sentences.append(" ".join(segs))
		except Exception as e:
			print(line)
			continue


# 生成无监督训练数据
sentences = []

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
model.save("gensim_word2vec.model")
model.wv['赛季']
model.wv.most_similar('赛季')