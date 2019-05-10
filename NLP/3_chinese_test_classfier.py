# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/5/10 9:33
@Project:NLP
@Filename:3_chinese_test_classfier.py
"""


import jieba
import pandas as pd


#朴素贝叶斯
df_technology = pd.read_csv("Data/technology.csv",encoding="utf-8")
df_technology = df_technology.dropna()

df_car = pd.read_csv("Data/car_news.csv",encoding="utf-8")
df_car = df_car.dropna()

df_entertainmenet = pd.read_csv("Data/entertainment.csv",encoding="utf-8")
df_entertainmenet = df_entertainmenet.dropna()

df_military = pd.read_csv("Data/military_news.csv",encoding="utf-8")
df_military = df_military.dropna()

df_sports = pd.read_csv("Data/sports_news.csv",encoding="utf-8")
df_sports = df_sports.dropna()

technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainmenet.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]


# 分词与中文文本处理
#停用词
stop_words = pd.read_csv("Data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'],encoding='utf-8')
stop_words = stop_words['stopword'].values

# 去停用词
def preprocess_text(content_line,sentences,category):
	for line in content_line:
		try:
			segs = jieba.lcut(line)
			segs = filter(lambda x:len(x) > 1,segs)
			segs = filter(lambda x: x not in stop_words,segs)
			sentences.append(" ".join(segs),category)
		except Exception as e:
			print(line)
			continue

# 生成训练数据
sentences = []

preprocess_text(technology,sentences,'technology')
preprocess_text(car,sentences,'car')
preprocess_text(entertainment,sentences,'entertainment')
preprocess_text(military,sentences,'military')
preprocess_text(sports,sentences,'sports')

# 生成训练集
import random
random.shuffle(sentences)

for sentence in sentences[:10]:
	print(sentence[0],sentence[1])


from sklearn.model_selection import train_test_split
x,y = zip(sentences)
x_tain,x_test,y_train,y_test = train_test_split(x,y,random_state=1234)

len(x_tain)

# 对文本抽取词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(analyzer='word',max_features=400)
vec.fit(x_tain)

def get_features(x):
	vec.transform(x)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_tain),y_train)

# 准确率
classifier.score(vec.transform(x_test),y_test)

len(x_test)

# 加入抽取2-gram和3-gram的统计特征

vec = CountVectorizer(analyzer='word',ngram_range=(1,4),max_features=20000)
vec.fit(x_tain)

def get_feature(x):
	vec.transform(x)

classifier = MultinomialNB()
classifier.fit(vec.transform(x_tain),y_train)
classifier.score(vec.transform(x_test),y_test)

# 交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score
import numpy as np

def straifiedkfold_cv(x,y,clf_class,shuffle=True,n_folds=5,**kwargs):
	straifiedk_fold = StratifiedKFold(y,n_folds,shuffle=shuffle)
	y_pred = y[:]
	for train_index,test_index in straifiedk_fold:
		X_train,X_test = x[train_index],x[test_index]
		y_train = y[train_index]
		clf = clf_class(**kwargs)
		clf.fit(X_train,y_train)
		y_pred[test_index] = clf.predict(X_test)
	return y_pred

NB = MultinomialNB
print(precision_score(y,straifiedkfold_cv(vec.transform(x),np.array(y),NB),average='macro'))

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class TextClassifier():
	def __init__(self,classifier=MultinomialNB()):
		self.classifier = classifier
		self.vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,4),max_features=20000)

	def features(self,X):
		return self.vectorizer.transform(X)

	def fit(self,X,y):
		self.vectorizer.fit(X)
		self.classifier.fit(self.features(X),y)

	def predict(self,x):
		return self.classifier.predict(self.features([x]))

	def score(self,X,y):
		return self.classifier.score(self.features(X),y)

test_classifier = TextClassifier()
test_classifier.fit(x_tain,y_train)
print(test_classifier.predict('这 是 有史以来 最 大 的一次 军舰 演习'))
print(test_classifier.score(x_test,y_test))


# SVM文本分类
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_tain),y_train)
svm.score(vec.transform(x_test),y_test)

svm = SVC()
svm.fit(vec.transform(x_tain),y_train)
svm.score(vec.transform(x_test),y_test)

# 换特征/模型

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class TextClassifier():
	def __init__(self,classifier=SVC(kernel='linear')):
		self.classifier = classifier
		self.vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=12000)

	def features(self,X):
		return self.vectorizer.transform(X)

	def fit(self,X,y):
		self.vectorizer.fit(X)
		self.classifier.fit(self.features(X),y)

	def predict(self,x):
		return self.classifier.predict(self.features([x]))

	def score(self,X,y):
		return self.classifier.score(self.features(X),y)

test_classifier = TextClassifier()
test_classifier.fit(x_tain,y_train)
print(test_classifier.predict('这 是 有史以来 最 大 的一次 军舰 演习'))
print(test_classifier.score(x_test,y_test))