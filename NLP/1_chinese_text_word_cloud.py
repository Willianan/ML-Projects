# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/28 9:45
@Project:NLP
@Filename:1_chinese_text_word_cloud.py
"""


import warnings
warnings.filterwarnings("ignore")
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0,5.0)
from wordcloud import WordCloud

# 导入娱乐新闻数据，分词
df = pd.read_csv('Data/entertainment_news.csv',sep=',',encoding='utf-8')
df = df.dropna()
content = df.content.values.tolist()
segment = []
for line in content:
	try:
		segs = jieba.lcut(line)
		for seg in segs:
			if len(seg) > 1 and segs !='\r\n':
				segment.append(seg)
	except:
		print(line)
		continue

# 去停用词
words_df = pd.DataFrame({'segment':segment})
stopwords = pd.read_csv("Data/stopwords.txt",index_col=False,quoting=3,sep='\t',names=['stopword'],encoding='utf-8')
words_df = words_df[words_df.segment.isin(stopwords.stopword)]

# 统计词频
words_stat = words_df.groupby(by=['segment'])['segment'].agg({'count':np.size})
words_stat = words_stat.reset_index().sort_values(by=['count'],ascending=False)
print(words_stat)

# 词云
wordcloud = WordCloud(font_path='Data/simhei.ttf',background_color="white",max_font_size=80)
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
plt.imshow(wordcloud)

# 自定义背景图词云
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (15.0,15.0)
from wordcloud import WordCloud,ImageColorGenerator
bimg = imread("image/entertainment.jpeg")
wordcloud = WordCloud(background_color='white',mask=bimg,font_path='Data/simhei.ttf',max_font_size=200)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
bimgColors = ImageColorGenerator(bimg)
plt.axis('off')
plt.imshow(wordcloud.recolor(color_func=bimgColors))

"""体育新闻"""

df = pd.read_csv("Data/sports_news.csv",encoding='utf-8')
df = df.dropna()
content = df.content.values.tolist()
segment = []
for line in content:
	try:
		segs = jieba.lcut(line)
		for seg in segs:
			if len(seg) > 1 and segs != "\r\n":
				segment.append(seg)
	except:
		print(line)
		continue

matplotlib.rcParams['figure.figsize'] = (10.0,8.0)
words_df = pd.DataFrame({'segment':segment})
stopwords = pd.read_csv('Data/stopwords.txt',index_col=False,quoting=3,sep='\t',names=['stopword'],encoding='utf-8')
words_df = words_df[words_df.segment.isin(stopwords.stopword)]
words_stat = words_df.groupby(by=['segment'])['segment'].agg({'count':np.size})
words_stat = words_stat.reset_index().sort_values(by=['count'],ascending=False)
words_stat.head()
wordcloud=WordCloud(font_path="data/simhei.ttf",background_color="black",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)

# 加入自定义图
from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from wordcloud import WordCloud,ImageColorGenerator
bimg=imread('image/sports.jpeg')
wordcloud=WordCloud(background_color="white",mask=bimg,font_path='data/simhei.ttf',max_font_size=200)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
bimgColors=ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))