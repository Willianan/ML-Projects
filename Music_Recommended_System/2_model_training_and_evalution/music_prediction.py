# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/15 16:06
@Project:Music_Recommended_System
@Filename:music_prediction.py
"""

from __future__ import absolute_import,division,print_function,unicode_literals
import os
import cProfile as pickle


from surprise import KNNBaseline,Reader
from surprise import Dataset

import cProfile as pickle
# 重建歌单id到歌单名的映射字典
id_name_dic = pickle.load(open("popular_playlist.pk1","rb"))
print("加载歌单id到歌单名的映射字典完成....")
# 重建歌单名到歌单id的映射字典
name_id_dic = {}
for playlist_id in id_name_dic:
	name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("加载歌单名到歌单id的映射字典完成.....")


file_path = os.path.expanduser("./popular_music_suprise_format.txt")
# 指定文件格式
reader = Reader(line_format="user item rating timestamp",sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path,reader=reader)
# 计算歌曲间的相似度
print("构建数据集......")
trainset = music_data.build_full_trainset()

# 模板之查找最近的user
print("开始训练模型.....")
algo = KNNBaseline()
algo.train(trainset)

current_playlist = name_id_dic.keys()[39]
print("歌单名称：",current_playlist)

# 取出近邻
playlist_id = name_id_dic[current_playlist]
print("歌单id：",playlist_id)
# 取出来对应的内部user id
playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
print("内部id：",playlist_inner_id)

playlist_neighbors = algo.get_neighbors(playlist_inner_id,k=10)

# 把歌曲id转成歌曲名字
playlist_neighbors = (algo.trainset.to_raw_uid(inner_id) for inner_id in playlist_neighbors)
playlist_neighbors = (id_name_dic[playlist_id] for playlist_id in playlist_neighbors)
print()
print("和歌单《" + current_playlist + "》，最接近的10首歌单为：")
for playlist in playlist_neighbors:
	print(playlist,algo.trainset.to_inner_uid(name_id_dic[playlist]))


# 针对用户进行预测
song_id_name_dic = pickle.load(open("popular_song.pk1","rb"))
print("加载歌曲id到歌曲名的映射字典完成......")
# 重建歌曲名到歌曲id的映射字典
song_name_id_dic = {}
for song_id in song_id_name_dic:
	song_name_id_dic[song_id_name_dic[song_id]] = song_id
print("加载歌曲名到歌曲id的映射字典完成......")

user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0],user_rating)
for song in items:
	print(algo.predict(user_inner_id,song_id,r_ui=1),song_id_name_dic[algo.trainset])

from collections import defaultdict
from surprise import SVD
from surprise import Dataset

def get_top_n(predictions,n=10):
	top_n = defaultdict(list)
	for uid,iid,true_r,est,_ in predictions:
		top_n[uid].append(iid,est)
	for uid,user_ratings in top_n.items():
		user_ratings.sort(key=lambda x:x[1],reverse=True)
		top_n[uid] = user_ratings[:n]
	return top_n

# 生成所有训练集中没有的（user,item）对
testset = trainset.build_anti_testset()
pickle.dump(testset,open("testset","wb"))

# 最粗暴的预测方式，把所有的pair都预测一遍，并按照得分排序
predictions = algo.test(testset)

top_n = get_top_n(predictions,n=10)
for uid,user_ratings in top_n.items():
	print(uid,[iid for (iid,_) in user_ratings])

# 2、用矩阵分解进行预测
# 使用NMF
from surprise import NMF,evaluate
from surprise import Dataset

file_path = os.path.expanduser("./popular_music_suprise_format.txt")
# 指定文件格式
reader = Reader(line_format='user item rating timestamp',sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path,reader=reader)
# 构建数据集和模型
algo = NMF()
trainset = music_data.build_full_trainset()
algo.train(trainset)

user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0],user_rating)
for song in items:
	print(algo.predict(algo.trainset.to_raw_uid(user_inner_id),algo.trainset))

# 模型存储
import surprise
surprise.dump.dump('./recommendation.model',algo=algo)
# 可以用下面的方式载入
algo = surprise.dump.load('./recommendation.model')

