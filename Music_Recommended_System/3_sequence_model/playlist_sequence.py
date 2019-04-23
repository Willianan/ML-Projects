# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/16 10:19
@Project:Music_Recommended_System
@Filename:playlist_sequence.py
"""

import multiprocessing
import gensim
from random import shuffle


def parse_playlist_get_sequence(in_line, playlist_sequence):
	song_sequence = []
	contents = in_line.strip().split("\t")
	# 解析歌单序列
	for song in contents[1:]:
		try:
			song_id, song_name, artist, popularity = song.split(":::")
			song_sequence.append(song_id)
		except:
			print("song frormat error")
			print(song)
	for i in range(len(song_sequence)):
		shuffle(song_sequence)
		playlist_sequence.append(song_sequence)


def train_song2vec(in_file, out_file):
	# 所有歌单序列
	playlist_sequence = []
	# 遍历所有歌单
	for line in open(in_file):
		parse_playlist_get_sequence(line, playlist_sequence)
	# 使用word2vec训练
	cores = multiprocessing.cpu_count()
	print("using all " + str(cores) + " cores")
	print("Training word2vec model...")
	model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150, min_count=3, window=7, workers=cores)
	print("Saving model.....")
	model.save(out_file)


import cProfile as pickle
song_dic = pickle.load(open("popular_song.pkl","rb"))
model_str = "./song2vec.model"
model = gensim.models.Word2Vec.load(model_str)

song_id_list = song_dic.keys()[1000:1500:50]
for song_id in song_id_list:
	result_song_list = model.most_similar(song_id)
	print(song_id,song_dic[song_id])
	print("相似歌曲和相似度分别为：")
	for song in result_song_list:
		print("\t",song_dic[song[0]],song[1],end="")
	print()
