# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/18 15:24
@Project:Music_Recommended_System
@Filename:user_based.py
"""


# 基于用户的协同过滤
# 使用余弦相似度
import sys
from collections import defaultdict
from itertools import combinations
import random
import numpy as np
import pdb

from pyspark import SparkContext

def parseVectorOnUser(line):
	# 解析数据 key是user，后面的是item和打分
	line = line.split("|")
	return line[0],(line[1],float(line[2]))

def parseVectorOnItem(line):
	# 解析数据 key是item，后面的是user和打分
	line = line.split("|")
	return line[1], (line[0], float(line[2]))

def sampleInteractions(item_id,users_with_rating,n):
	# 如果某个商品上用户行为特别多，可以选择适当做点下采样
	if len(users_with_rating) > n:
		return item_id,random.sample(users_with_rating,n)
	else:
		return item_id,users_with_rating

def findUserParirs(item_id,users_with_rating):
	# 对每个item，找到共同的打分的user对
	for user1,user2 in combinations(users_with_rating,2):
		return (user1[0],user2[0],(user1[1],user2[1]))

def calcSim(user_pair,rating_pairs):
	# 对每个user对，根据打分计算余弦距离，并返回共同打分的item个数
	sum_xx,sum_xy,sum_yy,sum_x,sum_y,n = (0.0,0.0,0.0,0.0,0.0,0)
	for rating_pair in rating_pairs:
		sum_xx += np.float(rating_pair[0] * np.float(rating_pair[0]))
		sum_yy += np.float(rating_pair[1] * np.float(rating_pair[1]))
		sum_xy += np.float(rating_pair[0] * np.float(rating_pair[1]))
		n += 1
	cos_sim = cosine(sum_xy,np.sqrt((sum_xx),np.sqrt(sum_yy))
	return user_pair,(cos_sim,n)
def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
	"""
	:param dot_product:向量A和B
	:param rating_norm_squared:A向量模
	:param rating2_norm_squared: B向量模
	:return: 向量余弦值->dotProduct(A,B) / (norm(A) * norm(B))
	"""
	numerator = dot_product
	denominator = rating_norm_squared * rating2_norm_squared
	return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstUser(user_pair,item_sim_data):
	# 对于每一个user-user对，用第一个user做key
	(user1_id,user2_id) = user_pair
	return user1_id,(user2_id,item_sim_data)
def nearesNeighbors(user,users_and_sims,n):
	# 选出相似度最高的N个邻居
	users_and_sims.sort(key=lambda x:x[1][0],reverse=True)
	return user,users_and_sims[:n]
def topNRecommenddations(user_id,user_sims,users_with_rating,n):
	# 根据最近的N个邻居进行推荐
	totals = defaultdict(int)
	sim_nums = defaultdict(int)
	for (neighbor,(sim,count)) in user_sims:
		# 遍历邻居的打分
		unscored_items = users_with_rating.get(neighbor,None)
		if unscored_items:
			for (item,rating) in unscored_items:
				if neighbor != item:
					# 更新推荐度和相似度
					totals[neighbor] += sim * rating
					sim_nums[neighbor] += sim
	# 归一化
	scord_items = [(total / sim_nums[item],item) for item,total in totals.items()]
	# 按照推荐度降序排列
	scord_items.sort(reverse=True)
	# 推荐度的item
	ranked_item = [x[1] for x in scord_items]
	return user_id,ranked_item[:n]

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print(>> sys.stderr,"Usage: PythonUSerCF <master> <file>")
		exit(-1)
	sc = SparkContext(sys.argv[1],"PythonUserCF")
	lines = sc.textFile(sys.argv[2])
	"""
	处理数据，获取稀疏item-user矩阵
	item_id ->((user_1,rating),(user_2,rating))
	"""
	item_user_pairs = lines.map(parseVectorOnItem).groupByKey().map(lambda p:sampleInteractions(p[0],p[1],500)).cache()
	"""
	获取2个用户所有的item-item对得分组合：
	(user1_id,user2_id) -> [(rating1,rating2),(rating1,rating2),(rating1,rating2),....]
	"""
	pairwise_users = item_user_pairs.filter(lambda p:len(p[1]) > 1).map(lambda p:findUserParirs(p[0],p[1])).groupByKey()
	"""
	计算余弦相似度，找到最近的N个邻居
	(user1,user2) -> (similarity.co_raters_count)
	"""
	user_sims = pairwise_users.map(lambda p:calcSim(p[0],p[1])).map(
		lambda p:keyOnFirstUser(p[0],p[1])).groupByKey().map(
		lambda p:nearesNeighbors(p[0],p[1],50))
	"""
	对每个用户的打分记录整理如下格式：
		user_id -> [(item_id_1,rating_1),(item_id_2,rating_2),....]
	"""
	user_item_hist = lines.map(parseVectorOnUser).groupByKey().collect()
	ui_dict = {}
	for (user,items) in user_item_hist:
		ui_dict[user] = items
	uib = sc.broadcast(ui_dict)
	"""
	为各个用户计算Top N的推荐
	user_id -> [item1,item2,item3,....]
	"""
	user_item_recs = user_sims.map(lambda p:topNRecommenddations(p[0],p[1],uib.value,100))
