# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/15 20:11
@Project:Music_Recommended_System
@Filename:Algorithm_Evaluate.py
"""

import os
from surprise import Reader,Dataset

# 指定文件路径
file_path = os.path.expanduser('./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp',sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path,reader=reader)
# 分成5折
music_data.split(n_folds=5)

# 使用NormalPredictor
from surprise import NormalPredictor,evaluate
algo1 = NormalPredictor()
perf1 = evaluate(algo1,music_data,measures=['RMSE','MAE'])

# 使用BaselineOnly
from surprise import BaselineOnly,evaluate
algo2 = BaselineOnly()
perf2 = evaluate(algo2,music_data,measures=['RMSE','MAE'])

# 使用基础版协同过滤
from surprise import KNNBasic,evaluate
algo3 = KNNBasic()
perf3 = evaluate(algo3,music_data,measures=['RMSE','MAE'])

# 使用均值协同过滤
from surprise import KNNWithMeans,evaluate
algo4 = KNNWithMeans()
perf4 = evaluate(algo4,music_data,measures=['RMSE','MAE'])

# 使用协同过滤baseline
from surprise import KNNBaseline,evaluate
algo5 = KNNBaseline()
perf5 = evaluate(algo5,music_data,measures=['RMSE','MAE'])
