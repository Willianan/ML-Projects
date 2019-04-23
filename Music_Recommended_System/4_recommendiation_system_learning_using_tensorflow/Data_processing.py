# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/18 10:25
@Project:Music_Recommended_System
@Filename:Data_processing.py
"""

import numpy as np
import pandas as pd
from __future__ import absolute_import, division, print_function


def read_data_and_process(fileName, sep="\t"):
	col_names = ["user", "item", "rate", "st"]
	df = pd.read_csv(fileName, sep=sep, header=None, names=col_names, engine="python")
	df["user"] -= 1
	df["item"] -= 1
	for col in ("user", "item"):
		df[col] = df[col].astype(np.int32)
	df["rate"] = df["rate"].astype(np.float32)
	return df


class ShuffleDataIterator(object):
	"""
	随机生成一个batch和一个batch数据
	"""
	# 初始化
	def __init__(self, inputs, batch_size=10):
		self.inputs = inputs
		self.batch_size = batch_size
		self.num_cols = len(self.inputs)
		self.len = len(self.inputs[0])
		self.inputs = np.transpose(np.vstack([np.array(self.inputs[i])]) for i in range(self.num_cols))
	# 总样本量
	def __len__(self):
		return self.len
	def __iter__(self):
		return self
	# 取出下一个batch
	def __next__(self):
		return self.next()
	# 随机生成batch_size个下标，取出对应的样本
	def next(self):
		ids = np.random.randint(0,self.len,(self.batch_size,))
		out =  self.inputs[ids,:]
		return [out[:,i] for i in range(self.num_cols)]

class OneEpochDataIterator(ShuffleDataIterator):
	"""
	顺序产出一个epoch的数据，在测试中可能会用到
	"""
	def __init__(self,inputs,batch_size=10):
		super(OneEpochDataIterator,self).__init__(inputs,batch_size=batch_size)
		if batch_size > 0:
			self.idx_group = np.array_split(np.arange(self.len),np.ceil(self.len / batch_size))
		else:
			self.idx_group = [np.arange(self.len)]
		self.group_id = 0

	def next(self):
		if self.group_id >= len(self.idx_group):
			self.group_id = 0
			raise StopIteration
		out = self.inputs[self.idx_group[self.group_id],:]
		self.group_id += 1
		return [out[:,i] for i in range(self.num_cols)]