# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/19 10:44
@Project:Machine_Translation_System
@Filename:utils.py
"""


import logging
import numpy as np
import torch.nn as nn
import torch
import io
from collections import Counter
import nltk

def load_data(in_file):
	en = []
	cn = []
	with io.open(in_file,"r",encoding="utf-8") as f:
		for line in f:
			line = line.strip().split()
			en.append(["BOS"]+nltk.word_tokenize(line[0]) + ["EOS"])
			cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
	return en,cn

def build_dict(sentences,max_words=50000):
	word_count =Counter()
	for sentence in sentences:
		for s in sentence:
			word_count[s] += 1
	ls = word_count.most_common(max_words)
	total_words = len(ls) + 1
	word_dict = {w[0]: index+1 for (index,w) in enumerate(ls)}
	word_dict["UNK"] = 0
	return word_dict,total_words

def encode(en_sentences,cn_sentences,en_dict,cn_dict,sort_by_len=True):
	'''
	:param en_sentences: 英文语句
	:param cn_sentences: 中文语句
	:param en_dict: 英文映射
	:param cn_dict: 中文映射
	:param sort_by_len: 排序
	:return: encode the sequences
	'''
	length = len(en_sentences)
	out_en_sentences = []
	out_cn_sentences = []

	for i in range(length):
		en_seq = [en_dict[w] if w in en_dict else 0 for w in en_sentences[i]]
		cn_seq = [cn_dict[w] if w in cn_dict else 0 for w in cn_sentences[i]]
		out_en_sentences.append(en_seq)
		out_cn_sentences.append(cn_seq)
	if sort_by_len:
		sorted_index = len_argsort(out_en_sentences)
		out_en_sentences = [out_en_sentences[i] for i in sorted_index]
		out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
	return out_en_sentences,out_cn_sentences


# sort sentences by english lengths
def len_argsort(seq):
	return sorted(range(len(seq)),key=lambda x:len(seq[x]))

def get_minibatches(n,minibatch_size,shuffle=False):
	idx_list = np.arange(0,n,minibatch_size)
	if shuffle:
		np.random.shuffle(idx_list)
	minibatches = []
	for idx in idx_list:
		minibatches.append(np.arange(idx,min(idx + minibatch_size,n)))
	return minibatches

def prepare_data(seqs):
	# convert to minibatch of seqs into numpy matrix
	lengths = [len(seq) for seq in seqs]
	n_samples = len(seqs)
	max_len = np.max(lengths)
	x = np.zeros((n_samples,max_len)).astype('int32')
	x_mask = np.zeros((n_samples,max_len)).astype('float32')
	for idx,seq in enumerate(seqs):
		x[idx,:lengths[idx]] = seq
		x_mask[idx,:lengths[idx]] = 1.0
	return x,x_mask

def get_examples(en_sentences,cn_sentences,batch_size):
	minibatches = get_minibatches(len(en_sentences), batch_size)
	all_ex = []
	for minibatch in minibatches:
		mb_en_sentences = [en_sentences[t] for t in minibatch]
		mb_cn_sentences = [cn_sentences[t] for t in minibatch]
		# convert to numpy array
		mb_x, mb_x_mask = prepare_data(mb_en_sentences)
		mb_y, mb_y_mask = prepare_data(mb_cn_sentences)
		all_ex.append((mb_x, mb_x_mask, mb_y, mb_y_mask))
	return all_ex


def to_contiguous(tensor):
	if tensor.is_contiguous():
		return tensor
	else:
		return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
	def __init__(self):
		super(LanguageModelCriterion, self).__init__()

	def forward(self, input, target, mask):
		input = to_contiguous(input).view(-1, input.size(2))
		target = to_contiguous(target).view(-1, 1)
		mask = to_contiguous(mask).view(-1, 1)
		output = -input.gather(1, target) * mask
		output = torch.sum(output) / torch.sum(mask)

		return output


class LinearND(nn.Module):
	def __init__(self, *args, **kwargs):
		"""
		A torch.nn.Linear layer modified to accept ND arrays.
		The function treats the last dimension of the input
		as the hidden dimension.
		"""
		super(LinearND, self).__init__()
		self.fc = nn.Linear(*args, **kwargs)

	def forward(self, x):
		size = x.size()
		out = x.view(-1, size[-1])
		out = self.fc(out)
		size = list(size)
		size[-1] = out.size()[-1]
		return out.view(size)
