# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/19 10:25
@Project:Machine_Translation_System
@Filename:main.py
"""


"""
1.preprocessing: word(en,cn) -> number(one hot vector)
2.model: encoder decoder model
3.training: feed the training data into the models,loss function,gradient descent
4.valuement on development set and keep the model with the best performance on dev set
5. test
6.translate
"""

import sys, os
import code
import numpy as np
from tqdm import tqdm
import pickle
import math
import nltk

from test import eval
from models import *
import utils
import config

import torch
from torch import optim
from torch.autograd import Variable


def main(args):
	# load sentences (English and Chinese words)
	train_en, train_cn = utils.load_data(args.train_file)
	dev_en, dev_cn = utils.load_data(args.dev_file)
	args.num_train = len(train_en)
	args.num_dev = len(dev_en)

	# build English and Chinese dictionary
	if os.path.isfile(args.vocab_file):
		en_dict, cn_dict, en_total_words, cn_total_words = pickle.load(open(args.vocab_file, "rb"))
	else:
		en_dict, en_total_words = utils.build_dict(train_en)
		cn_dict, cn_total_words = utils.build_dict(train_cn)
		pickle.dump([en_dict, cn_dict, en_total_words, cn_total_words], open(args.vocab_file, "wb"))
	# index to words dict
	inv_en_dict = {v: k for k, v in en_dict.items()}
	inv_cn_dict = {v: k for k, v in cn_dict.items()}
	args.en_total_words = en_total_words
	args.cn_total_words = cn_total_words



	# encode train and dev sentences into indieces
	train_en, train_cn = utils.encode(train_en, train_cn, en_dict, cn_dict)
	# convert to numpy tensors
	train_data = utils.gen_examples(train_en, train_cn, args.batch_size)

	dev_en, dev_cn = utils.encode(dev_en, dev_cn, en_dict, cn_dict)
	dev_data = utils.gen_examples(dev_en, dev_cn, args.batch_size)

	if os.path.isfile(args.model_file):
		model = torch.load(args.model_file)
	elif args.model == "EncoderDecoderModel":
		model = EncoderDecoderModel(args)

	if args.use_cuda:
		model = model.cuda()

	crit = utils.LanguageModelCriterion()
	learning_rate = args.learning_rate
	optimizer = optim.Adam(model.parameter(),lr=learning_rate)

	print("start evaluating on dev...")
	correct_count, loss, num_words = eval(model, dev_data, args, crit)

	loss = loss / num_words
	acc = correct_count / num_words
	print("dev loss %s" % (loss))
	print("dev accuracy %f" % (acc))
	print("dev total number of words %f" % (num_words))
	best_acc = acc

	learning_rate = args.learning_rate
	optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)

	total_num_sentences = 0.
	total_time = 0.

	for epoch in range(args.num_epoches):
		np.random.shuffle(train_data)
		total_train_loss = 0.
		total_num_words = 0.
		for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(train_data):

			batch_size = mb_x.shape[0]
			total_num_sentences += batch_size
			# convert numpy ndarray to PyTorch tensors and variables
			mb_x = Variable(torch.from_numpy(mb_x).long())
			mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
			hidden = model.init_hidden(batch_size)
			mb_input = Variable(torch.from_numpy(mb_y[:, :-1]).long())
			mb_out = Variable(torch.from_numpy(mb_y[:, 1:])).long()
			mb_out_mask = Variable(torch.from_numpy(mb_y_mask[:, 1:])).long()

			if args.use_cuda:
				mb_x = mb_x.cuda()
				mb_x_mask = mb_x_mask.cuda()
				mb_input = mb_input.cuda()
				mb_out = mb_out.cuda()
				mb_out_mask = mb_out_mask.cuda()

			mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)
			# calculate loss function
			loss = crit(mb_pred, mb_out, mb_out_mask)
			num_words = torch.sum(mb_out_mask).item()
			total_train_loss += loss.item() * num_words
			total_num_words += num_words
			# update the model
			optimizer.zero_grad()   # zero the previous gradient
			loss.backward()         # calculate gradient
			optimizer.step()        # gradient descent

		print("training loss: %f" % (total_train_loss / total_num_words))

		# evaluate every eval_epoch
		if (epoch + 1) % args.eval_epoch == 0:

			print("start evaluating on dev...")

			correct_count, loss, num_words = eval(model, dev_data, args, crit)

			loss = loss / num_words
			acc = correct_count / num_words
			print("dev loss %s" % (loss))
			print("dev accuracy %f" % (acc))
			print("dev total number of words %f" % (num_words))

			# save model if we have the best accuracy
			if acc >= best_acc:
				torch.save(model, args.model_file)
				best_acc = acc

				print("model saved...")
			else:
				learning_rate *= 0.5
				optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)

			print("best dev accuracy: %f" % best_acc)
			print("#" * 60)

	# load test data
	test_en, test_cn = utils.load_data(args.test_file)
	args.num_test = len(test_en)
	test_en, test_cn = utils.encode(test_en, test_cn, en_dict, cn_dict)
	test_data = utils.gen_examples(test_en, test_cn, args.batch_size)

	# evaluate on test
	correct_count, loss, num_words = eval(model, test_data, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("test loss %s" % (loss))
	print("test accuracy %f" % (acc))
	print("test total number of words %f" % (num_words))

	# evaluate on train
	correct_count, loss, num_words = eval(model, train_data, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("train loss %s" % (loss))
	print("train accuracy %f" % (acc))


if __name__ == "__main__":
	args = config.get_args()
	main(args)
