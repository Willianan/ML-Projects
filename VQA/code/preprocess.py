# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/5/15 10:27
@Project:VQA
@Filename:preprocess.py
"""


import os
from random import shuffle

import numpy as np
import scipy.io
import operator
from collections import defaultdict

from keras.models import Sequential,model_from_json
from keras.utils import generic_utils,np_utils
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers import Merge
from keras.layers.recurrent import LSTM

from sklearn import preprocessing
from sklearn.externals import joblib

from spacy.lang.en import English

from keras.utils.vis_utils import plot_model
from IPython.display import Image

from itertools import zip_longest

questions_train = open('data/preprocessed/questions_train2014.txt', 'rb').read().splitlines()
answers_train = open('data/preprocessed/answers_train2014.txt', 'rb').read().splitlines()
images_train = open('data/preprocessed/images_train2014.txt', 'rb').read().splitlines()

# 设定最多选取多少个回答
max_answers = 1000
answers_fq = defaultdict(int)
# 并为所有的回答，构造一个字典
for answer in answers_train:
	answers_fq[answer] += 1
# 按照出现次数排序
sorted_fq = sorted(answers_fq.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
top_answers, top_fq = zip(*sorted_fq)
new_answers_train = []
new_questions_train = []
new_images_train = []
# 只提取top1000问答相关
for answer, question, image in zip(answers_train, questions_train, images_train):
	if answer in top_answers:
		new_answers_train.append(answer)
		new_questions_train.append(question)
		new_images_train.append(image)
# 将新的数据赋值
questions_train = new_questions_train
answers_train = new_answers_train
images_train = new_images_train

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
nb_classes = len(list(labelencoder.classes_))
joblib.dump(labelencoder, 'data/labelencoder.pkl')

def get_answers_matrix(answers, encoder):
	# string转化成数字化表达
	y = encoder.transform(answers)
	nb_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, nb_classes)
	# 并构造成标准的matrix
	return Y

# input图片处理
vgg_model_path = 'Downloads/vgg_feats.mat'
# 导入下载好的vgg_features
features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
# 跟图片一一对应
image_ids = open('data/coco_vgg_IDMap.txt').read().splitlines()
id_map = {}
for ids in image_ids:
	id_split = ids.split()
	id_map[id_split[0]] = int(id_split[1])

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
	nb_samples = len(img_coco_ids)
	nb_dimensions = VGGfeatures.shape[0]
	image_matrix = np.zeros(nb_samples, nb_dimensions)
	for j in range(len(img_coco_ids)):
		image_matrix[j, :] = VGGfeatures[:, img_map[img_coco_ids[j]]]
	return image_matrix

# 提问问题的处理
# 载入Spacy的英语库
nlp = English()
# 图片的维度大小
img_dim = 4096
# 句子/单词的维度大小
word_vec_dim = 300

# 计算句子中所有word vector的总和
def get_questions_matrix_sum(questions, nlp):
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros(nb_samples, word_vec_dim)
	for i in range(len(questions)):
		tokens = nlp(questions[i])
		for j in range(len(tokens)):
			questions_matrix[i, :] += tokens[j].vector
	return questions_matrix

"""
VQA模型：MLP
"""
# 参数
num_hidden_units = 1024
num_hidden_layers = 3
dropout = 0.5
activation = 'tanh'
num_epochs = 100
model_save_interval = 10
batch_size = 128

# MLP模型
# 输入层
model = Sequential()
model.add(Dense(num_hidden_units, input_dim=img_dim + word_vec_dim, kernel_initializer='uniform'))
model.add(Activation(activation))
model.add(Dropout(dropout))
# 中间层
for i in range(num_hidden_layers - 1):
	model.add(Dense(num_hidden_units, kernel_initializer='uniform'))
	model.add(Activation(activation))
	model.add(Dropout(dropout))
# 输出层
model.add(Dense(nb_classes, kernel_initializer='uniform'))
model.add(Activation('softmax'))
# 打印模型
plot_model(model, to_file='data/model_mlp.png', show_shapes=True)
Image('data/model_mlp.png')

# 保存
json_string = model.to_json()
model_file_name = 'data/mlp_num_hidden_units_' + str(num_hidden_units) + '_num_hidden_layers_' + str(num_hidden_layers)
open(model_file_name + '.json', 'w').write(json_string)
# compile模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# 标准的chunk list方法
def grouper(iterable, n, fillvalue=None):
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)


# 开始训练
for k in range(num_epochs):
	# 给数据洗牌
	index_shuf = [i for i in range(len(questions_train))]
	shuffle(index_shuf)
	# 一一取出 问题，答案和图片
	questions_train = [questions_train[i] for i in index_shuf]
	answers_train = [answers_train[i] for i in index_shuf]
	images_train = [images_train[i] for i in index_shuf]
	progbar = generic_utils.Progbar(len(questions_train))
	# batch分组
	for qu_batch, an_batch, im_batch in zip(grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
	                                        grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
	                                        grouper(images_train, batch_size, fillvalue=images_train[-1])):
		X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
		X_i_batch = get_images_matrix(im_batch, id_map, VGGfeatures)
		X_batch = np.hstack(X_q_batch, X_i_batch)
		Y_batch = get_answers_matrix(an_batch, labelencoder)
		loss = model.train_on_batch(X_batch, Y_batch)
		progbar.add(batch_size, values=[("train loss", loss)])
	if k % model_save_interval == 0:
		model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))
model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

"""
VQA模型：LSTM
"""
def get_questions_tensor_timeseries(questions, nlp, timesteps):
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
	for i in range(len(questions)):
		tokens = nlp(questions[i])
		for j in range(len(tokens)):
			if j < timesteps:
				questions_tensor[i, j, :] = tokens[j].vector
	return questions_tensor

# 参数们
max_len = 30
word_vec_dim = 300
img_dim = 4096
dropout = 0.5
activation_mlp = 'tanh'
num_epochs = 1
model_save_interval = 5
num_hidden_units_mlp = 1024
num_hidden_units_lstm = 512
num_hidden_layers_mlp = 3
num_hidden_layers_lstm = 1
batch_size = 128

# 先造一个图片模型，也就是专门用来处理图片部分的
image_model = Sequential()
image_model.add(Reshape((img_dim,), input_shape=(img_dim,)))

# 在来一个语言模型，专门用来处理语言的
# 因为，只有语言部分，需要LSTM。
language_model = Sequential()
if num_hidden_layers_lstm == 1:
	language_model.add(
		LSTM(output_dim=num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))
else:
	language_model.add(
		LSTM(output_dim=num_hidden_units_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
	for i in range(num_hidden_layers_lstm - 2):
		language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=True))
	language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=False))

# 接下来，把楼上两个模型merge起来，
# 做最后的一步“分类”
model = Sequential()
model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
for i in range(num_hidden_layers_mlp):
	model.add(Dense(num_hidden_units_mlp, init='uniform'))
	model.add(Activation(activation_mlp))
	model.add(Dropout(dropout))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 同理，我们把模型结构存下来
json_string = model.to_json()
model_file_name = 'data/lstm_1_num_hidden_units_lstm_' + str(num_hidden_units_lstm) + \
                  '_num_hidden_units_mlp_' + str(num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
                  str(num_hidden_layers_mlp) + '_num_hidden_layers_lstm_' + str(num_hidden_layers_lstm)
open(model_file_name + '.json', 'w').write(json_string)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print('Compilation done')

plot_model(model, to_file='data/model_lstm.png', show_shapes=True)
Image(filename='data/model_lstm.png')

features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
print('loaded vgg features')
image_ids = open('data/coco_vgg_IDMap.txt').read().splitlines()
img_map = {}
for ids in image_ids:
	id_split = ids.split()
	img_map[id_split[0]] = int(id_split[1])

nlp = English()
print('loaded word2vec features...')
# training
print('Training started...')
for k in range(num_epochs):
	progbar = generic_utils.Progbar(len(questions_train))
	for qu_batch, an_batch, im_batch in zip(grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
	                                        grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
	                                        grouper(images_train, batch_size, fillvalue=images_train[-1])):
		X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, max_len)
		X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
		Y_batch = get_answers_matrix(an_batch, labelencoder)
		loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
		progbar.add(batch_size, values=[("train loss", loss)])

	if k % model_save_interval == 0:
		model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))
model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

"""
Demo
"""
# 在新的环境下：
# 载入NLP的模型
nlp = English()
# 以及label的encoder
labelencoder = joblib.load('data/labelencoder.pkl')

# 接着，把模型读进去
model = model_from_json(open(
	'data/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1.json').read())
model.load_weights(
	'data/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1_epoch_000.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
nlp = English()
labelencoder = joblib.load('data/labelencoder.pkl')
flag = True

# 所有需要外部导入的料
vggmodel = 'data/VGG_ILSVRC_19_layers.caffemodel'
prototxt = 'data/VGG-Copy1.prototxt'
img_path = 'data/test_img.png'
image_features = 'data/test_img_vgg_feats.mat'

while flag:
	# 首先，给出你要提问的图片
	img_path = str(raw_input('Enter path to image : '))
	# 对于这个图片，我们用caffe跑一遍VGG CNN，并得到4096维的图片特征
	os.system('python extract_features.py --caffe ' + ' --model_def '
	          + prototxt + ' --model ' + vggmodel + ' --image ' + img_path +
	          ' --features_save_to ' + image_features)
	print('Loading VGGfeats')
	# 把这个图片特征读入
	features_struct = scipy.io.loadmat(image_features)
	VGGfeatures = features_struct['feats']
	print("Loaded")
	# 然后，你开始问他问题
	question = unicode(raw_input("Ask a question: "))
	if question == "quit":
		flag = False
	timesteps = max_len
	X_q = get_questions_tensor_timeseries([question], nlp, timesteps)
	X_i = np.reshape(VGGfeatures, (1, 4096))
	# 构造成input形状
	X = [X_q, X_i]
	# 给出prediction
	y_predict = model.predict_classes(X, verbose=0)
	print(labelencoder.inverse_transform(y_predict))
