# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/5/13 11:07
@Project:CTR
@Filename:feature_engineering_and_machine_learning.py
"""

import pandas as pd
import numpy as np
import scipy as sp

# read file
def read_csv_file(f,logging=False):
	print("============================== reading data ==============================")
	data = pd.read_csv(f,'rb',encoding='utf-8')
	if logging:
		print(data.head(5))
		print(f,"  包含以下列......")
		print(data.columns.values)
		print(data.describe())
		print(data.info())
	return data

# 第一类编码
def categories_process_first_class(cate):
	cate = str(cate)
	if len(cate) == 1:
		if int(cate) == 0:
			return 0
	else:
		return int(cate[0])

# 第二类编码
def categories_process_second_class(cate):
	cate = str(cate)
	if len(cate) < 3:
		return 0
	else:
		return int(cate[1:])

# 年龄处理，切片
def age_process(age):
	age = int(age)
	if age == 0:
		return 0
	elif age < 15:
		return 1
	elif age < 25:
		return 2
	elif age < 40:
		return 3
	elif age < 60:
		return 4
	else:
		return 5

# 省份处理
def process_province(homeTown):
	homeTown = str(homeTown)
	province = int(homeTown[0:2])
	return province

# 城市处理
def process_city(homeTown):
	homeTown = str(homeTown)
	if len(homeTown) > 1:
		province = int(homeTown[2:])
	else:
		province = 0
	return province

# 时间
def get_time_day(t):
	t = str(t)
	t = int(t[0:2])
	return t
# 一天时间切成4段
def get_time_hour(t):
	t = str(t)
	t = int(t[2:4])
	if t < 6:
		return 0
	elif t < 12:
		return 1
	elif t < 18:
		return 2
	else:
		return 3

# 评估和计算logloss
def logloss(act,pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon,pred)
	pred = sp.minimum(1-epsilon,pred)
	ll = sum(act * sp.log(pred) + sp.subtract(1,act) * sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0 / len(act)
	return ll

# 特征工程和随机森林建模
from sklearn.preprocessing import Binarizer,MinMaxScaler

# 读取train_data和ad
# 特征工程
train_data = read_csv_file('./pre/train.csv',logging=True)
ad = read_csv_file('pre/ad.csv',logging=True)

# app
app_catrgories = read_csv_file('pre/app_categories.csv',logging=True)
app_catrgories["app_categories_first_class"] = app_catrgories['appCategory'].apply(categories_process_first_class)
app_catrgories["app_categories_second_class"] = app_catrgories['appCategory'].apply(categories_process_second_class)

# user
user = read_csv_file('pre/user.csv',logging=True)
user['age_process'] = user['age'].apply(age_process)
user["hometown_province"] = user['hometown'].apply(process_province)
user['hometown_city'] = user['hometown'].apply(process_city)
user['residence_province'] = user['residence'].apply(process_province)
user['residence_city'] = user['residence'].apply(process_city)

# 合并数据
# train_data
train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
train_data['clickTime_hour'] = train_data['clickTime'].apply(get_time_hour)
# test_data
test_data = read_csv_file('pre/test.csv',logging=True)
test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)
test_data['clickTime_hour'] = test_data['clickTime'].apply(get_time_hour)

train_user = pd.merge(train_data,user,on='userID')
train_user_ad = pd.merge(train_user,ad,on='creativeID')
train_user_ad_app = pd.merge(train_user_ad,app_catrgories,on='appID')

# 取出数据和label
# 特征部分
x_user_ad_app = train_user_ad_app.loc[:,['creativeID','userID','positionID','connectionType',
                                         'telecomsOperator','clickTime_day','clickTime_hour','age','gender','education',
                                         'marriageStatus','haveBaby','residence','age_process','hometown_province',
                                         'hometown_city','adID','camgaignID','advertiserID','appID','appPlatform',
                                         'app_categories_first_class','app_categories_second_class']]
x_user_ad_app = x_user_ad_app.values
x_user_ad_app = np.array(x_user_ad_app,dtype='int32')

# 标签部分
y_user_ad_app = train_user_ad_app.loc[:,['label']].values

# 随机森林建模&&特征重要度排序
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV

feat_labels = np.array(['creativeID','userID','positionID','connectionType',
                                         'telecomsOperator','clickTime_day','clickTime_hour','age','gender','education',
                                         'marriageStatus','haveBaby','residence','age_process','hometown_province',
                                         'hometown_city','adID','camgaignID','advertiserID','appID','appPlatform',
                                         'app_categories_first_class','app_categories_second_class'])
forest = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
forest.fit(x_user_ad_app,y_user_ad_app.reshape(y_user_ad_app.shape[0],))
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

import matplotlib.pyplot as plt

for f in range(x_user_ad_app.shape[1]):
	print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(x_user_ad_app.shape[1]),importances[indices],color='lightbule',align='center')
plt.xticks(range(x_user_ad_app.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1,x_user_ad_app.shape[1]])
plt.tight_layout()
plt.show()

# 随机森林调参
param_grid = {'n_estimators':[10,100,500,1000],
              'max_feature':[0.6,0.7,0.8,0.9]}

rf = RandomForestClassifier()
rfc = GridSearchCV(rf,param_grid=param_grid,scoring='neg_log_loss',cv=3,n_jobs=2)
rfc.fit(x_user_ad_app,y_user_ad_app.reshape(y_user_ad_app.shape[0],))
print(rfc.best_score_)
print(rfc.best_params_)

# Xgboost调参
import xgboost as xgb
import os
os.environ["OMP_NUM_THREADS"] = "8"         # 并行训练
rng = np.random.RandomState(4315)
import warnings
warnings.filterwarnings("ignore")

param_grid = {'max_depth':[3,4,5,7,9],
              'n_estimators':[10,50,100,400,800,1000,1200],
              'learning_rate':[0.1,0.2,0.3],
              'gamma':[0,0,2],
              'subsample':[0.8,1],
              'colsample_bylevel':[0.8,1]
              }

xgb_model = xgb.XGBClassifier()
rgs = GridSearchCV(xgb_model,param_grid,n_jobs=-1)
rgs.fit(X,y)
print(rgs.best_score_)
print(rgs.best_params_)

# 正负样本比

positive_num = train_user_ad_app[train_user_ad_app['label'] == 1].values.shape[0]
negative_num = train_user_ad_app[train_user_ad_app['label'] == 0].values.shape[0]
print(negative_num / float(positive_num))