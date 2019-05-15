# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/5/13 10:08
@Project:CTR
@Filename:BaseLine.py
"""

import zipfile
import pandas as pd
import numpy as np

# load data
dfTrain = pd.read_csv('pre/train.csv')
dfTest = pd.read_csv('pre/test.csv')
dfAd = pd.read_csv('pre/ad.csv')

# process data
dfTrain = pd.merge(dfTrain,dfAd,on='creativeID')
dfTest = pd.merge(dfTest,dfAd,on='creativeID')
y_train = dfTrain["label"].values

# model building
key = "appID"
dfCvr = dfTrain.groupby(key).apply(lambda df:np.mean(df["label"])).reset_index()
dfCvr.columns = [key,'avg_csr']
dfTest = pd.merge(dfTest,dfCvr,how='left',on=key)
dfTest["avg_cvr"].fillna(np.mean(dfTrain["label"]),inplace=True)

# submission
df = pd.DataFrame({"instanceID":dfTest["instanceID"].values,"proba":proba_test})
df.sort_values("instanceID",inplace=True)
df.to_csv("submission.csv",index=False)
with zipfile.ZipFile("submission.zip","w") as fout:
	fout.write("submission.csv",compress_type=zipfile.ZIP_DEFLATED)


"""
AD + RL
"""
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# load data
dfTrain = pd.read_csv('pre/train.csv')
dfTest = pd.read_csv('pre/test.csv')
dfAd = pd.read_csv('pre/ad.csv')

# process data
dfTrain = pd.merge(dfTrain,dfAd,on='creativeID')
dfTest = pd.merge(dfTest,dfAd,on='creativeID')
y_train = dfTrain["label"].values

# feature engineering/encoding
enc = OneHotEncoder
feats = ["creativeID","adID","camgaignID","advertiserID","AppID","appPlatform"]
for i,feat in enumerate(feats):
	x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1,1))
	x_test = enc.fit_transform(dfTest[feat].values.reshape(-1,1))
	if i == 0:
		X_train,X_test = x_train,x_test
	else:
		X_train,X_test = sparse.hstack((X_train,x_train)),sparse.hstack((X_test,x_test))

# model training
lr = LogisticRegression()
lr.fit(X_train,y_train)
proba_test = lr.predict_proba(X_test)[:,1]

# submission
df = pd.DataFrame({"instanceID":dfTest["instanceID"].values,"proba":proba_test})
df.sort_values("instanceID",inplace=True)
df.to_csv("submission.csv",index=False)
with zipfile.ZipFile("submission.zip","w") as fout:
	fout.write("submission.csv",compress_type=zipfile.ZIP_DEFLATED)
