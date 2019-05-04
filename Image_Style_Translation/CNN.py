# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/26 15:28
@Project:Image_Style_Translation
@Filename:CNN.py
"""

# simple CNN

import torch

class Net(torch.nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
		self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = torch.nn.Dropout2d()
		self.fc1 = torch.nn.Linear(320,50)
		self.fc2 = torch.nn.Linear(50,10)

	def forward(self,x):
		x = F.relu(F.max_pool2d(self.conv1(x),2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		x = x.view(-1,320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x,training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x)