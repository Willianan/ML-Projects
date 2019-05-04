# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/26 14:50
@Project:Image_Style_Translation
@Filename:nn.py
"""


import torch
from torch.autograd import Variable

N,D_in,H,D_out = 64,1000,100,10
x = Variable(torch.randn(N,D_in))
y = Variable(torch.randn(N,D_out),requires_grad=False)
# Define our model as a sequence of layers
model = torch.nn.Sequential(torch.nn.Linear(D_in,H),torch.nn.ReLU(),
                            torch.nn.Linear(H,D_out))
# nn also defines common loss functions
loss_fn = torch.nn.MSELoss(size_average=False)


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
	y_pred = model(x)
	loss = loss_fn(y_pred,y)

	model.zero_grad()
	loss.backward()
	# update all parameters after computing gradients
	optimizer.step()

	for param in model.parameters():
		param.data -= learning_rate * param.grad.data


# define model as single Module
class TwoLayerNet(torch.nn.Module):
	def __init__(self,D_in,H,D_out):
		super(TwoLayerNet,self).__init__()
		self.linear1 = torch.nn.Linear(D_in,H)
		self.linear2 = torch.nn.Linear(H,D_out)
	def forward(self,x):
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred
N,D_in,H,D_out = 64,1000,100,10

x = Variable(torch.randn(N,D_in))
y = Variable(torch.randn(N,D_out),requires_grad=False)

model = TwoLayerNet(D_in,H,D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)
for t in range(500):
	y_pred = model(x)
	loss = criterion(y_pred,y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()