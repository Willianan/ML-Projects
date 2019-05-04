# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/26 14:22
@Project:Image_Style_Translation
@Filename:two_layer_DNN_Autograd.py
"""


import torch
from torch.autograd import Variable

'''
A PyTorch Variable is a node in a computational graph
x.data is a Tensor
x.grad is a Variable of gradients (same shape as x.data)
x.grad.data is a Tensor of gradients
'''
N,D_in,H,D_out = 64,1000,100,10

# we will not want gradients with respect to data
x = Variable(torch.randn(N,D_in),requires_grad=False)
y = Variable(torch.randn(N,D_out),requires_grad=False)
# do want gradients with respect to weights
w1 = Variable(torch.randn(D_in,H),requires_grad=True)
w2 = Variable(torch.randn(H,D_out),requires_grad=True)

learning_rate = 1e-6

for t in range(500):
	# Forward pass looks exactly the same as the Tensor version,but everything is a variable now
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	loss = (y_pred - y).pow(2).sum()
	# Compute gradient of loss with respect to w1 and w2 (zero out grads first)
	if w1.grad: w1.grad.data.zero_()
	if w2.grad: w2.grad.data.zero_()
	loss.backward()
	# Make gradient step on weights
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data

# Define your own autograd functions by writing forward and backward for Tensors
# (similar to modular layers in A2)
class Relu(torch.autograd.Function):
	def forward(self,x):
		self.save_for_backward(x)
		return x.clamp(min=0)
	def backward(self,grad_y):
		x, = self.saved_tensors
		grad_input = grad_y.clone()
		grad_input[x < 0] = 0
		return grad_input