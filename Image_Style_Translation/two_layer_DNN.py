# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/26 14:10
@Project:Image_Style_Translation
@Filename:two_layer_DNN.py
"""


import torch

dtype = torch.FloatTensor

# create random tensors for data and weights
N,D_in,H,D_out = 64,1000,100,10
x = torch.randn(N,D_in).type(dtype)
y = torch.randn(N,D_out).type(dtype)
w1 =  torch.randn(D_in,H).type(dtype)
w2 = torch.randn(H,D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
	# Forward pass:compute predictions and loss
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)
	loss = (y_pred - y).pow(2).sum()

	#Backward pass:manually compute gradients
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)

	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2