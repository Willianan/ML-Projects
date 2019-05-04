# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/26 14:00
@Project:Image_Style_Translation
@Filename:sample_pytorch.py
"""


import torch
from torch.autograd import Variable

N,D = 3,4
x = Variable(torch.randn(N,D),requires_grad= True)
y = Variable(torch.randn(N,D),requires_grad= True)
z = Variable(torch.randn(N,D),requires_grad= True)

a = x * y
b = a + z
c = torch.sum(b)
c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)