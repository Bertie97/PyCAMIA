#!/usr/bin/env python3
import batorch as bt
print(bt.arange([5], 9, 2))
exit()
from micomputing.trans import Transformation

print(Transformation())
exit()

from pycamia import columns, factorial
import torch
import numpy as np
import batorch as bt

A = bt.rand({5}, 20, 30, 40, [3])
from micomputing import plt
plt.gridshow(A)
plt.show()

A = bt.rand(30, 30, [300, 300]).requires_grad_()
A.sum().backward()
smy = bt.display(A, show_thumb=True)
print(smy.Requires_Gradient)
smy.show()

A = bt.zeros({5}, [4, 2], 5, 6)
A += bt.randint_like[0, 2](A)
A = A / bt.randint_like[0, 20](A)
print(bt.summary(A))
exit()

X = bt.image_grid(100, 100).float()
Y = X + 1e-2*bt.randn_like(X)
print(bt.Jacobian(X, Y).movedim([], -1)[0])

A = bt.rand(4, 4)
print(columns(bt.up_scale(A, 2), A.amplify(2, 0).amplify(2, 1)))
print(columns(bt.down_scale(A, 2), A[::2, ::2]))

A = bt.arange(3, sz_feature_dim=1).duplicate({3}).long()
print(bt.uncross_matrix(bt.cross_matrix(A)))

print(bt.norm(bt.ones([3], 10, 10)))
print(bt.crop_as(np.zeros((100, 100)), (30, 30), (50, 50)))
print(bt.conv(bt.zeros([3], 100, 100), bt.zeros([3, 3], 3, 3)))

A = bt.rand({3}, [3, 3], 2) / 2
print(A.rank())
print(columns(A.movedim([], -1), bt.matpow(A, 1/3).movedim([], -1) @ bt.matpow(A, 1/3).movedim([], -1) @ bt.matpow(A, 1/3).movedim([], -1), line_width=140))
print(columns(A.movedim([], -1), bt.matlog(bt.matexp(A)).movedim([], -1).round(decimals=2), line_width=140))

A = bt.rand({3}, [3,4],5).requires_grad_(True)
print(A.split(dim=1))
print(bt.get_cpu_memory_used())
#print(A.min(1, 2).indices)
#exit()
#help(torch.full)
exit()
"""
I1=(4, 5, 6, 1)
I2=(5,6)
out[ijkl]=I1[I2[jk.],j,k,l]
"""
print(torch.stack((A.min(1).values.min(0).indices, A.min(1).indices.gather(0, A.min(1).values.min(0).indices.unsqueeze(0)).squeeze()), -1))
print(A[..., 0, 0])
#print(*(f for f in dir(torch.Tensor) if f.endswith('_')))
#help(torch.Tensor.append)
#exit()
#help(bt.Tensor.repeat)
#print(bt.rand([2,3,4], 5).range([]))

#print(bt.rand({3}, [4, 6], 4, 5, 6).sum([], keepdim=True))
#print(bt.rand([4, 5], 5).mergedims([], target={0}))

#f = lambda a, b: (lambda u, v: (u, u, v))(*a^b)
#print(f(bt.Size(3, 4), bt.Size({1}, 3, 1)))
#exit()
#print(bt.inv)
"""
(|1.22e+1||1.22e+1||1.22e+1||1.22e+1|
 |1.22e+1||1.22e+1||1.22e+1||1.22e+1|
 |1.22e+1||1.22e+1||1.22e+1||1.22e+1|)
[[(1.22e+1 1.22e+1 1.22e+1   ... "((1.22e+1 1.22e+1 1.22e+1 
   1.22e+1 1.22e+1 1.22e+1   ...    1.22e+1 1.22e+1 1.22e+1 
   1.22e+1 1.22e+1 1.22e+1   ...    1.22e+1 1.22e+1 1.22e+1)
   1.22e+1 1.22e+1 1.22e+1   ...   (1.22e+1 1.22e+1 1.22e+1 
   1.22e+1 1.22e+1 1.22e+1   ...    1.22e+1 1.22e+1 1.22e+1 
   1.22e+1 1.22e+1 1.22e+1), ...    1.22e+1 1.22e+1 1.22e+1))"],
  ...
 [|1.22e+1 1.22e+1 1.22e+1|  ... |1.22e+1 1.22e+1 1.22e+1|]
 [|1.22e+1 1.22e+1 1.22e+1|  ... |1.22e+1 1.22e+1 1.22e+1|]
 [|1.22e+1 1.22e+1 1.22e+1|, ... |1.22e+1 1.22e+1 1.22e+1|],]
"""
exit()
import torch
print(torch.zeros(3, 3, 5, 6).unsqueeze(4).shape)
print(bt.new_dim(bt.zeros({3}, [3], 5, 6), -1))
print(bt.tensor(3.4))
a = torch.tensor([10, 0.01])
print(torch.log2(torch.abs(a)).max() - torch.log2(torch.abs(a)).min())

import math
a = 12.23233242342
l = int(math.log(a, 10))
x = a / (10 ** l)
l = f'{l:2d}'.replace(' ', '+') if l >= 0 else f'{l:1d}'
print(f"{x:.2f}e{l}")
#print(bt.Tensor(2, 4, 5, 7)[:, [1,2,3], ..., bt.Tensor({4}, [1]).long()].shape)
#print(bt.torch.zeros(2, 4, 5, 7)[:, [1,2,3], ..., bt.torch.zeros(4, 1).long()].shape)
#print(bt.ones_like(bt.zeros(3,4,5)))
#print(bt.exist_dim(bt.Tensor({1}, [3], 4), ...))
#print(bt.Tensor(100, 100)[0][0])
#s.n_feature_dim = 1
#print(s.with_shape(bt.Size({}, [], [])).n_feature)#.special_from(5, [3], 6).shape)
#print(bt.new_dim(bt.Size(6, [5, 7]), {}))
