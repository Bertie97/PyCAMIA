#!/usr/bin/env python3

import batorch as bt
from pyoverload import dtype

print(bt.zeros({4}, [3], 100, 123).numel(0, 1))
print(bt.zeros({4}, [3], 100).merge_dims({}, -1))
from batorch.tensor import matmul_shape
print(matmul_shape(bt.Size({2}, 128, 128, 65, [1, 2]), bt.Size({2}, 128, 128, 65, [2])))
print(bt.stack(bt.zeros({4}, 3, 4), bt.ones({4}, 3, 3), crop=True))
cov_mat = bt.rand({4}, [3, 3], 4, 5)
ratio = cov_mat.det() / cov_mat.diag().prod([])
print(ratio)
a = bt.ones({4}, [3], 4, 5, 6)
b = 2 * bt.ones({4}, [3, 3], 4, 5, 6)
c = bt.ones({4}, [3], 4, 5, 6)
print(a.T)

print(bt.norm(bt.ones({4}, [3, 3], 4, 5, 6)))
print(bt.pow(2*bt.ones({3}, 2), 2))

params = bt.zeros({3}, [3, 4]) @ bt.zeros({3}, [3, 4, 5])
print(params.sum([1]))

X = bt.rand({3}, [4], 4, 5, 6)
print(X.func_size)
_range = bt.quantile(X.flatten(...).float(), bt.tensor([0.025, 0.975]), -1).movedim(0, -1)
print(_range)

#help(bt.Tensor.unsqueeze)
#print(bt.cat(bt.zeros(4), bt.ones(1)).unsqueeze({}, 1))
#print(bt.eye({2}, [2], [3, 3]))
#print(bt.zeros([3]).prod([]))
#print(bt.zeros([3]).sz_feature_dim_(-1).duplicate(3, {}).duplicate(5, -1))
#print(bt.zeros(1).dtype.is_floating_point)