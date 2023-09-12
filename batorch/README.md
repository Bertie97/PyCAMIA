# batorch

## Introduction

[batorch](https://github.com/Bertie97/pyctlib/tree/main/batorch) is a package affiliated to project [`PyCAMIA`](https://github.com/Bertie97/pycamia). We encapsulated a new type on top of `torch` tensers, which we call it `batorch.Tensor`. It has the same function as `torch.Tensor`, but it can automatically select the device it was on and provide batch or channel dimensions. Also, we try to provide more useful module for torch users to make deep learning to be implemented more easily. It relies `python v3.6+` with `torch v 1.7.0+`. ***Note that `torch v1.7.0` was released in 2020,*** *and it is necessary for this package as the inheritance behavior for this version is different from previous versions.* Most original `torch` functions should be able to be applied for `batorch` tensors. 

> Special features for `batorch` are still under development. If unknown errors pop our, please use traditional `torch` code to bypass it and meanwhile it would be very kind of you to let us know if anything is needed: please contact us by [e-mail](https://github.com/Bertie97/pycamia#Contributors). 

```python
>>> import batorch as bt
>>> import batorch.nn as nn
>>> bt.turn_off_autodevice()
>>> bt.manual_seed(0)
<torch._C.Generator object at 0x1071b6730>
>>> t = bt.randn([3000], 400, requires_grad=True)
>>> LP = nn.Linear(400, 400)
>>> a = LP(t)
>>> a.sum().sum().backward()
>>> print(t.grad)
Tensor([[-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        ...,
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702]], shape=batorch.Size([3000], 400))
```

`batorch` has all of following appealing features:

1. **Auto assign** the tensors to available `GPU` **device** by default. 
2. Use `[nbatch]` or `{nchannel}` to specify **the batch and channel dimensions**. i.e. `tp.rand([4], {2}, 20, 30)` returns a 2-channel feature tensor of $20\times30$ matrices with batch size 4. One may also use `tensor.batch_dimension` to access (or assign) batch dimension, channel dimension can be operated likewise. If you find it hard to remember the symbol, just remember brackets enclose paralleled items in matrices hence it represents the batch dimension for paralleled calculation; braces enclose equation systems which are highly related hence it represents the channel (or feature) dimension. 
3. Batch and channel dimension can help **auto matching the sizes** of two tensors in operations. For example, tensors of sizes `(3, [2], 4)` and `(3, 4)` can be automatically added together with axis of size 3 and 4 matched together. Some methods will also use this information. Sampling, for example, will take the batch dimension as priority.
4. The tensor object is **compatible with original `torch` functions**. 

## Installation

This package can be installed by `pip install batorch` or moving the source code to the directory of python libraries (the source code can be downloaded on [github](https://github.com/Bertie97/pycamia) or [PyPI](https://pypi.org/project/batorch/)). 

```shell
pip install batorch
```

## Usages

Not available yet, one may check the codes for usages.

## Acknowledgment

@ Yuncheng Zhou: Developer
@ Yiteng Zhang: Important functions extraction