
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'batorch',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.52',
    contact = 'bertiezhou@163.com',
    keywords = ['torch', 'batch', 'batched data'],
    description = "'batorch' is an extension of package torch, for tensors with batch dimensions. ",
    requires = ['pycamia', 'torch', 'pynvml'],
    update = '2023-07-10 15:53:47'
).check()
__version__ = '1.0.52'

import torch
distributed = torch.distributed
autograd = torch.autograd
random = torch.random
optim = torch.optim
utils = torch.utils
linalg = torch.linalg
__torchversion__ = torch.__version__

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    from .device import MPS
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    from .device import GPU, GPUs

from .device import free_memory_amount, all_memory_amount, AutoDevice
from .tensorfunc import conv, crop_as, decimal, divide, dot, down_scale, equals, gaussian_kernel, image_grid, matpow, matprod, one_hot, pad, permute_space, up_scale, norm, norm2, Fnorm, Fnorm2, frobenius_norm, meannorm, meannorm2, mean_norm, mean_norm2, Jacobian, grad_image, skew_symmetric, cross_matrix, uncross_matrix, summary, display #*
from .optimizer import CSGD, CADAM, Optimization, train, test #*
_user_replace = globals()
from .tensor import __all__ # do not expand
force_recover = ['tensor']
for obj in __all__:
    if obj not in globals() or obj in force_recover:
        exec(f"from .tensor import {obj}")
from . import nn
globals().update(_user_replace)

import math
e = tensor(math.e)
nan = tensor(math.nan)
inf = tensor(math.inf)
pi = tensor(math.pi)

from .torchext import __all__ as __torchall__
for obj in __torchall__:
    exec(f"from .torchext import {obj} as tmpo")
    setattr(torch, obj, tmpo)
