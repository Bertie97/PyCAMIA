
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'batorch',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.55',
    contact = 'bertiezhou@163.com',
    keywords = ['torch', 'batch', 'batched data'],
    description = "'batorch' is an extension of package torch, for tensors with batch dimensions. ",
    requires = ['pycamia', 'torch', 'pynvml', 'matplotlib', 'psutil', 'numpy', 'pyoverload', 'sympy'],
    update = '2024-12-24 16:23:03'
).check()
__version__ = '1.0.55'

import torch
distributed = torch.distributed
autograd = torch.autograd
random = torch.random
optim = torch.optim
utils = torch.utils
linalg = torch.linalg
nn = torch.nn
__torchversion__ = torch.__version__

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    from .device import MPS
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    from .device import GPU, GPUs

from .device import free_memory_amount, all_memory_amount, AutoDevice
from .tensorsize import new_dim, exist_dim, del_dim, iter_dim, linalg_dim, Size, FakeSize, FuncDim, BatchDim, SequeDim, FeatDim, SpaceDim, SpecDim, AllDim, SpecialDimensions, MajorSpecialDimensions, empty_special_dims
from .tensor import get_cpu_memory_used, get_gpu_memory_used, collect_memory, turn_on_autodevice, turn_off_autodevice, set_device, get_device, to_device, default_device, auto_device, new_dim, exist_dim, del_dim, iter_dim, linalg_dim, inverse, inv, diag, diagflat, diagonal, trace, tr, add, sub, mul, div, pow, fmod, log, ln, log2, log10, exp, square, sqrt, abs, sign, sin, cos, tan, cot, sec, csc, asin, arcsin, acos, arccos, atan, arctan, matmul, mm, bmm, smm, floor_divide, true_divide, equal, addmm, addbmm, saddmm, addcmul, clamp, floor, ceil, round, any, all, unique, isin, isnan, isinf, isposinf, isneginf, isfinite, unsqueeze, squeeze, flatten, transpose, t, permute, standard_shape, duplicate, amplify, repeated, repeat, gather, flip, detach, quantile, val_range, sum, prod, mean, std, cumsum, cumprod, min, max, median, cummin, cummax, argmin, argmax, split, sample, pick, eig, matpow, matexp, matlog, rank, matnorm, det, matrix_power, matrix_exp, matrix_log, matrix_rank, matrix_norm, Size, FakeSize, FuncDim, BatchDim, SequeDim, FeatDim, SpaceDim, SpecDim, AllDim, SpecialDimensions, MajorSpecialDimensions, empty_special_dims, broadcast, remove_dim, add_dim, Tensor, expand, expand_as, expand_to, complex, tensor, as_tensor, to_bttensor, empty, full, ones, zeros, tensor_to, empty_like, full_like, ones_like, zeros_like, tensor_like, rand, randn, rand_like, randn_like, randperm, arange, where, reshape, cat, stack, meshgrid, eye, eye_like, batch_arange, feature_arange, channel_arange, sequence_arange, batch_tensor, feature_tensor, channel_tensor, sequence_tensor, time_tensor, series_tensor, randint, randint_like, dtype, device, bfloat16, bool, cdouble, cfloat, chalf, complex128, complex32, complex64, double, half, float, float16, float32, float64, int, int16, int32, int64, int8, qint32, qint8, quint2x4, quint4x2, quint8, long, short, uint8, manual_seed #*
# from . import nn
from .tensorfunc import MultiVariate, grad, batorch_wrapper, conv, crop_as, decimal, divide, dot, equals, gaussian_kernel, image_grid, imagegrid, mask_center, one_hot, pad, permute_space, approx_quantile, normalize01, input_shape, grad_image, Jacobian, up_scale, down_scale, upper_matrix, upper_triangular_matrix, lower_matrix, lower_triangular_matrix, skew_symmetric, cross_matrix, skew_symmetric_params, uncross_matrix, norm, norm2, Fnorm, Fnorm2, frobenius_norm, meannorm, meannorm2, mean_norm, mean_norm2, summary, display, byte_size, get_memory_used, num_model_params #*
from .optimizer import CSGD, CADAM, Optimization, exp_lr, train, test #*
_user_replace = globals()
# from .tensor import __all__ # do not expand
# force_recover = ['tensor']
# for obj in __all__:
#     if obj not in globals() or obj in force_recover:
#         exec(f"from .tensor import {obj}")
# from . import nn
# globals().update(_user_replace)

save = torch.save
load = torch.load
no_grad = torch.no_grad
enable_grad = torch.enable_grad

import math
e = to_device(tensor(math.e))
nan = to_device(tensor(math.nan))
inf = to_device(tensor(math.inf))
pi = to_device(tensor(math.pi))
eps = to_device(tensor(1e-5))
one = to_device(tensor(1.))
zero = to_device(tensor(0.))

# from .torchext import __all__ as __torchall__
# for obj in __torchall__:
#     exec(f"from .torchext import {obj} as tmpo")
#     setattr(torch, obj, tmpo)
