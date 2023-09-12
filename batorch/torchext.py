
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "Extended torch functions for `batorch` package, all computed under original 'torch' core.",
    requires = "torch"
)

__all__ = """
    T
    pad
    crop_as
    image_grid
    grad_image
    Jacobian
    decimal
    divide
    equals
    gaussian_kernel
    one_hot
    movedim movedim_
    skew_symmetric
    cross_matrix
    uncross_matrix

    norm norm2
    Fnorm Fnorm2
    frobenius_norm
    meannorm meannorm2
    mean_norm mean_norm2
    meanFnorm meanFnorm2
    mean_Fnorm mean_Fnorm2

    eye
    diag
    trace
    normalize
    squeeze squeeze_
    unsqueeze unsqueeze_
    multiply multiple
    amplify ample
    repeat repeated
""".split()
    # matpow
    # matprod
    # down_scale
    # up_scale
    # permute_space

import math

with __info__:
    import torch
    from pyoverload import *
    from pycamia import restore_type_wrapper, avouch, alias, Version
    from pycamia import prod, to_tuple
    
def torch_tensor(x):
    if not isinstance(x, torch.Tensor): return torch.tensor(x)
    elif type(x) != torch.Tensor: return x.as_subclass(torch.Tensor)
    return x

@alias("meannorm2", "mean_norm2", root = False, mean = True)
@alias("meannorm", "mean_norm", mean = True)
@alias("meanFnorm2", "mean_Fnorm2", root = False, mean = True, dim = None)
@alias("meanFnorm", "mean_Fnorm", mean = True, dim = None)
@alias("norm2", root = False)
@alias("Fnorm2", root = False, dim = None)
@alias("Fnorm", "frobenius_norm", dim = None)
def norm(tensor, p = 2, dim = 1, keepdim = False, dtype=None, root = True, mean = False):
    """dim=None: all dimensions (except 0), this is also triggered by Fnorm."""
    tensor = torch_tensor(tensor)
    if hasattr(dim, 'len'): dim = list(dim)
    if isinstance(p, str):
        if not hasattr(dim, 'len') or len(dim) <= 1: dim = None
        if p == 'fro': p = 2; root = True; mean = False
    if dim is None: dim = list(range(1, tensor.ndim))
    if not isinstance(dim, list): dim = [dim]
    dim.sort()
    if isinstance(p, str):
        avouch(p == 'nuc' and len(dim) == 2)
        if not root: p = 2
        else:
            res = torch.norm(tensor, p, dim, keepdim=keepdim, dtype=dtype)
            if mean: res /= prod(dim) ** (1 / 2)
            if dtype: res.type(dtype)
            return res
    tensor = tensor.abs()
    if p == torch.inf:
        tensor_p = tensor
        op = lambda x, d: torch.max(x, d, keepdim=keepdim).values
    elif p == - torch.inf:
        tensor_p = tensor
        op = lambda x, d: torch.min(x, d, keepdim=keepdim).values
    else: 
        tensor_p = tensor ** p
        op = lambda x,d : torch.sum(x, d, keepdim=keepdim)
    n_data = 1
    for d in dim[::-1]:
        tensor_p = op(tensor_p, d)
        n_data *= tensor.size(d)
    if mean:
        if p in (float('inf'), float('-inf')): tensor_p /= n_data ** (1 / p)
        else: tensor_p /= n_data
    if root and p not in (float('inf'), float('-inf')): tensor_p = tensor_p ** (1 / p)
    if dtype: tensor_p.type(dtype)
    return tensor_p

def image_grid(*shape):
    """return standard grid coordinates of shape (n_dim, n@1, ..., n@n_dim) for input shape (n@1, ..., n@n_dim)."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
    if len(shape) == 1 and hasattr(shape[0], 'space'): shape = shape[0].space
    if len(shape) == 1 and hasattr(shape[0], 'shape'): shape = shape[0].shape
    kwargs = {'indexing': 'ij'} if torch.__version__ >= Version('1.10') else {}
    if any(x < 0 for x in shape): raise TypeError(f"Negative size in shape {shape} for image grid.")
    return torch.stack(torch.meshgrid(*[torch.arange(x) for x in shape], **kwargs), 0)

@overload
@restore_type_wrapper("roi")
def crop_as(x: array, y: (tuple, list), center: (tuple, null)=None, n_keepdim: int=0, fill: scalar=0):
    x = torch_tensor(x)
    size_x = x.shape
    size_y = (-1,) * n_keepdim + tuple(y)
    avouch(len(size_y) == len(size_x), "Mismatch dimensions: target size in 'crop_as', please use -1 if the dimension doesn't need to be cropped. ")
    size_y = tuple(a if b == -1 else b for a, b in zip(size_x, size_y))

    if center is None: center = (-1,) * len(size_x)
    else: center = (-1,) * n_keepdim + tuple(center)
    center = tuple(-1 if y < 0 else x for x, y in zip(center, size_y))
    avouch(len(center) == len(size_x), "Mismatch dimensions: center in 'crop_as', please use -1 if the dimension that is centered or doesn't need cropping. ")
    center = tuple(a / 2 if b == -1 else b for a, b in zip(size_x, center))

    z = fill * torch.ones(*size_y).type(x.type())
    def intersect(u, v):
        return max(u[0], v[0]), min(u[1], v[1])
    z_box = [intersect((0, ly), (- round(float(m - float(ly) / 2)), - round(float(m - float(ly) / 2)) + lx)) for m, lx, ly in zip(center, size_x, size_y)]
    x_box = [intersect((0, lx), (+ round(float(m - float(ly) / 2)), + round(float(m - float(ly) / 2)) + ly)) for m, lx, ly in zip(center, size_x, size_y)]
    # if the two boxes are seperated
    if any([r[0] >= r[1] for r in z_box]) or any([r[0] >= r[1] for r in x_box]): z.roi = None; return z
    region_z = tuple(slice(u, v) for u, v in z_box)
    region_x = tuple(slice(u, v) for u, v in x_box)
    z[region_z] = x[region_x]
    z.roi = region_x
    return z

@overload
def crop_as(x: array, y: array, center: tuple, fill: scalar=0):
    return crop_as(x, y.shape, center, fill)

@overload
def crop_as(x: array, y: union(tuple, list, array), fill: scalar=0):
    center = tuple(m/2 for m in x.shape)
    return crop_as(x, y, center, fill)

@overload
def crop_as(x: array, *y: int):
    center = tuple(m/2 for m in x.shape)
    return crop_as(x, y, center)

@restore_type_wrapper
def Jacobian(X, Y, dt=1, pad=False):
    """torch version: 
        The Jacobian matrix; Note that it is a transpose of grad_image if X is standard grid as it follows the orientation of Jacobian.
        X: (n_batch, n_dim, n@1, ..., n@n_dim)
        Y: (n_batch, n_feature, n@1, ..., n@n_dim)
        Jacobian(output, pad = True): (n_batch, n_feature, n_dim, n@1, ..., n@n_dim)
        Jacobian(output, pad = False): (n_batch, n_feature, n_dim, n@1 - dt, ..., n@n_dim - dt)
    """
    X, Y = torch_tensor(X), torch_tensor(Y)
    dX = grad_image(X, dx=dt, pad=pad) # (n_batch, n_dim, n_dim, n@1, ..., n@n_dim)
    dY = grad_image(Y, dx=dt, pad=pad) # (n_batch, n_dim, n_feature, n@1, ..., n@n_dim)
    # dyk/dxi = sum_j [(dyk/dtj) / (dxi/dtj)]: Jac[b, j, k, i, ...] = dY[b, j, k, *, ...] - dX[b, j, *, i, ...]
    return divide(dY.unsqueeze(3), dX.unsqueeze(2), 0.0).sum(1)

@restore_type_wrapper
def grad_image(array, dx=1, pad=False):
    '''torch version: 
        Gradient image / tensor of array (dx is the displacement for finite difference). 
        If pad is not True, the image will trim. The sizes should be: 
        array: (n_batch, n_feature, n@1, ..., n@n_dim)
        output (pad = True): (n_batch, n_dim, n_feature, n@1, ..., n@n_dim)
        output (pad = False): (n_batch, n_dim, n_feature, n@1 - dx, ..., n@n_dim - dx)
    '''
    array = torch_tensor(array)
    avouch(array.ndim >= 3, "torch 'grad_image' accept input of size (n_batch, n_feature, n@1, ..., n@n_dim), which is at least 3D. ")
    output = []
    n_batch, n_feature, *size = array.shape
    n_dim = len(size)
    *size_levels, n_data = torch.tensor([1] + size[::-1]).cumprod(0)
    trim_size = tuple(x - dx for x in size)
    n_data_trim = prod(trim_size)
    data_index = (image_grid(*trim_size) * torch.tensor(size_levels[::-1]).view((n_dim,) + (1,) * n_dim)).sum(0).flatten()\
        .unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n_batch, n_dim, n_feature, n_data_trim)
    data_index = data_index + dx * torch.tensor(size_levels[::-1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    array_plus = array.unsqueeze(1).flatten(3).expand(n_batch, n_dim, n_feature, n_data).gather(3, data_index.to(array.device))
    grad = (array_plus - array.unsqueeze(1)[(slice(None),) * 3 + tuple(slice(x) for x in trim_size)].flatten(3).expand(n_batch, n_dim, n_feature, n_data_trim)) / dx
    grad = grad.view(n_batch, n_dim, n_feature, *trim_size)
    if pad: return crop_as(grad, size, n_keepdim=3)
    return grad

@restore_type_wrapper
def decimal(array):
    array = torch_tensor(array)
    return array - torch.floor(array)

@restore_type_wrapper
def divide(a, b, limit=1, tol=1e-6):
    a, b = torch_tensor(a), torch_tensor(b)
    avouch(a.ndim == b.ndim, f"torch 'divide' needs inputs of the same dimension. Not {a.shape} and {b.shape}")
    shape = tuple(max(x, y) for x, y in zip(a.shape, b.shape))
    return torch.where(b.abs() < tol, torch.where(a.abs() < tol, torch.zeros(shape), limit * torch.ones(shape)), a / torch.where(b.abs() < tol, tol * torch.ones(shape), b))

@restore_type_wrapper
def equals(a, b):
    a, b = torch_tensor(a), torch_tensor(b)
    return ((torch.abs(a - b) / (a.abs() + b.abs() + 1e-4)) < 1e-4) | (a.abs() + b.abs() < 1e-4)

def gaussian_kernel(n_dims = 2, kernel_size = 3, sigma = 0, normalize = True):
    radius = (kernel_size - 1) / 2
    if sigma == 0: sigma = radius * 0.6
    grid = image_grid(*(kernel_size,) * n_dims).float()
    kernel = torch.exp(- ((grid - radius) ** 2).sum(0) / (2 * sigma ** 2))
    return (kernel / kernel.sum()) if normalize else kernel

@restore_type_wrapper
def dot(a, b):
    """a, b: (n_batch, n_channel, n@1, ..., n@n_dim)"""
    a, b = torch_tensor(a), torch_tensor(b)
    avouch(a.shape == b.shape, "torch 'dot' product recieve two tensor of a same shape. Use 'X.expand_to' to adjust if necessary. ")
    return (a * b).sum(1)

def one_hot(k, n):
    """
    Create an one-hot vector as a zero `torch.Tensor` with length `n` and the `k`-th element 1. 
    e.g. k == -1 gives tensor([0, 0, ..., 0, 1])
    """
    l = [0] * n; l[k] = 1
    return torch.tensor(l)

@restore_type_wrapper
def pad(x, p = 1, n_keepdim = 0, fill = 0):
    x = torch_tensor(x)
    p = to_tuple(p)
    if len(p) == 1: p *= x.ndim - n_keepdim
    if len(p) < x.ndim: p = (0,) * n_keepdim + p
    return crop_as(x, tuple(s + 2 * q for s, q in zip(x.shape, p)), fill = fill)

@restore_type_wrapper
def movedim(x, p, q):
    x = torch_tensor(x)
    if q > p: q += 1
    else: p += 1
    return x.unsqueeze(q).transpose(p, q).squeeze(p)

@restore_type_wrapper
def movedim_(x, p, q):
    x = torch_tensor(x)
    if q > p: q += 1
    else: p += 1
    x.unsqueeze_(q).transpose_(p, q).squeeze_(p)

@alias("skew_symmetric")
def cross_matrix(axis):
    '''
    axis: (n_batch, n_dim * (n_dim - 1) / 2)
    output: (n_batch, n_dim, n_dim)
    '''
    axis = torch_tensor(axis)
    n_batch = axis.size(0)
    n_dim = int(math.sqrt(2 * axis.size(1))) + 1
    output = torch.zeros(n_batch, n_dim, n_dim)
    if n_dim == 2:
        output[:, 0, 1] = axis[:, 0]
        output[:, 1, 0] = - axis[:, 0]
    elif n_dim == 3:
        output[:, 1, 2] = axis[:, 0]
        output[:, 2, 0] = axis[:, 1]
        output[:, 0, 1] = axis[:, 2]
        output[:, 2, 1] = - axis[:, 0]
        output[:, 0, 2] = - axis[:, 1]
        output[:, 1, 0] = - axis[:, 2]
    return output
    
def uncross_matrix(cross_matrix):
    '''
    cross_matrix: (n_batch, n_dim, n_dim)
    output: (n_batch, n_dim * (n_dim - 1) / 2)
    '''
    cross_matrix = torch_tensor(cross_matrix)
    n_batch = cross_matrix.size(0)
    n_dim = cross_matrix.size(1)
    axis = torch.zeros(n_batch, n_dim * (n_dim - 1) // 2)
    if n_dim == 2:
        axis[:, 0] = cross_matrix[:, 0, 1]
    elif n_dim == 3:
        axis[:, 0] = cross_matrix[:, 1, 2]
        axis[:, 1] = cross_matrix[:, 2, 0]
        axis[:, 2] = cross_matrix[:, 0, 1]
    return axis

def eye(*size: tuple, **kwargs):
    if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
    if len(size) == 1 and hasattr(size[0], 'shape'): size = size[0].shape
    if len(size) < 1: raise TypeError("Empty size not valid for 'eye'. ")
    if len(size) == 1: size = size * 2
    if len(size) == 2 and size[0] != size[1]: size += (size[1],)
    elif len(size) == 2: size = (1,) + size
    if len(size) > 3: raise TypeError("No more than 2-D is allowed for 'eye'. ")
    n = min(size[1], size[2])
    out = torch.zeros(size, **kwargs)
    out[:, torch.arange(n), torch.arange(n)] = 1
    return out

@restore_type_wrapper
def diag(x):
    x = torch_tensor(x)
    if x.ndim == 1: return torch.diag(x)
    n = min(x.size(-1), x.size(-2))
    return x[..., torch.arange(n), torch.arange(n)]

@restore_type_wrapper
def trace(x):
    x = torch_tensor(x)
    return diag(x).sum(-1)

@restore_type_wrapper
def T(x):
    x = torch_tensor(x)
    return x.transpose(-1, -2)

@restore_type_wrapper
def normalize(x):
    x = torch_tensor(x)
    M, m = x.max(), x.min()
    if M == m: return x
    return (x - m) / (M - m)

@restore_type_wrapper
def squeeze(x, index=None):
    if index is None: index = [i for i in x.shape if i==1]
    if isinstance(index, int): index = [index]
    x = torch_tensor(x)
    for i in index: x = x.squeeze(i)
    return x

@restore_type_wrapper
def squeeze_(x, index=None):
    if index is None: index = [i for i in x.shape if i==1]
    if isinstance(index, int): index = [index]
    x = torch_tensor(x)
    for i in index: x.squeeze_(i)
    return x

@restore_type_wrapper
def unsqueeze(x, index=None):
    if index is None: index = [0]
    if isinstance(index, int): index = [index]
    x = torch_tensor(x)
    for i in index: x = x.unsqueeze(i)
    return x

@restore_type_wrapper
def unsqueeze_(x, index=None):
    if index is None: index = [0]
    if isinstance(index, int): index = [index]
    x = torch_tensor(x)
    for i in index: x.unsqueeze_(i)
    return x

@alias("multiple")
@restore_type_wrapper
def multiply(x, size, dim=0):
    x = torch_tensor(x)
    return repeat(x.unsqueeze(dim), size, dim)

@alias("ample")
@restore_type_wrapper
def amplify(x, size, dim=0):
    x = torch_tensor(x)
    return multiply(x, size, dim+1).flatten(dim, dim+1)

@alias("repeated")
@restore_type_wrapper
def repeat(x, size, dim=0):
    x = torch_tensor(x)
    n_repeat = [1] * x.ndim
    n_repeat[dim] = size
    return x.repeat(n_repeat)
