
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch. This is the template to file tensor.py, please modify in this file instead of the auto-generated tensor.py",
    requires = "torch"
)

__all__ = """
    get_cpu_memory_used           get_gpu_memory_used           collect_memory
    turn_on_autodevice            turn_off_autodevice
    set_device     get_device     to_device      default_device auto_device
    new_dim        exist_dim      del_dim        iter_dim       linalg_dim
    inverse inv    diag           diagflat       diagonal       trace tr
    add            sub            mul            div            pow
    fmod           log            ln             log2           log10
    exp            sqrt           abs            sign
    sin            cos            tan            cot            sec            csc
    asin           arcsin         acos           arccos         atan           arctan
    matmul         mm             bmm            smm
    floor_divide   true_divide    equal
    addmm          addbmm         saddmm         addcmul
    clamp          floor          ceil           round
    any            all            unique         isin
    isnan          isinf          isposinf       isneginf       isfinite
    unsqueeze      squeeze
    flatten        transpose      t              permute        standard_shape
    duplicate      amplify        repeated       repeat
    gather         flip           detach
    quantile       val_range
    sum            prod           mean           std            norm
    cumsum         cumprod        min            max            median
    cummin         cummax         argmin         argmax
    split          sample         pick
    eig            matpow         matexp         matlog         rank           matnorm
    det            matrix_power   matrix_exp     matrix_log     matrix_rank    matrix_norm

    FuncDim        BatchDim       SequeDim       FeatDim        SpaceDim
    Size           FakeSize
    broadcast      remove_dim     add_dim

    Tensor
    expand         expand_as      expand_to
    complex        tensor         as_tensor      to_bttensor
    empty          full           ones           zeros          tensor_to
    empty_like     full_like      ones_like      zeros_like     tensor_like
    rand           randn          rand_like      randn_like     randperm
    arange         where          reshape
    cat            stack          meshgrid
    eye            eye_like
    batch_arange   feature_arange channel_arange sequence_arange
    batch_tensor   feature_tensor channel_tensor sequence_tensor
    time_tensor    series_tensor
    randint        randint_like
    
    dtype          device
    bfloat16       bool
    cdouble        cfloat         chalf          
    complex128     complex32      complex64
    double         half
    float          float16        float32        float64
    int            int16          int32          int64          int8
    qint32         qint8          quint2x4       quint4x2       quint8
    long           short          uint8
    manual_seed
""".split()

import builtins, re, sys, math
from abc import ABCMeta
from collections import defaultdict
from functools import wraps
from typing import Generator
from .device import GB, AutoDevice, SleepingDevice
from .tensorsize import new_dim, exist_dim, del_dim, iter_dim, linalg_dim, Size, FakeSize, FuncDim, BatchDim, SequeDim, FeatDim, SpaceDim

with __info__:
    import torch
    import batorch as bt
    from pyoverload import null, to_torch_dtype, dtype as dtype_, method
    from pycamia import ByteSize, Version
    from pycamia import avouch, touch, alias, void
    from pycamia import execblock, get_num_indent, add_lineno
    from pycamia import tokenize, token_replace, identity_function
    from pycamia import get_alphas, arg_extract, max_argmax
    from pycamia import argmax as argmax_, prod as prod_, item, to_list
    from pycamia import get_reference_line

int_ = builtins.int
min_ = builtins.min
max_ = builtins.max
abs_ = builtins.abs
any_ = builtins.any
all_ = builtins.all
sum_ = builtins.sum
bool_ = builtins.bool
round_ = builtins.round
range_ = builtins.range
float_ = builtins.float
complex_ = builtins.complex
num_ = (int_, float_)

_total_cpu_memory_used = 0
_total_gpu_memory_used = 0
_device = AutoDevice(verbose=True, always_proceed=True)

"""
    TODO:
    sparse-related
    device-related
"""

def get_cpu_memory_used():
    global _total_cpu_memory_used
    return ByteSize(_total_cpu_memory_used)

def get_gpu_memory_used():
    global _total_gpu_memory_used
    return ByteSize(_total_gpu_memory_used)

def collect_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret.device == torch.device('cpu'):
            global _total_cpu_memory_used
            _total_cpu_memory_used += ret.byte_size()
        else:
            global _total_gpu_memory_used
            _total_gpu_memory_used += ret.byte_size()
        return ret
    return wrapper

def set_device(device):
    global _device
    if isinstance(device, AutoDevice): _device = device
    elif isinstance(device, torch.device): _device = SleepingDevice(device)
    else: raise TypeError("Invalid device type. ")

def get_device():
    global _device
    return _device

def default_device():
    global _device
    return _device.device

def auto_device(x):
    global _device
    return x.to(_device.device)

@collect_memory
def to_device(x):
    global _device
    return x.to(_device.main_device)

def turn_on_autodevice(): _device.turn_on()
def turn_off_autodevice(): _device.turn_off()

def torch_super(self, func_name):
    return method(getattr(torch.Tensor, func_name), self)

size_mapping = defaultdict(lambda: identity_function,
    unsqueeze = lambda s, d: Size(add_dim(FakeSize(s), d)).update_special_from(d),
    squeeze = lambda s, d: remove_dim(s, d).update_special_from(d),
    permute = lambda s, d: s.permute(*d),
    transpose = lambda s, d1, d2: s.permute(*range_(min_(d1[0], d2[0])), max_(d1[0], d2[0]), *range_(min_(d1[0], d2[0])+1, max_(d1[0], d2[0])), min_(d1[0], d2[0]), *range_(max_(d1[0], d2[0])+1, len(s))) if d1[0] != d2[0] else s,
    any = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    all = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    sum = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    prod = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    min = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    max = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    median = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    mean = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    std = lambda s, d, **k: s if k.get('keepdim', False) else remove_dim(s, d),
    cumsum = lambda s, d: s,
    cumprod = lambda s, d: s,
    cummax = lambda s, d: s,
    cummin = lambda s, d: s,
    gather = lambda s, d: s,
    flip = lambda s, d: s
)

def matmul_shape(self_shape, other_shape):
    self_shape = Size(self_shape)
    other_shape = Size(other_shape)
    if self_shape.n_dim < 1 or other_shape.n_dim < 1:
        x_shape, y_shape = self_shape ^ other_shape
        return x_shape, x_shape, y_shape
    shape_last_2 = self_shape[-2:]
    if shape_last_2.n_feature_dim == 2 or shape_last_2.n_sequence_dim == 2 or shape_last_2.n_space_dim == 2:
        repr_self_shape = self_shape.with_dim_size(-1, -1).with_dim_size(-2, -1)
        l_size = 2
    else:
        repr_self_shape = self_shape.with_dim_size(-1, -1)
        l_size = 1
    shape_last_2 = other_shape[-2:]
    if shape_last_2.n_feature_dim == 2 or shape_last_2.n_sequence_dim == 2 or shape_last_2.n_space_dim == 2:
        repr_other_shape = other_shape.with_dim_size(-1, -1).with_dim_size(-2, -1)
        r_size = 2
    else:
        repr_other_shape = other_shape.with_dim_size(-1, -1)
        r_size = 1
    x_shape, y_shape = repr_self_shape ^ repr_other_shape
    z_shape = Size(max_(x, y) for x, y in zip(x_shape, y_shape)).special_from(x_shape)
    if l_size == r_size == 1:
        x_shape = x_shape[:-1] + Size(1).special_from(self_shape[-1:]) + self_shape[-1:]
        y_shape = y_shape[:-1] + other_shape[-1:] + Size(1).special_from(other_shape[-1:])
        ref_shape = z_shape[:-1]
    elif l_size == 1:
        x_shape = x_shape[:-2] + Size(1).special_from(self_shape[-1:]) + self_shape[-1:]
        y_shape = y_shape[:-r_size] + other_shape[-r_size:]
        ref_shape = z_shape[:-r_size] + other_shape[-1:]
    elif r_size == 1:
        x_shape = x_shape[:-l_size] + self_shape[-l_size:]
        y_shape = y_shape[:-2] + other_shape[-1:] + Size(1).special_from(other_shape[-1:])
        ref_shape = z_shape[:-l_size] + self_shape[-l_size:-1]
    else:
        x_shape = x_shape[:-l_size] + self_shape[-l_size:]
        y_shape = y_shape[:-r_size] + other_shape[-r_size:]
        ref_shape = z_shape[:-l_size] + self_shape[-l_size:-1] + other_shape[-r_size:-1]
    return ref_shape, x_shape, y_shape

size_mapping_op = defaultdict(lambda: (lambda a, b: (lambda u, v: (u, u, v))(*a^b)),
    __matmul__ = matmul_shape,
    mm = matmul_shape,
    bmm = matmul_shape,
    smm = matmul_shape,
    addmm = lambda a, b, c: (lambda a, _, u, v: (lambda u, v: (u, u, v))(*a^u) + ((a^v)[1],))(a, *matmul_shape(b, c)),
    addbmm = lambda a, b, c: (lambda a, _, u, v: (lambda u, v: (u, u, v))(*a^u) + ((a^v)[1],))(a, *matmul_shape(b, c)),
    saddmm = lambda a, b, c: (lambda a, _, u, v: (lambda u, v: (u, u, v))(*a^u) + ((a^v)[1],))(a, *matmul_shape(b, c)),
    addcmul = lambda a, b, c: (lambda b: (b,) + tuple(b.updated_sizes))(broadcast(a, b, c, with_size_updates=True)),
    quantile = lambda s, q, d, **k: ((Size(FuncDim(1)) if q.n_dim > 0 else Size()) + (s if k.get('keepdim', False) else remove_dim(s, d)), None, None),
    where = lambda c, x, y: (lambda b: (b,) + tuple(b.updated_sizes))(broadcast(c, x, y, with_size_updates=True)),
)

### START BIWAY AUTO GENERATION
@alias("inv")
def inverse(input: 'Tensor', dim: linalg_dim[2]=None):
    """
    conpute the inverse matrix for dimensions (the first available condition):
    (1) the last 2 feature dimensions (if n_feature_dim >= 2);
    (2) the last 2 space dimensions (if n_space_dim >= 2); 
    (3) the last 2 sequence dimensions (if n_sequence_dim >= 2). 
    """
    avouch(len(dim) == 2 and dim[0] != dim[1], TypeError("bt.inverse accepts only two dimensions for inversion. "))
    if not input.dtype.is_floating_point: input = input.type(bt.float)
    with input.hide_special(), torch._C.DisableTorchFunction():
        inv_output = Tensor.inherit_from(torch.linalg.inv(input.move_dim(dim, -1)), input, shape=[])
    return inv_output.move_dim([-2, -1], dim).special_from(input)

def diag(input: 'Tensor', diagonal=0, dim: linalg_dim[1,2]=None, *, out=None):
    """
    Compute the diagonal of a 2D matrix or a 2D matrix with diagonal elements from 1D input.
    Regarding the shape of input, the first available condition is performed:
    (1) create 2D feature for 1D feature; 
    (2) get 1D diagonal for the last 2 feature dimensions;
    (3) create 2D space for 1D space;
    (4) get 1D diagonal for the last 2 space dimensions;
    (5) create 2D sequence for 1D sequence;
    (6) get 1D diagonal for the last 2 sequence dimensions.

    `diagonal` controls the i-th diagonal, positive for those above the main diagonal, e.g. `diagonal=1` means:
    [0 * 0 0 0 0 0]
    [0 0 * 0 0 0 0]
    [0 0 0 * 0 0 0]
    [0 0 0 0 * 0 0]
    [0 0 0 0 0 * 0]
    [0 0 0 0 0 0 *]
    [0 0 0 0 0 0 0]
    this argument works for both 1->2D and 2->1D. 
    """
    size = input.shape
    if len(dim) == 1:
        n = size[dim[0]]
        design_mat = cat(zeros(size.with_dim_size(dim[0], 1), device=input.device, dtype=input.dtype), input, dim[0])
        index_mat = zeros(n + abs_(diagonal), n + abs_(diagonal), device=input.device).long()
        if diagonal >= 0: index_mat[arange(n, device=input.device), arange(diagonal, diagonal+n, device=input.device)] = arange(n, device=input.device) + 1
        else: index_mat[arange(-diagonal, -diagonal+n, device=input.device), arange(n, device=input.device)] = arange(n, device=input.device) + 1
        index_mat.add_special_dim(0, size[dim[0]:dim[0]+1])
        index_mat.add_special_dim(1, size[dim[0]:dim[0]+1])
        return design_mat[(slice(None),) * dim[0] + (index_mat,)]
    if len(dim) == 2:
        n = min_(size[dim[0]], size[dim[1]]) - abs_(diagonal)
        if dim[0] > dim[1]: dim = dim[::-1]
        if diagonal >= 0: index_x = arange(n, device=input.device).special_from(size[dim[0]:dim[0]+1]); index_y = arange(diagonal, diagonal+n, device=input.device).special_from(size[dim[1]:dim[1]+1])
        else: index_x = arange(-diagonal, -diagonal+n, device=input.device).special_from(size[dim[0]:dim[0]+1]); index_y = arange(n, device=input.device).special_from(size[dim[1]:dim[1]+1])
        with input.hide_special(), index_x.hide_special(), index_y.hide_special():
            diag_mat = input[(slice(None),) * dim[0] + (index_x,) + (slice(None),) * (dim[1] - dim[0] - 1) + (index_y,)]
        if dim[1] - dim[0] - 1 == 0: return diag_mat.special_from(remove_dim(input.shape, [dim[1]]))
        else: return diag_mat.move_dim(0, dim[0]).special_from(remove_dim(input.shape, [dim[1]]))

def diagflat(input: 'Tensor', diagonal=0, dim: exist_dim=[]):
    return diag(input.flatten(*dim), diagonal=diagonal) # suppress: special_from

def diagonal(input: 'Tensor', diagonal=0, dim1=0, dim2=1):
    return diag(input, diagonal=diagonal, dim=(dim1, dim2)) # suppress: special_from

@alias('tr')
def trace(input: 'Tensor', dim=None):
    if dim is None:
        if input.has_feature: dim = exist_dim(input, [])
        elif input.has_space: dim = exist_dim(input, ...)
        elif input.has_sequence: dim = exist_dim(input, '')
        else: raise TypeError(f"Invalid size {input.shape} for bt.trace: at least one non-batch dimension needed. ")
    else: dim = exist_dim(input, dim)
    if len(dim) > 2: dim = dim[-2:]
    return diag(input, dim=dim).sum(dim[0]) # suppress: special_from

def det(input: 'Tensor', *, out=None, dim:linalg_dim[2]=None):
    avouch(len(dim) == 2 and dim[0] != dim[1], TypeError("bt.det accepts only two dimensions for determinant. "))
    ref_shape = remove_dim(input.shape, dim)
    with input.hide_special(), torch._C.DisableTorchFunction():
        return Tensor.inherit_from(torch.det(input.move_dim(dim, -1)), input, shape=ref_shape) # suppress: special_from

# operations
def add(self: 'Tensor', other: 'Tensor', *, alpha=1, out=None): ...
def sub(self: 'Tensor', other: 'Tensor', *, alpha=1, out=None): ...
def mul(self: 'Tensor', value: 'Tensor', out=None): ...
def div(self: 'Tensor', value: 'Tensor', *, rounding_mode=None, out=None): ...
def pow(input: 'Tensor', exponent, *, out=None): ...
def fmod(self: 'Tensor', other: 'Tensor', *, out=None): ...
def log(input: 'Tensor', base=torch.e, out=None):
    with torch._C.DisableTorchFunction():
        return torch.log(input).as_subclass(torch.Tensor) / torch.log(torch.tensor(base))
def ln(input: 'Tensor', *, out=None):
    with torch._C.DisableTorchFunction():
        return torch.log(input).as_subclass(torch.Tensor)
def log2(input: 'Tensor', *, out=None): ...
def log10(input: 'Tensor', *, out=None): ...
def exp(input: 'Tensor', *, out=None): ...
def sqrt(input: 'Tensor', *, out=None): ...
def abs(input: 'Tensor', *, out=None): ...
def sign(input: 'Tensor', *, out=None): ...
def sin(input: 'Tensor', *, out=None): ...
def cos(input: 'Tensor', *, out=None): ...
def tan(input: 'Tensor', *, out=None): ...
def cot(input: 'Tensor', *, out=None):
    with torch._C.DisableTorchFunction():
        return 1 / torch.tan(input)
def sec(input: 'Tensor', *, out=None):
    with torch._C.DisableTorchFunction():
        return 1 / torch.cos(input)
def csc(input: 'Tensor', *, out=None):
    with torch._C.DisableTorchFunction():
        return 1 / torch.sin(input)
def asin(input: 'Tensor', *, out=None): ...
def acos(input: 'Tensor', *, out=None): ...
def atan(input: 'Tensor', *, out=None): ...
def arcsin(input: 'Tensor', *, out=None): ...
def arccos(input: 'Tensor', *, out=None): ...
def arctan(input: 'Tensor', *, out=None): ...
def mm(input: 'Tensor', other: 'Tensor', *, out=None): ...
def bmm(input: 'Tensor', other: 'Tensor', *, out=None): ...
def smm(input: 'Tensor', other: 'Tensor'): ...
def floor_divide(input: 'Tensor', other: 'Tensor', *, out=None): ...
def true_divide(dividend: 'Tensor', divisor: 'Tensor', *, out=None): ...
def equal(self: 'Tensor', other: 'Tensor'): ...
def addmm(input: 'Tensor', mat1: 'Tensor', mat2: 'Tensor', *, beta=1, alpha=1, out=None): ...
def addbmm(input: 'Tensor', batch1: 'Tensor', batch2: 'Tensor', *, beta=1, alpha=1, out=None):
    ref_shape, input_shape, batch1_shape, batch2_shape = size_mapping_op['addbmm'](input_shape, batch1_shape, batch2_shape)
    input = input.view(input_shape).squeeze({})
    batch1 = batch1.view(batch1_shape)
    batch2 = batch2.view(batch2_shape)
    with torch._C.DisableTorchFunction():
        return Tensor.inherit_from(torch.addbmm(input,batch1,batch2, beta=beta,alpha=alpha,out=out), input)
def saddmm(input: 'Tensor', mat1: 'Tensor', mat2: 'Tensor', *, beta=1, alpha=1, out=None): ...
def addcmul(input: 'Tensor', tensor1: 'Tensor', tensor2: 'Tensor', *, value=1, out=None): ...

# value operations
def clamp(input: 'Tensor', min=None, max=None, *, out=None): ...
def floor(input: 'Tensor', *, out=None): ...
def ceil(input: 'Tensor', *, out=None): ...
def round(input: 'Tensor', *, decimals=0, out=None): ...
def any(input: 'Tensor', *dims: del_dim[...], keepdim=False, out=None): ...
def all(input: 'Tensor', *dims: del_dim[...], keepdim=False, out=None): ...
def unique(input: 'Tensor', sorted=True, return_inverse=False, return_counts=False, dim=None):
    with torch._C.DisableTorchFunction():
        ret = torch.unique(input, sorted=sorted,return_inverse=return_inverse,return_counts=return_counts,dim=dim)
    
    if isinstance(ret, tuple):
        if len(ret) >= 1 and ret[0].ndim == 1: ret[0] = Tensor.inherit_from(ret[0], input, shape=...)
        elif len(ret) >= 1: ret[0] = Tensor.inherit_from(ret[0], input, shape=input_shape)
        if len(ret) >= 2 and return_inverse: ret[1] = Tensor.inherit_from(ret[1], input, shape=input_shape)
        elif len(ret) >= 2: ret[1] = Tensor.inherit_from(ret[1], input, shape=...)
        if len(ret) >= 3: ret[2] = Tensor.inherit_from(ret[2], input, shape=...)
        return ret
    elif ret.ndim == 1:
        return Tensor.inherit_from(ret, input, shape=[])
    else: return Tensor.inherit_from(ret, input, shape=input_shape)
def isin(elements: 'Tensor', test_elements: 'Tensor', *, assume_unique=False, invert=False): ...
def isnan(input: 'Tensor'): ...
def isinf(input: 'Tensor'): ...
def isposinf(input: 'Tensor'): ...
def isneginf(input: 'Tensor'): ...
def isfinite(input: 'Tensor'): ...

# dimension manipulations
def unsqueeze(self: 'Tensor', *dims: new_dim[...]): ...
def squeeze(self: 'Tensor', *dims: del_dim[...]):
    valid_dims = []
    with torch._C.DisableTorchFunction():
        for d in dims[::-1]:
            if self.size(d) == 1:
                valid_dims.append(d)
                self = torch_super(self, 'squeeze')(d)
    dims = tuple(valid_dims)
    return self
def flatten(self: 'Tensor', *dims: exist_dim):
    if len(dims) > 1: flat_range = min_(dims), max_(dims) + 1
    elif len(dims) == 1: flat_range = dims[0], self.n_dim
    else: flat_range = 0, self.n_dim
    ref_shape = self.shape[:flat_range[0] + 1] + self.shape[flat_range[1]:]
    if len(ref_shape) == 0: ref_shape = bt.Size(1).with_func_dim(True)
    with torch._C.DisableTorchFunction():
        return Tensor.inherit_from(torch_super(self, 'flatten')(flat_range[0], flat_range[1]-1), self, shape=ref_shape)
def transpose(self: 'Tensor', dim0: exist_dim[1], dim1: exist_dim[1]): ...
def t(self): ...
def permute(self: 'Tensor', *dims):
    ref_shape = Size(*dims)
    dims = exist_dim(self, *dims)
    with torch._C.DisableTorchFunction():
        obj = torch.permute(self, dims)
    obj = Tensor.inherit_from(obj, self, shape=size_mapping['permute'](self_shape, dims))
    if ref_shape.has_special: obj.special_from(ref_shape)
    return obj
def standard_shape(self):
    permutation = (
        ([] if self.sz_func_dim == 0 else ([0] if self.sz_func_dim > 0 else [self.n_dim-1])) + 
        ([] if self.sz_batch_dim == 0 else ([self.size_start] if self.sz_batch_dim > 0 else [self.size_stop])) + 
        ([] if self.sz_feature_dim == 0 else list(range_(self.feature_start, self.feature_stop))) + 
        list(range_(self.space_start, self.space_stop)) + 
        ([] if self.sz_sequence_dim == 0 else list(range_(self.sequence_start, self.sequence_stop))))
    return permute(self, *permutation) # suppress: special_from

def duplicate(self: 'Tensor', num=2, dim: new_dim[1]={}):
    """
    data.duplicate(num, 0): data(n_1, n_2) => (num, n_1, n_2)
    Duplicate a tensor by `num` times and stack them as a new tensor. 

    Args:
        num (int, optional): The number of duplications. Defaults to 2. 
        dim (int/new_dim, optional): The dimension to stack along. Defaults to batch.
    """
    return self.unsqueeze(dim).repeat((1,) * dim[0] + (num,) + (1,) * (self.ndim - dim[0])).special_from(dim)

def amplify(self: 'Tensor', num=2, dim: exist_dim[1]={}):
    """
    data.amplify(num, 0): data(n_1, n_2) => (n_1 * num, n_2)
    Amplify a dimension of a tensor by enlarging with duplications: amplifying [a, b, c] with number 2 results in [a, a, b, b, c, c].
    Note that one should use 'repeated' (for one dimension) or 'repeat' (for all dimensions) to duplicate the whole tensor and 
        concatenate the duplications together ([a, b, c] => [a, b, c, a, b, c]). 
    
    Args: 
        num (int, optional): The number of duplications. Defaults to 2. 
        dim (int/new_dim, optional): The dimension to stack along. Defaults to batch.
    """
    dim = dim[0]
    with self.hide_special():
        output = self.duplicate(num, dim+1).flatten(dim, dim + 1)
    return output.special_from(self)

def repeated(self: 'Tensor', num=2, dim: exist_dim[1]={}):
    """
    data.repeated(num, 0): data(n_1, n_2) => (num * n_1, n_2)
    Repeat a tensor by `num` times along one dimension `dim` (use 'repeat' for multiple dimensions) and concatenate them as a new tensor.
    Note that repeating [a, b, c] with number 2 results in [a, b, c, a, b, c].
        One should use 'amplify' to amplify to [a, a, b, b, c, c].
    
    Args: 
        num (int, optional): The number of duplications. Defaults to 2. 
        dim (int/new_dim, optional): The dimension to stack along. Defaults to batch.
    """
    dim = dim[0]
    with self.hide_special():
        output = self.duplicate(num, dim).flatten(dim, dim + 1)
    return output.special_from(self)

def repeat(self: 'Tensor', *size: 'Size'):
    with torch._C.DisableTorchFunction():
        return Tensor.inherit_from(torch_super(self, 'repeat')(*size), self).update_special_from(size)

# resamplers
def gather(self: 'Tensor', dim: exist_dim[1], index, *, sparse_grad=False, out=None): ...
def flip(self: 'Tensor', *dims: exist_dim): ...

# properties
def detach(self: 'Tensor'): ...
def quantile(self: 'Tensor', q: 'Tensor', dim=None, keepdim=False, *, interpolation='linear'):
    n_dim_count = None
    if dim is not None:
        dim = exist_dim(self, dim)
        if len(dim) > 1: self = self.flatten(dim); n_dim_count = len(dim)
        ref_shape, _, _ = size_mapping_op['quantile'](self.shape, q_shape, dim[:1], keepdim=keepdim)
    with torch._C.DisableTorchFunction():
        if dim is None: res = Tensor.inherit_from(torch.quantile(self, q, keepdim=keepdim,interpolation=interpolation), self, shape=q); dim = [int_(q.n_dim > 0)]
        else: res = Tensor.inherit_from(torch.quantile(self, q, dim[0], keepdim=keepdim,interpolation=interpolation), self, shape=ref_shape)
    if keepdim:
        d = dim[0] + int_(q.n_dim > 0)
        return res.split_dim(d, res.shape[d:d+1] * n_dim_count)
    else: return res
def val_range(self, dim: exist_dim=None):
    """
    Compute the range in dimensions `dim`, resulting in a squeeze of these dimensions 
        to a sole functional dimension of 2, i.e. the minimal and maximal values. 

    Args: dim (int/exist_dim, optional): The dimensions to find range. Defaults to None.
    Output: ((2), ...[size without the specified dimnsions])
    """
    return stack(self.min(dim), self.max(dim), 0).with_func_dim(0) # suppress: special_from

# reductions
def sum(input: 'Tensor', *dim: del_dim[:], keepdim=False, dtype=None): ...
def prod(input: 'Tensor', dim: del_dim[...]=None, keepdim=False, dtype=None): ...
def mean(input: 'Tensor', *dim: del_dim[:], keepdim=False, dtype=None): ...
def std(input: 'Tensor', *dim: del_dim[:], correction=1, keepdim=False): ...
def norm(input: 'Tensor', p='fro', dim: del_dim[...]=None, keepdim=False, dtype=None): ...
def cumsum(input: 'Tensor', dim: del_dim[1], dtype=None, out=None): ...
def cumprod(input: 'Tensor', dim: del_dim[1], dtype=None, out=None): ...

def __indexed_reduce__(func_name, self: 'Tensor', *dim, keepdim=None, out=None, ret_index_only=False):
    if len(dim) == 1 and isinstance(dim[0], torch.Tensor):
        other = dim[0]
        subclass = Tensor.get_tensor_subclass(self)
        other = other.as_subclass(subclass).special_from(other.shape) if not isinstance(other, subclass) else other
        self_shape = Size(self.shape); other_shape = Size(other.shape)
        ref_shape, self_shape, other_shape = size_mapping_op[func_name](self_shape, other_shape)
        with torch._C.DisableTorchFunction():
            return Tensor.inherit_from(torch_super(self, func_name)(other, **(dict(out=out) if locals().get('out', None) is not None else {})), self, shape=ref_shape)
    else:
        dim = del_dim(self, *dim)
        indices = None
        num_dim = 0
        init_shape = self.shape
        with torch._C.DisableTorchFunction():
            for d in dim[::-1]:
                result = torch_super(self, func_name)(d, **dict(keepdim=keepdim) if keepdim is not None else {})
                self = result.values
                res_indices = Tensor.inherit_from(result.indices, self, shape=[])
                if indices is None: indices = res_indices.unsqueeze(0, sz_func_dim=1)
                elif keepdim or keepdim is None: indices = cat(res_indices.unsqueeze(0, sz_func_dim=1), indices.gather(d+1, res_indices.duplicate(num_dim, 0, sz_func_dim=1)), 0)
                else: indices = cat(res_indices.unsqueeze(0, sz_func_dim=1), indices.gather(d+1, res_indices.unsqueeze(d).duplicate(num_dim, 0, sz_func_dim=1)).squeeze_(d+1), 0)
                num_dim += 1
        if keepdim is False: cur_shape = remove_dim(init_shape, dim)
        else: cur_shape = init_shape
        if ret_index_only:
            if indices is None: return
            init_shape_tensor = tensor_to(init_shape, self)
            dim_tensor = tensor(dim)
            dim_size = cat(init_shape_tensor[dim_tensor][1:].flip().cumprod(0).flip(), tensor_like(ones(1), init_shape_tensor)).with_func_dim(True)
            flatten_indices = (dim_size * indices).sum(FuncDim)
            indices = indices.special_from(Size(Func_dim(1)) + cur_shape)
            flatten_indices = flatten_indices.special_from(cur_shape)
            if len(dim) == 1: indices.squeeze_(0)
            flatten_indices.indices = indices
            flatten_indices.values = Tensor.inherit_from(self, self, shape=cur_shape)
            if out is not None:
                out.zero_().add_(flatten_indices)
            return flatten_indices
        else:
            self = Tensor.inherit_from(self, self, shape=cur_shape)
            self.indices = indices.special_from(Size(FuncDim(1)) + cur_shape) if indices is not None else None
            if len(dim) == 1: self.indices.squeeze_(0)
            # indices_tuple = indices.special_from(cur_shape + (1,)).split(dim=-1, squeezedim=True)
            # self.indices = indices_tuple if len(dim) > 1 else indices_tuple[0]
            self.values = self
            if out is not None:
                if isinstance(out, tuple):
                    out[0].zero_().add_(self.values)
                    if len(out) > 1: out[1].zero_().add_(self.indices)
                else: out.zero_().add_(self.values)
            return self

def min(input: 'Tensor', *dim, keepdim=False, out=None):
    return __indexed_reduce__('min', input, *dim, keepdim=keepdim, **dict(out=out) if locals().get('out', None) is not None else {}) # suppress: special_from
def max(input: 'Tensor', *dim, keepdim=False, out=None):
    return __indexed_reduce__('max', input, *dim, keepdim=keepdim, **dict(out=out) if locals().get('out', None) is not None else {}) # suppress: special_from
def median(input: 'Tensor', *dim, keepdim=False, out=None):
    return __indexed_reduce__('median', input, *dim, keepdim=keepdim, **dict(out=out) if locals().get('out', None) is not None else {}) # suppress: special_from
def cummin(input: 'Tensor', dim: exist_dim[1]=None, *, out=None):
    return __indexed_reduce__('cummin', input, *dim, **dict(out=out) if locals().get('out', None) is not None else {}) # suppress: special_from
def cummax(input: 'Tensor', dim: exist_dim[1]=None, *, out=None):
    return __indexed_reduce__('cummax', input, *dim, **dict(out=out) if locals().get('out', None) is not None else {}) # suppress: special_from
def argmin(input: 'Tensor', *dim, keepdim=False):
    return __indexed_reduce__('min', input, *dim, keepdim=keepdim, ret_index_only=True) # suppress: special_from
def argmax(input: 'Tensor', *dim, keepdim=False):
    return __indexed_reduce__('max', input, *dim, keepdim=keepdim, ret_index_only=True) # suppress: special_from

# slicing functions
def split(self, split_size: int=1, dim: exist_dim[1] = {}, squeezedim=False):
    """
    split(self, split_size: int=1, dim: exist_dim = {}) -> Tensor
    Split a tensor into a tuple of tensors, along `dim`, with split_size for each tensor.

    Args:
        split_size (int or list, optional): The split size for each tensor, using a list of integers adding up to size to split the dimension accordingly. Defaults to 1.
        dim (int/exist_dim, optional): The dimension to split along. Defaults to batch.
    """
    dim = dim[0]
    with torch._C.DisableTorchFunction():
        if squeezedim:
            avouch(split_size == 1 or all_(s == 1 for s in split_size), TypeError("Keyword argument 'squeezedim' is only active for 'split_size=1' in bt.Tensor.split. "))
            return tuple(Tensor.inherit_from(x, self).squeeze_(dim) for x in torch_super(self, 'split')(split_size, dim))
        else: return tuple(Tensor.inherit_from(x, self) for x in torch_super(self, 'split')(split_size, dim))

def sample(self, number: int = 1, dim: exist_dim = {}, random: bool = True):
    """
    sample(self, numbder: int = 1, dim: int = self.batch_dimension, random: bool = True) -> Tensor

    Sample `number` of slices from a given dimension.
    
    Args:
        number (int, optional): the number of slices. Defaults to `1`.
        dim (int/exist_dim, keyword argument): the dimension to slice or select. Defaults to batch dimension.
        random (bool, optional): whether to randomly pick the slices or not. Defaults to True.
    
    Note:
        Using `sample(num, dim)` for data of size (n_1, n_2, ..., n_r) would result in
            a tensor of size (n_1, n_2, ..., n_{dim-1}, num, n_{dim+1}, ..., n_r)
    
    Examples::
        >>> data.shape
        batorch.Size([4, 3], 5, 6)
        >>> data.sample(1, 2, random=False).shape
        batorch.Size([4, 3], 1, 6)
        >>> # The above is equivalant to data[:, :, :1, ...].shape.
        >>> data.sample(7, [], random=False).shape
        batorch.Size(7, 5, 6)
        >>> # The above is equivalant to data.flatten(0, 1)[:7].shape.
    """
    if len(dim) > 1: self = self.merge_dims(*dim, target=dim[0])
    sample_indices = [slice(None)] * self.n_dim
    dim = dim[0]
    if random:
        import random
        n_total = self.size(dim)
        n_round = number // n_total
        n_remain = number % n_total
        samples = cat(randperm({n_round}, n_total, device=self.device).flatten().view(-1), tensor(random.sample(range_(n_total), k = n_remain), device=self.device), 0)
    else:
        avouch(number <= self.size(dim), TypeError(f"Too many elements needed to be sampled from dimension {dim}"))
        samples = tensor(range_(number))
    sample_indices[dim] = samples.special_from(self.shape[dim:dim+1])
    return self[tuple(sample_indices)] # suppress: special_from

def pick(self, index: int = None, dim: exist_dim = {}, random: bool = False):
    """
    pick(self, index: int = 0, dim: int = self.batch_dimension, random: bool = False) -> Tensor

    Pick one slice on dimension `dim` for big tensors. 
    
    Args:
        index (int, optional): the slice index to pick. Defaults to `None`.
        dim (int/exist_dim, keyword argument): the dimension to slice or select. Defaults to batch dimension.
        random (bool, optional): whether to randomly pick the slice or not. Defaults to False.
    
    Note:
        Using `pick(index, dim)` for data of size (n_1, n_2, ..., n_r) would result in
            a tensor of size (n_1, n_2, ..., n_{dim-1}, n_{dim+1}, ..., n_r)
    
    Examples::
        >>> data.shape
        batorch.Size(4, 3, 5, 6)
        >>> data.pick(-1, 2, random=False).shape
        batorch.Size(4, 3, 6)
        >>> # The above is equivalant to data[:, :, 4, ...].shape.
    """
    if len(dim) > 1: self = self.merge_dims(*dim, target=dim[0])
    dim = dim[0]
    if random:
        avouch(index is None, "'index' should be None if random pick is enabled. Use keyword argument 'dim=xx' to identify the dimension.")
        import random
        index = random.randint(0, self.size(dim)-1)
    else:
        avouch(isinstance(index, int_) and -self.size(dim) <= index < self.size(dim) or isinstance(index, (slice, Tensor)), 
               TypeError(f"Invalid index for picking from dimension {dim} of size {self.size(dim)}: {index} ({index.__class__}). "))
    return self[(slice(None),) * dim + (index,)] # suppress: special_from

def eig(input: 'Tensor', dim: linalg_dim[2]=None, out=None):
    """
    Find eigen values and vectors for matrix `input`: (for the first available condition):
    (1) in feature dimensions if more than 2D is available; 
    (2) in space dimensions if more than 2D is available; 
    (3) in sequence dimensions if more than 2D is available. 
    """
    has_batch = False
    with input.hide_special(), torch._C.DisableTorchFunction():
        A = input.move_dim(dim, -1)
        if A.n_dim > 2: A = A.flatten(0, -3); has_batch = True
        if torch.__version__ >= Version('1.10'):
            L, V = torch.linalg.eig(A)
        else:
            K, P = torch.eig(A, eigenvectors=True)
            L = torch.complex(K[:, 0], K[:, 1])
            Vr = torch.where((K[:, 1] < 0).reshape((1, -1)), torch.cat((P[:, :1], P[:, :-1]), 1), P)
            Vi = (K[:, 1] > 0).reshape((1, -1)) * torch.cat((P[:, 1:], P[:, -1:]), 1) - (K[:, 1] < 0).reshape((1, -1)) * P
            V = torch.complex(Vr, Vi)
        L = Tensor.inherit_from(L, input, shape=[])
        V = Tensor.inherit_from(V, input, shape=[])
    dim_type = input.shape[dim[0]:dim[0]+1]
    if has_batch:
        L = L.split_dim(0, remove_dim(input.shape, dim))
        V = V.split_dim(0, remove_dim(input.shape, dim))
    L = L.move_dim(-1, dim[0]).add_special_dim(dim[0], dim_type)
    V = V.move_dim([-2, -1], dim).add_special_dim(dim[0], dim_type).add_special_dim(dim[0]+1, dim_type)
    return L, V # suppress: special_from

def matmul(input: 'Tensor', other: 'Tensor', *, dim1=None, dim2=None, out=None):
    """perform matmul for the best linear dimensions, justlike other mat** functions do. """
    if input.has_feature and other.has_feature:
        dim1 = exist_dim(input, [])
        dim2 = exist_dim(other, [])
    elif input.has_space and other.has_space:
        dim1 = exist_dim(input, ...)
        dim2 = exist_dim(other, ...)
    elif input.has_sequence and other.has_sequence:
        dim1 = exist_dim(input, '')
        dim2 = exist_dim(other, '')
    else: raise TypeError(f"Cannot perform matmul alignment for shapes {input_shape} and {other_shape}. ")
    dim1 = dim1[-2:]
    dim2 = dim2[-2:]
    size = 2 if len(dim1) == len(dim2) else 1
    input = input.move_dim(dim1, -1)
    other = other.move_dim(dim2, -1)
    return (input @ other).movedim(list(range_(-size, 0)), dim2[0]) # suppress: special_from

@alias("matrix_power")
def matpow(input: 'Tensor', k, *, dim: linalg_dim[2]=None):
    """return a matrix power of A^k."""
    L, V = eig(input, dim=dim)
    L = L.move_dim(dim[0], -1)
    V = V.move_dim(dim, -1)
    L_k = where((L.real < 0) & (L.imag.abs() < 1e-6), -complex((-L.real) ** k, L.imag), L ** k)
    R = V @ diag(L_k, dim=-1) @ V.inv()
    if R.is_complex() and not input.is_complex(): R = R.real
    return R.move_dim([-2, -1], dim).type(input.dtype) # suppress: special_from

@alias("matrix_exp")
def matexp(input: 'Tensor', *, dim: linalg_dim[2]=None):
    """return a matrix exponential of e^A."""
    L, V = eig(input, dim=dim)
    L = L.move_dim(dim[0], -1)
    V = V.move_dim(dim, -1)
    R = V @ diag(exp(L), dim=-1) @ V.inv()
    if R.is_complex() and not input.is_complex(): R = R.real
    return R.move_dim([-2, -1], dim).type(input.dtype) # suppress: special_from

@alias("matrix_log")
def matlog(input: 'Tensor', *, dim: linalg_dim[2]=None):
    """return a matrix exponential of e^A."""
    L, V = eig(input, dim=dim)
    L = L.move_dim(dim[0], -1)
    V = V.move_dim(dim, -1)
    R = V @ diag(log(L), dim=-1) @ V.inv()
    if R.is_complex() and not input.is_complex(): R = R.real
    return R.move_dim([-2, -1], dim).type(input.dtype) # suppress: special_from

@alias("matrix_rank")
def rank(input: 'Tensor', *, atol=None, rtol=None, hermitian=False, dim: linalg_dim[2]=None, out=None):
    A = input.move_dim(dim, -1)
    with torch._C.DisableTorchFunction():
        return Tensor.inherit_from(torch.linalg.matrix_rank(A, atol=atol, rtol=rtol, hermitian=hermitian, **dict(out=out) if locals().get('out', None) is not None else {}), input, shape=A.shape[:-2])

@alias("matrix_norm")
def matnorm(input: 'Tensor', ord='fro', dim: linalg_dim[2]=None, keepdim=False, *, dtype=None, out=None):
    A = input.move_dim(dim, -1)
    with torch._C.DisableTorchFunction():
        return Tensor.inherit_from(torch.linalg.matrix_norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, **dict(out=out) if locals().get('out', None) is not None else {}), input, shape=A.shape if keepdim else A.shape[:-2])
### STOP BIWAY AUTO GENERATION

def broadcast(*sizes, with_size_updates=False):
    if len(sizes) == 0:
        broadcasted = Size()
        if with_size_updates: broadcasted.updated_sizes = []
    elif len(sizes) == 1:
        broadcasted = sizes[0]
        if with_size_updates: broadcasted.updated_sizes = [sizes[0]]
    else:
        x, y = sizes[0] ^ sizes[1]
        broadcasted = Size(max_(u, v) for u, v in zip(x, y)).special_from(y)
        updated_sizes = [x, y]
        for z in sizes[2:]:
            x, y = broadcasted ^ z
            broadcasted = Size(max_(u, v) for u, v in zip(x, y)).special_from(y)
            updated_sizes.append(y)
        if with_size_updates: broadcasted.updated_sizes = updated_sizes
    return broadcasted

def remove_dim(size, list_of_removals):
    if not isinstance(list_of_removals, list): list_of_removals = list(list_of_removals)
    list_of_removals.sort()
    return sum_([size[slice(x + 1, y)] for x, y in zip([-1] + list_of_removals, list_of_removals + [None])], Size())

def add_dim(size, list_of_adds):
    if not isinstance(list_of_adds, list): list_of_adds = list(list_of_adds)
    for i in list_of_adds: size = size[:i] + (1,) + size[i:]
    return size

class Tensor(torch.Tensor):
    
    @classmethod
    def get_tensor_subclass(cls, target):
        if not isinstance(target, type): target = target.__class__
        if not issubclass(target, Tensor): target = Tensor
        return target
    
    @classmethod
    def inherit_from(cls, data, target, shape=None):
        subclass = Tensor.get_tensor_subclass(target)
        if shape is None: shape = target.shape
        if shape == ...: result = data.as_subclass(subclass)
        elif len(shape) == 0: result = data.as_subclass(subclass).init_special()
        else: result = data.as_subclass(subclass).special_from(shape)
        result.inherited = getattr(target, "inherited", {})
        return result
    
    @staticmethod
    def _make_subclass(cls, 
        torch_tensor, requires_grad=None, device=None, 
        with_func=None, func_dim=void, sz_func_dim=None,
        with_batch=None, batch_dim=void, sz_batch_dim=None,
        with_channel=None, channel_dim=void, n_feature_dim=None,
        with_feature=None, feature_dim=void, sz_feature_dim=None,
        with_sequence=None, sequence_dim=void, n_sequence_dim=None,
        sz_sequence_dim=None, ref_size=None
    ):
        # Inherit the inheritting properties
        recorded_inherit = getattr(torch_tensor, 'inherited', {})
        # First move the data to device, eliminating argument 'device'
        if device is not None and torch_tensor.device != device:
            if isinstance(device, AutoDevice): device = device.main_device
            torch_tensor = torch_tensor.to(device)
        if requires_grad is None: requires_grad = torch_tensor.requires_grad
        # Create the resulting tensor
        if torch_tensor.__class__ == cls: self = torch_tensor.clone()
        else: self = torch.Tensor._make_subclass(cls, torch_tensor, requires_grad)
        # Compress the controller arguments into sz numbers
        n_dim = self.ndim
        if not hasattr(self, 'sz_func_dim'): self.sz_func_dim = 0
        if not hasattr(self, 'sz_batch_dim'): self.sz_batch_dim = 0
        if not hasattr(self, 'sz_feature_dim'): self.sz_feature_dim = 0
        if not hasattr(self, 'sz_sequence_dim'): self.sz_sequence_dim = 0
        avouch(sum_([with_func is not None, func_dim != void, sz_func_dim is not None, ref_size is not None and ref_size.sz_func_dim != 0]) <= 1, "Only one input is accepted in 'with_func', 'func_dim', 'sz_func_dim' and 'ref_size'. ")
        avouch(sum_([with_batch is not None, batch_dim != void, sz_batch_dim is not None, ref_size is not None and ref_size.sz_batch_dim != 0]) <= 1, "Only one input is accepted in 'with_batch', 'batch_dim', 'sz_batch_dim' and 'ref_size'. ")
        avouch(sum_([with_channel is not None, with_feature is not None, channel_dim != void, feature_dim != void, n_feature_dim is not None, sz_feature_dim is not None, ref_size is not None and ref_size.sz_feature_dim != 0]) <= 1, "Only one input is accepted in 'with_channel', 'with_feature', 'channel_dim', 'feature_dim', 'n_feature_dim', 'sz_feature_dim', and 'ref_size'. ")
        avouch(sum_([with_sequence is not None, sequence_dim != void, n_sequence_dim is not None, sz_sequence_dim is not None, ref_size is not None and ref_size.sz_sequence_dim != 0]) <= 1, "Only one input is accepted in 'with_sequence', 'sequence_dim', 'n_sequence_dim', 'sz_sequence_dim', and 'ref_size'. ")
        if with_func is not None: self.sz_func_dim = int_(with_func)
        elif func_dim != void:
            cands = (0, -n_dim, n_dim-1, -1)
            avouch(func_dim in cands, TypeError(f"'func_dim' for tensor should be one of {cands[0]} and {cands[-1]}. "))
            self.sz_func_dim = 1 if func_dim in cands[:2] else -1
        elif sz_func_dim is not None:
            avouch(sz_func_dim in (0, 1, -1), TypeError("'sz_func_dim' for tensor should be one of (0, 1, -1). "))
            self.sz_func_dim = sz_func_dim
        elif ref_size is not None and ref_size.sz_func_dim != 0: self.sz_func_dim = ref_size.sz_func_dim
        if with_batch is not None: self.sz_batch_dim = int_(with_batch)
        elif batch_dim != void:
            cands = (max_(self.sz_func_dim, 0), max_(self.sz_func_dim, 0)-n_dim, n_dim-1+min_(self.sz_func_dim, 0), -1+min_(self.sz_func_dim, 0))
            avouch(batch_dim in cands, TypeError(f"'batch_dim' for tensor should be one of {cands[0]} and {cands[-1]}. "))
            self.sz_batch_dim = 1 if batch_dim in cands[:2] else -1
        elif sz_batch_dim is not None:
            avouch(sz_batch_dim in (0, 1, -1), TypeError("'sz_batch_dim' for tensor should be one of (0, 1, -1). "))
            self.sz_batch_dim = sz_batch_dim
        elif ref_size is not None and ref_size.sz_batch_dim != 0: self.sz_batch_dim = ref_size.sz_batch_dim
        if with_channel is not None: self.sz_feature_dim = int_(with_channel)
        elif with_feature is not None: self.sz_feature_dim = int_(with_feature)
        elif channel_dim != void:
            cands = (max_(self.sz_func_dim, 0)+max_(self.sz_batch_dim, 0), 
                     max_(self.sz_func_dim, 0)+max_(self.sz_batch_dim, 0)-n_dim, 
                     n_dim-1+min_(self.sz_func_dim, 0)+min_(self.sz_batch_dim, 0), 
                     -1+min_(self.sz_func_dim, 0)+min_(self.sz_batch_dim, 0))
            avouch(channel_dim in cands, TypeError(f"'channel_dim' for tensor should be one of {cands[0]} and {cands[-1]}. "))
            self.sz_feature_dim = 1 if channel_dim in cands[:2] else -1
        elif feature_dim != void:
            cands = (max_(self.sz_func_dim, 0)+max_(self.sz_batch_dim, 0), 
                     max_(self.sz_func_dim, 0)+max_(self.sz_batch_dim, 0)-n_dim, 
                     n_dim-1+min_(self.sz_func_dim, 0)+min_(self.sz_batch_dim, 0), 
                     -1+min_(self.sz_func_dim, 0)+min_(self.sz_batch_dim, 0))
            avouch(feature_dim in cands, TypeError(f"'feature_dim' for tensor should be one of {cands[0]} and {cands[-1]}. "))
            self.sz_feature_dim = 1 if feature_dim in cands[:2] else -1
        elif n_feature_dim is not None:
            avouch(isinstance(n_feature_dim, int_) and n_feature_dim >= 0, TypeError(f"'n_feature_dim' for tensor should be non-negative integer, not {n_feature_dim}. "))
            self.sz_feature_dim = n_feature_dim
        elif sz_feature_dim is not None:
            avouch(isinstance(sz_feature_dim, int_), TypeError(f"'sz_feature_dim' for tensor should be an integer, not {sz_feature_dim}. "))
            self.sz_feature_dim = sz_feature_dim
        elif ref_size is not None and ref_size.sz_feature_dim != 0: self.sz_feature_dim = ref_size.sz_feature_dim
        if with_sequence is not None: self.sz_sequence_dim = int_(with_sequence)
        elif sequence_dim != void:
            cands = (max_(self.sz_func_dim, 0)+max_(self.sz_batch_dim, 0)+max_(self.sz_feature_dim, 0), 
                     max_(self.sz_func_dim, 0)+max_(self.sz_batch_dim, 0)+max_(self.sz_feature_dim, 0)-n_dim, 
                     n_dim-1+min_(self.sz_func_dim, 0)+min_(self.sz_batch_dim, 0)+min_(self.sz_feature_dim, 0), 
                     -1+min_(self.sz_func_dim, 0)+min_(self.sz_batch_dim, 0)+min_(self.sz_feature_dim, 0))
            avouch(sequence_dim in cands, TypeError(f"'sequence_dim' for tensor should be one of {cands[0]} and {cands[-1]}. "))
            self.sz_sequence_dim = 1 if sequence_dim in cands[:2] else -1
        elif n_sequence_dim is not None:
            avouch(isinstance(n_sequence_dim, int_) and n_sequence_dim >= 0, TypeError(f"'n_sequence_dim' for tensor should be non-negative integer, not {n_sequence_dim}. "))
            self.sz_sequence_dim = -n_sequence_dim
        elif sz_sequence_dim is not None:
            avouch(isinstance(sz_sequence_dim, int_), TypeError(f"'sz_sequence_dim' for tensor should be an integer, not {sz_sequence_dim}. "))
            self.sz_sequence_dim = sz_sequence_dim
        elif ref_size is not None and ref_size.sz_sequence_dim != 0: self.sz_sequence_dim = ref_size.sz_sequence_dim
        self.inherited = recorded_inherit
        return self
    
    def as_subclass(self, *args, **kwargs):
        recorded_inherit = getattr(self, 'inherited', {})
        result = super().as_subclass(*args, **kwargs)
        result.inherit = recorded_inherit
        return result
    
    @collect_memory
    def __new__(cls, *args, **kwargs):
        """bt.Tensor
        Usages:
            bt.Tensor(tensor: tuple/list/torch.Tensor/bt.Tensor/(tensor with 'shape')/(tensor with method '__tensor__'), requires_grad=None, device=None, **kwargs)
            bt.Tensor(shape: tuple, requires_grad=None, device=None, **kwargs)
            bt.Tensor(*shape: int, requires_grad=None, device=None, **kwargs)
            
        Args:
            tensor (tuple or list or Tensor): Create from a tensor or iterative object. 
            shape (tuple or *tuple): Create a tensor memory from a shape tuple. 
            **kwargs includes:
                func controllers:
                    with_func (bool): Whether the tensor has a functional dimension, at the first dimension. Defaults to False. 
                    func_dim (int:0/n_dim-1 or None): The functional dimension index (0 or -1 only, as the functional dimension can only be the first/last dimension) of the tensor. Defaults to None.
                    sz_func_dim (int): The sz number of functional dimension: 0 for no functional dimension, 1 for one func on the left, and -1 for one on the right. Defaults to 0. 
                batch controllers:
                    with_batch (bool): Whether the tensor has a batch dimension, at the first dimension (apart from functional dimension). Defaults to False. 
                    batch_dim (int or None): The batch dimension index (the first/last dimension apart from functional dimension) of the tensor. Defaults to None.
                    sz_batch_dim (int): The sz number of batch dimension: 0 for no batch dimension, 1 for one batch on the left, and -1 for one on the right. Defaults to 0. 
                feature controllers:
                    with_channel (bool): Whether the tensor has a channel dimension, at the first dimension apart from batch. Defaults to False. 
                    with_feature (bool): Whether the tensor has a feature dimension, at the first dimension apart from batch. Defaults to False. 
                    channel_dim (int or None): The channel dimension index (the first/last dimension apart from batch) of the tensor. Defaults to None. 
                    feature_dim (int or None): The feature dimension index (the first/last dimension apart from batch) of the tensor. Defaults to None. 
                    n_feature_dim (int+): The number of feature dimensions, on the left part of the size. Defaults to 0. 
                    sz_feature_dim (int): The sz number of feature dimensions: 0 for no features, positive for features on the left, and negative for features on the right. Defaults to 0.
                sequence controllers:
                    with_sequence (bool): Whether the tensor has one sequence dimension, at the first dimension apart from batch and feature. Defaults to False. 
                    sequence_dim (int or None): The sequence dimension index (the first/last dimension apart from batch and feature) of the tensor. Defaults to None. 
                    n_sequence_dim (int+): The number of sequence dimensions, on the left part of the size. Defaults to 0. 
                    sz_sequence_dim (int): The sz number of sequence dimensions: 0 for no sequences, positive for sequences on the left, and negative for sequences on the right. Defaults to 0.
        """
        if len(args) >= 1 and isinstance(args[0], torch.Tensor): return Tensor._make_subclass(cls, *args, **kwargs)
        if len(args) >= 1 and hasattr(args[0], '__tensor__'): return Tensor._make_subclass(cls, args[0].__tensor__(), *args[1:], **kwargs)
        device = kwargs.pop('device', _device)
        if isinstance(device, AutoDevice): device = device.main_device
        if len(args) >= 1 and hasattr(args[0], 'shape') or isinstance(args[0], (list, tuple)): return Tensor._make_subclass(cls, torch.tensor(args[0], requires_grad=False, device=device), *args[1:], **kwargs)
        shape = Size(*args)
        if shape.has_special:
            if shape.sz_func_dim != 0: kwargs.setdefault('sz_func_dim', shape.sz_func_dim)
            if shape.sz_batch_dim != 0: kwargs.setdefault('sz_batch_dim', shape.sz_batch_dim)
            if shape.sz_feature_dim != 0: kwargs.setdefault('sz_feature_dim', shape.sz_feature_dim)
            if shape.sz_sequence_dim != 0: kwargs.setdefault('sz_sequence_dim', shape.sz_sequence_dim)
            return Tensor._make_subclass(cls, super().__new__(torch.Tensor, *shape.tuple(), device=device), **kwargs)
        return Tensor._make_subclass(cls, super().__new__(torch.Tensor, *args, device=device), **kwargs)

    def __init__(self, *args, **kwargs):
        self.sz_func_dim = self.sz_func_dim
        self.sz_batch_dim = self.sz_batch_dim
        self.sz_feature_dim = self.sz_feature_dim
        self.sz_sequence_dim = self.sz_sequence_dim
    
    # Inherit all the properties for special dimensions from 'bt.Size'. 
    has_func = property(Size.has_func.fget)
    nfuncdim = n_func_dim = property(Size.n_func_dim.fget)
    is_funcdim = is_func_dim = Size.is_func_dim
    sz_func_dim_ = with_szfuncdim = with_sz_func_dim = Size.with_sz_func_dim
    n_func_dim_ = with_nfuncdim = with_n_func_dim = Size.with_n_func_dim
    func_dimension = func_dim = property(Size.func_dim.fget).setter(Size.func_dim.fset)
    func_dim_ = func_dimension_ = \
    with_funcdim = with_func_dim = Size.with_func_dim
    nfunc = func_size = n_func = property(lambda self: Size.n_func.fget(self.shape))
    size_start = property(Size.size_start.fget)
    size_stop = property(Size.size_stop.fget)

    has_batch = property(Size.has_batch.fget)
    nbatchdim = n_batch_dim = property(Size.n_batch_dim.fget)
    is_batchdim = is_batch_dim = Size.is_batch_dim
    sz_batch_dim_ = with_szbatchdim = with_sz_batch_dim = Size.with_sz_batch_dim
    n_batch_dim_ = with_nbatchdim = with_n_batch_dim = Size.with_n_batch_dim
    batch_dimension = batch_dim = property(Size.batch_dim.fget).setter(Size.batch_dim.fset)
    batch_dim_ = batch_dimension_ = \
    with_batchdim = with_batch_dim = Size.with_batch_dim
    nbatch = batch_size = n_batch = property(lambda self: Size.n_batch.fget(self.shape))
    non_bat_start = property(Size.non_bat_start.fget)
    non_bat_stop = property(Size.non_bat_stop.fget)

    has_channel = property(Size.has_channel.fget)
    nchanneldim = n_channel_dim = property(Size.n_channel_dim.fget)
    is_channeldim = is_channel_dim = Size.is_channel_dim
    channel_dimension = channel_dim = property(Size.channel_dim.fget).setter(Size.channel_dim.fset)
    channel_dim_ = channel_dimension_ = \
    with_channeldim = with_channel_dim = Size.with_channel_dim
    nchannel = channel_size = n_channel = property(lambda self: Size.n_channel.fget(self.shape))
    
    has_feature = property(Size.has_feature.fget)
    is_featuredim = is_feature_dim = Size.is_feature_dim
    nfeaturedim = n_feature_dim = property(Size.n_feature_dim.fget).setter(Size.n_feature_dim.fset)
    sz_feature_dim_ = with_szfeaturedim = with_sz_feature_dim = Size.with_sz_feature_dim
    n_feature_dim_ = with_nfeaturedim = with_n_feature_dim = Size.with_n_feature_dim
    feature_start = property(Size.feature_start.fget).setter(Size.feature_start.fset)
    feature_stop = property(Size.feature_stop.fget).setter(Size.feature_stop.fset)
    feature_range = property(Size.feature_range.fget).setter(Size.feature_range.fset)
    nfeature = n_feature = property(lambda self: Size.n_feature.fget(self.shape))
    feature_size = feature = property(lambda self: Size.feature.fget(self.shape))
    seq_spc_start = property(Size.seq_spc_start.fget)
    seq_spc_stop = property(Size.seq_spc_stop.fget)

    has_time = has_series = has_sequence = property(Size.has_sequence.fget)
    is_timedim = is_seriesdim = is_sequencedim = \
    is_time_dim = is_series_dim = is_sequence_dim = Size.is_sequence_dim
    ntimedim = nseriesdim = nsequencedim = \
    n_time_dim = n_series_dim = n_sequence_dim = property(Size.n_sequence_dim.fget).setter(Size.n_sequence_dim.fset)
    sz_time_dim_ = sz_series_dim_ = sz_sequence_dim_ = \
    with_sztimedim = with_szseriesdim = with_szsequencedim = \
    with_sz_time_dim = with_sz_series_dim = with_sz_sequence_dim = Size.with_sz_sequence_dim
    n_time_dim_ = n_series_dim_ = n_sequence_dim_ = \
    with_ntimedim = with_nseriesdim = with_nsequencedim = \
    with_n_time_dim = with_n_series_dim = with_n_sequence_dim = Size.with_n_sequence_dim
    time_dim_ = series_dim_ = sequence_dim_ = \
    with_timedim = with_seriesdim = with_sequencedim = \
    with_time_dim = with_series_dim = with_sequence_dim = Size.with_sequence_dim
    time_start = series_start = sequence_start = property(Size.sequence_start.fget).setter(Size.sequence_start.fset)
    time_stop = series_stop = sequence_stop = property(Size.sequence_stop.fget).setter(Size.sequence_stop.fset)
    time_range = series_range = sequence_range = property(Size.sequence_range.fget).setter(Size.sequence_range.fset)
    ntime = ntimeline = nseries = nsequence = \
    n_time = n_timeline = n_series = n_sequence = property(lambda self: Size.n_sequence.fget(self.shape))
    time_size = series_size = sequence_size = \
    time = series = sequence = property(lambda self: Size.sequence.fget(self.shape))
    
    has_space = property(Size.has_space.fget)
    is_spacedim = is_space_dim = Size.is_space_dim
    nspacedim = n_space_dim = property(Size.n_space_dim.fget)
    space_start = property(Size.space_start.fget)
    space_stop = property(Size.space_stop.fget)
    space_range = property(Size.space_range.fget)
    nspace = n_space = property(lambda self: Size.n_space.fget(self.shape))
    space_size = space = property(lambda self: Size.space.fget(self.shape))

    has_special = property(Size.has_special.fget)
    nspecialdim = n_special_dim = property(Size.nspecialdim.fget)
    special_dims = property(Size.special_dims.fget)
    def add_special_dim(self, *args, **kwargs):
        self.shape = Size.add_special_dim(self.shape, *args, **kwargs)
        return self
    def change_special_dim(self, *args, **kwargs):
        self.shape = Size.change_special_dim(self.shape, *args, **kwargs)
        return self
    special_from = Size.special_from
    update_special_from = Size.update_special_from
    init_special = Size.init_special
    is_specialdim = is_special_dim = Size.is_special_dim
    nele = n_ele = property(lambda self: Size.n_ele.fget(self.shape))
    ndim = n_dim = property(lambda self: super().ndim)
    python_repr = property(lambda self: self.shape.python_repr)
    
    @alias("free_special")
    def hide_special(self):
        class hide_special_operator:
            def __init__(self, this):
                self.this = this
            def __enter__(self):
                self.sz_func_dim = self.this.sz_func_dim
                self.sz_batch_dim = self.this.sz_batch_dim
                self.sz_feature_dim = self.this.sz_feature_dim
                self.sz_sequence_dim = self.this.sz_sequence_dim
                self.this.init_special()
            def __exit__(self, exc_type, exc_value, traceback):
                self.this.sz_func_dim = self.sz_func_dim
                self.this.sz_batch_dim = self.sz_batch_dim
                self.this.sz_feature_dim = self.sz_feature_dim
                self.this.sz_sequence_dim = self.sz_sequence_dim
        return hide_special_operator(self)
    
    def numel(self, *dim: exist_dim):
        dim = exist_dim(self, *dim)
        if len(dim) > 1: return prod_(self.size(*dim))
        elif len(dim) > 0: return self.size(dim[0])
        return 1
    def dim(self): return super().dim()

    # Get and assign the shape property. The assignment is only accessible for special dimensions. Assigning inconsistent shape would result in an error. 
    @property
    def shape(self):
        if not hasattr(self, 'sz_func_dim'):
            pyfile, lineno, line = get_reference_line(search_more=True, with_line_info=True)
            raise AttributeError(f"Getting batorch shape from an uninitialized Tensor object of size {super().shape}, in line {lineno}, {pyfile}: \n{line}")
        return Size.__new_raw__(super().shape, sz_func_dim=self.sz_func_dim, sz_batch_dim=self.sz_batch_dim, sz_feature_dim=self.sz_feature_dim, sz_sequence_dim=self.sz_sequence_dim)
    
    @shape.setter
    def shape(self, *x): return self.with_shape(*x)

    def size(self, *dim: exist_dim):
        dim = exist_dim(self, dim)
        with torch._C.DisableTorchFunction():
            sizes = tuple(torch_super(self, 'size')(d) for d in dim)
        if len(dim) > 1: return sizes
        else: return sizes[0]
    
    def with_shape(self, *x):
        x = arg_extract(x)
        if isinstance(x, Tensor): x = x.shape
        if not isinstance(x, Size): x = Size(x)
        with torch._C.DisableTorchFunction():
            avouch(all_(u == v or v == -1 for u, v in zip(super().shape, x.tuple())), f"Cannot assign shape {x} to tensor with data shape {tuple(super().shape)}, due to unequal sizes. ")
        self.special_from(x)
        return self
    
    # Control the slicing system. 
    def __getitem__(self, indices):
        shapes = []
        if isinstance(indices, (slice, torch.Tensor)) or indices is ...: indices = (indices,)
        if isinstance(indices, tuple):
            squeeze_dims = []
            unsqueeze_dims = []
            offset = 0
            for i, x in enumerate(indices):
                i += offset
                if x is ...: offset += self.n_dim - len(indices); continue
                if isinstance(x, int_): squeeze_dims.append(i); continue
                if isinstance(x, slice): continue
                if isinstance(x, torch.Tensor):
                    if issubclass(x.dtype, dtype_(bool)):
                        avouch(self.shape[i:i+x.n_dim] == x.shape, TypeError("Bool indices for tensor should be of exact same size as the input tensor. "))
                        offset += x.n_dim
                        unsqueeze_dims.append((i + len(unsqueeze_dims) - len(squeeze_dims), x.shape[:1]))
                        squeeze_dims.extend(range_(i, i + x.n_dim))
                        continue
                    shapes.append(x.shape)
                else: shapes.append(Tensor(x).shape)
                squeeze_dims.append(i)
        elif isinstance(indices, int_): squeeze_dims = [0]; unsqueeze_dims = []
        else: raise TypeError(f"Invalid indices = {indices} for tensor indexing. ")
        if squeeze_dims and all_(y - x == 1 for x, y in zip(squeeze_dims[:-1], squeeze_dims[1:])):
            new_shape = self.shape[:squeeze_dims[0]] + broadcast(*shapes) + self.shape[squeeze_dims[-1]+1:]
        else: new_shape = broadcast(*shapes) + remove_dim(self.shape, squeeze_dims)
        for i, x in unsqueeze_dims:
            new_shape = new_shape[:i] + x + new_shape[i:]
        with torch._C.DisableTorchFunction():
            return Tensor.inherit_from(torch.Tensor.__getitem__(self, indices), self, shape=new_shape).grad_fn_name_("indexing")
            # return Tensor._make_subclass(Tensor, super().__getitem__(indices).as_subclass(torch.Tensor), ref_size=new_shape).grad_fn_name_("indexing")
    
    def __iter__(self):
        i = 0
        while True:
            try: yield self[i]
            except IndexError: break
            i += 1
        
    # Manipulate dimensions.
    @property
    def T(self: 'Tensor', dim: linalg_dim[1:]=None):
        shape = self.shape
        dim = linalg_dim[1:](self.shape, dim)
        dim_order = [i if i not in dim else dim[-dim.index(i) - 1] for i in range_(self.n_dim)]
        unsq_dim = self.shape[dim[0]:dim[0]+1].with_dim_size(0, dim[0]).python_repr
        if len(dim) == 1: return self.permute(*dim_order).unsqueeze(unsq_dim).grad_fn_name_('transpose')
        return self.permute(*dim_order).grad_fn_name_('transpose')
    
    def t_(self: 'Tensor', dim: linalg_dim[1:]=None):
        shape = self.shape
        dim = linalg_dim[1:](self.shape, dim)
        dim_order = [i if i not in dim else dim[-dim.index(i) - 1] for i in range_(self.n_dim)]
        unsq_dim = self.shape[dim[0]:dim[0]+1].with_dim_size(0, dim[0]).python_repr
        if len(dim) == 1: return self.permute(*dim_order).unsqueeze(unsq_dim).grad_fn_name_('transpose')
        return self.permute_(*dim_order).grad_fn_name_('transpose_')
    
    def permute_(self: 'Tensor', *dims: exist_dim):
        dims = exist_dim(self, *dims)
        avouch(len(dims) == self.ndim, RuntimeError(f"permute_(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = {self.n_dim} is not equal to len(dims) = {len(dims)}"))
        cur_order = list(range_(self.ndim))
        special_shape = self.shape
        self.init_special()
        for i in range_(len(cur_order)):
            j = cur_order.index(dims[i])
            if j == i: continue
            cur_order[i], cur_order[j] = cur_order[j], cur_order[i]
            self.transpose_(i, j)
        return self.special_from(special_shape.permute(*dims)).grad_fn_name_('permute_')

    @alias("movedim")
    def move_dim(self, dim1: del_dim, dim2: exist_dim):
        """
        movedim(self, dim1, dim2) -> Tensor

        move dim1 to dim2 (specified in the targeting size)
        data of size (2, 3, 4, 5) can be transform to (2, 4, 5, 3) by data.movedim(1, -1) or data.movedim(1, 3)
        """
        ref_shape = Size(dim2)
        dim1 = del_dim(self.shape, dim1)
        dim2 = exist_dim(self.shape, dim2)
        avouch(len(dim1) == len(dim2) or len(dim2)== 1, "Tensor.move_dim only takes dimension of same size or one target dim.")
        if len(dim2) == 1:
            d2 = dim2[0]
            if all_(d > d2 for d in dim1):
                dimensions = list(range_(d2)) + list(dim1) + [i for i in range_(d2, self.n_dim) if i not in dim1]
            else:
                dimensions = [i for i in range_(d2+1) if i not in dim1] + list(dim1) + [i for i in range_(d2+1, self.n_dim) if i not in dim1]
            res = self.permute(*dimensions).add_special_dim(d2, dim2)
        else:
            dimensions = [0] * self.n_dim
            assigned = [False] * self.n_dim
            for i in dim1:
                j = dim2[dim1.index(i)]
                dimensions[j] = i
                assigned[j] = True
            for i in range_(self.n_dim):
                if i in dim1: continue
                j = assigned.index(False)
                dimensions[j] = i
                assigned[j] = True
            avouch(all_(assigned), RuntimeError(f"Not permute for dimension move if dim1={dim1} and dim2={dim2}. "))
            res = self.permute(*dimensions)
            for i in range_(len(dim2)):
                res.add_special_dim(dim2[i], dim2[i:i+1])
        return res.grad_fn_name_('move_dim')
            
        # d1 = dim1[0]
        # d2 = dim2[0]

        # if d1 < d2: return self.permute(*range(d1), *range(d1+1, d2+1), d1, *range(d2+1, self.n_dim)).add_special_dim(d2, dim2)
        # elif d1 > d2: return self.permute(*range(d2), d1, *range(d2, d1), *range(d1+1, self.n_dim)).add_special_dim(d2, dim2)
        # return self.add_special_dim(d2, dim2).grad_fn_name_('move_dim')

    @alias("movedim_")
    def move_dim_(self, dim1: del_dim, dim2: exist_dim):
        """
        In-place operation for movedim
        """
        dim1 = del_dim(self.shape, dim1)
        dim2 = exist_dim(self.shape, dim2)
        avouch(len(dim1) == len(dim2) == 1, "Tensor.move_dim only takes integers as inputs.")
        d1 = dim1[0]
        d2 = dim2[0]

        if d1 < d2: return self.permute_(*range(d1), *range(d1+1, d2+1), d1, *range(d2+1, self.n_dim)).add_special_dim(d2, dim2)
        elif d1 > d2: return self.permute_(*range(d2), d1, *range(d2, d1), *range(d1+1, self.n_dim)).add_special_dim(d2, dim2)
        return self.add_special_dim(d2, dim2).grad_fn_name_('move_dim_')

    @alias("joindims", "join_dims", "mergedims")
    def merge_dims(self, *dims: exist_dim, target: new_dim=None):
        """
        mergedims(self, *source, target) -> Tensor

        merge dims into one dimension: target (the last argument)
        data of size (2, 3, 4, 5) can be transform to (24, 5) with a Cartesian of 3 x 2 x 4 by:
            data.mergedims([1, 0, 2], target=0) / data.mergedims(1, 0, 2, target=0)
        Note that one can only omit the target dimension if no order of dimension is changed. 
            the automatically chosen target is the new position of the last dimension one gives. 
            e.g. data.mergedims(1, 0, 3) result in (4, 30) and it follows a Cartesian of 2 x 3 x 5.
        """
        input_dims = dims
        dims = exist_dim(self.shape, *dims)
        if target is None:
            target_repr = Size(dims[-1] - sum_([1 if d < dims[-1] else 0 for d in dims[:-1]])).update_special_from(Size(input_dims[-1])).python_repr
            dims = sorted(dims)
        else: target_repr = (target,)
        target = new_dim(remove_dim(self.shape, dims), *target_repr)
        avouch(len(dims) >= 2, f"Please input at least two dims to be merged for method 'mergedims', not {dims}. ")
        avouch(len(target) == 1, f"At most one 'target' argument is allowed for method 'mergedims', not {target_repr}. ")

        res = self.clone()
        other_dims = [i for i in range_(self.n_dim) if i not in dims]
        out_dims = other_dims[:target[0]] + dims + other_dims[target[0]:]
        prev_shape = res.shape
        with res.hide_special():
            res.permute_(out_dims)
            res = res.flatten(target[0], target[0] + len(dims) - 1)
        post_shape = sum_((prev_shape[i:i+1] for i in other_dims[:target[0]]), Size())
        post_shape += res.shape.special_from(target)[target[0]:target[0]+1]
        post_shape += sum_((prev_shape[i:i+1] for i in other_dims[target[0]:]), Size())
        return res.special_from(post_shape).grad_fn_name_('merge_dims')

    @alias("splitdim")
    def split_dim(self, source: del_dim, *size: Size):
        """
        splitdim(self, source, *target_size) -> Tensor

        split one dimension source into multiple dimensions: target
        data of size (2, 4, 5) can be transform to (2, 2, 2, 5) with data.splitdim(1, 2, 2).
        Note that batch representations for source and target are different
            (splitdim([1], [2], 2) means split the batchdim at index 1 into a size of ([2], 2), which is 2x2 with batchdim at index 0).
            One can use -1 for arbitrary size. 
        """
        size = Size(*size)
        source = del_dim(self, source)
        # avouch(len(size) >= 2, f"Please input an at-least-two-dim-shape to split dimension {source} into in method 'splitdim', not {size}. ")
        if len(source) > 1: self = self.merge_dims(*source, target=source[0])

        new_size = self.shape[:source[0]] + size.with_n_ele(self.shape[source[0]]) + self.shape[source[0] + 1:]
        return self.view(new_size).grad_fn_name_('split_dim')
    
    def expand(self, *sizes: Size):
        return self.expand_to(*sizes).grad_fn_name_('expand')

    def expand_as(self, other: 'Tensor'):
        return self.expand_to(other).grad_fn_name_('expand_as')
        
    def expand_to(self, *target, assign_to_dims: exist_dim=None, dims_allow_mismatch: exist_dim=None):
        if len(target) == 1 and isinstance(target[0], torch.Tensor): target = target[0].shape
        avouch(isinstance(target, tuple), TypeError(f"Invalid input for bt.Tensor.expand_to: {target}, should be a 'tuple' or 'Size'."))
        if not isinstance(target, Size): target = Size(*target)
        if assign_to_dims is None:
            new_shape, _ = self.shape ^ target
            avouch(len(new_shape) == len(target), TypeError(f"Cannot expand tensor with shape {self.shape} to {target}. "))
        else:
            assign_to_dims = list(exist_dim(target, assign_to_dims))
            new_shape = Size(*(self.shape[assign_to_dims.index[i]] if i in assign_to_dims else 1 for i in range_(len(target))).special_from(target))
        if dims_allow_mismatch is None: dims_allow_mismatch = tuple()
        else: dims_allow_mismatch = tuple(exist_dim(target, dims_allow_mismatch))
        avouch(all_(i in dims_allow_mismatch or x == y or x == 1 or y in (1, -1) for i, (x, y) in enumerate(zip(new_shape, target))), 
               TypeError(f"Size mismatch in 'expand_to': {self.shape} (expanded to {new_shape}) and {target}. "))
        n_repeats = tuple(y if i not in dims_allow_mismatch and x == 1 else 1 for i, (x, y) in enumerate(zip(new_shape, target)))
        if len(n_repeats) > 0:
            return self.view(new_shape).repeat(*n_repeats).grad_fn_name_('expand_to')
        else: return self.view(new_shape).grad_fn_name_('expand_to')
    
    def unsqueeze_to(self: 'Tensor', *target:Size, assign_to_dims: exist_dim=None):
        return self.expand_to(*target, assign_to_dims=assign_to_dims, dims_allow_mismatch=tuple()).grad_fn_name_('unsqueeze_to')

    # Control the output of the tensor. 
    def tobytes(self): return self.detach().cpu().numpy().tobytes()
    
    def __hash__(self): return super().__hash__()
    
    @classmethod
    def __block_repr__(cls, rpr: str, by=' '):
        n_col = max_(len(line) for line in rpr.split('\n'))
        return '\n'.join(l + by * (n_col - len(l)) for l in rpr.split('\n'))
    
    @classmethod
    def __corner_block_repr__(cls, rpr: str, by=' '):
        lines = rpr.split('\n')
        n_col = max_(len(line) for line in lines)
        n_row = len(lines)
        return '\n'.join(
            ('｢' if i == 0 else (' ' if i == n_row - 1 else '|')) + 
            l + by * (n_col - len(l)) + 
            ('｣' if i == n_row - 1 else (' ' if i == 0 else '|'))
        for i, l in enumerate(rpr.split('\n')))
    
    @classmethod
    def __shift_repr__(cls, rpr: str, shift: int=1, ignore_first: bool=True, by=' '):
        if ignore_first: return ('\n' + by * shift).join(rpr.split('\n'))
        return '\n'.join(by * shift + l for l in rpr.split('\n'))
    
    def __raw_repr__(self, cell_format=None):
        criteria = {
            'batch': (1, (1, 0)),
            'sequence': (2, 1),
            '<n-4': (4, 1),
            '<n-2': (6, 2),
            '<n-1': (10, 4),
            '<n': (20, 8)
        }
        cell_len_exp = 8
        cell_len_str = 3

        if cell_format is None:
            if self.n_ele == 0: return "[]"
            # if self.n_dim > 1:
            #     # permute the dimensions to (batch, sequence, space, feature, func) for display. 
            #     dimensions = (
            #         ([self.batch_dim] if self.has_batch else []) + 
            #         (list(range_(*self.sequence_range)) if self.has_sequence else []) + 
            #         (list(range_(*self.space_range)) if self.has_space else []) + 
            #         (list(range_(*self.feature_range)) if self.has_feature else []) + 
            #         ([self.func_dim] if self.has_func else [])
            #     )
            #     self = self.permute(*dimensions)

            display_tensor = None
            for d in range_(self.n_dim):
                if self.is_batch_dim(d): max_size, pad_size = criteria['batch']
                elif self.is_sequence_dim(d): max_size, pad_size = criteria['sequence']
                elif d < self.n_dim - 4: max_size, pad_size = criteria['<n-4']
                elif d < self.n_dim - 2: max_size, pad_size = criteria['<n-2']
                elif d < self.n_dim - 1: max_size, pad_size = criteria['<n-1']
                else: max_size, pad_size = criteria['<n']
                if self.size(d) <= max_size: continue
                if isinstance(pad_size, int_): pad_size = (pad_size, pad_size)
                if display_tensor is None: display_tensor = cat(self[(slice(None),) * d + (slice(None, pad_size[0]),)], self[(slice(None),) * d + (slice(self.size(d)-pad_size[1], None),)], d)
                else: display_tensor = cat(display_tensor[(slice(None),) * d + (slice(None, pad_size[0]),)], display_tensor[(slice(None),) * d + (slice(self.size(d)-pad_size[1], None),)], d)
            if display_tensor is None: display_tensor = self.flatten()
            else: display_tensor = display_tensor.flatten()
            if display_tensor.is_complex(): display_tensor = cat(display_tensor.real, display_tensor.imag)
            str_ele = False
            if any(isnan(display_tensor)):
                display_tensor = display_tensor[~isnan(display_tensor)]
                str_ele = True
            if any(isinf(display_tensor)):
                display_tensor = display_tensor[~isinf(display_tensor)]
                str_ele = True
            if display_tensor.n_ele == 0:
                if not str_ele: raise RuntimeError("str_ele=False after eliminating nan/inf to an empty tensor.")
                cell_format = ('int', cell_len_str + any(self < 0), self.shape)
            elif issubclass(display_tensor.dtype, dtype_(bool)):
                cell_format = ('bool', 4 if all(display_tensor) else 5, self.shape)
            elif not display_tensor.dtype.is_floating_point:
                cell_len = int_(max(display_tensor.clamp(min=1).log(10).floor()).item()) + 1
                cell_len = max_(cell_len, int_(max(display_tensor.clamp(max=-1e-1).abs().log(10).floor()).item()) + 1 + 1)
                cell_len = min_(cell_len, cell_len_exp)
                if str_ele: cell_len = max_(cell_len, cell_len_str)
                cell_format = ('int', cell_len, self.shape)
            else:
                zero_to_one = lambda x: ones(1)[0] if x == 0 else x.log(10)
                if sum((display_tensor.abs() > 1e-64) & (display_tensor.abs() < 1e-4)) > display_tensor.n_ele / 2 or \
                    any(display_tensor >= 1e6) or any(display_tensor <= -1e5): cell_format = ('exp', cell_len_exp, self.shape)
                elif abs(zero_to_one(display_tensor.abs().max()) - zero_to_one(display_tensor.abs().min())).item() > 5: cell_format = ('exp', cell_len_exp, self.shape)
                elif all((display_tensor - display_tensor.round()).abs() < 1e-4) or any(display_tensor >= 1e4) or any(display_tensor <= -1e3):
                    cell_len = int_(max((-display_tensor.sign()).clamp(min=0) + display_tensor.abs().clamp(min=1).log(10).floor()).item()) + 1 + 1
                    # cell_len = max_(cell_len, int_(max(display_tensor.clamp(max=-1e-1).abs().log(10).floor()).item()) + 1 + 1 + 1)
                    cell_len = min_(cell_len, cell_len_exp)
                    if str_ele: cell_len = max_(cell_len, cell_len_str)
                    cell_format = ('.0f', cell_len, self.shape)
                elif all((display_tensor - display_tensor.round(decimals=2)).abs() < 1e-4) or any(display_tensor >= 1e2) or any(display_tensor <= -1e1):
                    cell_len = int_(max((-display_tensor.sign()).clamp(min=0) + display_tensor.abs().clamp(min=1).log(10).floor()).item()) + 1 + 3
                    # cell_len = max_(cell_len, int_(max(display_tensor.clamp(max=-1e-1).abs().log(10).floor()).item()) + 1 + 1 + 3)
                    cell_len = min_(cell_len, cell_len_exp)
                    if str_ele: cell_len = max_(cell_len, cell_len_str)
                    cell_format = ('.2f', cell_len, self.shape)
                else:
                    cell_len = int_(max((-display_tensor.sign()).clamp(min=0) + display_tensor.abs().clamp(min=1).log(10).floor()).item()) + 1 + 5
                    # cell_len = max_(cell_len, int_(max(display_tensor.clamp(max=-1e-1).abs().log(10).floor()).item()) + 1 + 1 + 5)
                    cell_len = min_(cell_len, cell_len_exp)
                    if str_ele: cell_len = max_(cell_len, cell_len_str)
                    cell_format = ('.4f', cell_len, self.shape)
        cell_fname, cell_len, total_size = cell_format
        elps = ("{:^%ds}"%cell_len).format('...')
        if self.n_dim == 0:
            val = self.item()
            def val_str(val):
                if repr(val) == 'nan': return ("{:>%ds}"%cell_len).format("NaN")
                elif repr(val) == 'inf': return ("{:>%ds}"%cell_len).format("Inf")
                elif repr(val) == '-inf': return ("{:>%ds}"%cell_len).format("-Inf")
                elif cell_fname == 'bool': return ("{:^%ds}"%cell_len).format(str(val))
                elif cell_fname == 'int': return ("{:^%dd}"%cell_len).format(val)
                elif cell_fname == '.0f': return ("{:%ds}"%cell_len).format(str(round_(val)) + '.')
                elif cell_fname == '.2f': return ("{:%d.2f}"%cell_len).format(val)
                elif cell_fname == '.4f': return ("{:%d.4f}"%cell_len).format(val)
                elif cell_fname == 'exp':
                    if val == 0: lev = 0
                    else: lev = int_(math.log(abs_(val), 10))
                    base = val / (10 ** lev)
                    lev_str = f'{lev:2d}'.replace(' ', '+') if lev >= 0 else f'{lev:1d}'
                    return f"{base:5.2f}e{lev_str}"
            if isinstance(val, complex_):
                return val_str(val.real) + '+' + val_str(val.imag) + 'i'
            else:
                return val_str(val)
        if self.n_dim == 1:
            max_size, pad_size = criteria['<n']
            if isinstance(pad_size, int_): pad_size = (pad_size, pad_size)
            n_size = self.size(0)
            display_samples = range_(n_size) if n_size <= max_size else list(range_(pad_size[0])) + [...] + list(range_(n_size - pad_size[1], n_size))
            if self.has_func: return f"({' '.join(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format) for i in display_samples)})"
            elif self.has_batch:
                if total_size.n_func_dim + total_size.n_batch_dim == total_size.n_dim:
                    return f"{{{self[0].__raw_repr__(cell_format=cell_format)}, ...}}"
                else: return f"{{{self[0].__raw_repr__(cell_format=cell_format)}}}"
            elif self.has_feature:
                if total_size.n_feature_dim == 1:
                    return Tensor.__corner_block_repr__('\n'.join(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format) for i in display_samples))
                return f"{' '.join(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format) for i in display_samples)}"
            elif self.has_sequence:
                if total_size.n_space_dim > 0: return f">{self[0].__raw_repr__(cell_format=cell_format)}>"
                elif n_size == 1: return f"> {self[0].__raw_repr__(cell_format=cell_format)} >"
                elif n_size == 2: return f"'{self[0].__raw_repr__(cell_format=cell_format)} > {self[-1].__raw_repr__(cell_format=cell_format)}'"
                else: return f"'{self[0].__raw_repr__(cell_format=cell_format)} > ... > {self[-1].__raw_repr__(cell_format=cell_format)}'"
            else: return f"[{', '.join(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format) for i in display_samples)}]"
        if self.n_dim > 4: max_size, pad_size = criteria['<n-4']
        elif self.n_dim > 2: max_size, pad_size = criteria['<n-2']
        else: max_size, pad_size = criteria['<n-1']
        if isinstance(pad_size, int_): pad_size = (pad_size, pad_size)
        n_size = self.size(0)
        display_samples = range_(n_size) if n_size <= max_size else list(range_(pad_size[0])) + [...] + list(range_(n_size - pad_size[1], n_size))
        if self.shape[:1].has_func: return '(' + '\n '.join(' ' + elps if i == ... else Tensor.__shift_repr__(self[i].__raw_repr__(cell_format=cell_format)) for i in display_samples) + ')'
        elif self.shape[:1].has_batch:
            if len(self.shape) <= 2 or len(self.shape) == 3 and self.shape.has_sequence:
                return f"{{{Tensor.__shift_repr__(self[0].__raw_repr__(cell_format=cell_format))}, ...}}"
            return f"{{{Tensor.__shift_repr__(self[0].__raw_repr__(cell_format=cell_format))},\n...}}"
        elif self.shape[:1].has_feature:
            if total_size.n_feature_dim <= 2:
                return Tensor.__corner_block_repr__('\n'.join(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format) for i in display_samples))
            elif total_size.n_dim - len(self.shape) == total_size.feature_stop-1 and total_size.sz_feature_dim < 0 and not self.shape[1:].has_func:
                if total_size.n_feature_dim > 1:
                    return f"{' '.join(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format) for i in display_samples)}"
                return f"[{' '.join(elps if i == ... else Tensor.__shift_repr__(self[i].__raw_repr__(cell_format=cell_format)) for i in display_samples)}]"
            else: return '｢' + '\n '.join(' ' + elps if i == ... else Tensor.__shift_repr__(self[i].__raw_repr__(cell_format=cell_format)) for i in display_samples) + '｣'
        elif self.shape[:1].has_sequence:
            if total_size.n_dim - len(self.shape) == total_size.sequence_stop-1 and total_size.sz_sequence_dim < 0:
                return f">{Tensor.__shift_repr__(self[0].__raw_repr__(cell_format=cell_format))}>"
            elif n_size == 1: return '> ' + Tensor.__shift_repr__(self[0].__raw_repr__(cell_format=cell_format), 3) + ' >'
            else:
                item1 = Tensor.__shift_repr__(self[0].__raw_repr__(cell_format=cell_format))
                itemn = Tensor.__shift_repr__(self[-1].__raw_repr__(cell_format=cell_format))
                if n_size == 2: return "'''>" + item1 + '\n ' + itemn + ">'''"
                else: return "'''" + item1 + '\n ' + ' ' * re.search(r'\d', item1).span()[0] + 'v...v\n ' + itemn + "'''"
        else:
            if total_size.n_dim - len(self.shape) == total_size.space_stop-1:
                columns = []
                n_row = None
                for i in display_samples:
                    if i == ...: columns.append('\n'.join([' ' * cell_format[1]] * (n_row - 1) + [elps])); continue
                    columns.append(Tensor.__block_repr__(self[i].__raw_repr__(cell_format=cell_format)))
                    if n_row is None: n_row = len(columns[0].split('\n'))
                return '[' + Tensor.__shift_repr__('\n'.join((', ' if i == n_row - 1 else '  ').join(rows) for i, rows in enumerate(zip(*[c.split('\n') for c in columns])))) + ']'
            return '[' + ',\n '.join(Tensor.__shift_repr__(elps if i == ... else self[i].__raw_repr__(cell_format=cell_format)) for i in display_samples) + ']'
    
    @property
    def mean_std(self):
        return f"{self.mean().detach().cpu().item():.6f} ± {self.std().detach().cpu().item():.6f}"
    
    def __str__(self):
        if not hasattr(self, 'sz_func_dim'):
            raise RuntimeError(f"Not initialized batorch.Tensor of shape: {self.as_subclass(torch.Tensor).shape}")
        if self.n_dim == 0:
            val = self.item()
            if not self.dtype.is_floating_point: return str(val)
            elif abs_(val - self.floor().item()) < 1e-4: return f"{int_(val)}."
            elif repr(val) == 'nan': return "NaN"
            elif repr(val) == 'inf': return "Inf"
            return "%6.4f"%self.item()
        prefix = "Tensor("
        parts = [Tensor.__shift_repr__(self.__raw_repr__(), len(prefix))]
        raw_shape_str = str(self.shape).split('Size')[-1]
        parts.append(f"shape={raw_shape_str}")
        if self.device.type != 'cpu':    
            device_str = 'cpu' if self.device.type == 'cpu' else f"{self.device.type}:{self.device.index}"
            parts.append(f"device=[{device_str}]")
        if self.grad_fn: parts.append(f"grad_fn={self.grad_fn}")
        if self.requires_grad: parts.append(f"requires_grad={self.requires_grad}")
        return prefix + ', '.join(parts) + ')'

    def __repr__(self):
        if not hasattr(self, 'sz_func_dim'):
            raise RuntimeError(f"<Not initialized batorch.Tensor of shape: {self.as_subclass(torch.Tensor).shape}>")
        num_nans = isnan(self).sum()
        num_infs = isinf(self).sum()
        special_dim_str = f"[{self.n_special_dim}]+" if self.n_special_dim > 0 else ''
        device_str = 'cpu' if self.device.type == 'cpu' else f"{self.device.type}:{self.device.index}"
        raw_shape_str = str(self.shape).split('Size')[-1]
        valid_nums = self.flatten()
        valid_nums = valid_nums[~isnan(valid_nums)]
        valid_nums = valid_nums[~isinf(valid_nums)]
        if valid_nums.n_ele == 0:
            valid_val_str = 'nothing'
        elif self.is_complex():
            valid_val_str = f"Re(min:{valid_nums.real.min()}, med:{valid_nums.real.median()}, max:{valid_nums.real.max()}), "
            valid_val_str += f"Im(min:{valid_nums.imag.min()}, med:{valid_nums.imag.median()}, max:{valid_nums.imag.max()})"
        elif self.dtype == bool:
            if valid_nums.min() == valid_nums.max():
                valid_val_str = f"{valid_nums.min()}"
            else: valid_val_str = "True, False"
        else:
            valid_val_str = f"min:{valid_nums.min()}, med:{valid_nums.median()}, max:{valid_nums.max()}"
        nan_str = ''; inf_str = ''
        if num_nans > 0: nan_str = f"{num_nans} NaN"
        if num_nans > 1: nan_str += 's'
        if num_infs > 0: inf_str = f"{num_infs} Inf"
        if num_infs > 1: inf_str += 's'
        error_val_str = ', '.join([x for x in (nan_str, inf_str) if x])
        if num_nans + num_infs == 0: val_range_str = valid_val_str
        elif (num_nans + num_infs) / self.n_ele < 0.5: val_range_str = f"{valid_val_str}, {error_val_str}"
        else: val_range_str = error_val_str
        return f"<{special_dim_str}{self.n_space_dim}D {str(self.dtype).split('.')[-1]} Tensor on {device_str}: shape={raw_shape_str}, requires_grad={self.requires_grad}, val=[{val_range_str}]>"
    
    ## Other utilities
    def byte_size(self):
        return ByteSize(self.element_size() * self.numel())

    def rename(self, *args, **kwargs):
        with torch._C.DisableTorchFunction():
            output = Tensor.inherit_from(torch_super(self, 'rename')(*args, **kwargs), self)
        for i, n in enumerate(output.names):
            if n is None: continue
            if 'func' in n.lower(): output.add_special_dim(i, func_dim)
            if 'batch' in n.lower(): output.add_special_dim(i, {})
            elif 'channel' in n.lower() or 'feature' in n.lower(): output.add_special_dim(i, [])
            elif 'time' in n.lower() or 'series' in n.lower() or 'sequence' in n.lower(): output.add_special_dim(i, '')
        return output.grad_fn_name_('rename')

    def refine_names(self, *args):
        with torch._C.DisableTorchFunction():
            output = Tensor.inherit_from(torch_super(self, 'refine_names')(*args), self)
        for i, n in enumerate(output.names):
            if n is None: continue
            if 'func' in n.lower(): output.add_special_dim(i, func_dim)
            if 'batch' in n.lower(): output.add_special_dim(i, {})
            elif 'channel' in n.lower() or 'feature' in n.lower(): output.add_special_dim(i, [])
            elif 'time' in n.lower() or 'series' in n.lower() or 'sequence' in n.lower(): output.add_special_dim(i, '')
        return output.update_special_from(self)

    def normalize_(self):
        m, M = self.min(), self.max()
        if m == M:
            if M >= 1: return self.zero_().add_(1)
            if m <= 0: return self.zero_()
            return self
        self.sub_(m)
        self.div_(M-m)
        return self.grad_fn_name_('normalize_')

    def normalize(self):
        m, M = self.min(), self.max()
        if m == M:
            if M >= 1: return ones_like(self)
            if m <= 0: return zeros_like(self)
            return self
        return ((self - m) / (M - m)).grad_fn_name_('normalize')

    @alias('extend')
    def append(self, value):
        avouch(self.n_dim == 1, "Only 1-dimensional tensor can use 'append' to concat. ")
        tensor_value = tensor(value, device=self.device, dtype=self.dtype) if not isinstance(value, torch.Tensor) else (value.as_subclass(Tensor, device=self.device, dtype=self.dtype) if not isinstance(value, Tensor) else value)
        avouch(tensor_value.n_dim <= 1, "Only scalar or 1-dimensional tensors can by concat using append/extend. ")
        return cat(self, tensor_value, dim=0).grad_fn_name_('append')
    
    def setitem_(self, ind, val):
        self[ind] = val
        return self.grad_fn_name_('setitem_')
    
    def grad_fn_name_(self, name):
        self.grad_fn_name = name
        return self
    
    @alias('concatenate')
    def cat(self, *other, dim=0):
        return cat(self, *other, dim=dim).grad_fn_name_('cat')

    def stack(self, *other, dim=0):
        return stack(self, *other, dim=dim).grad_fn_name_('stack')
    
    ## dtypes
    @alias("as_type")
    def astype(self, dt):
        """
            numpy dtype v.s. torch dtype:
            ==============================
            numpy type // torch type
            ------------------------------
            void0, void::void // 
            object0, object_::object // 
            bool8, bool_::bool // torch.bool
            byte, int8::int8 // torch.int8
            short, int16::int16 // torch.short, torch.int16
            int32, intc::int32 // torch.int, torch.int32
            int0, int_, int64, intp, longlong, signedinteger::int64 // torch.long, torch.int64
            ubyte, uint8::uint8 // torch.uint8
            ushort, uint16::uint16 // 
            uint32, uintc::uint32 // 
            uint, uint0, uint64, Uint64, uintp, ulonglong::uint64 // 
            // torch.bfloat16 # 16bit, 范围大如32bit但精度低
            half, float16::float16 // torch.half, torch.float16
            single, float32::float32 // torch.float, torch.float32
            double, float64, float_, longdouble, longfloat, number::float64 // torch.double, torch.float64
            // torch.complex32
            csingle, complex64, singlecomplex::complex64 // torch.cfloat, torch.complex64
            cdouble, cfloat, clongdouble, clongfloat, complex_, complex128, longcomplex::complex128 // torch.cdouble, torch.complex128
            str0, str_, Str0::str // 
            bytes0, bytes_, string_::bytes // 
            datetime64::datetime64 // 
            timedelta64::timedelta64 // 
            # 量子计算类型
            // torch.qint8
            // torch.qint32
            // torch.quint8
            // torch.quint4x2
        """
        torch_dt = to_torch_dtype(dt)
        with torch._C.DisableTorchFunction():
            return Tensor.inherit_from(torch_super(self, 'type')(torch_dt), self).grad_fn_name_('astype')
        # if isinstance(dt, str): return Tensor.inherit_from(super().type(dt.replace('bt.', 'torch.')), self, shape=...)
        # if hasattr(dt, 'dtype'): dt = dt.dtype
        # if isinstance(dt, torch.dtype): return Tensor.inherit_from(super().type(dt), self, shape=...)
        # import numpy as np
        # dt_name = np.dtype(dt).name
        # dtype_map = {'uint16': "int32", 'uint32': "int64", 'uint64': "int64"}
        # torch_dt = getattr(torch, dtype_map.get(dt_name, dt_name), None)
        # avouch(torch_dt is not None, f"Invalid dtype {dt}: {dt_name} cannot be converted into torch dtype.")
        # return Tensor.inherit_from(super().type(torch_dt), self, shape=...)

    def type(self, dt=None):
        with torch._C.DisableTorchFunction():
            if dt is None: return torch_super(self, 'type')().replace("torch.", "batorch.")
            else: return self.astype(dt)

    def __getattribute__(self, key):
        with torch._C.DisableTorchFunction():
            g = super(Tensor, self).__getattribute__(key)
        if key in ('grad', 'real', 'imag'):
            if not isinstance(g, torch.Tensor): return g
            return Tensor.inherit_from(g, self)
        elif key == 'grad_fn':
            if g is None: return g
            class gfn:
                def __init__(self, fn, fn_name): self.fn = fn; self.fn_name = fn_name
                def __call__(self, *args, **kwargs): return self.fn(*args, **kwargs)
                def __getattribute__(self, key):
                    if key in ('fn', 'fn_name', '__init__', '__call__', '__getattribute__', '__repr__'):
                        return super().__getattribute__(key)
                    attr = getattr(self.fn, key, None)
                    if attr is not None: return attr
                    return super().__getattribute__(key)
                def __repr__(self): return f"<backward: {self.fn_name} at 0x{id(self.fn):0x}>"
            return gfn(g, getattr(self, 'grad_fn_name', 'Unknown'))
        return g
    
    def __setattr__(self, key, value):
        if key in ('grad', 'real', 'imag'):
            if isinstance(value, Tensor):
                value = value.as_subclass(torch.Tensor)
        super().__setattr__(key, value)
        
    def as_func_tensor(self):
        avouch(self.n_dim == 1, TypeError("Only 1D tensor can be converted to functional tensor. "))
        self.init_special()
        self.sz_func_dim = self.n_dim
        return self

    def as_batch_tensor(self):
        avouch(self.n_dim == 1, TypeError("Only 1D tensor can be converted to batch tensor. "))
        self.init_special()
        self.sz_batch_dim = self.n_dim
        return self

    def as_feature_tensor(self):
        self.init_special()
        self.sz_feature_dim = self.n_dim
        return self

    def as_sequence_tensor(self):
        self.init_special()
        self.sz_sequenc_dim = self.n_dim
        return self

    def auto_device(self):
        global _device
        return self.to(_device.main_device)
    
    def numpy(self):
        return torch.Tensor.numpy(super().detach().cpu())
    
    ### START METHOD AUTO GENERATION
    # operators
    def __add__(self: 'Tensor', other: 'Tensor'): ...
    def __iadd__(self: 'Tensor', other: 'Tensor'): ...
    def __sub__(self: 'Tensor', other: 'Tensor'): ...
    def __isub__(self: 'Tensor', other: 'Tensor'): ...
    def __mul__(self: 'Tensor', other: 'Tensor'): ...
    def __imul__(self: 'Tensor', other: 'Tensor'): ...
    def __div__(self: 'Tensor', other: 'Tensor'): ...
    def __idiv__(self: 'Tensor', other: 'Tensor'): ...
    def __pow__(self: 'Tensor', other: 'Tensor'): ...
    def __ipow__(self: 'Tensor', other: 'Tensor'): ...
    def __mod__(self: 'Tensor', other: 'Tensor'): ...
    def __imod__(self: 'Tensor', other: 'Tensor'): ...
    def __truediv__(self: 'Tensor', other: 'Tensor'): ...
    def __itruediv__(self: 'Tensor', other: 'Tensor'): ...
    def __floordiv__(self: 'Tensor', other: 'Tensor'): ...
    def __ifloordiv__(self: 'Tensor', other: 'Tensor'): ...
    def __neg__(self: 'Tensor'): ...
    def __eq__(self: 'Tensor', other: 'Tensor'): ...
    def __ne__(self: 'Tensor', other: 'Tensor'): ...
    def __or__(self: 'Tensor', other: 'Tensor'): ...
    def __ior__(self: 'Tensor', other: 'Tensor'): ...
    def __and__(self: 'Tensor', other: 'Tensor'): ...
    def __iand__(self: 'Tensor', other: 'Tensor'): ...
    def __xor__(self: 'Tensor', other: 'Tensor'): ...
    def __ixor__(self: 'Tensor', other: 'Tensor'): ...
    def __invert__(self: 'Tensor'): ...
    def __lt__(self: 'Tensor', other: 'Tensor'): ...
    def __le__(self: 'Tensor', other: 'Tensor'): ...
    def __gt__(self: 'Tensor', other: 'Tensor'): ...
    def __ge__(self: 'Tensor', other: 'Tensor'): ...
    def __matmul__(self: 'Tensor', other: 'Tensor'): ...
    
    # reversed operators
    def __radd__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__add__')(self)
    def __rsub__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__sub__')(self)
    def __rmul__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__mul__')(self)
    def __rdiv__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__div__')(self)
    def __rpow__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__pow__')(self)
    def __rmod__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__mod__')(self)
    def __rtruediv__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__truediv__')(self)
    def __rfloordiv__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__floordiv__')(self)
    def __ror__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__or__')(self)
    def __rand__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__and__')(self)
    def __rxor__(self: 'Tensor', other: 'Tensor'):
        self_shape, other_shape = self_shape ^ other_shape
        self = self.view(self_shape)
        other = other.view(other_shape)
        ref_shape = self_shape
        with torch._C.DisableTorchFunction():
            return torch_super(other, '__xor__')(self)
    
    # inplace methods: operations
    def add_(self: 'Tensor', other: 'Tensor', *, alpha=1): ...
    def sub_(self: 'Tensor', other: 'Tensor', *, alpha=1): ...
    def multiply(self: 'Tensor', value: 'Tensor'): ...
    def mul_(self: 'Tensor', value: 'Tensor'): ...
    def div_(self: 'Tensor', value: 'Tensor', *, rounding_mode=None): ...
    def pow_(self: 'Tensor', other: 'Tensor'): ...
    def fmod_(self: 'Tensor', other: 'Tensor'): ...

    # inplace methods: initializers
    def zero_(self: 'Tensor'): ...
    def one_(self): return self.fill_(1) # suppress: special_from
    def fill_(self: 'Tensor', value): ...
    def normal_(self: 'Tensor', mean=0, std=1, *, generator=None): ...

    # inplace methods: dimension manipulations
    def unsqueeze_(self: 'Tensor', *dims: new_dim[...]): ...
    def squeeze_(self: 'Tensor', *dims: del_dim[...]):
        valid_dims = []
        with torch._C.DisableTorchFunction():
            for d in dims[::-1]:
                if self.size(d) == 1:
                    valid_dims.append(d)
                    torch_super(self, 'squeeze_')(d)
        dims = tuple(valid_dims)
        return self
    def transpose_(self: 'Tensor', dim0: exist_dim[1], dim1: exist_dim[1]): ...
    
    # properties
    @collect_memory
    def to(self: 'Tensor', *args, **kwargs): ...
    def clone(self: 'Tensor', *, memory_format=torch.preserve_format): ...
    def int(self: 'Tensor', memory_format=torch.preserve_format): ...
    def long(self: 'Tensor', memory_format=torch.preserve_format): ...
    def float(self: 'Tensor', memory_format=torch.preserve_format): ...
    def double(self: 'Tensor', memory_format=torch.preserve_format): ...
    def cpu(self: 'Tensor'): ...
    def cuda(self: 'Tensor'): ...

    # shapes:
    def reshape(self, *size: Size): ...
    def reshape_as(self, other: 'Tensor'): ...
    def view(self, *size: Size):
        with torch._C.DisableTorchFunction():
            return torch_super(self, 'view')(size)
    def view_as(self, other: 'Tensor'): ...
    def where(self: 'Tensor', condition: 'Tensor', other: 'Tensor'=None, *, equals: 'Tensor'=None):
        if equals is None:
            ref_shape = broadcast(self_shape, condition_shape, other_shape, with_size_updates=True)
            self = self.view(ref_shape.updated_sizes[0])
            condition = condition.view(ref_shape.updated_sizes[1])
            other = other.view(ref_shape.updated_sizes[2])
            with torch._C.DisableTorchFunction():
                obj = torch_super(self, 'where')(condition, other)
            return Tensor.inherit_from(obj, self, shape=ref_shape)
        else:
            ref_shape = broadcast(self_shape, condition_shape, equals_shape, with_size_updates=True)
            self = self.view(ref_shape.updated_sizes[0])
            condition = condition.view(ref_shape.updated_sizes[1])
            equals = equals.view(ref_shape.updated_sizes[2])
            with torch._C.DisableTorchFunction():
                obj = torch_super(self, 'where')(~condition, equals)
            return Tensor.inherit_from(obj, self, shape=ref_shape)
    def down_scale(self: 'Tensor', factor=1):
        return self[(slice(None),) * self.space_start + (slice(None, None, factor),) * self.n_space_dim] # suppress: special_from
    def up_scale(self: 'Tensor', factor=1):
        for d in range_(*self.space_range):
            self = self.amplify(factor, d)
        return self # suppress: special_from
    ### STOP METHOD AUTO GENERATION
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        try:
            # if Tensor in types and cls != Tensor: return NotImplemented
            
            self = args[0] if len(args) > 0 else None
            # if func.__qualname__.startswith("_VariableFunctionsClass"): # basic functions 'torch.*'
            #     if func.__name__ in globals(): self = None

            # if func.__qualname__.startswith("_TensorBase"): # tensor functions 'torch.Tensor;*'
            #     if hasattr(Tensor, func.__name__, func.__name__): self = None

            types = tuple(cls if t.__name__ == "Parameter" else t for t in types)
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, torch.Tensor):
                avouch(isinstance(ret, cls), RuntimeError(f"Error in having return value not of subclass '{cls.__name__}', this should be done by PyTorch >= 1.6. "))
                if hasattr(ret, 'sz_func_dim'): return ret
                ret.init_special()
                if isinstance(self, Tensor) and ret.n_dim == self.n_dim: ret.special_from(self)
            
            return ret
            
        except Exception as e:
            raise e.__class__(f"[In function {func.__qualname__}]\t" + str(e))

def expand(self: 'Tensor', *sizes: Size): return self.expand(*sizes)
def expand_as(self: 'Tensor', other: 'Tensor'): return self.expand_as(other)
def expand_to(self: 'Tensor', *target, assign_to_dims: exist_dim=None, dims_allow_mismatch: exist_dim=None): return self.expand_to(*target, assign_to_dims=assign_to_dims, dims_allow_mismatch=dims_allow_mismatch)

### START GLOBAL AUTO GENERATION
def complex(real: 'Tensor', imag: 'Tensor', *, out=None): ...
def tensor(data, *, dtype=None, device=_device.main_device, requires_grad=False, pin_memory=False): ...
def as_tensor(data: 'Tensor', dtype=None, device=None): ...
@collect_memory
def empty(*size: Size, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format): ...
@collect_memory
def full(*size, fill_value=None, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False):
    if len(size) == 2 and isinstance(size[0], tuple): size, fill_value = size[0]
    elif len(size) == 1 and isinstance(size[0], tuple): size = size[0]
    if not isinstance(size, Size): size = Size(size)
    if fill_value is None: fill_value = 0
    with torch._C.DisableTorchFunction():
        res = torch.full(size, fill_value=fill_value, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    return res.as_subclass(Tensor).special_from(size)
@collect_memory
def ones(*size: Size, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False): ...
@collect_memory
def zeros(*size: Size, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False): ...
@collect_memory
def empty_like(input: 'Tensor', *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def full_like(input: 'Tensor', fill_value=0, *, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def ones_like(input: 'Tensor', *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def zeros_like(input: 'Tensor', *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...

@collect_memory
def rand(*size: Size, generator=None, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, pin_memory=False): ...
@collect_memory
def randn(*size: Size, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, pin_memory=False): ...
@collect_memory
def rand_like(input: 'Tensor', *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def randn_like(input: 'Tensor', *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def randperm(*n: Size, generator=None, out=None, dtype=torch.int64,layout=torch.strided, device=_device.main_device, requires_grad=False, pin_memory=False):
    avouch(n.n_space_dim == 1, TypeError("'torch.randperm' only accepts 1 space dimension for permutation. "))
    n_batch = (n.n_batch if n.has_batch else 1) * (n.n_feature if n.has_feature else 1) * (n.n_sequence if n.has_sequence else 1)
    with torch._C.DisableTorchFunction():
        result = stack([torch.randperm(*n.space,generator=generator,out=out,dtype=dtype,layout=layout,device=device,requires_grad=requires_grad,pin_memory=pin_memory).as_subclass(Tensor).init_special() for _ in range_(n_batch)], {})
    if n_batch == 0: result = zeros({0}, n.space[0]).long()
    return result.split_dim({}, n.with_space()).move_dim(-1, n.space_start)
@collect_memory
def arange(*start_stop_step, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False):
    start_stop_step = Size(*start_stop_step)
    with torch._C.DisableTorchFunction():
        obj = torch.arange(*start_stop_step,out=out,dtype=dtype,layout=layout,device=device,requires_grad=requires_grad)
    obj = obj.as_subclass(Tensor).special_from(start_stop_step[:1]) # suppress: special_from

def where(condition: 'Tensor', input: 'Tensor', other: 'Tensor', *, out=None): ...

def reshape(self, *size: Size): ...

@alias('concatenate')
def cat(*tensors, dim=None, crop=False, out=None):
    """
    Concatenate tensors along dimension `dim`. 

    Args:
        dim (int/exist_dim, optional): The dimension for concatenation. 
            Defaults to auto concatenation at
            (1) the first feature dimension if tensors have feature;
            (2) the batch dimension if tensors have batch;
            (3) the first sequence dimension if tensors have sequence;
            (4) the first spacial dimension otherwise.
        crop (bool): Whether to crop the sizes automatically when necessary. 
    """
    from .tensorfunc import crop_as
    if len(tensors) == 0: return tensor([])
    elif len(tensors) == 1 and isinstance(tensors[0], (list, tuple)): tensors = tuple(tensors[0])
    elif len(tensors) == 2 and isinstance(tensors[0], (list, tuple)) and isinstance(tensors[1], exist_dim):
        avouch(dim is None, TypeError(f"Cannot concatenate tensors by multiple dimensions."))
        dim = tensors[1]
        tensors = tuple(tensors[0])
    elif len(tensors) > 2 and isinstance(tensors[-1], exist_dim):
        avouch(dim is None, TypeError(f"Cannot concatenate tensors by multiple dimensions."))
        dim = tensors[-1]
        tensors = tuple(tensors[:-1])
    avouch(all_(isinstance(x, torch.Tensor) for x in tensors), TypeError(f"'bt.cat' can only concatenate torch.Tensor objects. "))
    if len(tensors) == 0: return tensor([])
    pivot = tensors[argmax_([x.n_dim for x in tensors])[0]]
    if dim is None:
        if not isinstance(pivot, Tensor): dim = (0,)
        elif pivot.has_feature: dim = exist_dim(pivot, [0])
        elif pivot.has_batch: dim = exist_dim(pivot, {})
        elif pivot.has_sequence: dim = exist_dim(pivot, '0')
        else: dim = (0,)
    else: dim = exist_dim(pivot, dim)
    avouch(len(dim) == 1, TypeError(f"Cannot concat tensors in dimensions {dim}, please flatten them first or use '[0]' and "'0'" to specify the first feature/sequence dimension."))
    dim = dim[0]
    
    if crop: dims_allow_mismatch = (dim,) + tuple(range_(pivot.space_start, pivot.space_stop))
    else: dims_allow_mismatch = dim
    try: tensors = [x.expand_to(pivot, dims_allow_mismatch=dims_allow_mismatch) for x in tensors if x.n_ele > 0]
    except TypeError as e:
        if "Cannot expand tensor" in str(e) or "Size mismatch in 'expand_to'" in str(e):
            raise TypeError("Tensors can only be concatenated when all of them have a same shape except for one dimension. " + f"Currently: {[x.shape for x in tensors]}")
        else: raise e
    if crop: tensors = [x if x.shape[:dim] == pivot.shape[:dim] and x.shape[dim+1:] == pivot.shape[dim+1:] else crop_as(x, pivot.space) for x in tensors]
    
    bt_tensors = [t for t in tensors if isinstance(t, Tensor)]
    with torch._C.DisableTorchFunction():
        if len(bt_tensors) == 0: return torch.cat(tensors, dim, out=out).as_subclass(Tensor)
        else: return Tensor.inherit_from(torch.cat(tensors, dim, out=out), bt_tensors[0])

def stack(*tensors, dim=None, crop=False, out=None):
    """
    Stack tensors along a new dimension `dim`. 

    Args:
        dim (int/new_dim, optional): The dimension for stacking. 
            Defaults to auto stack at
            (1) a new batch dimension if tensors have no batch;
            (2) a new feature dimension if tensors have batch dimension;
        crop (bool): Whether to crop the sizes automatically when necessary. 
    """
    from .tensorfunc import crop_as
    if len(tensors) == 0: return tensor([])
    elif len(tensors) == 1 and isinstance(tensors[0], (list, tuple)): tensors = tuple(tensors[0])
    elif len(tensors) == 2 and isinstance(tensors[0], (list, tuple)) and isinstance(tensors[1], new_dim):
        avouch(dim is None, TypeError(f"Cannot stack tensors by multiple dimensions."))
        dim = tensors[1]
        tensors = tuple(tensors[0])
    elif len(tensors) > 2 and isinstance(tensors[-1], new_dim):
        avouch(dim is None, TypeError(f"Cannot stack tensors by multiple dimensions."))
        dim = tensors[-1]
        tensors = tuple(tensors[:-1])
    avouch(all_(isinstance(x, torch.Tensor) for x in tensors), TypeError(f"'bt.stack' can only stack torch.Tensor objects. "))
    if len(tensors) == 0: return tensor([])
    pivot = tensors[argmax_([x.n_dim for x in tensors])[0]]
    if dim is None:
        if not isinstance(pivot, Tensor): dim = new_dim(pivot, {})
        elif not pivot.has_batch: dim = new_dim(pivot, {})
        else: dim = new_dim(pivot, [pivot.non_bat_start])
    else: dim = new_dim(pivot, dim)
    avouch(len(dim) == 1, TypeError(f"Cannot concat tensors in dimensions {dim}, please flatten them first or use '[0]' and "'0'" to specify the first feature/sequence dimension."))
    
    if crop: dims_allow_mismatch = tuple(range_(pivot.space_start, pivot.space_stop))
    else: dims_allow_mismatch = None
    try: tensors = [x.expand_to(pivot, dims_allow_mismatch=dims_allow_mismatch) for x in tensors if x.n_ele > 0]
    except TypeError as e:
        if "Cannot expand tensor" in str(e) or "Size mismatch in 'expand_to'" in str(e):
            raise TypeError("Tensors can only be stacked when all of them have a same shape. " + f"Currently: {[x.shape for x in tensors]}")
        else: raise e
    if crop: tensors = [x if x.shape == pivot.shape else crop_as(x, pivot.space) for x in tensors]
    
    bt_tensors = [t for t in tensors if isinstance(t, Tensor)]
    with torch._C.DisableTorchFunction():
        if len(bt_tensors) == 0: return torch.stack(tensors, dim[0], out=out).as_subclass(Tensor)
        else: return Tensor.inherit_from(torch.stack(tensors, dim[0], out=out), bt_tensors[0], shape=dim)

def meshgrid(*tensors, indexing: str = None):
    """
    Create the mesh grid using 1D tensors. 

    Args:
        tensors (tuple of Tensors): The tensors used for mesh grid. 
            output[j][i_0, ..., i_{k-1}] = tensors[j][i_{j}],
            e.g. output_0, output_1 = meshgrid(arange(2), arange(3), indexing='ij') =>
                output_0 = Tensor([[0, 0, 0],
                                   [1, 1, 1]])
                output_1 = Tensor([[0, 1, 2],
                                   [0, 1, 2]])
        indexing (str, optional): The indexing criteria.
            indexing = 'ij' means the index for an element goes as (i_row, j_column).
            indexing = 'xy' means the index for an element goes as (x+_coordinate (in column), y-_coordinate (in row)),
                note that y_coordinate=0 for the first row and increases for lower rows.
            Altering indexing from 'ij' and 'xy' will result in a transpose of results.
            Defaults to 'ij' in PyTorch < 1.10 and 'xy' in future versions (following PyTorch).
    """
    avouch(all_(isinstance(x, torch.Tensor) and x.ndim == 1 for x in tensors), TypeError(f"'bt.meshgrid' can only span torch.Tensor objects. "))
    with torch._C.DisableTorchFunction():
        ret = tuple(t.as_subclass(Tensor).init_special() for t in torch.meshgrid(*tensors, indexing=indexing))
    for i, t in enumerate(tensors):
        for r in ret: r.add_special_dim(i, t.shape)
    return ret # suppress: special_from

@collect_memory
def eye(*size: Size, dim=None, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False):
    """
    create identity matrix in (the first available condition):
    (1) feature dimensions if size has at least one; 
    (2) space dimensions if size has at least one; 
    (3) sequence dimensions if size has at least one. 
    """
    avouch(len(size) > 0, TypeError("bt.eye receives at least one size input. "))
    if dim is None:
        if size.has_feature:
            dim = exist_dim(size, [])
            if len(dim) > 2: dim = dim[-2:]
        elif size.has_space: dim = exist_dim(size, ...)
        elif size.has_sequence: dim = exist_dim(size, '')
        else: raise TypeError(f"Invalid size {size} for bt.eye: at least one non-batch dimension needed. ")
    if len(dim) == 1:
        size = size[:dim[0]] + size[dim[0]:dim[0]+1] + size[dim[0]:]
        dim = (dim[0], dim[0]+1)
    avouch(len(dim) == 2, TypeError("bt.eye can only be created in two-dimensional space, please make sure the shape has 2 space dimensions or use keyword argument 'dim' to identify them. "))
    
    kwargs = dict(out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    if dim[0] > dim[1]: dim = dim[::-1]
    new_size = size[:dim[1]].with_dim_size(dim[0], min_(size[dim[0]], size[dim[1]])) + size[dim[1]+1:]
    eye_output = ones(new_size, **kwargs).diag(dim=(dim[0],)).move_dim(dim[0]+1, dim[1]).special_from(size)
    if size[dim[1]] > size[dim[0]]:
        return cat(eye_output, zeros(eye_output.shape.with_dim_size(dim[1], size[dim[1]]-size[dim[0]]), **kwargs), dim[1]).special_from(size)
    elif size[dim[1]] < size[dim[0]]:
        return eye_output[(slice(None),) * dim[1] + (slice(0, size[dim[1]]),)].special_from(size)
    else: return eye_output.special_from(size)

@collect_memory
def eye_like(input: 'Tensor', dim=None):
    """
    create identity matrix from shape of `input` in (the first available condition):
    (1) feature dimensions if size has at least one; 
    (2) space dimensions if size has at least one; 
    (3) sequence dimensions if size has at least one. 
    """
    return eye(input_shape, dim=dim, dtype=input.dtype, device=input.device, layout=input.layout)

### STOP GLOBAL AUTO GENERATION
@collect_memory
def tensor_like(input, target: 'Tensor', *, dtype=None, device=None, requires_grad=None):
    """
    bt.tensor_like(input, target) creates a tensor with the same `dtype`, `device` and `requires_grad` as `target` (namely in the same characteristics). 
    Note that an expanding reshape will also be performed just like `ones/zeros_like`. Use `tensor_to` to create a tensor with only the same `dtype` and `device`. 
    """
    target = to_bttensor(target)
    if dtype is None: dtype = target.dtype
    if device is None: device = target.device
    if requires_grad is None: requires_grad = target.requires_grad
    if not isinstance(input, torch.Tensor): input = torch.tensor(input, dtype=dtype, device=device, requires_grad=requires_grad)
    if not isinstance(input, Tensor): input = as_tensor(input.as_subclass(Tensor).init_special(), dtype=dtype, device=device).requires_grad_(requires_grad)
    else: input = as_tensor(input, dtype=dtype, device=device).requires_grad_(requires_grad)
    if input.n_dim == target.n_dim: input = input.special_from(target)
    else: input = input.expand_to(target)
    return input

@collect_memory
def tensor_to(input, target: 'Tensor', *, dtype=None, device=None, requires_grad=None):
    """
    bt.tensor_to(input, target) creates a tensor with the same `dtype` and `device` as `target` (namely in the same scope). 
    Note that no shape changes will be performed (similar to method `to`) but special dimensions would be inherited from `target` when they have the same number of dimensions. 
    """
    target = to_bttensor(target)
    if dtype is None: dtype = target.dtype
    if device is None: device = target.device
    if requires_grad is not None:
        if not isinstance(input, torch.Tensor): input = torch.tensor(input, dtype=dtype, device=device, requires_grad=requires_grad)
        if not isinstance(input, Tensor): input = as_tensor(input.as_subclass(Tensor).init_special(), dtype=dtype, device=device).requires_grad_(requires_grad)
        else: input = as_tensor(input, dtype=dtype, device=device).requires_grad_(requires_grad)
    else:
        if not isinstance(input, torch.Tensor): input = torch.tensor(input, dtype=dtype, device=device)
        if not isinstance(input, Tensor): input = as_tensor(input.as_subclass(Tensor).init_special(), dtype=dtype, device=device)
        else: input = as_tensor(input, dtype=dtype, device=device)
    if input.n_dim == target.n_dim: input = input.special_from(target)
    return input

def to_bttensor(data, *, dtype=None, device=_device.main_device, requires_grad=False, pin_memory=False):
    if data is None: return
    elif not isinstance(data, torch.Tensor):
        return tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    elif not isinstance(data, Tensor):
        return data.as_subclass(Tensor).init_special()
    else: return data

@collect_memory
def batch_arange(*start_stop_step, out=None, dtype=None, layout=torch.strided, device=_device.device, requires_grad=False):
    return arange(*start_stop_step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad).with_sz_batch_dim(1)

@collect_memory
@alias('channel_arange')
def feature_arange(*start_stop_step, out=None, dtype=None, layout=torch.strided, device=_device.device, requires_grad=False):
    return arange(*start_stop_step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad).with_sz_feature_dim(1)

@collect_memory
def sequence_arange(*start_stop_step, out=None, dtype=None, layout=torch.strided, device=_device.device, requires_grad=False):
    return arange(*start_stop_step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad).with_sz_sequence_dim(-1)

def batch_tensor(data, *, dtype=None, device=_device.device, requires_grad=False, pin_memory=False):
    self = tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    avouch(self.n_dim == 1, TypeError(f"Cannot create 'batch_tensor' from {data}: dimension is not 1. "))
    return self.with_sz_batch_dim(1)

@alias("channel_tensor", one_dim_only=True)
def feature_tensor(data, *, dtype=None, device=_device.device, requires_grad=False, pin_memory=False, one_dim_only=False):
    self = tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    if one_dim_only: avouch(self.n_dim == 1, TypeError(f"Cannot create 'channel/feature_tensor' from {data}: dimension is not 1. "))
    return self.with_sz_feature_dim(self.n_dim)

@alias("time_tensor", "series_tensor")
def sequence_tensor(data, *, dtype=None, device=_device.device, requires_grad=False, pin_memory=False):
    self = tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    return self.with_sz_sequence_dim(-self.n_dim)

class _Randint:

    def __init__(self):
        """Please use randint[lower, upper] to specify the range with upper end excluded. """
        self.range = (0, 2)

    def __getitem__(self, *t):
        if len(t) == 1 and isinstance(t[0], slice):
            t = t[0]
            if t.step is not None: raise TypeError(f"Please use randint_like[lower:upper] to specify the range with upper end excluded. ")
            t = (t.start if t.start is None else 0, t.stop if t.stop is not None else 2)
        elif len(t) == 1 and isinstance(t[0], tuple): t = t[0]
        if len(t) == 0: t = (0, 2)
        elif len(t) == 1: t = (0, t[0])
        if len(t) > 2 or t[0] >= t[1]: raise TypeError(f"Please use randint_like[lower, upper] to specify the range with upper end excluded. ")
        self.range = t
        return self

    def __call__(self, *size, generator=None, dtype=None, device=None, requires_grad=False):
        if len(size) <= 3 and isinstance(size[-1], tuple):
            *t, size = size
            if len(t) == 0: t = (0, 2)
            elif len(t) == 1: t = (0, t[0])
            elif len(t) > 2: raise TypeError(f"Please use randint[lower, upper] to specify the range with upper end excluded. ")
            self.range = t
        size = Size(*size)
        with torch._C.DisableTorchFunction():
            return torch.randint(self.range[0], self.range[1], size, generator=generator, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(Tensor).special_from(size)

class _Randint_like:

    def __init__(self):
        """Please use randint_like[lower, upper] to specify the range with upper end excluded. """
        self.range = (0, 2)

    def __getitem__(self, *t):
        if len(t) == 1 and isinstance(t[0], slice):
            t = t[0]
            if t.step is not None: raise TypeError(f"Please use randint_like[lower:upper] to specify the range with upper end excluded. ")
            t = (t.start if t.start is None else 0, t.stop if t.stop is not None else 2)
        elif len(t) == 1 and isinstance(t[0], tuple): t = t[0]
        if len(t) == 0: t = (0, 2)
        elif len(t) == 1: t = (0, t[0])
        if len(t) > 2 or t[0] >= t[1]: raise TypeError(f"Please use randint_like[lower, upper] to specify the range with upper end excluded. ")
        self.range = t
        return self

    def __call__(self, data, *t, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        if 0 < len(t) <= 2:
            if len(t) == 1: t = (0, t[0])
            elif len(t) > 2: raise TypeError(f"Please use randint[lower, upper] to specify the range with upper end excluded. ")
            self.range = t
        with torch._C.DisableTorchFunction():
            if layout is None:
                return torch.randint_like(data, self.range[0], self.range[1], memory_format=memory_format, dtype=dtype, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(Tensor).special_from(data.shape)
            else:
                return torch.randint_like(data, self.range[0], self.range[1], memory_format=memory_format, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(Tensor).special_from(data.shape)

randint = _Randint()
randint_like = _Randint_like()

##########################
##  Direct inheritance  ##
##########################

dtype = torch.dtype;              device = torch.device

# D-Types
bfloat16 = torch.bfloat16;        bool = torch.bool
cdouble = torch.cdouble;          cfloat = torch.cfloat;             chalf = torch.chalf
complex128 = torch.complex128;    complex64 = torch.complex64;       complex32 = torch.complex32
double = torch.double;            half = torch.half
float = torch.float;              float16 = torch.float16;           float32 = torch.float32;           float64 = torch.float64
int = torch.int;                  int16 = torch.int16;               int32 = torch.int32;               int64 = torch.int64;               int8 = torch.int8
qint32 = torch.qint32;            qint8 = torch.qint8;               quint2x4 = torch.quint2x4;         quint4x2 = torch.quint4x2;         quint8 = torch.quint8
long = torch.long;                short = torch.short;               uint8 = torch.uint8

# functions
def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = True
