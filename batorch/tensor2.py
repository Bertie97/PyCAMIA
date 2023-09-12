
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch.",
    requires = "torch"
)

__all__ = """
    CPU
    GPU
    GPUs
    Tensor
    Size
    set_device
    get_device
    to_device
    turn_on_autodevice
    turn_off_autodevice
    get_cpu_memory_used
    get_gpu_memory_used
    inv
    diag
    batch_tensor
    channel_tensor
""".split()

import builtins, sys
from collections import defaultdict
from functools import wraps
from typing import Generator
from .tensorfunc import __all__ as tf_list
from .torch_namespace import *
from .device import GB, CPU, GPU, GPUs, AutoDevice, FixedDevice

with __info__:
    import torch
    import batorch as bt
    from pycamia import ByteSize
    from pycamia import avouch, touch, alias, execblock, void
    from pycamia import get_alphas, arg_extract, max_argmax
    from pycamia import argmax as _argmax, item, to_list, tokenize, identity_function

_int = builtins.int
_min = builtins.min
_max = builtins.max
_abs = builtins.abs
_any = builtins.any
_all = builtins.all
_sum = builtins.sum
_range = builtins.range
_float = builtins.float
_num = (_int, _float)

_device = AutoDevice(verbose=True, always_proceed=True)
_total_cpu_memory_used = 0
_total_gpu_memory_used = 0

new_dim_inherit_methods = """multiply multiple""".split()
new_dim_methods = """unsqueeze unsqueeze_ multiply multiple""".split()
old_dim_methods = """
    squeeze squeeze_ transpose transpose_ movedim movedim_ splitdim repeated amplify ample sample pick split flip
    cummin cummax cumsum cumprod sum prod min max median mean std argmin argmax
""".split()
one_dim_methods_last = """multiply multiple repeated amplify ample sample split flip""".split()
one_dim_methods_first = """splitdim""".split()
avouch(all(x in new_dim_methods for x in new_dim_inherit_methods))

with open(__file__) as fp:
    __mirror__ = [None] + fp.read().split('\n')

def set_device(device):
    if isinstance(device, AutoDevice): new_device = device
    elif isinstance(device, torch.device): new_device = FixedDevice(device)
    else: raise TypeError("Invalid device type. ")
    global _device
    _device = new_device

def get_device():
    global _device
    return _device

def to_device(x):
    global _device
    return _device(x)

def turn_on_autodevice(): _device.turn_on()
def turn_off_autodevice(): _device.turn_off()

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
        if ret.device == CPU.main_device:
            global _total_cpu_memory_used
            _total_cpu_memory_used += ret.byte_size()
        else:
            global _total_gpu_memory_used
            _total_gpu_memory_used += ret.byte_size()
        return ret
    return wrapper

def kwget(kwargs, key, default):
    if kwargs is None: return default
    else: return kwargs.get(key, default)


class Size(tuple):

    @classmethod
    def __new_shape__(cls, shape, **kwargs):
        avouch(isinstance(shape, tuple), f"Invalid initialization of bt.Size (need tuple): {shape}.")
        with_batch = kwargs.get("with_batch", False)
        feature_range = kwargs.get("feature_range", None)
        n_sequence_dim = kwargs.get("n_sequence_dim", 0)
        avouch(isinstance(with_batch, bool), f"Invalid initialization of bt.Size (non bool argument 'with_batch': {type(with_batch)}; If you did not input it manually, reach out for developers with Error Code: B522).")
        avouch(isinstance(n_sequence_dim, int), f"Invalid initialization of bt.Size (non int argument 'n_sequence_dim': {type(n_sequence_dim)}; If you did not input it manually, reach out for developers with Error Code: B523).")
        avouch(feature_range is None or isinstance(feature_range, list) and len(feature_range) == 2, f"Invalid initialization of bt.Size (non list argument 'feature_range': {type(feature_range)}; If you did not input it manually, reach out for developers with Error Code: B524).")
        input_feature = feature_range is not None

        raw_size = []
        began_sequence = False
        for i, x in enumerate(shape):
            # time/sequence dimensions
            if x == ...:
                avouch(not began_sequence, f"Invalid initialization of bt.Size (more than one conflict ellipsis): {shape}.")
                began_sequence = True
                avouch(n_sequence_dim == 0, f"Invalid initialization of bt.Size (positive n_sequence_dim conflicts with '...' expression): {shape}.")
                continue
            # normal/space dimensions
            if isinstance(x, _int):
                avouch(x >= -1, f"Invalid initialization of bt.Size (Size cannot have negative values except -1 indicating an arbitrary number): {shape}.")
                raw_size.append(x)
                if began_sequence: n_sequence_dim += 1
                continue
            if began_sequence:
                avouch(None, f"Invalid initialization of bt.Size (no special dimension is allowed in time dimensions): {shape}.")
            # batch dimensions
            if isinstance(x, dict) and len(x) == 0:
                avouch(i == 0, f"Invalid initialization of bt.Size (batch dimension can only be the first dimension): {shape}.")
                raw_size.append(-1); with_batch = True; continue
            if isinstance(x, set) and len(x) == 1:
                avouch(i == 0, f"Invalid initialization of bt.Size (batch dimension can only be the first dimension): {shape}.")
                raw_size.append(x.pop()); with_batch = True; continue
            # feature dimensions
            if feature_range is not None: i += feature_range[1] - feature_range[0] - 1
            if isinstance(x, list):
                avouch(feature_range is None or feature_range[1] == 0 or feature_range[1] == i, f"Invalid initialization of bt.Size (feature dimensions should be neighboring dimensions): {shape}.")
                avouch(all([isinstance(y, _int) for y in x]), f"Invalid initialization of bt.Size (representation for feature dimensions should be a list of integers): {shape}.")
                if len(x) == 0: raw_size.append(-1); len_feat = 1
                else: raw_size.extend(x); len_feat = len(x)
                if feature_range is None: feature_range = [i, i + len_feat]
                else: feature_range[1] += len_feat
                continue
            avouch(None, f"Invalid initialization of bt.Size (unacceptable type, please contact the developers for more information; Error Code: B521): {shape}.")

        n_dim = len(raw_size)
        if input_feature: feature_range = [x if x >= 0 else x + n_dim for x in feature_range]

        return cls.__new_raw__(raw_size, with_batch, feature_range, n_sequence_dim)

    @classmethod
    def __new_size__(cls, size, **kwargs):
        avouch(isinstance(size, Size), f"Invalid initialization of bt.Size (need bt.Size object): {size}")
        kwargs.setdefault("with_batch", size.with_batch)
        kwargs.setdefault("feature_range", size.feature_range)
        kwargs.setdefault("n_sequence_dim", size.n_sequence_dim)
        return cls.__new_raw__(list(size), **kwargs)

    @classmethod
    def __new_raw__(cls, size, with_batch=False, feature_range=None, n_sequence_dim=0):
        if feature_range is not None: avouch(isinstance(feature_range, list) and len(feature_range) == 2, f"Invalid initialization of bt.Size (argument 'feature_range' has length {len(feature_range)} instead of 2; If you did not input it manually, reach out for developers with Error Code: B525).")
        avouch(not with_batch or feature_range is None or feature_range[0] > 0, f"Invalid initialization of bt.Size (conflict betach and feature dimensions; If you did not input it manually, reach out for developers with Error Code: B526).")
        self = super().__new__(cls, size)
        self.with_batch = with_batch
        self.feature_range = feature_range
        self.n_sequence_dim = n_sequence_dim
        self.n_dim = self.ndim = len(size)
        return self

    def __new__(cls, *args, **kwargs):
        """bt.Size
        Usages:
            bt.Size(shape: torch.Tensor/bt.Tensor/bt.Size/generator/tuple/str, with_batch=False, feature_range=None)
            bt.Size(*shape: python_repr[int/len0-list/len1-list/len0-dict/len1-set], with_batch=False, feature_range=None)
        """
        if len(args) == 1 and hasattr(args[0], 'shape'): return Size.__new_shape__(arg.shape, **kwargs)
        if len(args) == 1 and isinstance(args[0], Generator): return Size.__new_shape__(tuple(args[0]), **kwargs)
        if len(args) == 1 and isinstance(args[0], tuple): return Size.__new_shape__(args[0], **kwargs)
        if len(args) == 1 and isinstance(args[0], str): return Size.__new_shape__(eval(args[0]), **kwargs)
        if len(args) == 1 and isinstance(args[0], Size): return Size.__new_size__(args[0], **kwargs)
        return Size.__new_shape__(args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.with_batch = self.with_batch
        self.feature_range = self.feature_range
        self.n_sequence_dim = self.n_sequence_dim
        self.n_dim = self.n_dim
        self.ndim = self.n_dim

    ## batch dimension:
    @property
    def has_batch(self): return self.with_batch

    @alias("batch_dim_", "batch_dimension_", "with_batchdim")
    def with_batch_(self, wbatch):
        if wbatch == 0: wbatch = True
        avouch(isinstance(wbatch, bool), "'bt.Size.with_batch_' only takes input bool or integer 0.")
        self.with_batch = wbatch
        return self

    @alias("nbatch", "batch_size")
    @property
    def n_batch(self):
        avouch(self.with_batch, f"'bt.Size' object {self} does not have batch dimension.")
        return self[0]

    @alias("nbatchdim")
    @property
    def n_batch_dim(self):
        return _int(self.with_batch)

    ## channel dimension: 
    @property
    def has_channel(self): return self.n_feature_dim == 1

    @alias("channel_dimension")
    @property
    def channel_dim(self):
        avouch(self.n_feature_dim == 1, f"Cannot get channel dimension from size with {self.n_feature_dim} feature dimensions.")
        return self.n_feature_dim[0]

    @channel_dim.setter
    def channel_dim(self, dim):
        avouch(dim is None or isinstance(dim, _int) and - self.n_dim <= dim < self.n_dim, f"Channel dimension of {self} should in [{-self.n_dim}, {self.n_dim - 1}].")
        return self.channel_dim_(dim)

    @alias("channel_dimension_", "with_channeldim")
    def channel_dim_(self, dim):
        avouch(dim is None or isinstance(dim, _int) and - self.n_dim <= dim < self.n_dim, f"Channel dimension of {self} should in [{-self.n_dim}, {self.n_dim - 1}].")
        if dim is None: self.feature_range = None
        else: self.feature_range = [dim, dim + 1]
        return self

    @alias("nchannel", "channel_size")
    @property
    def n_channel(self):
        avouch(self.n_feature_dim == 1, f"Cannot get channel dimension from size with {self.n_feature_dim} feature dimensions.")
        return self[self.channel_dim]

    @alias("nchanneldim")
    @property
    def n_channel_dim(self):
        return _int(self.has_channel)

    ## feature dimensions:
    @property
    def has_feature(self): return self.feature_range is not None

    @alias("with_featurerange")
    def feature_range_(self, frange):
        avouch(frange is None or isinstance(frange, list) and len(frange) == 2, "'bt.Size.feature_range_' only takes input list of length 2.")
        if frange is not None: avouch(self.n_batch_dim <= frange[0] < self.sequence_start and self.n_batch_dim <= frange[1] < self.sequence_start, 
                                      f"Feature range should be between {self.n_batch_dim} and {self.sequence_start}, or there will be conflict. ")
        self.feature_range = frange
        return self

    @property
    def feature_start(self): return self.sequence_start if self.feature_range is None else self.feature_range[0]

    @feature_start.setter
    def feature_start(self, dim):
        if dim < 0: dim += self.n_dim
        avouch(self.n_batch_dim <= dim < self.sequence_start, f"Feature start should be between {self.n_batch_dim} and {self.sequence_start}, or there will be conflict. ")
        if self.feature_range is None: self.feature_range = [dim, self.n_dim]
        else: self.feature_range[0] = dim

    @property
    def feature_stop(self): return self.sequence_start if self.feature_range is None else self.feature_range[1]

    @feature_stop.setter
    def feature_stop(self, dim):
        if dim < 0: dim += self.n_dim
        avouch(self.n_batch_dim <= dim < self.sequence_start, f"Feature stop should be between {self.n_batch_dim} and {self.sequence_start}, or there will be conflict. ")
        if self.feature_range is None: self.feature_range = [self.n_batch_dim, dim]
        else: self.feature_range[1] = dim

    @alias("nfeaturedim")
    @property
    def n_feature_dim(self):
        return self.feature_stop - self.feature_start

    @alias("nfeature")
    @property
    def n_feature(self):
        avouch(self.feature_range is not None, f"Cannot get feature dimensions from size {self}.")
        p = 1
        for i in range(*self.feature_range): p *= self[i]
        return p

    @alias("feature_size")
    @property
    def feature(self):
        return self[slice(self.feature_start, self.feature_stop)]

    def with_feature(self, size):
        avouch(len(size) == self.n_feature_dim, f"Cannot substitute feature in {self} by {size} as their dimensions are not the same.")
        return (self[:self.feature_start] + size + self[self.feature_stop:]).feature_range_(self.feature_range)

    ## sequence dimensions:
    @alias("has_time")
    @property
    def has_sequence(self): return self.n_sequence_dim > 0

    @alias("with_ntimedim")
    @alias("with_nsequencedim")
    @alias("n_time_dim_")
    def n_sequence_dim_(self, number):
        avouch(number is None or isinstance(number, int), "'bt.Size.n_sequence_dim_' only takes integer.")
        if number is None: number = 0
        self.n_sequence_dim = number
        return self

    @alias("with_timedim")
    @alias("with_sequencedim")
    @alias("time_dim_")
    def sequence_dim_(self, with_seq):
        avouch(with_seq is None or isinstance(with_seq, bool), "'bt.Size.sequence_dim_' only takes bool.")
        self.n_sequence_dim = 1 if with_seq else 0
        return self

    @property
    def sequence_start(self): return self.n_dim - self.n_sequence_dim

    @sequence_start.setter
    def sequence_start(self, dim):
        if dim < 0: dim += self.n_dim
        self.n_sequence_dim = self.n_dim - dim

    @alias("ntime")
    @alias("ntimeline")
    @alias("nsequence")
    @alias("n_time")
    @alias("n_timeline")
    @property
    def n_sequence(self):
        avouch(self.n_sequence_dim > 0, f"Cannot get sequence dimensions from size {self}.")
        p = 1
        for i in range(-self.n_sequence_dim, 0): p *= self[i]
        return p

    @alias("time_size")
    @alias("sequence_size")
    @property
    def sequence(self):
        return self[self.sequence_start:]

    @alias("with_time")
    def with_sequence(self, size):
        avouch(len(size) == self.n_sequence_dim, f"Cannot substitute sequence in {self} by {size} as their dimensions are not the same.")
        return (self[: self.sequence_start] + size).n_sequence_dim_(self.n_sequence_dim)

    ## space dimensions:
    @property
    def has_space(self): return self.n_space_dim > 0

    @alias("nspacedim")
    @property
    def n_space_dim(self):
        return self.n_dim - self.feature_stop + self.feature_start - self.n_batch_dim - self.n_sequence_dim

    @alias("leftspacedim")
    @property
    def left_space_dim(self):
        return self.feature_start - self.n_batch_dim

    @alias("rightspacedim")
    @property
    def right_space_dim(self):
        return self.n_dim - self.feature_stop - self.n_sequence_dim

    @alias("nspace")
    @property
    def n_space(self):
        avouch(self.n_space_dim > 0, f"Cannot get space dimensions from size {self}.")
        p = 1
        for i in range(self.n_batch_dim, self.feature_start): p *= self[i]
        for i in range(self.feature_stop, self.sequence_start): p *= self[i]
        return p

    @alias("space_size")
    @property
    def space(self):
        return self[self.n_batch_dim: self.feature_start] + self[self.feature_stop: self.sequence_start]

    def with_space(self, size):
        avouch(len(size) == self.n_space_dim, f"Cannot substitute space in {self} by {size} as their dimensions are not the same.")
        return self[:self.n_batch_dim] + size[:self.feature_start - self.n_batch_dim] + self.feature + size[self.feature_start - self.n_batch_dim:] + self.sequence

    ## special dimensions:
    @property
    def has_special(self): return self.has_batch or self.has_feature or self.has_sequence

    @alias("nspecialdim")
    @property
    def n_special_dim(self):
        return self.n_batch_dim + self.feature_stop - self.feature_start + self.n_sequence_dim

    @property
    def special_dims(self):
        return ([0] if self.with_batch else []) + list(range(self.feature_start, self.feature_stop)) + list(range(self.sequence_start, self.n_dim))

    def special_from(self, other):
        avouch(isinstance(other, (Size, Tensor)), f"Invalid input for Size.special_from: {type(other)}. ")
        avouch(self.n_dim == other.n_dim, f"Dimension mismatch when inheriting special dimensions: {other} to {self}. ")
        self.with_batch = other.with_batch
        self.feature_range = other.feature_range
        self.n_sequence_dim = other.n_sequence_dim
        return self

    def remove_special(self):
        self.with_batch = False
        self.feature_range = None
        self.n_sequence_dim = 0
        return self

    ## all dimensions:
    @alias("nele")
    @property
    def n_ele(self):
        p = 1
        for i in range(self.n_dim): p *= self[i]
        return p
    
    def transpose(self, i, j):
        if i > j: i, j = j, i
        return self[:i] + self[j:j+1] + self[i+1:j] + self[i:i+1] + self[j+1:]

    ## methods:
    @alias("clone")
    def copy(self): return Size(self)

    @alias("raw")
    def tuple(self): return tuple(self)

    @property
    def python_repr(self):
        batch = set(list(self[:self.with_batch]))
        feature = list(self[self.feature_start: self.feature_stop])
        sequence = (...,) + tuple(self[self.sequence_start:])
        return (((batch,) if self.with_batch else tuple()) + self[self.n_batch_dim: self.feature_start] + 
                ((feature,) if len(feature) > 0 else tuple()) + self[self.feature_stop: self.sequence_start] + (sequence if self.n_sequence_dim > 0 else tuple())).tuple()

    @alias("__repr__")
    def __str__(self):
        rep = self.python_repr
        return f"batorch.Size{rep}".replace(',)', ')').replace('Ellipsis', '...')
    
    ## operations:
    def __getitem__(self, k):
        if isinstance(k, _int): return super().__getitem__(k)
        avouch(isinstance(k, slice), f"Slicing of 'bt.Size' only takes integers or slices, not {k}. ")
        s, e = k.start, k.stop
        if s is None: s = 0
        if e is None: e = self.n_dim
        if s < 0: s += self.n_dim
        if e < 0: e += self.n_dim
        with_batch = s == 0 and e > 0 and self.with_batch
        if self.feature_range is None: feature_range = None
        else:
            feature_range = [_min(_max(x - s, 0), e - s) for x in self.feature_range]
            if feature_range[0] == feature_range[1]: feature_range = None
        n_sequence_dim = _min(_max(self.n_sequence_dim + e - self.n_dim, 0), e - s)
        return self.__class__.__new_raw__(super().__getitem__(k), with_batch=with_batch, feature_range=feature_range, n_sequence_dim=n_sequence_dim)
    
    @alias('__iadd__')
    def __add__(self, other):
        avouch(isinstance(other, tuple), "Only Size + tuple is available for 'bt.Size' as a python tuple, please use `size << 2` to increase the space for size numerically. ")
        feature_range = self.feature_range
        n_sequence_dim = 0 if len(other) > 0 else self.n_sequence_dim
        if isinstance(other, Size) and len(other) > 0:
            if self.feature_range is None:
                if other.feature_range is not None: feature_range = [x + self.n_dim for x in other.feature_range]
            elif other.feature_range is not None and self.feature_range[1] == self.n_dim and other.feature_range[0] == 0: feature_range = [self.feature_range[0], other.feature_range[1] + self.n_dim]
            if other.n_sequence_dim == other.n_dim: n_sequence_dim = other.n_sequence_dim + self.n_sequence_dim
            else: n_sequence_dim = other.n_sequence_dim
        return self.__class__.__new_raw__(super().__add__(other), with_batch=self.with_batch, feature_range=feature_range, n_sequence_dim=n_sequence_dim)
        
    def __radd__(self, other):
        avouch(isinstance(other, tuple), "Only tuple + Size is available for 'bt.Size' as a python tuple, please use `size <<(>>) 2` to increase(decrease) the space for size numerically. ")
        if isinstance(other, Size): return other.__add__(self)
        return self.__class__.__new_raw__(other + self.tuple(), with_batch=False if len(other) > 0 else self.with_batch, feature_range=[x + len(other) for x in self.feature_range] if self.feature_range is not None else None, n_sequence_dim=self.n_sequence_dim)
    
    @alias('__imul__', '__rmul__')
    def __mul__(self, other):
        avouch(isinstance(other, _int), "Only Size * int is available for 'bt.Size' as a python tuple, please use `size **(//) 2` to multiply(devide) the space for size numerically. ")
        return self.__class__.__new_raw__(super().__mul__(other), with_batch=self.with_batch, 
                                          feature_range=[0, self.n_dim * other] if self.n_feature_dim == self.n_dim else self.feature_range,
                                          n_sequence_dim=self.n_dim * other if self.n_sequence_dim == self.n_dim else self.n_sequence_dim)
    
    ## element-wise operations:
    @staticmethod
    def __op__(self, other, *, operation):
        avouch(isinstance(self, Size), "Inner problem: if 'bt.Size.__op__' is not called manually, please contact the developers with Error Code: B526")
        avouch(isinstance(other, (_num, tuple)), f"Element-wise operations are only used for numbers or tuples, not {type(other)}.")
        op = lambda x, y: _max(_int(operation(x, y)), 1) if x > 0 else -1
        if isinstance(other, _num): return self.with_space(op(x, other) for x in self.space)
        other_batch = 0
        other_feature = (0,)
        other_sequence = (0,)
        other_space = (0,)
        if isinstance(other, Size):
            if other.with_batch: other_batch = other.n_batch
            if other.has_feature: other_feature = other.feature
            if other.has_sequence: other_sequence = other.sequence
            if other.has_space: other_space = other.space
        elif isinstance(other, tuple): other_space = other
        else: avouch(None, f"Cannot perform element-wise operation between types {type(self)} and {type(other)}. ")
        if len(other_feature) == 1: other_feature *= self.n_feature_dim
        if len(other_sequence) == 1: other_sequence *= self.n_sequence_dim
        if len(other_space) == 1: other_space *= self.n_space_dim
        avouch(isinstance(other_batch, _num), f"Invalid operation between {self} and {other}: conflict in batch dimension. ")
        avouch(isinstance(other_feature, tuple) and len(other_feature) == self.n_feature_dim, f"Invalid operation between {self} and {other}: conflict in feature size. ")
        avouch(isinstance(has_sequence, tuple) and len(has_sequence) == self.n_sequence_dim, f"Invalid operation between {self} and {other}: conflict in sequence size. ")
        avouch(isinstance(other_space, tuple) and len(other_space) == self.n_space_dim, f"Invalid operation between {self} and {other}: conflict in space size. ")
        return self.__class__.__new_raw__(tuple(
            op(x, other_batch) if i == 0 and self.with_batch else (
            op(x, other_space[i - self.n_batch_dim]) if self.n_batch_dim <= i < self.feature_start else (
            op(x, other_feature[i - self.feature_start]) if self.feature_start <= i < self.feature_stop else (
            op(x, other_space[i - self.feature_stop + self.feature_start - self.n_batch_dim]) if self.feature_stop <= i < self.sequence_start else (
            op(x, other_sequence[i - self.sequence_start])
        )))) for i, x in enumerate(self)), with_batch=self.with_batch, feature_range=self.feature_range, n_sequence_dim=self.n_sequence_dim)
        
    @alias('__ilshift__', '__rlshift__')
    def __lshift__(self, other): return Size.__op__(self, other, operation=lambda x, y: x + y)
    @alias('__irshift__')
    def __rshift__(self, other): return Size.__op__(self, other, operation=lambda x, y: x - y)
    def __rrshift__(self, other): return Size.__op__(self, other, operation=lambda x, y: y - x)
    @alias('__ipow__', '__rpow__')
    def __pow__(self, other): return Size.__op__(self, other, operation=lambda x, y: x * y)
    @alias('__ifloordiv__')
    def __floordiv__(self, other): return Size.__op__(self, other, operation=lambda x, y: x // y)
    def __rfloordiv__(self, other): return Size.__op__(other, self, operation=lambda x, y: y // x)
    
    def __xor__(self, other):
        avouch(isinstance(self, Size) and isinstance(other, tuple), "xor for bt.Size only accept two tuples.")
        if not isinstance(other, Size): other = Size.__new_raw__(other)
        # batch:
        swap = False
        if not self.has_batch and other.has_batch: self, other = other, self; swap = True
        if self.has_batch and not other.has_batch: other = Size({1}) + other
        if swap: self, other = other, self
        
        # sequence:
        swap = False
        if not self.has_sequence and other.has_sequence: self, other = other, self; swap = True
        if self.has_sequence and not other.has_sequence: other = other + Size((...,) + (1,) * self.n_sequence_dim)
        if swap: self, other = other, self
        avouch(self.n_sequence_dim == other.n_sequence_dim, f"Mismatched sequence dimensions: {self.sequence} and {other.sequence}.")
        
        # feature:
        swap = False
        if not self.has_feature and other.has_feature: self, other = other, self; swap = True
        if self.has_feature and not other.has_feature:
            avouch(other.n_space_dim == 0 or other.n_space_dim == self.n_space_dim, f"Mismatched space dimensions: {self.space} and {other.space}.")
            if other.n_space_dim == 0: other = other[:other.n_batch_dim] + Size((1,) * (self.feature_start - self.n_batch_dim) + ([1],) * (self.n_feature_dim) + (1,) * (self.n_dim - self.n_sequence_dim - self.feature_stop)) + other[other.sequence_start:]
            else: other = other[:self.feature_start] + Size([1] * self.n_feature_dim) + other[self.feature_start:]
        if swap: self, other = other, self
        avouch(self.n_feature_dim == other.n_feature_dim, f"Mismatched feature dimensions: {self.feature} and {other.feature}.")
        
        # left space:
        swap = False
        if self.left_space_dim == 0 and other.left_space_dim > 0: self, other = other, self; swap = True
        if self.left_space_dim > 0 and other.left_space_dim == 0: other = other[:other.n_batch_dim] + Size((1,) * self.left_space_dim) + other[other.feature_start:]
        if swap: self, other = other, self
        avouch(self.left_space_dim == other.left_space_dim, f"Mismatched space dimensions: {self.space} and {other.space}.")
        
        # righ space:
        swap = False
        if self.right_space_dim == 0 and other.right_space_dim > 0: self, other = other, self; swap = True
        if self.right_space_dim > 0 and other.right_space_dim == 0: other = other[:other.feature_stop] + Size((1,) * self.right_space_dim) + other[other.feature_stop:]
        if swap: self, other = other, self
        avouch(self.right_space_dim == other.right_space_dim, f"Mismatched space dimensions: {self.space} and {other.space}.")
        
        return self, other

size_mapping = defaultdict(lambda: identity_function,
)

def get_updated_code(func, mode='function'):
    line_no = func.__code__.co_firstlineno
    declaration = __mirror__[line_no].strip()
    if declaration.startswith('@'): avouch(None, "Internal error: no decorator is allowed for auto generated functions in batorch.tensor (Please contact the developer with Error Code: B531). ")
    _def, _fname, _args, *_tail = tokenize(declaration, sep=[' ', '(', ')', '\n'])
    inner_codes = ""
    if not _tail[-1] in ('...', 'pass'):
        l = line_no
        indent = __mirror__[l].split('def')[0]
        inner_codes = [''] # Due to the indent issue, a blank line was added at the front. 
        while True:
            l += 1
            line = __mirror__[l]
            if line.strip() and line.split('def')[0] == indent:
                break
            inner_line = line[len(indent):]
            if inner_line.strip().startswith('return'):
                inner_line = inner_line.replace('return', 'obj =')
            if inner_line.startswith('\t'):
                count = 0
                for x in inner_line:
                    if x != '\t': break
                    count += 1
                inner_line = '    ' * count + inner_line.strip('\t')
            inner_line = '    ' + inner_line
            inner_codes.append(inner_line)
        inner_codes = '\n'.join(inner_codes)
    ant = func.__annotations__
    tensor_args = []
    size_args = []
    for a, t in ant.items():
        if t == 'Tensor': tensor_args.append(a)
        if t == Size: size_args.append(a)
    parts = []
    for x in _args.split(','):
        x = x.strip()
        if x == '*': continue
        if ':' in x: x = x.split(':')[0]
        if '=' not in x: parts.append(x); continue
        param = x.split('=', 1)[0]
        parts.append(param + '=' + param.strip())
    if mode == 'method': x, *parts = parts
    inherit_args = ','.join(parts)

    reshape_op = ""
    if len(tensor_args) >= 2:
        x = tensor_args[0]
        reshape_op = [f"x_shape = {x}.shape"]
        for y in tensor_args[1:]:
            reshape_op.append(f"x_shape, y_shape = x_shape ^ {y}.shape")
            reshape_op.append(f"{y} = {y}.view(y_shape)")
        for y in tensor_args[1:-1]:
            reshape_op.append(f"_, y_shape = x_shape ^ {y}.shape")
            reshape_op.append(f"{y} = {y}.view(y_shape)")
        reshape_op.append(f"{x} = {x}.view(x_shape)")
        reshape_op = '; '.join(reshape_op)
    cast = '; '.join([f"{x} = torch.tensor({x}) if not isinstance({x}, torch.Tensor) else {x}; {x} = {x}.as_subclass(Tensor).with_shape({x}.shape) if not isinstance({x}, Tensor) else {x}" for x in tensor_args])
    get_size = '; '.join([f"{x}_shape={x}.shape" for x in tensor_args] + [f'{x}={x}[0] if isinstance({x}, tuple) and isinstance({x}[0], Size) else Size({x})' for x in size_args])
    if len(tensor_args) >= 2:
        size_reference = "x_shape"
    else:
        size_reference = ', '.join([f'{x}_shape' for x in tensor_args] + size_args)
        size_reference = f"size_mapping['{func.__name__}']({size_reference})"
    parent = f'super(Tensor, {x})' if mode == 'method' else 'torch'
    if not inner_codes: inner_codes = f"obj = {parent}.{_fname}({inherit_args})"
    if '.with_shape' not in inner_codes:
        inner_codes += f"\n        obj = obj.as_subclass(Tensor).with_shape({size_reference})"
    return f"""
    {_def} {_fname}({_args}, with_batch=None, feature_range=void, n_sequence_dim=0):
        {cast} # Cast all arguments to the 'Tensor' or 'Size' type as we want. 
        {reshape_op} # For '__operations__', reshape all the available tensors to a same size. 
        {get_size} # Obtain available sized in arguments (which will be fed into size function). 
        {inner_codes} # Use the given inner codes if they are provided. 
        if with_batch is not None: obj.with_batch = with_batch
        if feature_range is not void: obj.feature_range = feature_range
        if n_sequence_dim != 0: obj.n_sequence_dim = n_sequence_dim
        return obj
    """

class Tensor(torch.Tensor):
    
    @staticmethod
    def _make_subclass(cls, torch_tensor, requires_grad=None, device=None, with_batch=None, feature_range=void, n_sequence_dim=0, **_):
        avouch(not _, "bt.Tensor only accept keyword arguments requires_grad, device, with_batch and feature_range")
        if device is not None and torch_tensor.device != device:
            avouch(isinstance(device, AutoDevice), "Please use `AutoDevice` in batorch.Tensor._make_subclass. Please contact the developers if you did not use Tensor._make_subclass directly (Error Code: B530). ")
            torch_tensor = device(torch_tensor)
        if isinstance(torch_tensor, Tensor) and cls == Tensor:
            if not hasattr(torch_tensor, 'requires_grad'): torch_tensor.requires_grad = False
            if not hasattr(torch_tensor, 'feature_range'): torch_tensor.feature_range = None
            if not hasattr(torch_tensor, 'n_sequence_dim'): torch_tensor.n_sequence_dim = 0
            if requires_grad is not None: torch_tensor.requires_grad = requires_grad
            if with_batch is not None: torch_tensor.with_batch = with_batch
            if feature_range is not void: torch_tensor.feature_range = feature_range
            if n_sequence_dim != 0: torch_tensor.n_sequence_dim = n_sequence_dim
            return torch_tensor
        if requires_grad is None: requires_grad = torch_tensor.requires_grad
        self = torch.Tensor._make_subclass(cls, torch_tensor, requires_grad)
        if with_batch is None: with_batch = torch_tensor.with_batch if isinstance(torch_tensor, Tensor) else False
        if feature_range is void: feature_range = torch_tensor.feature_range if isinstance(torch_tensor, Tensor) else None
        self.with_batch = with_batch
        self.feature_range = feature_range
        self.n_sequence_dim = n_sequence_dim
        return self

    @collect_memory
    def __new__(cls, *args, **kwargs):
        """bt.Tensor
        Usages:
            bt.Tensor(tensor: list/torch.Tensor/bt.Tensor/tensor with 'shape'/tensor with method '__tensor__', requires_grad=None, device=None, with_batch=None, feature_range=void, n_sequence_dim=0)
            bt.Tensor(shape: tuple, requires_grad=None, device=None, with_batch=None, feature_range=void, n_sequence_dim=0)
            bt.Tensor(*shape: int, requires_grad=None, device=None, with_batch=None, feature_range=void, n_sequence_dim=0)
        """
        if len(args) >= 1 and isinstance(args[0], torch.Tensor): return Tensor._make_subclass(cls, *args, **kwargs)
        if len(args) >= 1 and hasattr(args[0], '__tensor__'): return Tensor._make_subclass(cls, args[0].__tensor__(), *args[1:], **kwargs)
        device = kwargs.pop('device', _device)
        if isinstance(device, AutoDevice): device = device.main_device
        if len(args) >= 1 and hasattr(args[0], 'shape') or isinstance(args[0], list): return Tensor._make_subclass(cls, torch.tensor(args[0], requires_grad=False, device=device), *args[1:], **kwargs)
        return Tensor._make_subclass(cls, super().__new__(torch.Tensor, *args, device=device), **kwargs)

    def __init__(self, *args, **kwargs):
        self.with_batch = self.with_batch
        self.feature_range = self.feature_range
        self.n_sequence_dim = self.n_sequence_dim

    @property
    def shape(self): return Size.__new_raw__(super().shape, with_batch=self.with_batch, feature_range=self.feature_range, n_sequence_dim=self.n_sequence_dim)
    
    def with_shape(self, *x):
        x = arg_extract(x)
        if isinstance(x, Tensor): x = x.shape
        if not isinstance(x, Size): x = Size(x)
        self.with_batch = x.with_batch
        self.feature_range = x.feature_range
        self.n_sequence_dim = x.n_sequence_dim
        return self

    @alias("__str__")
    def __repr__(self, *args, **kwargs):
        string = super().__repr__(*args, **kwargs)
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")
    
    ## utilities
    def byte_size(self):
        return ByteSize(self.element_size() * self.numel())
    
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
        if isinstance(dt, str): return super().type(dt.replace('bt.', 'torch.')).as_subclass(self.__class__)
        if hasattr(dt, 'dtype'): dt = dt.dtype
        if isinstance(dt, torch.dtype): return super().type(dt).as_subclass(self.__class__)
        import numpy as np
        dt_name = np.dtype(dt).name
        dtype_map = {'uint16': "int32", 'uint32': "int64", 'uint64': "int64"}
        torch_dt = getattr(torch, dtype_map.get(dt_name, dt_name), None)
        avouch(torch_dt is not None, f"Invalid dtype {dt}: {dt_name} cannot be converted into torch dtype.")
        return super().type(torch_dt).as_subclass(self.__class__)

    def type(self, dt=None):
        if dt is None: return super().type().replace("torch.", "bt.")
        else: return self.astype(dt)

    basic_vars = list(locals().keys()) + ['basic_vars']
    def __add__(self: 'Tensor', other: 'Tensor'): ...
    def __iadd__(self: 'Tensor', other: 'Tensor'): ...
    def __radd__(self: 'Tensor', other: 'Tensor'): ...
    def __sub__(self: 'Tensor', other: 'Tensor'): ...
    def __isub__(self: 'Tensor', other: 'Tensor'): ...
    def __rsub__(self: 'Tensor', other: 'Tensor'): ...
    def __mul__(self: 'Tensor', other: 'Tensor'): ...
    def __imul__(self: 'Tensor', other: 'Tensor'): ...
    def __rmul__(self: 'Tensor', other: 'Tensor'): ...
    def __div__(self: 'Tensor', other: 'Tensor'): ...
    def __idiv__(self: 'Tensor', other: 'Tensor'): ...
    def __rdiv__(self: 'Tensor', other: 'Tensor'): ...
    def __pow__(self: 'Tensor', other: 'Tensor'): ...
    def __ipow__(self: 'Tensor', other: 'Tensor'): ...
    def __rpow__(self: 'Tensor', other: 'Tensor'): ...
    def __mod__(self: 'Tensor', other: 'Tensor'): ...
    def __imod__(self: 'Tensor', other: 'Tensor'): ...
    def __rmod__(self: 'Tensor', other: 'Tensor'): ...
    def __truediv__(self: 'Tensor', other: 'Tensor'): ...
    def __itruediv__(self: 'Tensor', other: 'Tensor'): ...
    def __rtruediv__(self: 'Tensor', other: 'Tensor'): ...
    def __floordiv__(self: 'Tensor', other: 'Tensor'): ...
    def __ifloordiv__(self: 'Tensor', other: 'Tensor'): ...
    def __rfloordiv__(self: 'Tensor', other: 'Tensor'): ...
    def __eq__(self: 'Tensor', other: 'Tensor'): ...
    def __ieq__(self: 'Tensor', other: 'Tensor'): ...
    def __req__(self: 'Tensor', other: 'Tensor'): ...
    def __ne__(self: 'Tensor', other: 'Tensor'): ...
    def __ine__(self: 'Tensor', other: 'Tensor'): ...
    def __rne__(self: 'Tensor', other: 'Tensor'): ...
    def __or__(self: 'Tensor', other: 'Tensor'): ...
    def __ior__(self: 'Tensor', other: 'Tensor'): ...
    def __ror__(self: 'Tensor', other: 'Tensor'): ...
    def __and__(self: 'Tensor', other: 'Tensor'): ...
    def __iand__(self: 'Tensor', other: 'Tensor'): ...
    def __rand__(self: 'Tensor', other: 'Tensor'): ...
    def __xor__(self: 'Tensor', other: 'Tensor'): ...
    def __ixor__(self: 'Tensor', other: 'Tensor'): ...
    def __rxor__(self: 'Tensor', other: 'Tensor'): ...
    def __lt__(self: 'Tensor', other: 'Tensor'): ...
    def __le__(self: 'Tensor', other: 'Tensor'): ...
    def __gt__(self: 'Tensor', other: 'Tensor'): ...
    def __ge__(self: 'Tensor', other: 'Tensor'): ...
    def reshape(self, *size: Size): ...
    def reshape_as(self, other: 'Tensor'): ...
    def view(self, *size: Size): ...
    def view_as(self, other: 'Tensor'): ...
    def t(self): ...
    additional_methods = list(set(locals()) - set(basic_vars))
    
    for f in additional_methods:
        execblock(get_updated_code(eval(f), mode='method'))

    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     try:
    #         if Tensor in types and cls != Tensor: return NotImplemented # Donnot accept calls from outside batorch
    #         if len(args) == 0: return super().__torch_function__(func, types, args, kwargs) # let non-torch-operation torch functions to run
            
    #         func_name = func.__name__
    #         if func_name in ('__get__', '__set__', '__delete__'):
    #             return super().__torch_function__(func, types, args, kwargs)
            
    #         with torch._C.DisableTorchFunction():
    #             obj = super().__torch_function__(func, types, args, kwargs)
    #         return obj.as_subclass(Tensor).with_shape(shape_mapping[func_name]())

            # sfunc = str(func)
            # if sfunc.startswith('<attribute') or sfunc.startswith('<property'):
            #     return super().__torch_function__(func, types, args, kwargs)

            # func_name = sfunc.split(' of ')[0].split(' at ')[0].split()[-1].strip("'").split('.')[-1]
            # if func_name in ('__get__', '__set__', '__delete__'):
            #     return super().__torch_function__(func, types, args, kwargs)

            # self = args[0]
            # types = tuple(cls if t in [torch.nn.Parameter, bt.nn.Parameter] else t for t in types)
            # torch_func_name = Tensor.__torch_function_map__.get(func_name, None)
            # if isinstance(self, Tensor) and self.init and self.has_special: pass
            # elif torch_func_name in ('__torch_function_resizing_func__', '__torch_function_full_func__', '__torch_function_resizing_as_func__', '__torch_function_randint_func__'): pass
            # else:
            #     with torch._C.DisableTorchFunction():
            #         ret = super().__torch_function__(func, types, args, kwargs)
            #     def apply(r): r.special_from_()
            #     return Tensor.__torch_function_convert_apply__(ret, apply, cls)

            # if torch_func_name is None: return Tensor.__torch_function_default_func__(func, types, args, kwargs)
            # else: return getattr(Tensor, torch_func_name)(func, types, args, kwargs)
            
        # except Exception as e:
        #     print(f"In function {func}:")
        #     raise e#.with_traceback(None)

basic_locals = list(locals().keys()) + ['basic_locals']
def tensor(data: Tensor, *, dtype=None, device=None, requires_grad=False, pin_memory=False): ...
def as_tensor(data: Tensor, dtype=None, device=None): ...
def empty(*size: Size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format): ...
def ones(*size: Size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False): ...
def zeros(*size: Size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False): ...
def empty_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format): ...
def ones_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format): ...
def zeros_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format): ...
additional_functions = [f for f in locals() if f not in basic_locals]

for f in additional_functions:
    execblock(get_updated_code(eval(f)))

class _Randint:

    def __init__(self):
        """Please use randint[lower, upper] to specify the range with upper end excluded. """
        self.range = (0, 2)

    def __getitem__(self, t):
        if len(t) == 0: t = (0, 2)
        elif len(t) == 1: t = (0, t[0])
        elif len(t) > 2: raise TypeError(f"Please use randint[lower, upper] to specify the range with upper end excluded. ")
        self.range = t
        return self

    def __call__(self, *size, generator=None, dtype=None, device=None, requires_grad=False):
        size = Size(size)
        return torch.randint(self.range[0], self.range[1], size, generator=generator, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(Tensor).with_shape(size)

class _Randint_like:

    def __init__(self):
        """Please use randint_like[lower, upper] to specify the range with upper end excluded. """
        self.range = (0, 2)

    def __getitem__(self, t):
        if len(t) == 0: t = (0, 2)
        elif len(t) == 1: t = (0, t[0])
        elif len(t) > 2: raise TypeError(f"Please use randint_like[lower, upper] to specify the range with upper end excluded. ")
        self.range = t
        return self

    def __call__(self, data, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        if layout is None:
            return torch.randint_like(data, self.range[0], self.range[1], memory_format=memory_format, dtype=dtype, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(Tensor).with_shape(data.shape)
        else:
            return torch.randint_like(data, self.range[0], self.range[1], memory_format=memory_format, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(Tensor).with_shape(data.shape)

randint = _Randint()
randint_like = _Randint_like()
