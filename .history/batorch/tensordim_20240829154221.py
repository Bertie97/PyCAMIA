
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch.",
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
    any            all            unique
    isnan          isinf
    unsqueeze      squeeze
    flatten        transpose      t              permute        standard_shape
    duplicate      amplify        repeated       repeat
    gather         flip           detach
    quantile       val_range
    sum            prod           mean           std
    cumsum         cumprod        min            max            median
    cummin         cummax         argmin         argmax
    split          sample         pick
    eig            matpow         matexp         matlog         rank           matnorm
    det            matrix_power   matrix_exp     matrix_log     matrix_rank    matrix_norm

    Size           FakeSize       func_dim_size  func_dim
    broadcast      remove_dim     add_dim

    Tensor
    expand         expand_as      expand_to
    complex        tensor         as_tensor      to_bttensor
    empty          full           ones           zeros
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

with open(__file__) as fp:
    __mirror__ = [None] + fp.read().split('\n')

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

"""
Dimensions: there are two types of dimension indicators for batorch functions. 
1. new_dim: a new dimension that did not exist in previous tensor. 
    For a size (n_0, n_1, ..., n_{r-1}), the index for the new dimension is,
               ^    ^    ^    ^       ^
               0    1    2   r-1      r
    If the indicator is a special-dim representation, it means the creation of a special dimension of such type. 
        e.g. creating dimension "[2]" for shape ({3}, [4], 5, 6) would result in shape ({3}, [4, 1], 5, 6)
2. exist_dim: an existed dimension. 
    For a size (n_0, n_1, ..., n_{r-1}), the index for the dimension is,
                 ^    ^           ^
                 0    1          r-1
    If the indicator is a special-dim representation, it is indexed in this special dimension scope. 
        e.g. dimension "[1]" for shape ({3}, [4, 5], 6, 7) is not the dimension with size 4, but the dimension of size 5: ({3}, [4, >5<], 6, 7). 
    Two sub-types of `exist_dim` are available: 
    2.1. del_dim: an existed dimension that would be removed in the output of a function. The dimensions would be called in a reversed order for iterative performance. 
    2.2. linalg_dim: the best existing dimensions for linear algebra, selected in the order of feature dimension -> space dimension -> sequence dimension. 
Adding sub-scripts for the dimension types would result in new behaviors.
*_dim[:] for '*' argument means the star should be removed in the call of torch function. 
*_dim[...] means the call of torch function should iteratively performed for each of the dimensions. 
*_dim[1] means the dimension representation should uniquely identify one dimension. 
linalg_dim[l: u]; linalg_dim[l, u]; linalg_dim[t] = linalg_dim[t, t] means the dimensions for linear algebra,
    which indicates at least l dimensions and at most u dimensions. 
"""

class new_dim_meta(ABCMeta):

    def __instancecheck__(self, item):
        if isinstance(item, tuple): return all_(self.__instancecheck__(x) for x in item)
        if isinstance(item, dict) and len(item) == 0: return True
        if isinstance(item, set) and len(item) == 1 and isinstance(item[0], int_): return True
        if isinstance(item, list) and all_(isinstance(x, int_) for x in item): return True
        if isinstance(item, str) and len(item) == 0: return True
        if isinstance(item, str):
            try:
                item = eval(item)
                if isinstance(item, int_): return True
                item = tuple(item)
            except: return False
            if all_(isinstance(x, int_) for x in item): return True
        if isinstance(item, int_): return True
        return False

class new_dim(metaclass=new_dim_meta):
    def __new__(this, self, *args):
        """
        Conver the dimension representations to actual dimension indices to new dimensions.
        Integers in special dimension marks represent the dimension to create the special dim, 
            e.g. '{3}' represents putting a batch dimension at dimension 3. Note that errors would be 
            raised if this causes invalid representations. 

        Exapmles: 
            >>> bt.Size({1}, 1, 3, 4).special_from(bt.new_dim(bt.Size({}, 3, 4), []))
            batorch.Size({1}, [1], 3, 4)
        """
        if len(args) == 1 and (isinstance(args[0], list) and len(args[0]) > 1 or isinstance(args[0], tuple)): args = args[0]
        if len(args) == 0: args = ({},) if self.sz_batch_dim == 0 else (([],) if self.sz_feature_dim == 0 else (('',) if self.sz_sequence_dim == 0 else (self.space_start,)))
        if isinstance(args, FakeSize): return FakeSize(tuple((x + self.n_dim) if x < 0 else x for x in args), sz_batch_dim=args.sz_batch_dim, sz_feature_dim=args.sz_feature_dim, sz_sequence_dim=args.sz_sequence_dim)
        if isinstance(args, Size): args = args.python_repr
        if not (hasattr(self, 'n_dim') and hasattr(self, 'sz_batch_dim')):
            if isinstance(self, torch.Tensor): self = self.as_subclass(torch.Tensor)
            else: self = tuple(self)
            raise AttributeError(f"Cannot get special dimension from {self}. Possible reasons are:\n(1) The input object is not bt.Tensor/Size. \n(2) Special dimensions are lost during unreimplemented torch functions. ")
        n_dim = self.n_dim
        total_n_dim = self.n_dim + len(args)
        sz_func_dim = self.sz_func_dim
        sz_batch_dim = self.sz_batch_dim
        sz_feature_dim = self.sz_feature_dim
        sz_sequence_dim = self.sz_sequence_dim
        int_args = []
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 0:
                # unspecified functional dimension
                avouch(sz_func_dim == 0, TypeError("Cannot add new functional dimension for tensor with func dimension. "))
                int_args.append(0)
                sz_func_dim = 1
            elif isinstance(arg, tuple) and len(arg) == 1:
                # specified functional dimension
                avouch(sz_func_dim == 0, TypeError("Cannot add new functional dimension for tensor with func dimension. "))
                i_func = arg[0]
                avouch(isinstance(i_func, int_), TypeError(f"Invalid new dimension {(i_func,)}: must be integer in format (...,). "))
                avouch(-n_dim-1 <= i_func <= n_dim, IndexError(f"Invalid new dimension {(i_func,)}: dimension out of range, should be in [0, {n_dim}]. "))
                avouch(i_func in (0, -total_n_dim, total_n_dim-1, -1), TypeError("New functional dimension should be the first/last dimension. "))
                int_args.append(i_func)
                sz_func_dim = 1 if i_func == 0 else -1
            elif isinstance(arg, dict) and len(arg) == 0:
                # unspecified batch dimension
                avouch(sz_batch_dim == 0, TypeError("Cannot add new batch dimension for tensor with batch. "))
                int_args.append(max_(sz_func_dim, 0))
                sz_batch_dim = 1
            elif isinstance(arg, set) and len(arg) == 1:
                # specified batch dimension
                avouch(sz_batch_dim == 0, TypeError("Cannot add new batch dimension for tensor with batch. "))
                i_batch = arg.pop()
                avouch(isinstance(i_batch, int_), TypeError(f"Invalid new dimension {set([i_batch])}: must be integer in {{}}. "))
                avouch(-n_dim-1 <= i_batch <= n_dim, IndexError(f"Invalid new dimension {set([i_batch])}: dimension out of range, should be in [0, {n_dim}]. "))
                if i_batch < 0: i_batch += total_n_dim
                # new batch dimension
                for pm_n in [1, -1]:
                    if i_batch in (max_(pm_n * sz_func_dim, 0), n_dim + min_(pm_n * sz_func_dim, 0)):
                        sz_func_dim = pm_n * sz_func_dim
                        break
                else: raise TypeError("New batch dimension should be the first/last dimension apart from the functional dimension. ")
                # avouch(i_batch in (max_(sz_func_dim, 0), max_(sz_func_dim, 0)-total_n_dim, total_n_dim-1+min_(sz_func_dim, 0), min_(sz_func_dim, 0)-1), TypeError("New batch dimension should be the first/last dimension apart from the functional dimension. "))
                int_args.append(i_batch)
                sz_batch_dim = 1 if i_batch == max_(sz_func_dim, 0) else -1
            elif isinstance(arg, list) and len(arg) == 0:
                # unspecified feature dimension
                avouch(sz_feature_dim == 0, TypeError("Cannot add new feature dimension with size 1 for tensor already with feature: multiple choice in placing new dimension (use [*] to identify). "))
                int_args.append(max_(sz_func_dim, 0) + max_(sz_batch_dim, 0))
                sz_feature_dim = 1
            elif isinstance(arg, list) and len(arg) == 1:
                # specified feature dimension
                i_new_feature = arg[0]
                avouch(isinstance(i_new_feature, int_), TypeError(f"Invalid new dimension [{i_new_feature}]: must be integer in []. "))
                avouch(-n_dim-1 <= i_new_feature <= n_dim, IndexError(f"Invalid new dimension [{i_new_feature}]: dimension out of range, should be in [0, {n_dim}]. "))
                if i_new_feature < 0: i_new_feature += total_n_dim
                if sz_feature_dim != 0:
                    start = max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) if sz_feature_dim > 0 else n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + sz_feature_dim
                    stop = max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + sz_feature_dim if sz_feature_dim > 0 else n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0)
                    avouch(start <= i_new_feature <= stop, TypeError("New feature dimension should not be apart from existing feature dimensions. "))
                else:
                    # new feature dimension
                    pms = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    for pm_n, pm_b in pms:
                        if pm_n < 0 and (sz_batch_dim != 0): continue
                        start = max_(pm_n * sz_func_dim, 0) + max_(pm_b * sz_batch_dim, 0)
                        stop = n_dim + min_(pm_n * sz_func_dim, 0) + min_(pm_b * sz_batch_dim, 0)
                        if i_new_feature in (start, stop):
                            sz_func_dim = pm_n * sz_func_dim
                            sz_batch_dim = pm_b * sz_batch_dim
                            break
                    else: raise TypeError("New feature dimension should be the first/last dimension apart from the batch dimension. ")
                    # avouch(i_new_feature in (max_(sz_func_dim, 0) + max_(sz_batch_dim, 0), n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0)), 
                    #        TypeError("New feature dimension should be the first/last dimension apart from the batch dimension. "))
                int_args.append(i_new_feature)
                if sz_feature_dim > 0: sz_feature_dim += 1
                elif sz_feature_dim < 0: sz_feature_dim -= 1
                else: sz_feature_dim = 1 if i_new_feature == max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) else -1
            elif isinstance(arg, str) and len(arg) == 0:
                # unspecified sequence dimension
                avouch(sz_sequence_dim == 0, TypeError("Cannot add new sequence dimension with size 1 for tensor already with sequence: multiple choice in placing new dimension (use '*' to identify). "))
                int_args.append(n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0))
                sz_sequence_dim = -1
            elif isinstance(arg, str) and ',' not in arg:
                # specified sequence dimension
                i_new_sequence = touch(lambda: eval(arg), default=None)
                i_new_sequence = touch(lambda: i_new_sequence[0], default=i_new_sequence)
                avouch(i_new_sequence is not None, TypeError(f"Invalid new dimension '{arg}': must be integer in ''. "))
                avouch(isinstance(i_new_sequence, int_), TypeError(f"Invalid new dimension '{i_new_sequence}': must be integer in ''. "))
                avouch(-n_dim-1 <= i_new_sequence <= n_dim, IndexError(f"Invalid new dimension [{i_new_sequence}]: dimension out of range, should be in [0, {n_dim}]. "))
                if i_new_sequence < 0: i_new_sequence += total_n_dim
                if sz_sequence_dim != 0:
                    start = max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0) if sz_sequence_dim > 0 else n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0) + sz_sequence_dim
                    stop = max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0) + sz_sequence_dim if sz_sequence_dim > 0 else n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0)
                    avouch(start <= i_new_sequence <= stop, TypeError("New sequence dimension should not be apart from existing sequence dimensions. "))
                else:
                    # new sequential dimension
                    pms = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                           (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
                    for pm_n, pm_b, pm_f in pms:
                        if pm_n < 0 and (sz_batch_dim != 0 or sz_feature_dim != 0): continue
                        if pm_b < 0 and (sz_feature_dim != 0): continue
                        start = max_(pm_n * sz_func_dim, 0) + max_(pm_b * sz_batch_dim, 0) + max_(pm_f * sz_feature_dim, 0)
                        stop = n_dim + min_(pm_n * sz_func_dim, 0) + min_(pm_b * sz_batch_dim, 0) + min_(pm_f * sz_feature_dim, 0)
                        if i_new_sequence in (start, stop):
                            sz_func_dim = pm_n * sz_func_dim
                            sz_batch_dim = pm_b * sz_batch_dim
                            sz_feature_dim = pm_f * sz_feature_dim
                            break
                    else: raise TypeError('New sequence dimension should be the first/last dimension apart from the batch and feature dimensions. ')
                    # avouch(i_new_sequence in (max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0), n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0)), 
                    #        TypeError('New sequence dimension should be the first/last dimension apart from the batch and feature dimensions. '))
                int_args.append(i_new_sequence)
                if sz_sequence_dim > 0: sz_sequence_dim += 1
                elif sz_sequence_dim < 0: sz_sequence_dim -= 1
                else: sz_sequence_dim = 1 if i_new_sequence == (max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0)) else -1
            elif isinstance(arg, int_):
                # specified spatial dimension
                avouch(isinstance(arg, int_), TypeError(f"Invalid new dimension {arg}: must be an integer. "))
                avouch(-n_dim-1 <= arg <= n_dim, IndexError(f"Invalid new dimension [{arg}]: dimension out of range, should be in [0, {n_dim}]. "))
                if arg < 0: arg += total_n_dim
                start = max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0) + max_(sz_sequence_dim, 0)
                stop = n_dim + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0) + min_(sz_sequence_dim, 0)
                if stop - start > 0:
                    avouch(start <= arg <= stop, TypeError("New space dimension should not be apart from existing space dimensions. "))
                else:
                    # new spatial dimension
                    pms = [(1, 1, 1, 1), (1, 1, 1, -1), (1, 1, -1, 1), (1, 1, -1, -1), (1, -1, 1, 1), (1, -1, 1, -1), (1, -1, -1, 1), (1, -1, -1, -1),
                           (-1, 1, 1, 1), (-1, 1, 1, -1), (-1, 1, -1, 1), (-1, 1, -1, -1), (-1, -1, 1, 1), (-1, -1, 1, -1), (-1, -1, -1, 1), (-1, -1, -1, -1)]
                    for pm_n, pm_b, pm_f, pm_s in pms:
                        if pm_n < 0 and (sz_batch_dim != 0 or sz_feature_dim != 0 or sz_sequence_dim != 0): continue
                        if pm_b < 0 and (sz_feature_dim != 0 or sz_sequence_dim != 0): continue
                        if pm_f < 0 and (sz_sequence_dim != 0): continue
                        start = max_(pm_n * sz_func_dim, 0) + max_(pm_b * sz_batch_dim, 0) + max_(pm_f * sz_feature_dim, 0) + max_(pm_s * sz_sequence_dim, 0)
                        stop = n_dim + min_(pm_n * sz_func_dim, 0) + min_(pm_b * sz_batch_dim, 0) + min_(pm_f * sz_feature_dim, 0) + min_(pm_s * sz_sequence_dim, 0)
                        if arg in (start, stop):
                            sz_func_dim = pm_n * sz_func_dim
                            sz_batch_dim = pm_b * sz_batch_dim
                            sz_feature_dim = pm_f * sz_feature_dim
                            sz_sequence_dim = pm_s * sz_sequence_dim
                            break
                    else: raise TypeError('New space dimension should be adjacent to sequence/ feature/ batch dimensions. ')
                int_args.append(arg)
            else: raise TypeError(f"Ivalid dimension indicator {arg}: should be integers meaning dimensions, or list/set/dict/str of one number (one dimension once). ")
            n_dim += 1
        return FakeSize(int_args, sz_func_dim = sz_func_dim, sz_batch_dim = sz_batch_dim, sz_feature_dim = sz_feature_dim, sz_sequence_dim = sz_sequence_dim)
    
    @classmethod
    def __class_getitem__(cls, arg):
        return iter_dim(cls, arg)

class exist_dim_meta(ABCMeta):

    def __instancecheck__(self, item):
        if isinstance(item, tuple): return all_(self.__instancecheck__(x) for x in item)
        if isinstance(item, dict) and len(item) == 0: return True
        if isinstance(item, set) and len(item) == 1 and isinstance(item[0], int_): return True
        if isinstance(item, list) and all_(isinstance(x, int_) for x in item): return True
        if isinstance(item, str) and len(item) == 0: return True
        if isinstance(item, str):
            try: item = eval(item)
            except: return False
            try: item = tuple(item)
            except: item = (item,)
            if all_(isinstance(x, int_) for x in item): return True
        if isinstance(item, int_): return True
        if item is ... or item is None: return True
        return False

class exist_dim(metaclass=exist_dim_meta):
    def __new__(this, self, *args):
        """
        Conver the dimension representations to actual dimension indices for existed dimensions.
        Integers in special dimension marks represent the index of the dimension OF THIS KIND. 
        Blank marks means all the dimensions of this kind. 
        
        Warning:
            Instead of meaning dimension 1 happens to be a feature dimension, representation '[1]' means
                the second feature dimension (which is not dimension 1 when a tensor has a batch dimension in the front). 

        Exapmles: 
            >>> bt.exist_dim(bt.Size({}, [3, 4]), [])
            [1, 2]
            >>> bt.exist_dim(bt.Size({}, [3, 4]), [1], {})
            [2, 0]
        """
        if len(args) == 1 and (isinstance(args[0], list) and len(args[0]) > 1 or isinstance(args[0], tuple)): args = args[0]
        if len(args) == 0: args = (None,)
        if isinstance(args, FakeSize): return FakeSize(tuple((x + self.n_dim) if x < 0 else x for x in args), sz_batch_dim=args.sz_batch_dim, sz_feature_dim=args.sz_feature_dim, sz_sequence_dim=args.sz_sequence_dim)
        if isinstance(args, Size): args = args.python_repr
        if not (hasattr(self, 'n_dim') and hasattr(self, 'sz_batch_dim')):
            if isinstance(self, torch.Tensor): self = self.as_subclass(torch.Tensor)
            else: self = tuple(self)
            raise AttributeError(f"Cannot get special dimension from {self}. Possible reasons are:\n(1) The input object is not bt.Tensor/Size. \n(2) Special dimensions are lost during unreimplemented torch functions. ")
        self_repr = getattr(self, 'shape', self)
        combined_args = []
        for i, arg in enumerate(args):
            if i == 0: combined_args.append(arg); continue
            if isinstance(arg, int_): combined_args.append(arg)
            elif isinstance(arg, dict) and len(arg) == 0:
                if isinstance(combined_args[-1], (dict, set)): combined_args[-1] = {}
                else: combined_args.append({})
            elif isinstance(arg, set) and len(arg) == 1:
                if isinstance(combined_args[-1], set): raise TypeError(f"Multiple batch dimensions: {{{combined_args[-1]}}} and {arg}. ")
                if isinstance(combined_args[-1], dict): continue
                combined_args.append(arg)
            elif isinstance(arg, list) and isinstance(combined_args[-1], list):
                if len(arg) == 0 or len(combined_args[-1]) == 0: combined_args[-1] = []
                else: combined_args[-1].extend(arg)
            elif isinstance(arg, str) and isinstance(touch(lambda: eval(arg)), (int_, tuple)) and isinstance(combined_args[-1], str):
                if len(arg) == 0 or len(combined_args[-1]) == 0: combined_args[-1] = ''
                else: combined_args = ', '.join([combined_args[-1], arg])
            else: combined_args.append(arg)
        int_args = []
        for arg in combined_args:
            if isinstance(arg, dict) and len(arg) == 0:
                avouch(self.has_batch, TypeError(f"Cannot find batch dimension in {self_repr}. "))
                int_args.append(self.batch_dim)
            elif isinstance(arg, set) and len(arg) == 1:
                i_batch = arg.pop()
                avouch(self.has_batch, TypeError(f"Cannot find batch dimension in {self_repr}. "))
                avouch(i_batch in (0, -1), TypeError(f"Cannot identify the {i_batch}-th batch dimension. "))
                int_args.append(self.batch_dim)
            elif isinstance(arg, list) and len(arg) == 0:
                avouch(self.has_feature, TypeError(f"Cannot find feature dimensions in {self_repr}. "))
                int_args.extend(range_(*self.feature_range))
            elif isinstance(arg, list):
                avouch(self.has_feature, TypeError(f"Cannot find feature dimensions in {self_repr}. "))
                avouch(all_(-self.n_feature_dim <= a < self.n_feature_dim for a in arg), IndexError(f"Cannot identify feature dimensions {arg}: index out of range (the indices are only counted for feature dimensions). "))
                arg = [a + self.n_feature_dim if a < 0 else a for a in arg]
                int_args.extend(a + self.feature_start for a in arg)
            elif isinstance(arg, str) and len(arg) == 0:
                avouch(self.has_sequence, TypeError(f"Cannot find sequence dimensions in {self_repr}. "))
                int_args.extend(range_(*self.sequence_range))
            elif isinstance(arg, str) and arg in self.names: int_args.append(self.names.index(arg))
            elif isinstance(arg, str):
                avouch(self.has_sequence, TypeError(f"Cannot find sequence dimensions in {self_repr}. "))
                try: arg = eval(arg)
                except: raise TypeError(f"Invalid representation for sequence dimensions: '{arg}'. ")
                try: arg = tuple(arg)
                except: arg = (arg,)
                avouch(all_(-self.n_sequence_dim <= a < self.n_sequence_dim for a in arg), IndexError(f"Cannot identify sequence dimensions {arg}: index out of range (the indices are only counted for sequence dimensions). "))
                arg = [a + self.n_sequence_dim if a < 0 else a for a in arg]
                int_args.extend(a + self.sequence_start for a in arg)
            elif arg is ...:
                avouch(self.has_space, TypeError(f"Cannot find space dimensions in {self_repr}. "))
                int_args.extend(range_(*self.space_range))
            elif arg is None:
                avouch(len(int_args) == 0, TypeError(f"Cannot use 'None' along with other dimensions in {self_repr}. "))
                int_args = list(range_(self.n_dim))
                break
            elif isinstance(arg, int_):
                avouch(-self.n_dim <= arg < self.n_dim, IndexError(f"Cannot find dimension {arg} in {self_repr}: dimension out of range, should be in [0, {self.n_dim}). "))
                if arg < 0: arg += self.n_dim
                int_args.append(arg)
            else: raise TypeError(f"Invalid identifier for dimension: {arg!r} in {args!r}.")
        return FakeSize(int_args)
    
    @classmethod
    def __class_getitem__(cls, arg):
        return iter_dim(cls, arg)

class del_dim(exist_dim):
    def __new__(this, self, *args):
        return super().__new__(this, self, *args)
    
class iter_dim:
    def __init__(this, cls, arg):
        avouch(isinstance(arg, int_) or arg in [..., slice(None)], TypeError(f"Invalid subscript for '{cls.__name__}': {arg}, should be int, ... or : ."))
        this.cls = cls
        this.arg = arg
        if arg == ...: arg_str = '...'
        elif arg == slice(None): arg_str = ':'
        else: arg_str = str(arg)
        this.__name__ = f"{cls.__name__}[{arg_str}]"

    def __call__(this, self, *args):
        dims = this.cls(self, *args)
        if isinstance(this.arg, int_):
            avouch(len(dims) == this.arg, TypeError(f"Too many dimensions identified: {dims} by {args}, should be of length {this.arg}. "))
        return dims
    
    def __repr__(this):
        return f"IterativelyPerformedDim<{this.cls.__name__}[{this.arg}]>"
    
class linalg_dim(metaclass=exist_dim_meta):
    def __new__(this, input, *dim, min_n_dim=2, max_n_dim=2):
        """
        Conver the dimension representations to actual dimension indices for existed dimensions.
        It is a specifically designed for linear algebra, hence find the 2D space to perform linalg methods.
        All other rules are the same as 'exist_dim'. 

        Warning:
            Instead of meaning dimension 1 happens to be a feature dimension, representation '[1]' means
                the second feature dimension (which is not dimension 1 when a tensor has a batch dimension in the front). 

        Exapmles: 
            >>> bt.linalg_dim(bt.Size({}, [3, 4]), [])
            [1, 2]
            >>> bt.linalg_dim(bt.Size({}, [3, 4]), [1], {})
            [2, 0]
            >>> bt.linalg_dim(bt.Size(3, 4, 5))
            [1, 2]
            >>> bt.linalg_dim[2](bt.Size([3], 3, 4, 5), [])
            [...]
            TypeError: ...
        """
        if min_n_dim is None: min_n_dim = 1
        if len(dim) == 0 or len(dim) == 1 and dim[0] is None:
            if input.n_feature_dim >= min_n_dim: dim = exist_dim(input, [])
            elif input.n_space_dim >= min_n_dim: dim = exist_dim(input, ...)
            elif input.n_sequence_dim >= min_n_dim: dim = exist_dim(input, '')
            else: raise TypeError(f"Invalid size {input.shape} for linalg_dim: at least {min_n_dim} non-batch dimension needed. ")
        else: dim = exist_dim(input, *dim)
        if max_n_dim is not None and max_n_dim > 0 and len(dim) > max_n_dim: dim = dim[-max_n_dim:]
        return dim
    
    @classmethod
    def __class_getitem__(cls, arg):
        if isinstance(arg, slice):
            avouch(arg.step is None, TypeError("'linalg_dim' cannot accept 2 colons in subscript. "))
            arg = arg.start, arg.stop
        if not isinstance(arg, tuple): arg = arg, arg
        avouch(len(arg) == 2 and (arg[0] is None or isinstance(arg[0], int_)) and (arg[1] is None or isinstance(arg[1], int_)), 
               TypeError("'linalg_dim' takes only subscripts of (int, int), indicating the min/max number of dimensions. "))
        ret = lambda *a: linalg_dim(*a, min_n_dim=arg[0], max_n_dim=arg[1])
        ret.__name__ = f"linalg_dim[{arg[0]}, {arg[1]}]"
        return ret

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
    # if self_shape.has_feature and other_shape.has_feature \
    #     and self_shape.n_feature_dim + other_shape.n_feature_dim >= 3:
    #     dims = exist_dim(self_shape, [])[-2:]
    #     l_shape = [self_shape[d] for d in dims]
    #     repr_self_shape = self_shape.with_dim_size(dims[0], -1).with_dim_size(dims[-1], -1)
    #     dims = exist_dim(other_shape, [])[-2:]
    #     r_shape = [other_shape[d] for d in dims]
    #     repr_other_shape = other_shape.with_dim_size(dims[0], -1).with_dim_size(dims[-1], -1)
    # elif self_shape.has_space and other_shape.has_space \
    #     and self_shape.n_space_dim + other_shape.n_space_dim >= 3:
    #     dims = exist_dim(self_shape, ...)[-2:]
    #     l_shape = [self_shape[d] for d in dims]
    #     repr_self_shape = self_shape.with_dim_size(dims[0], -1).with_dim_size(dims[-1], -1)
    #     dims = exist_dim(other_shape, ...)[-2:]
    #     r_shape = [other_shape[d] for d in dims]
    #     repr_other_shape = other_shape.with_dim_size(dims[0], -1).with_dim_size(dims[-1], -1)
    # elif self_shape.has_sequence and other_shape.n_sequence_dim \
    #     and self_shape.n_sequence_dim + other_shape.n_sequence_dim >= 3:
    #     dims = exist_dim(self_shape, '')[-2:]
    #     l_shape = [self_shape[d] for d in dims]
    #     repr_self_shape = self_shape.with_dim_size(dims[0], -1).with_dim_size(dims[-1], -1)
    #     dims = exist_dim(other_shape, '')[-2:]
    #     r_shape = [other_shape[d] for d in dims]
    #     repr_other_shape = other_shape.with_dim_size(dims[0], -1).with_dim_size(dims[-1], -1)
    # else: raise TypeError(f"Cannot perform matmul alignment for shapes {self_shape} and {other_shape}")
    # avouch(l_shape[-1] == r_shape[0] or len(l_shape) == len(r_shape) == 1, 
    #        TypeError(f"Cannot perform matmul alignment for shapes {self_shape} and {other_shape}"))
    # res_shape = [l_shape[0], r_shape[-1]]
    
    # print(self_shape, other_shape, repr_self_shape, repr_other_shape, x_shape, y_shape, l_shape, r_shape, res_shape)
    # dims = [d for d, s in enumerate(x_shape) if s < 0]
    # avouch(len(dims) in (1, 2), RuntimeError(f"Error in matmul alignment for shapes {self_shape} and {other_shape}"))
    # if len(dims) == 2: x_shape = x_shape.with_dim_size(dims[0], res_shape[0]).with_dim_size(dims[1], res_shape[1])
    # else: x_shape = x_shape[:dims[0]] + Size(res_shape).add_special_dim(0, x_shape[dims[0]:dims[0]+1]).add_special_dim(1, x_shape[dims[0]:dims[0]+1]) + x_shape[dims[0]+1:]
    # dims = [d for d, s in enumerate(y_shape) if s < 0]
    # avouch(len(dims) in (1, 2), RuntimeError(f"Error in matmul alignment for shapes {self_shape} and {other_shape}"))
    # if len(dims) == 2: y_shape = y_shape.with_dim_size(dims[0], res_shape[0]).with_dim_size(dims[1], res_shape[1])
    # else: y_shape = y_shape[:dims[0]] + Size(res_shape).add_special_dim(0, y_shape[dims[0]:dims[0]+1]).add_special_dim(1, y_shape[dims[0]:dims[0]+1]) + y_shape[dims[0]+1:]
    # return x_shape, x_shape, y_shape

size_mapping_op = defaultdict(lambda: (lambda a, b: (lambda u, v: (u, u, v))(*a^b)),
    __matmul__ = matmul_shape,
    mm = matmul_shape,
    bmm = matmul_shape,
    smm = matmul_shape,
    addmm = lambda a, b, c: (lambda a, _, u, v: (lambda u, v: (u, u, v))(*a^u) + ((a^v)[1],))(a, *matmul_shape(b, c)),
    addbmm = lambda a, b, c: (lambda a, _, u, v: (lambda u, v: (u, u, v))(*a^u) + ((a^v)[1],))(a, *matmul_shape(b, c)),
    saddmm = lambda a, b, c: (lambda a, _, u, v: (lambda u, v: (u, u, v))(*a^u) + ((a^v)[1],))(a, *matmul_shape(b, c)),
    addcmul = lambda a, b, c: (lambda b: (b,) + tuple(b.updated_sizes))(broadcast(a, b, c, with_size_updates=True)),
    quantile = lambda s, q, d, **k: ((func_dim if q.n_dim > 0 else Size()) + (s if k.get('keepdim', False) else remove_dim(s, d)), None, None),
    where = lambda c, x, y: (lambda b: (b,) + tuple(b.updated_sizes))(broadcast(c, x, y, with_size_updates=True)),
)

def get_updated_code(func, mode='function', ignore_args=[]):
    """
    Get codes for inheritance.
    
    Args:
        func (function): the function object to be expanded. 
        mode (str: function|method): the string indicating the block is a function or a method. 
        ignore_arg (list): the list of arguments that are ignored in auto generated codes.
    """
    ant = func.__annotations__
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    f_code = func.__code__
    line_no = f_code.co_firstlineno
    declaration = __mirror__[line_no].strip()
    decorators = []
    while declaration.startswith('@'):
        decorators.append(declaration)
        line_no += 1
        declaration = __mirror__[line_no].strip()
    _def, _fname, _args, *_tail = tokenize(declaration, sep=[' ', '(', ')', '\n'])

    for ig in ignore_args:
        _args = token_replace(_args, lambda x: x.lstrip('*').startswith(ig), '', sep=[','], strip=' ').replace(',,', ',').rstrip(',')

    is_inplace = _fname.endswith('_') and _fname[-2:] != '__'
    args_name = f_code.co_varnames[f_code.co_argcount + f_code.co_kwonlyargcount] if f_code.co_flags & 0x04 else ''
    
    # Copy the manually defined 'inner_code' in the declaration. 
    num_indent = get_num_indent(__mirror__[line_no])
    indent = lambda k: " " * 4 * k
    inner_codes = ""
    if not _tail[-1] in ('...', 'pass'):
        qts = '\'"'
        nesters = r"()[]{}"
        find_left = {r:l for l, r in zip(nesters[::2], nesters[1::2])}
        find_right = {l:r for l, r in zip(nesters[::2], nesters[1::2])}

        inner_codes = [''] # Due to the indent issue, a blank line was added at the front. 
        depth = {}
        scopes = []
        cur_line_block = []
        force_indent = False
        l = line_no - 1
        while True:
            l += 1
            line = __mirror__[l]
            if l == line_no: line = tokenize(line, sep=':', by="()[]{}$$``''\"\"##")[-1].strip()
            if not line.strip(): continue
            if l == line_no:
                inner_codes.append(indent(num_indent + 1) + line)
                break
            skip_char = False
            for i, c in enumerate(line):
                if skip_char: skip_char = False; continue
                in_str = depth.get('"', 0) + depth.get("'", 0) + depth.get('"""', 0) + depth.get("'''", 0) > 0
                if c == '#' and not in_str: break
                if c == '\\' and not in_str:
                    if depth[c] > 0: raise SyntaxError("unexpected character after line continuation character")
                    depth[c] = 1
                elif c == '\\': skip_char = True; continue
                elif c in qts:
                    if depth.get(qts[1-qts.index(c)], 0) + depth.get(qts[1-qts.index(c)]*3, 0) == 0:
                        if line[i:i+3] == c*3: depth[c*3] = 1 - depth.get(c*3, 0)
                        else: depth[c] = 1 - depth.get(c, 0)
                elif c in find_right:
                    depth[c] = depth.get(c, 0) + 1
                elif c in find_left:
                    c_left = find_left[c]
                    if depth.get(c_left, 0) <= 0: raise SyntaxError(f"unmatched {c!r}")
                    depth[c_left] -= 1
            cur_line_block.append(line)
            if sum_(depth.values()) == 0:
                notab_line = cur_line_block[0].lstrip('\t')
                cur_line_block[0] = indent(len(cur_line_block[0]) - len(notab_line)) + notab_line
                if get_num_indent(cur_line_block[0]) <= num_indent: cur_line_block = []; break
                inner_line = '\n'.join(cur_line_block)
                if get_num_indent(cur_line_block[0]) >= num_indent + len(scopes) + 2: raise IndentationError("unexpected indent")
                while True:
                    if get_num_indent(cur_line_block[0]) >= num_indent + len(scopes) + 1: break
                    scopes.pop(-1)
                if force_indent:
                    if get_num_indent(cur_line_block[0]) == num_indent + len(scopes) + 1: force_indent = False
                    else: raise IndentationError(f"expected an indented block after function definition in line:\n {inner_codes[-1]}\n>{inner_line}")
                if inner_line.rstrip().endswith(':'):
                    scopes.append(inner_line.lstrip().split(maxsplit=1)[0])
                    force_indent = True
                inner_codes.append(inner_line)
                cur_line_block = []
            elif depth.get('\\', 0) > 0: depth['\\'] = 0
        if sum_(depth.values()) > 0:
            c = [k for k, n in depth.items() if n > 0][0]
            raise SyntaxError(f"{c!r} was never closed")
        if cur_line_block: inner_codes.append('\n'.join(cur_line_block))
        for i, inner_line in enumerate(inner_codes):
            nosp_line = inner_line.lstrip(' ')
            prefix = ' ' * (len(inner_line) - len(nosp_line))
            nosp_line = ';'.join(f'obj = {x[7:]}' if x.startswith('return ') else x for x in tokenize(nosp_line, sep=';'))
            nosp_line = ':'.join(f'obj = {x[7:]}' if x.startswith('return ') else x for x in tokenize(nosp_line, sep=':'))
            inner_codes[i] = prefix + nosp_line
        inner_codes = '\n'.join(inner_codes) + '\n'
        delta_indent = 1 - num_indent
        new_inner_codes = []
        for line in inner_codes.split('\n'):
            if delta_indent > 0: line = indent(delta_indent) + line
            else: line = line[len(indent(1)) * (-delta_indent):]
            new_inner_codes.append(line)
        inner_codes = '\n'.join(new_inner_codes) + '\n'
    if inner_codes.strip() in ('...', 'pass'): inner_codes = ""
    # if not inner_codes and decorators:
    #     raise RuntimeError("Internal error: no decorator is allowed for auto generated functions in batorch.tensor (Please contact the developer with Error Code: B531). ")
    tensor_args = []
    size_args = []
    dim_args = []
    iter_dim_args = []
    rev_dim_args = []
    rmv_star_args = []
    for a, t in ant.items():
        if isinstance(t, iter_dim) and t.arg==slice(None):
            dim_args.append(a)
            rmv_star_args.append(a)
        elif isinstance(t, iter_dim):
            dim_args.append(a)
            iter_dim_args.append(a)
            if t.cls == del_dim:
                rev_dim_args.append(a)
        elif t == 'Tensor' or isinstance(t, type) and t.__name__ == 'Tensor': tensor_args.append(a)
        elif t == 'Size' or isinstance(t, type) and t.__name__ == 'Size': size_args.append(a)
        elif t in (new_dim, del_dim, exist_dim): dim_args.append(a)
        elif getattr(t, '__name__', '').startswith('linalg_dim'): dim_args.append(a)
    parts = []
    for x in _args.split(','):
        x = x.strip()
        if x == '*': continue
        if ':' in x: x = x.split(':')[0]
        if '=' not in x: parts.append(x); continue
        param = x.split('=', 1)[0]
        parts.append(param + '=' + param.strip())
    if mode == 'method': self, *parts = parts
    else: self = parts[0]
    inherit_args = ','.join(p for p in parts if not p.startswith('*') and not '=' in p)
    inherit_kwargs = ','.join([p for p in parts if p.startswith('*') or '=' in p])

    # Deal with the shapes and special dimensions, as well as generate the major code blocks. 
    reshape_op = ""
    if len(tensor_args) >= 2:
        self = tensor_args[0]
        reshape_op = [f"x_shape = {self}.shape"]
        for y in tensor_args[1:]:
            reshape_op.append(f"x_shape, y_shape = x_shape ^ {y}.shape")
            reshape_op.append(f"{y} = {y}.view(y_shape)")
        for y in tensor_args[1:-1]:
            reshape_op.append(f"_, y_shape = x_shape ^ {y}.shape")
            reshape_op.append(f"{y} = {y}.view(y_shape)")
        reshape_op.append(f"{self} = {self}.view(x_shape)")
        reshape_op = '; '.join(reshape_op)
    cast = ('\n' + indent(2)).join(
        [('\n' + indent(2)).join([
            f"{x} = torch.tensor({x}) if {x} is not None and not isinstance({x}, torch.Tensor) else {x}",
            f"{x} = {x}.as_subclass(Tensor).special_from({x}.shape) if {x} is not None and not isinstance({x}, Tensor) else {x}",
            f"if {x}.device != {tensor_args[0]}.device and {x}.device.type == 'cpu': {x} = {x}.to({tensor_args[0]}.device)" if i > 0 else '',
        ]) for i, x in enumerate(tensor_args)] + 
        [f"{x} = {ant[x].__name__}({self}, %s{x})"%('*' if x == args_name else '') for x in dim_args]
    )
    get_size = ('\n' + indent(2)).join([f"{x}_shape=None if {x} is None else Size({x}.shape)" for x in tensor_args] + [f'{x}=Size(%s{x})'%('*' if x == args_name else '') for x in size_args])
    size_fname = _fname
    if is_inplace: size_fname = size_fname.rstrip('_')
    size_arguments = ', '.join([f'{x}_shape' for x in tensor_args] + size_args + dim_args)
    kwargs_dict = "dict(" + ', '.join(f"{x}={x}" for x in f_code.co_varnames[:f_code.co_argcount + f_code.co_kwonlyargcount + ((f_code.co_flags & 0x04) >> 2) + ((f_code.co_flags & 0x08) >> 3)] if x not in tensor_args and x not in size_args + dim_args) + ")"
    if len(tensor_args) >= 2:
        size_reference = "ref_shape"
    else:
        kwarg = f", **{kwargs_dict}" if size_mapping[size_fname].__code__.co_flags & 0x08 else ''
        size_reference = f"size_mapping['{size_fname}']({size_arguments}{kwarg})"
    
    # Generate auto 'inner_codes'. 
    if not inner_codes:
        if len(tensor_args) >= 2:
            kwarg = f", **{kwargs_dict}" if size_mapping_op[size_fname].__code__.co_flags & 0x08 else ''
            inner_codes = "ref_shape, " + ', '.join([f'{x}_shape' for x in tensor_args]) + \
                f" = size_mapping_op['{size_fname}']({size_arguments}{kwarg})\n"
            for x in tensor_args:
                inner_codes += indent(2) + f"{x} = {x}.view({x}_shape)\n"
            inner_codes += indent(2)
        else: inner_codes = ""
        parent = f"torch_super({self}, '{_fname}')" if mode == 'method' else f'torch.{_fname}'
        if len(iter_dim_args) > 0:
            lists = ', '.join([iarg + '[::-1]' if iarg in rev_dim_args else iarg for iarg in iter_dim_args])
            iters = ', '.join([d + '_iter' for d in iter_dim_args])
            if len(iter_dim_args) == 1: iters += ','
            for d in iter_dim_args:
                inherit_args = token_replace(inherit_args, d, d+'_iter', sep=[',', '='], strip=' ')
                inherit_kwargs = token_replace(inherit_kwargs, lambda x: x.lstrip('*') == d, d+'_iter', sep=[',', '='], strip=' ')
            inner_codes += "with torch._C.DisableTorchFunction():\n"
            inner_codes += indent(3) + f"for {iters} in zip({lists}):\n"
            inner_codes += indent(4) + f"auto_generated_args = tuple(x for x in [{inherit_args}] if x is not None)\n"
            if is_inplace: inner_codes += indent(4) + f"{parent}(*auto_generated_args, {inherit_kwargs})\n"
            else: inner_codes += indent(4) + f"{self} = {parent}(*auto_generated_args, {inherit_kwargs})\n"
            inner_codes += indent(2) + f"obj = {self}\n"
        else:
            for d in rmv_star_args:
                inherit_kwargs = token_replace(inherit_kwargs, lambda x: x.lstrip('*') == d, d, sep=[',', '='], strip=' ')
            inner_codes += "with torch._C.DisableTorchFunction():\n"
            inner_codes += indent(3) + f"auto_generated_args = tuple(x for x in [{inherit_args}] if x is not None)\n"
            if is_inplace:
                inner_codes += indent(3) + f"{parent}(*auto_generated_args, {inherit_kwargs})\n"
                inner_codes += indent(3) + f"obj = {self}\n"
            else: inner_codes += indent(3) + f"obj = {parent}(*auto_generated_args, {inherit_kwargs})\n"
    if 'special_from' not in inner_codes:
        if is_inplace: inner_codes += indent(2) + f"obj.special_from({size_reference})"
        else: inner_codes += indent(2) + f"obj = obj.as_subclass(Tensor).special_from({size_reference}, allow_view=True)"
        
    # Deal with **kwargs:
    if not f_code.co_flags & 0x08:
        _args += ", sz_func_dim=0, sz_batch_dim=0, sz_feature_dim=0, sz_sequence_dim=0"
        manual_param = """
        if sz_func_dim != 0: obj.sz_func_dim = sz_func_dim
        if sz_batch_dim != 0: obj.sz_batch_dim = sz_batch_dim
        if sz_feature_dim != 0: obj.sz_feature_dim = sz_feature_dim
        if sz_sequence_dim != 0: obj.sz_sequence_dim = sz_sequence_dim
        """
    else:
        kwarg = f_code.co_varnames[f_code.co_argcount + f_code.co_kwonlyargcount + ((f_code.co_flags & 0x04) >> 2)]
        manual_param = f"""
        if {kwarg}.get('sz_func_dim', 0) != 0: obj.sz_func_dim = {kwarg}['sz_func_dim']
        if {kwarg}.get('sz_batch_dim', 0) != 0: obj.sz_batch_dim = {kwarg}['sz_batch_dim']
        if {kwarg}.get('sz_feature_dim', 0) != 0: obj.sz_feature_dim = {kwarg}['sz_feature_dim']
        if {kwarg}.get('sz_sequence_dim', 0) != 0: obj.sz_sequence_dim = {kwarg}['sz_sequence_dim']
        """
    
    if decorators: decorators = ('\n' + indent(num_indent)).join(decorators)
    else: decorators = ''

    # Return the auto-generated codes for the function. 
    return f"""
    {decorators}
    {_def} {_fname}({_args}):
        # Cast all arguments to the wanted 'Tensor' or 'Size' type, and convert the dim arguments. 
        {cast}
        # Obtain available sized in arguments (which will be fed into size function).  
        {get_size}
        # Use the given inner codes if they are provided. 
        {inner_codes}
        {manual_param}
        if getattr(obj, 'grad_fn', None) is not None:
            obj.grad_fn_name = "{_fname}"
        return obj
    """
    
basic_locals = list(locals().keys()) + ['basic_locals', '__indexed_reduce__']

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
        inv_output = torch.linalg.inv(input.move_dim(dim, -1)).as_subclass(Tensor).init_special()
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
        return torch.det(input.move_dim(dim, -1)).as_subclass(Tensor).special_from(ref_shape) # suppress: special_from

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
        return torch.addbmm(input,batch1,batch2, beta=beta,alpha=alpha,out=out).as_subclass(Tensor).special_from(input.shape)
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
        if len(ret) >= 1 and ret[0].ndim == 1: ret[0] = ret[0].as_subclass(Tensor)
        elif len(ret) >= 1: ret[0] = ret[0].as_subclass(Tensor).special_from(input_shape)
        if len(ret) >= 2 and return_inverse: ret[1] = ret[1].as_subclass(Tensor).special_from(input_shape)
        elif len(ret) >= 2: ret[1] = ret[1].as_subclass(Tensor)
        if len(ret) >= 3: ret[2] = ret[2].as_subclass(Tensor)
        return ret
    elif ret.ndim == 1:
        return ret.as_subclass(Tensor).init_special()
    else: return ret.as_subclass(Tensor).special_from(input_shape)
def isnan(input: 'Tensor'): ...
def isinf(input: 'Tensor'): ...

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
        return torch_super(self, 'flatten')(flat_range[0], flat_range[1]-1).as_subclass(Tensor).special_from(ref_shape)
def transpose(self: 'Tensor', dim0: exist_dim[1], dim1: exist_dim[1]): ...
def t(self): ...
def permute(self: 'Tensor', *dims):
    ref_shape = Size(*dims)
    dims = exist_dim(self, *dims)
    with torch._C.DisableTorchFunction():
        obj = torch.permute(self, dims)
    obj = obj.as_subclass(Tensor).special_from(size_mapping['permute'](self_shape, dims))
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
        return torch_super(self, 'repeat')(*size).as_subclass(Tensor).special_from(self).update_special_from(size)

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
        if dim is None: res = torch.quantile(self, q, keepdim=keepdim,interpolation=interpolation).as_subclass(Tensor).special_from(q); dim = [int_(q.n_dim > 0)]
        else: res = torch.quantile(self, q, dim[0], keepdim=keepdim,interpolation=interpolation).as_subclass(Tensor).special_from(ref_shape)
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
def cumsum(input: 'Tensor', dim: del_dim[1], dtype=None, out=None): ...
def cumprod(input: 'Tensor', dim: del_dim[1], dtype=None, out=None): ...

def __indexed_reduce__(func_name, self: 'Tensor', *dim, keepdim=None, out=None, ret_index_only=False):
    if len(dim) == 1 and isinstance(dim[0], torch.Tensor):
        other = dim[0]
        other = other.as_subclass(Tensor).special_from(other.shape) if not isinstance(other, Tensor) else other
        self_shape = Size(self.shape); other_shape = Size(other.shape)
        ref_shape, self_shape, other_shape = size_mapping_op[func_name](self_shape, other_shape)
        with torch._C.DisableTorchFunction():
            return torch_super(self, func_name)(other, **(dict(out=out) if locals().get('out', None) is not None else {})).as_subclass(Tensor).special_from(ref_shape)
    dim = del_dim(self, *dim)
    indices = None
    num_dim = 0
    init_shape = self.shape
    with torch._C.DisableTorchFunction():
        for d in dim[::-1]:
            result = torch_super(self, func_name)(d, **dict(keepdim=keepdim) if keepdim is not None else {})
            self = result.values
            res_indices = result.indices.as_subclass(Tensor).init_special()
            if indices is None: indices = res_indices.unsqueeze(0, sz_func_dim=1)
            elif keepdim or keepdim is None: indices = cat(res_indices.unsqueeze(0, sz_func_dim=1), indices.gather(d+1, res_indices.duplicate(num_dim, 0, sz_func_dim=1)), 0)
            else: indices = cat(res_indices.unsqueeze(0, sz_func_dim=1), indices.gather(d+1, res_indices.unsqueeze(d).duplicate(num_dim, 0, sz_func_dim=1)).squeeze_(d+1), 0)
            num_dim += 1
    if keepdim is False: cur_shape = remove_dim(init_shape, dim)
    else: cur_shape = init_shape
    if ret_index_only:
        indices = indices.special_from(func_dim + cur_shape) if indices is not None else None
        if len(dim) == 1: indices.squeeze_(0)
        indices.indices = indices
        indices.values = self.as_subclass(Tensor).special_from(cur_shape)
        if out is not None:
            out.zero_().add_(indices)
        return indices
    else:
        self = self.as_subclass(Tensor).special_from(cur_shape)
        self.indices = indices.special_from(func_dim + cur_shape) if indices is not None else None
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
            return tuple(x.as_subclass(Tensor).special_from(self).squeeze_(dim) for x in torch_super(self, 'split')(split_size, dim))
        else: return tuple(x.as_subclass(Tensor).special_from(self) for x in torch_super(self, 'split')(split_size, dim))

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
        L = L.as_subclass(Tensor).init_special()
        V = V.as_subclass(Tensor).init_special()
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

def matpow(input: 'Tensor', k, *, dim: linalg_dim[2]=None):
    """return a matrix power of A^k."""
    L, V = eig(input, dim=dim)
    L = L.move_dim(dim[0], -1)
    V = V.move_dim(dim, -1)
    L_k = where((L.real < 0) & (L.imag.abs() < 1e-6), -complex((-L.real) ** k, L.imag), L ** k)
    R = V @ diag(L_k, dim=-1) @ V.inv()
    if R.is_complex() and not input.is_complex(): R = R.real
    return R.move_dim([-2, -1], dim).type(input.dtype) # suppress: special_from

def matexp(input: 'Tensor', *, dim: linalg_dim[2]=None):
    """return a matrix exponential of e^A."""
    L, V = eig(input, dim=dim)
    L = L.move_dim(dim[0], -1)
    V = V.move_dim(dim, -1)
    R = V @ diag(exp(L), dim=-1) @ V.inv()
    if R.is_complex() and not input.is_complex(): R = R.real
    return R.move_dim([-2, -1], dim).type(input.dtype) # suppress: special_from

def matlog(input: 'Tensor', *, dim: linalg_dim[2]=None):
    """return a matrix exponential of e^A."""
    L, V = eig(input, dim=dim)
    L = L.move_dim(dim[0], -1)
    V = V.move_dim(dim, -1)
    R = V @ diag(log(L), dim=-1) @ V.inv()
    if R.is_complex() and not input.is_complex(): R = R.real
    return R.move_dim([-2, -1], dim).type(input.dtype) # suppress: special_from

def rank(input: 'Tensor', *, atol=None, rtol=None, hermitian=False, dim: linalg_dim[2]=None, out=None):
    A = input.move_dim(dim, -1)
    with torch._C.DisableTorchFunction():
        return torch.linalg.matrix_rank(A, atol=atol, rtol=rtol, hermitian=hermitian, **dict(out=out) if locals().get('out', None) is not None else {}).as_subclass(Tensor).special_from(A.shape[:-2])

def matnorm(input: 'Tensor', ord='fro', dim: linalg_dim[2]=None, keepdim=False, *, dtype=None, out=None):
    A = input.move_dim(dim, -1)
    with torch._C.DisableTorchFunction():
        return torch.linalg.matrix_norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, **dict(out=out) if locals().get('out', None) is not None else {}).as_subclass(Tensor).special_from(A.shape if keepdim else A.shape[:-2])

biway_functions = [f for f in locals() if f not in basic_locals]

class Size(tuple):

    @classmethod
    def __new_raw__(cls, shape, sz_func_dim: int = 0, sz_batch_dim: int = 0, sz_feature_dim: int = 0, sz_sequence_dim: int = 0):
        """
        The raw construction function defined by the inner parameters.

        Args:
            shape (tuple of ints): The raw tuple structure. 
            sz_func_dim (int, optional): An inner parameter for functional dimension, it can only be 0, 1, or -1. Defaults to 0.
            sz_batch_dim (int, optional): An inner parameter for batch dimension, it can only be 0, 1, or -1. Defaults to 0.
            sz_feature_dim (int, optional): An inner parameter for feature dimensions, being positive when they are in front of the space-sequence dimensions. Defaults to 0.
            sz_sequence_dim (int, optional): An inner parameter for sequence dimensions, being positive when they are in front of the space dimensions. Defaults to 0.
        """
        avouch(isinstance(shape, tuple) and all_(isinstance(s, num_) for s in shape), TypeError(f"Invalid 'shape = {shape}' for bt.Size, should be a tuple. "))
        avouch(sz_func_dim in (0, 1, -1), TypeError(f"Invalid 'sz_func_dim = {sz_func_dim}' for bt.Size, should be 0, 1, or -1. "))
        avouch(sz_batch_dim in (0, 1, -1), TypeError(f"Invalid 'sz_batch_dim = {sz_batch_dim}' for bt.Size, should be 0, 1, or -1. "))
        avouch(isinstance(sz_feature_dim, int_), TypeError(f"Invalid 'sz_feature_dim = {sz_feature_dim}' for bt.Size, should be an integer. "))
        avouch(isinstance(sz_sequence_dim, int_), TypeError(f"Invalid 'sz_sequence_dim = {sz_sequence_dim}' for bt.Size, should be an integer. "))
        avouch(len(shape) >= abs_(sz_func_dim) + abs_(sz_batch_dim) + abs_(sz_feature_dim) + abs_(sz_sequence_dim), TypeError(f"Too many special dimensions for shape of length {len(shape)}: sz_func_dim = {sz_func_dim}, sz_batch_dim = {sz_batch_dim}, sz_feature_dim = {sz_feature_dim}, sz_sequence_dim = {sz_sequence_dim}. "))
        self = super().__new__(cls, shape)
        self.sz_func_dim = sz_func_dim
        self.sz_batch_dim = sz_batch_dim
        self.sz_feature_dim = sz_feature_dim
        self.sz_sequence_dim = sz_sequence_dim
        self.n_dim = self.ndim = len(shape)
        return self

    @classmethod
    def __new_size__(cls, size, **kwargs):
        """The construction function for a bt.Size object. """
        avouch(isinstance(size, Size), TypeError(f"Invalid 'size = {size}' for bt.Size, should be a bt.Size object. "))
        kwargs.setdefault("sz_func_dim", size.sz_func_dim)
        kwargs.setdefault("sz_batch_dim", size.sz_batch_dim)
        kwargs.setdefault("sz_feature_dim", size.sz_feature_dim)
        kwargs.setdefault("sz_sequence_dim", size.sz_sequence_dim)
        return cls.__new_raw__(tuple(size), **kwargs)

    @classmethod
    def __new_tuple__(cls, shape, func_dim: (int, null) = None, batch_dim: (int, null) = None, channel_dim: (int, null) = None, sequence_dim: (int, null) = None, n_feature_dim: int = None, n_sequence_dim: int = None, sz_func_dim: int = None, sz_batch_dim: int = None, sz_feature_dim: int = None, sz_sequence_dim: int = None):
        """
        The construction function for a tuple with readable parameters. 

        Args:
            shape (tuple of ints): the raw tuple structure. 
            func_dim (int, null, optional): The index of the functional dimension, having a domain of 0 or n_dim - 1, the first or last dimension. Defaults to None.
            batch_dim (int, null, optional): The index of the batch dimension, having a domain of 0 or n_dim - 1, the first or last dimension. Defaults to None.
            channel_dim (int, null, optional): The index of the channel dimension, being the first or last dimension apart from the batch dimension. Defaults to None.
            sequence_dim (int, null, optional): The index of the sequence dimension, having the first or last dimension apart from the batch and channel dimension. Defaults to None.
            n_feature_dim (int, optional): The number of feature dimensions (to the left of space dimensions), conflict to argument 'channel_dim'. Defaults to None.
            n_sequence_dim (int, optional): The number of sequence dimensions (to the right of space dimensions), conflict to argument 'sequence_dim'. Defaults to None.
            sz_func_dim (int, optional): The sz number of the functional dimension, conflict to argument 'func_dim'. Defaults to None.
            sz_batch_dim (int, optional): The sz number of the batch dimension, conflict to argument 'batch_dim'. Defaults to None.
            sz_feature_dim (int, optional): The sz number of feature dimensions, conflict to arguments 'channel_dim' and 'n_feature_dim'. Defaults to None.
            sz_sequence_dim (int, optional): The sz number of sequence dimensions, conflict to arguments 'sequence_dim' and 'n_sequence_dim'. Defaults to None.
        """
        avouch(isinstance(shape, tuple), TypeError(f"Invalid 'shape = {shape}' for bt.Size, should be a tuple. "))
        if len(shape) > 0 and not all_(isinstance(x, num_) for x in shape):
            raw_shape = cls.__new_repr__(shape)
            shape = tuple(raw_shape)
            avouch(func_dim is None or raw_shape.sz_func_dim == 0, TypeError(f"Invalid 'shape = {shape}; func_dim = {func_dim}' for bt.Size (conflict in func dimension)."))
            avouch(batch_dim is None or raw_shape.sz_batch_dim == 0, TypeError(f"Invalid 'shape = {shape}; batch_dim = {batch_dim}' for bt.Size (conflict in batch dimension)."))
            avouch(channel_dim is None and n_feature_dim is None or raw_shape.sz_feature_dim == 0 and (channel_dim is None or n_feature_dim is None), 
                   TypeError(f"Invalid 'shape = {shape}; channel_dim = {channel_dim}; n_feature_dim = {n_feature_dim}' for bt.Size (conflict in feature dimensions)."))
            avouch(sequence_dim is None and n_sequence_dim is None or raw_shape.sz_sequence_dim == 0 and (sequence_dim is None or n_sequence_dim is None), 
                   TypeError(f"Invalid 'shape = {shape}; sequence_dim = {sequence_dim}; n_sequence_dim = {n_sequence_dim}' for bt.Size (conflict in sequence dimensions)."))
            if raw_shape.sz_func_dim != 0:
                avouch(sz_func_dim is None, TypeError("Conflict arguments during the creation of Size: sz_func_dim"))
                sz_func_dim = raw_shape.sz_func_dim
            if raw_shape.sz_batch_dim != 0:
                avouch(sz_batch_dim is None, TypeError("Conflict arguments during the creation of Size: sz_batch_dim"))
                sz_batch_dim = raw_shape.sz_batch_dim
            if raw_shape.sz_feature_dim != 0:
                avouch(sz_feature_dim is None, TypeError("Conflict arguments during the creation of Size: sz_feature_dim"))
                sz_feature_dim = raw_shape.sz_feature_dim
            if raw_shape.sz_sequence_dim != 0:
                avouch(sz_sequence_dim is None, TypeError("Conflict arguments during the creation of Size: sz_sequence_dim"))
                sz_sequence_dim = raw_shape.sz_sequence_dim
        if sz_func_dim is None: sz_func_dim = 0
        if sz_batch_dim is None: sz_batch_dim = 0
        if sz_feature_dim is None: sz_feature_dim = 0
        if sz_sequence_dim is None: sz_sequence_dim = 0
        n_dim = len(shape)
        # Deal with the func dimension
        if sz_func_dim == 0:
            func_cands = (None, 0, -n_dim, n_dim-1, -1)
            avouch(func_dim in func_cands, TypeError(f"Invalid 'func_dim = {func_dim}' for bt.Size, should be None or {func_cands[1]} (before the space-sequence dimensions), or {func_cands[-1]} (after the space-sequence dimensions). "))
            sz_func_dim = 0 if func_dim is None else (1 if func_dim in (0, -n_dim) else -1)
        # Deal with the batch dimension
        if sz_batch_dim == 0:
            batch_cands = (None, 
                           max_(sz_func_dim, 0), 
                           max_(sz_func_dim, 0) - n_dim, 
                           n_dim - 1 + min_(sz_func_dim, 0), 
                           min_(sz_func_dim, 0) - 1)
            avouch(batch_dim in batch_cands, TypeError(f"Invalid 'batch_dim = {batch_dim}' for bt.Size, should be None or {batch_cands[1]} (before the space-sequence dimensions), or {batch_cands[-1]} (after the space-sequence dimensions). "))
            sz_batch_dim = 0 if batch_dim is None else (1 if batch_dim in (0, -n_dim) else -1)
        # Deal with the feature dimension(s)
        if sz_feature_dim == 0:
            if n_feature_dim is None:
                channel_cands = (None, 
                                 max_(sz_func_dim, 0) + max_(sz_batch_dim, 0), 
                                 max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) - n_dim, 
                                 n_dim - 1 + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0), 
                                 min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) - 1)
                avouch(channel_dim in channel_cands, TypeError(f"Invalid 'channel_dim = {channel_dim}' for bt.Size, should be None or {channel_cands[1]} (before the space-sequence dimensions), or {channel_cands[-1]} (after the space-sequence dimensions). "))
                sz_feature_dim = 0 if channel_dim is None else (1 if channel_dim in channel_cands[1:3] else -1)
            elif channel_dim is not None: raise TypeError("Argument 'channel_dim' is conflict to 'n_feature_dim' for bt.Size, they cannot be assigned simultaneously. ")
            else:
                avouch(isinstance(n_feature_dim, int_) and n_feature_dim >= 0, TypeError(f"Invalid 'n_feature_dim = {n_feature_dim}' for bt.Size, should be an integer. "))
                sz_feature_dim = n_feature_dim
        # Deal with the sequence dimension(s)
        if sz_sequence_dim == 0:
            if n_sequence_dim is None:
                sequence_cands = (None, 
                                  max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0), 
                                  max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0) - n_dim, 
                                  n_dim - 1 + min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0), 
                                  min_(sz_func_dim, 0) + min_(sz_batch_dim, 0) + min_(sz_feature_dim, 0) - 1)
                avouch(sequence_dim in sequence_cands, TypeError(f"Invalid 'sequence_dim = {sequence_dim}' for bt.Size, should be one of None, {sequence_cands[1]} (before the space dimensions), or {sequence_cands[-1]} (after the space dimensions)"))
                sz_sequence_dim = 0 if sequence_dim is None else (1 if sequence_dim in sequence_cands[1:3] else -1)
            elif sequence_dim is not None: raise TypeError("Argument 'sequence_dim' is conflict to 'n_sequence_dim' for bt.Size, they cannot be assigned simultaneously. ")
            else:
                avouch(isinstance(n_sequence_dim, int_) and n_sequence_dim >= 0, TypeError(f"Invalid 'n_sequence_dim = {n_sequence_dim}' for bt.Size, should be an integer. "))
                sz_sequence_dim = - n_sequence_dim
        # Check dimension consistency
        avouch(abs_(sz_func_dim) + abs_(sz_batch_dim) + abs_(sz_feature_dim) + abs_(sz_sequence_dim) <= n_dim, TypeError(f"Too many special dimensions for shape of length {n_dim}. "))
        return cls.__new_raw__(shape, sz_func_dim=sz_func_dim, sz_batch_dim=sz_batch_dim, sz_feature_dim=sz_feature_dim, sz_sequence_dim=sz_sequence_dim)

    @classmethod
    def __new_repr__(cls, shape):
        """
        The constructor using python representations. Including:
        1. (n_func,) for functional dimension, 
        2. {n_batch} for batch dimension, 
        3. [n_feature] for feature dimensions,
        4. 'n_sequence' for sequence dimensions,
        5. integers for ordinary space dimensions. 

        Examples::
            >>> s = bt.Size({2}, [3], [4, 5], 6, 7, '8')
            >>> s
            batorch.Size({2}, [3, 4, 5], 6, 7, '8')
            >>> s.feature
            batorch.Size([3, 4, 5])
            >>> s.with_feature(2)
            batorch.Size({2}, [2], 6, 7, '8')
            >>> s << 2 # padding
            batorch.Size({2}, [3, 4, 5], 8, 9, '8')
            >>> s ** 2 # repeat to enlarge
            batorch.Size({2}, [3, 4, 5], 12, 14, '8')
        """
        sz_func_dim = 0
        sz_batch_dim = 0
        sz_feature_dim = 0
        sz_sequence_dim = 0
        raw_size = []
        cum_i = 0
        ends_with_func = False
        ends_with_batch = False
        ends_with_feature = False
        ends_with_sequence = False
        for i, s in enumerate(shape):
            if isinstance(s, tuple): # functional dimension
                avouch(sz_func_dim == 0, TypeError(f"Invalid 'shape = {shape}' for bt.Size (conflict of multiple functional dimensions)."))
                avouch(len(s) == 1 and isinstance(s[0], num_) or len(s) == 0, TypeError(f"Invalid 'shape = {shape}' for bt.Size (only (x,) is available for functional dimension)."))
                avouch(i in (0, len(shape) - 1), TypeError(f"Invalid 'shape = {shape}' for bt.Size (functional dimension can only be the first/last dimension)."))
                if len(s) == 0: raw_size.append(-1)
                else: raw_size.append(s[0])
                if i == 0:
                    sz_func_dim = 1
                else:
                    sz_func_dim = -1
                    ends_with_func = True
                cum_i += 1
            elif isinstance(s, (dict, set)): # batch dimension
                avouch(sz_batch_dim == 0, TypeError(f"Invalid 'shape = {shape}' for bt.Size (conflict of multiple batch dimensions)."))
                avouch(isinstance(s, set) and len(s) == 1 or len(s) == 0, TypeError(f"Invalid 'shape = {shape}' for bt.Size (no dict item is allowed, only {{x}} or {{}} are available)."))
                avouch(not ends_with_func, TypeError(f"Invalid 'shape = {shape}' for bt.Size (batch dimension can only be the first/last dimension apart from the functional dimension)."))
                avouch(i in (max_(sz_func_dim, 0), len(shape) - 1 + min_(sz_func_dim, 0)), TypeError(f"Invalid 'shape = {shape}' for bt.Size (batch dimension can only be the first/last dimension apart from the functional dimension)."))
                if len(s) == 0: raw_size.append(-1)
                else: x = s.pop(); raw_size.append(x); shape[i].add(x)
                if i == max_(sz_func_dim, 0):
                    sz_batch_dim = 1
                else:
                    sz_batch_dim = -1
                    ends_with_batch = True
                cum_i += 1
            elif isinstance(s, list): # feature dimensions
                avouch(sz_feature_dim <= 0 or isinstance(shape[i-1], (tuple, dict, set, list)), 
                       TypeError(f"Invalid 'shape = {shape}' for bt.Size (feature dimensions should be neighboring dimensions)."))
                avouch(all_([isinstance(y, num_) for y in s]), TypeError(f"Invalid 'shape = {shape}' for bt.Size (representation for feature dimensions should be a list of integers)."))
                avouch(not ends_with_batch and not ends_with_func, TypeError(f"Invalid 'shape = {shape}' for bt.Size (batch dimension can only be the first/last dimension apart from the functional dimension)."))
                if len(s) == 0: raw_size.append(-1); len_feat = 1
                else: raw_size.extend(s); len_feat = len(s)
                if sz_feature_dim == 0:
                    if cum_i == max_(sz_func_dim, 0) + max_(sz_batch_dim, 0): sz_feature_dim = len_feat
                    else: sz_feature_dim = -len_feat
                elif sz_feature_dim > 0: sz_feature_dim += len_feat
                else: sz_feature_dim -= len_feat
                if sz_feature_dim < 0: ends_with_feature = True
                cum_i += len_feat
            elif isinstance(s, str): # sequence dimensions
                s_val = -1 if s == '' else touch(lambda: eval(s))
                avouch(sz_sequence_dim <= 0 or isinstance(shape[i-1], (tuple, dict, set, list, str)),
                       TypeError(f"Invalid 'shape = {shape}' for bt.Size (sequence dimensions should be neighboring dimensions)."))
                avouch(s_val is not None, TypeError(f"Invalid 'shape = {shape}' for bt.Size (representation for sequence dimensions should be a list of integers)."))
                avouch(not ends_with_feature and not ends_with_batch and not ends_with_func, TypeError(f"Invalid 'shape = {shape}' for bt.Size (feature dimensions can only be the first/last dimensions apart from the functional/batch dimensions)."))
                if not isinstance(s_val, tuple): s_val = (s_val,)
                s_val = list(s_val)
                avouch(all_([isinstance(y, num_) for y in s_val]), TypeError(f"Invalid 'shape = {shape}' for bt.Size (representation for sequence dimensions should be a list of integers)."))
                raw_size.extend(s_val); len_sqs = len(s_val)
                if sz_sequence_dim == 0:
                    ends_with_sequence = cum_i > max_(sz_func_dim, 0) + max_(sz_batch_dim, 0) + max_(sz_feature_dim, 0)
                    sz_sequence_dim = -len_sqs
                elif sz_sequence_dim > 0: sz_sequence_dim += len_sqs
                else: sz_sequence_dim -= len_sqs
                cum_i += len_sqs
            elif isinstance(s, num_):
                avouch(not ends_with_sequence and not ends_with_feature and not ends_with_batch and not ends_with_func, TypeError(f"Invalid 'shape = {shape}' for bt.Size (sequence dimensions can only be the first/last dimensions apart from the functional/batch/feature dimensions)."))
                if sz_sequence_dim < 0: sz_sequence_dim = -sz_sequence_dim
                raw_size.append(s)
                cum_i += 1
            else: raise TypeError(f"Invalid 'shape = {shape}' for bt.Size (only (x,)(functional dimension), {{x}}(batch dimension), {{}}(batch dimension with arbitrary size), [x, y, ...](feature dimensions), [](feature dimension with arbitrary size)), 'x, y, ...'(sequence dimensions), and ''(sequence dimension with arbitrary size) are allowed as special dimensions in bt.Size).")

        return cls.__new_raw__(tuple(raw_size), sz_func_dim=sz_func_dim, sz_batch_dim=sz_batch_dim, sz_feature_dim=sz_feature_dim, sz_sequence_dim=sz_sequence_dim)

    def __new__(cls, *args, **kwargs):
        """
        The construction function for 'bt.Size'. 

        Usages:
            bt.Size(shape: torch.Tensor/bt.Tensor/bt.Size/generator/tuple/str, batch_dim=False, n_feature_dim=None, n_sequence_dim=n_sequence_dim)
            bt.Size(*shape: python_repr[int, dict[0], set[1], list[], str], batch_dim=False, n_feature_dim=None, n_sequence_dim=n_sequence_dim)
            One may use 'channel_dim=*' to replace n_feature_dim if there is only one feature dimension. 
            and 'sequence_dim=*' to replace n_sequence_dim if there is only one sequence dimension. 
        
        Warning:
            Please be careful using private usages including keywords starting with 'sz_' such as 'sz_batch_dim'. 
        Note that one cannot create a functional dimension by python representations, please use argument `sz_func_dim` instead. 

        Examples::
            >>> s = bt.Size({2}, [3], [4, 5], 6, 7, '8')
            >>> s
            batorch.Size({2}, [3, 4, 5], 6, 7, '8')
            >>> s.feature
            batorch.Size([3, 4, 5])
            >>> s.with_feature(2)
            batorch.Size({2}, [2], 6, 7, '8')
            >>> s << 2 # padding
            batorch.Size({2}, [3, 4, 5], 8, 9, '8')
            >>> s ** 2 # repeat to enlarge
            batorch.Size({2}, [3, 4, 5], 12, 14, '8')
        """
        if len(args) == 1 and hasattr(args[0], 'shape'): args = (args[0].shape,)
        if len(args) == 1 and isinstance(args[0], Generator): return cls.__new_tuple__(tuple(args[0]), **kwargs)
        if len(args) == 1 and isinstance(args[0], FakeSize): return cls.__new_raw__(tuple(args[0]), **kwargs).special_from(args[0])
        if len(args) == 1 and isinstance(args[0], Size): return cls.__new_size__(args[0], **kwargs)
        if len(args) == 1 and isinstance(args[0], tuple): return cls.__new_tuple__(args[0], **kwargs)
        if len(args) == 1 and isinstance(args[0], str):
            if args[0] == '':
                kwargs['sz_sequence_dim'] = 1
                return cls.__new_tuple__((-1,), **kwargs)
            if touch(lambda: int_(args[0])) is not None:
                kwargs['sz_sequence_dim'] = 1
                return cls.__new_tuple__((int_(args[0]),), **kwargs)
            self = cls.__new_tuple__(eval(args[0]), **kwargs)
            if self.n_special_dim > 0 or args[0].startswith('('): return self
            return self.sz_sequence_dim_(-self.n_dim)
        return cls.__new_tuple__(args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.sz_func_dim = self.sz_func_dim
        self.sz_batch_dim = self.sz_batch_dim
        self.sz_feature_dim = self.sz_feature_dim
        self.sz_sequence_dim = self.sz_sequence_dim
        self.n_dim = self.n_dim
        self.ndim = self.n_dim

    ## functional dimension:
    @property
    def has_func(self): return self.sz_func_dim != 0

    @alias("nfuncdim")
    @property
    def n_func_dim(self): return abs_(self.sz_func_dim)
    
    @alias("is_funcdim")
    def is_func_dim(self, i):
        avouch(isinstance(i, int_), TypeError(f"Invalid call 'is_func_dim({i})': the argument is not an integer. "))
        if not self.has_func: return False
        return self.sz_func_dim == 1 and i in (0, -self.n_dim) or self.sz_func_dim == -1 and i in (self.n_dim-1, -1)

    @property
    def func_dim(self):
        return None if self.sz_func_dim == 0 else (0 if self.sz_func_dim > 0 else self.n_dim-1)
    
    @alias("func_dimension")
    @func_dim.setter
    def func_dim(self, ifunc):
        return self.with_func_dim(ifunc)

    @alias("func_dim_", "func_dimension_")
    @alias("with_funcdim")
    def with_func_dim(self, ifunc: (int, bool, null)):
        avouch(ifunc is None or isinstance(ifunc, bool_) or ifunc in (0, -self.n_dim, self.n_dim-1, -1), TypeError("'bt.Size.with_func_dim' only takes input bool or integer 0, -1."))
        if ifunc or isinstance(ifunc, int_):
            avouch(self.n_batch_dim + self.n_feature_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set func_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
            self.sz_func_dim = 1 if ifunc in (0, -self.n_dim, True) else -1
        else: self.sz_func_dim = 0
        return self

    @alias("sz_func_dim_", "with_szfuncdim")
    def with_sz_func_dim(self, nfcdim):
        avouch(nfcdim is None or isinstance(nfcdim, int_), TypeError("'bt.Size.with_sz_func_dim' only takes input of an integer."))
        if nfcdim is None: self.sz_func_dim = 0
        else:
            avouch(self.n_batch_dim + self.n_feature_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set func_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
            self.sz_func_dim = nfcdim
        return self

    @alias("n_func_dim_", "with_nfuncdim")
    def with_n_func_dim(self, nfcdim):
        avouch(nfcdim >= 0, TypeError("'bt.Size.with_n_func_dim' accept only positive number of dimensions (before the space dimensions). "))
        return self.with_sz_func_dim(nfcdim)

    @alias("nfunc", "func_size")
    @property
    def n_func(self):
        return self[self.func_dim] if self.has_func else None

    def with_func(self, n_func):
        if n_func is None: return self[self.size_start:self.size_stop]
        avouch(isinstance(n_func, int_), TypeError("Func size should be an integer. "))
        if self.sz_func_dim >= 0:
            return func_dim_size(n_func) + self[self.n_func_dim:]
        else: return self[:-self.n_func_dim] + func_dim_size(n_func)

    @property
    def size_start(self): return max_(self.sz_func_dim, 0)
    @property
    def size_stop(self): return self.n_dim + min_(self.sz_func_dim, 0)

    ## batch dimension:
    @property
    def has_batch(self): return self.sz_batch_dim != 0

    @alias("nbatchdim")
    @property
    def n_batch_dim(self): return abs_(self.sz_batch_dim)
    
    @alias("is_batchdim")
    def is_batch_dim(self, i):
        avouch(isinstance(i, int_), TypeError(f"Invalid call 'is_batch_dim({i})': the argument is not an integer. "))
        if not self.has_batch: return False
        return self.sz_batch_dim == 1 and i in (self.size_start, self.size_start-self.n_dim) or self.sz_batch_dim == -1 and i in (self.size_stop-1, self.size_stop-1-self.n_dim)

    @property
    def batch_dim(self):
        return None if self.sz_batch_dim == 0 else (self.size_start if self.sz_batch_dim > 0 else self.size_stop-1)

    @alias("batch_dimension")
    @batch_dim.setter
    def batch_dim(self, ibatch):
        return self.with_batch_dim(ibatch)

    @alias("batch_dim_", "batch_dimension_")
    @alias("with_batchdim")
    def with_batch_dim(self, ibatch: (int, bool, null)):
        avouch(ibatch is None or isinstance(ibatch, bool_) or ibatch in (self.size_start, self.size_start-self.n_dim, self.size_stop-1, self.size_stop-1-self.n_dim), TypeError(f"'bt.Size.with_batch_dim' only takes input bool or integers {self.size_start}, {self.size_stop-1}."))
        if ibatch or isinstance(ibatch, int_):
            avouch(self.n_func_dim + self.n_feature_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set batch_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
            self.sz_batch_dim = 1 if ibatch in (self.size_start, self.size_start-self.n_dim, True) else -1
        else: self.sz_batch_dim = 0
        return self

    @alias("sz_batch_dim_", "with_szbatchdim")
    def with_sz_batch_dim(self, nbdim):
        avouch(nbdim is None or isinstance(nbdim, int_), TypeError("'bt.Size.with_sz_batch_dim' only takes input of an integer."))
        if nbdim is None: self.sz_batch_dim = 0
        else:
            avouch(self.n_func_dim + self.n_feature_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set batch_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
            self.sz_batch_dim = nbdim
        return self

    @alias("n_batch_dim_", "with_nbatchdim")
    def with_n_batch_dim(self, nbdim):
        avouch(nbdim >= 0, TypeError("'bt.Size.with_n_batch_dim' accept only positive number of dimensions (before the space dimensions). "))
        return self.with_sz_batch_dim(nbdim)

    @alias("nbatch", "batch_size")
    @property
    def n_batch(self):
        return self[self.batch_dim] if self.has_batch else None

    def with_batch(self, n_batch):
        if n_batch is None:
            if self.sz_func_dim == 0: return self[self.non_bat_start:self.non_bat_stop]
            elif self.sz_func_dim > 0: return self[:1] + self[self.non_bat_start:self.non_bat_stop]
            else: return self[self.non_bat_start:self.non_bat_stop] + self[-1:]
        avouch(isinstance(n_batch, int_), TypeError("Batch size should be an integer. "))
        if self.sz_batch_dim > 0:
            return self[:self.size_start] + Size({n_batch}) + self[self.size_start + self.n_batch_dim:]
        else: return self[:self.size_stop-self.n_batch_dim] + Size({n_batch}) + self[self.size_stop:]

    @property
    def non_bat_start(self): return max_(self.sz_func_dim, 0) + max_(self.sz_batch_dim, 0)
    @property
    def non_bat_stop(self): return self.n_dim + min_(self.sz_func_dim, 0) + min_(self.sz_batch_dim, 0)

    ## channel dimension: 
    @property
    def has_channel(self): return self.sz_feature_dim in (1, -1)

    @alias("nchanneldim")
    @property
    def n_channel_dim(self): return int_(self.has_channel)

    @alias("is_channeldim")
    def is_channel_dim(self, i):
        avouch(isinstance(i, int_), TypeError(f"Invalid call 'is_channel_dim({i})': the argument is not an integer. "))
        if not self.has_feature: return False
        return (self.sz_feature_dim == 1 and i in (self.non_bat_start, self.non_bat_start - self.n_dim) or 
                self.sz_feature_dim == -1 and i in (self.non_bat_stop - 1, self.non_bat_stop - self.n_dim - 1))

    @property
    def channel_dim(self):
        avouch(self.sz_feature_dim in (1, -1), TypeError(f"Cannot get channel dimension from size with {self.n_feature_dim} feature dimensions."))
        return self.non_bat_start if self.sz_feature_dim > 0 else self.non_bat_stop - 1

    @alias("channel_dimension")
    @channel_dim.setter
    def channel_dim(self, ichannel):
        return self.with_channel_dim(ichannel)

    @alias("channel_dim_", "channel_dimension_")
    @alias("with_channeldim")
    def with_channel_dim(self, ichannel: (int, null)):
        avouch(ichannel is None or ichannel in (self.non_bat_start, self.non_bat_start - self.n_dim, self.non_bat_stop - 1, self.non_bat_stop - self.n_dim - 1), 
               TypeError(f"Channel dimension of {self} should be the first or last dimension apart from the batch dimension, which are {self.non_bat_start} and {self.non_bat_stop - 1}."))
        if ichannel is None: self.sz_feature_dim = 0
        else:
            avouch(self.n_func_dim + self.n_batch_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set channel_dim for size {self} of non-special dimension 0."))
            self.sz_feature_dim = 1 if ichannel in (self.non_bat_start, self.non_bat_start - self.n_dim) else -1
        return self

    @alias("nchannel", "channel_size")
    @property
    def n_channel(self):
        avouch(self.n_feature_dim == 1, TypeError(f"Cannot get channel dimension from size with {self.n_feature_dim} feature dimensions."))
        return self[self.channel_dim]

    def with_channel(self, n_channel):
        avouch(isinstance(n_channel, int_), TypeError("Channel size should be an integer. "))
        if self.sz_feature_dim > 0:
            return self[:self.non_bat_start] + Size([n_channel]) + self[self.non_bat_start + self.n_feature_dim:]
        else: return self[:self.non_bat_stop - self.n_feature_dim] + Size([n_channel]) + self[self.non_bat_stop:]

    ## feature dimensions:
    @property
    def has_feature(self): return self.n_feature_dim != 0

    @alias("is_featuredim")
    def is_feature_dim(self, i):
        avouch(isinstance(i, int_), TypeError(f"Invalid call 'is_feature_dim({i})': the argument is not an integer. "))
        if not self.has_feature: return False
        if i < 0: i += self.n_dim
        return self.feature_start <= i < self.feature_stop

    @property
    def n_feature_dim(self): return abs_(self.sz_feature_dim)

    @alias("nfeaturedim")
    @n_feature_dim.setter
    def n_feature_dim(self, n): return self.with_n_feature_dim(n)

    @alias("sz_feature_dim_", "with_szfeaturedim")
    def with_sz_feature_dim(self, nfdim):
        avouch(nfdim is None or isinstance(nfdim, int_), TypeError("'bt.Size.with_sz_feature_dim' only takes input of an integer."))
        if nfdim is None: self.sz_feature_dim = 0
        else:
            avouch(self.n_func_dim + self.n_batch_dim + abs_(nfdim) + self.n_sequence_dim <= self.n_dim, TypeError(f"Cannot assign {abs_(nfdim)} features in size {self} with non-special dimension of {self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_sequence_dim}, or there will be conflict. "))
            self.sz_feature_dim = nfdim
        return self

    @alias("n_feature_dim_", "with_nfeaturedim")
    def with_n_feature_dim(self, nfdim):
        avouch(nfdim >= 0, TypeError("'bt.Size.with_n_feature_dim' accept only positive number of dimensions (before the space dimensions). "))
        return self.with_sz_feature_dim(nfdim)

    @property
    def feature_start(self):
        avouch(self.has_feature, TypeError("Cannot get feature start from size without feature dimensions."))
        return None if self.sz_feature_dim == 0 else (self.non_bat_start if self.sz_feature_dim > 0 else self.non_bat_stop + self.sz_feature_dim)

    @feature_start.setter
    def feature_start(self, dim):
        avouch(dim is None or isinstance(dim, int_), TypeError(f"Feature start should be an integer. "))
        if dim is None: self.sz_feature_dim = 0; return
        if dim < 0: dim += self.n_dim
        avouch(self.non_bat_start <= dim < self.non_bat_stop, TypeError(f"Feature start should avoid the batch dimensions, which is between {self.non_bat_start} and {self.non_bat_stop - 1}, or there will be conflict. "))
        self.sz_feature_dim = dim - self.non_bat_stop

    @property
    def feature_stop(self):
        avouch(self.has_feature, TypeError("Cannot get feature start from size without feature dimensions."))
        return None if self.sz_feature_dim == 0 else (self.non_bat_start + self.sz_feature_dim if self.sz_feature_dim > 0 else self.non_bat_stop)

    @feature_stop.setter
    def feature_stop(self, dim):
        avouch(dim is None or isinstance(dim, int_), TypeError(f"Feature stop should be an integer. "))
        if dim is None: self.sz_feature_dim = 0; return
        if dim < 0: dim += self.n_dim
        avouch(self.non_bat_start <= dim < self.non_bat_stop, TypeError(f"Feature stop should avoid the batch dimensions, which is between {self.non_bat_start} and {self.non_bat_stop - 1}, or there will be conflict. "))
        self.sz_feature_dim = dim - self.non_bat_start

    @property
    def feature_range(self):
        return (self.feature_start, self.feature_stop)
    
    @feature_range.setter
    def feature_range(self, *args):
        avouch(len(args) == 2 or len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2, 
               TypeError("Only two values are allowed in the assignment of 'feature_range', indicating the start and end dimensions. "))
        if len(args) == 1: args = args[0]
        avouch(args[0] == self.non_bat_start or args[1] == self.non_bat_stop, 
               TypeError(f"Feature dimensions are the first or last dimensions (starting from {self.non_bat_start} or ending at {self.non_bat_stop}). "))
        if args[0] == self.non_bat_start: self.feature_stop = args[1]
        else: self.feature_start = args[0]

    @alias("nfeature")
    @property
    def n_feature(self):
        avouch(self.has_feature, TypeError(f"Cannot get feature dimensions from size {self}."))
        p = 1
        for i in range_(*self.feature_range): p *= self[i]
        return p

    @alias("feature_size")
    @property
    def feature(self):
        return self[self.feature_start: self.feature_stop]

    def with_feature(self, *size):
        if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
        avouch(all_(isinstance(x, int_) for x in size), TypeError("feature size should be a tuple of integers. "))
        if not self.has_feature: start = self.non_bat_start; stop = self.non_bat_start
        else: start = self.feature_start; stop = self.feature_stop
        # avouch(len(size) == self.n_feature_dim, f"Cannot substitute feature in {self} by {size} as their dimensions are not the same.")
        return self[:start] + Size(size, sz_feature_dim=len(size)) + self[stop:]
        
    @property
    def seq_spc_start(self): return max_(self.sz_func_dim, 0) + max_(self.sz_batch_dim, 0) + max_(self.sz_feature_dim, 0)
    @property
    def seq_spc_stop(self): return self.n_dim + min_(self.sz_func_dim, 0) + min_(self.sz_batch_dim, 0) + min_(self.sz_feature_dim, 0)

    ## sequence dimensions:
    @alias("has_time", "has_series")
    @property
    def has_sequence(self): return self.sz_sequence_dim != 0

    @alias("is_timedim", "is_seriesdim", "is_sequencedim")
    @alias("is_time_dim", "is_series_dim")
    def is_sequence_dim(self, i):
        avouch(isinstance(i, int_), TypeError(f"Invalid call 'is_sequence_dim({i})': the argument is not an integer. "))
        if not self.has_sequence: return False
        if i < 0: i += self.n_dim
        return self.sequence_start <= i < self.sequence_stop

    @property
    def n_sequence_dim(self): return abs_(self.sz_sequence_dim)

    @alias("ntimedim", "nseriesdim", "nsequencedim")
    @alias("n_time_dim", "n_series_dim")
    @n_sequence_dim.setter
    def n_sequence_dim(self, n): return self.with_n_sequence_dim(n)

    @alias("sz_time_dim_", "sz_series_dim_", "sz_sequence_dim_")
    @alias("with_sztimedim", "with_szseriesdim", "with_szsequencedim")
    @alias("with_sz_time_dim", "with_sz_series_dim")
    def with_sz_sequence_dim(self, nsdim):
        avouch(nsdim is None or isinstance(nsdim, int_), TypeError("'bt.Size.with_sz_sequence_dim' only takes input of an integer."))
        if nsdim is None: self.sz_sequence_dim = 0
        else:
            avouch(self.n_func_dim + self.n_batch_dim + self.n_feature_dim + abs_(nsdim) <= self.n_dim, TypeError(f"Cannot assign {abs_(nsdim)} sequence dimensions in size {self} with non-special dimension of {self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_feature_dim}, or there will be conflict. "))
            self.sz_sequence_dim = nsdim
        return self

    @alias("n_time_dim_", "n_series_dim_", "n_sequence_dim_")
    @alias("with_ntimedim", "with_nseriesdim", "with_nsequencedim")
    @alias("with_n_time_dim", "with_n_series_dim")
    def with_n_sequence_dim(self, nsdim):
        avouch(nsdim >= 0, TypeError("'bt.Size.with_n_sequence_dim' accept only positive number of dimensions (before the space dimensions). "))
        return self.with_sz_sequence_dim(-nsdim)

    @alias("time_dim_", "series_dim_", "sequence_dim_")
    @alias("with_timedim", "with_seriesdim", "with_sequencedim")
    @alias("with_time_dim", "with_series_dim")
    def with_sequence_dim(self, dim):
        avouch(dim is None or isinstance(dim, (bool_, int_)), TypeError("'bt.Size.with_sequence_dim' only takes integer or bool."))
        if isinstance(dim, bool_): dim = -1 if dim else None
        if dim is None: self.sz_sequence_dim = 0; return self
        if dim < 0: dim += self.n_dim
        avouch(dim in (self.seq_spc_start, self.seq_spc_stop - 1), TypeError(f"Sequence dimension can only be the first or last dimension ({self.seq_spc_start} or {self.seq_spc_stop-1}) apart from batch and feature dimensions."))
        self.sz_sequence_dim = 1 if dim == self.seq_spc_start else -1
        return self

    @property
    def sequence_start(self):
        avouch(self.has_sequence, TypeError("Cannot get sequence start from size without sequence dimensions."))
        return None if self.sz_sequence_dim == 0 else (self.seq_spc_start if self.sz_sequence_dim > 0 else self.seq_spc_stop + self.sz_sequence_dim)

    @alias("time_start", "series_start")
    @sequence_start.setter
    def sequence_start(self, dim):
        avouch(dim is None or isinstance(dim, int_), TypeError(f"Sequence start should be an integer. "))
        if dim is None: self.sz_sequence_dim = 0; return
        if dim < 0: dim += self.n_dim
        avouch(self.seq_spc_start <= dim < self.seq_spc_stop, TypeError(f"Sequence start should avoid the batch/feature dimensions, which is between {self.seq_spc_start} and {self.seq_spc_stop - 1}, or there will be conflict. "))
        self.sz_sequence_dim = dim - self.seq_spc_stop

    @property
    def sequence_stop(self):
        avouch(self.has_sequence, TypeError("Cannot get sequence start from size without sequence dimensions."))
        return None if self.sz_sequence_dim == 0 else (self.seq_spc_start + self.sz_sequence_dim if self.sz_sequence_dim > 0 else self.seq_spc_stop)

    @alias("time_stop", "series_stop")
    @sequence_stop.setter
    def sequence_stop(self, dim):
        avouch(dim is None or isinstance(dim, int_), TypeError(f"Sequence stop should be an integer. "))
        if dim is None: self.sz_sequence_dim = 0; return
        if dim < 0: dim += self.n_dim
        avouch(self.seq_spc_start <= dim < self.seq_spc_stop, TypeError(f"Sequence stop should avoid the batch/feature dimensions, which is between {self.seq_spc_start} and {self.seq_spc_stop - 1}, or there will be conflict. "))
        self.sz_sequence_dim = dim - self.seq_spc_start

    @property
    def sequence_range(self):
        return (self.sequence_start, self.sequence_stop)
    
    @alias("time_range", "series_range")
    @sequence_range.setter
    def sequence_range(self, *args):
        avouch(len(args) == 2 or len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2, 
               "Only two values are allowed in the assignment of 'sequence_range', indicating the start and end dimensions. ")
        if len(args) == 1: args = args[0]
        avouch(args[0] == self.seq_spc_start or args[1] == self.seq_spc_stop, 
               TypeError(f"Feature dimensions are the first or last dimensions (starting from {self.seq_spc_start} or ending at {self.seq_spc_stop}). "))
        if args[0] == self.seq_spc_start: self.sequence_stop = args[1]
        else: self.sequence_start = args[0]

    @alias("ntime", "ntimeline", "nseries", "nsequence")
    @alias("n_time", "n_timeline", "n_series")
    @property
    def n_sequence(self):
        avouch(self.has_sequence > 0, TypeError(f"Cannot get sequence dimensions from size {self}."))
        p = 1
        for i in range_(*self.sequence_range): p *= self[i]
        return p

    @alias("time_size", "series_size", "sequence_size")
    @alias("time", "series")
    @property
    def sequence(self):
        return self[self.sequence_start:self.sequence_stop]

    @alias("with_time", "with_series")
    def with_sequence(self, *size):
        if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
        avouch(all_(isinstance(x, int_) for x in size), TypeError("sequence size should be a tuple of integers. "))
        if not self.has_sequence: start = self.seq_spc_stop; stop = self.seq_spc_stop
        else: start = self.sequence_start; stop = self.sequence_stop
        # avouch(len(size) == self.n_sequence_dim, f"Cannot substitute sequence in {self} by {size} as their dimensions are not the same.")
        return self[:start] + Size(size, sz_sequence_dim=len(size)) + self[stop:]

    ## space dimensions:
    @property
    def has_space(self): return self.n_space_dim > 0
    
    @alias("is_spacedim")
    def is_space_dim(self, i):
        return self.space_start <= i < self.space_stop

    @alias("nspacedim")
    @property
    def n_space_dim(self):
        return self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_feature_dim - self.n_sequence_dim
    
    @property
    def space_start(self):
        return max_(self.sz_func_dim, 0) + max_(self.sz_batch_dim, 0) + max_(self.sz_feature_dim, 0) + max_(self.sz_sequence_dim, 0)
    
    @property
    def space_stop(self):
        return self.n_dim + min_(self.sz_func_dim, 0) + min_(self.sz_batch_dim, 0) + min_(self.sz_feature_dim, 0) + min_(self.sz_sequence_dim, 0)

    @property
    def space_range(self):
        return (self.space_start, self.space_stop)

    @alias("nspace")
    @property
    def n_space(self):
        avouch(self.has_space, TypeError(f"Cannot get space dimensions from size {self}."))
        p = 1
        for i in range_(*self.space_range): p *= self[i]
        return p

    @alias("space_size")
    @property
    def space(self):
        return self[self.space_start:self.space_stop]

    def with_space(self, *size):
        if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
        avouch(all_(isinstance(x, int_) for x in size), TypeError("space size should be a tuple of integers. "))
        # avouch(len(size) == self.n_space_dim, f"Cannot substitute space in {self} by {size} as their dimensions are not the same.")
        return self[:self.space_start] + size + self[self.space_stop:]

    ## special dimensions:
    @property
    def has_special(self): return self.has_func or self.has_batch or self.has_feature or self.has_sequence

    @alias("nspecialdim")
    @property
    def n_special_dim(self):
        return self.n_func_dim + self.n_batch_dim + self.n_feature_dim + self.n_sequence_dim

    @property
    def special_dims(self):
        sdim_list = (([] if self.sz_func_dim == 0 else ([0] if self.sz_func_dim > 0 else [self.n_dim-1])) + 
                     ([] if self.sz_batch_dim == 0 else ([self.size_start] if self.sz_batch_dim > 0 else [self.size_stop])) + 
                     ([] if self.sz_feature_dim == 0 else list(range_(self.feature_start, self.feature_stop))) + 
                     ([] if self.sz_sequence_dim == 0 else list(range_(self.sequence_start, self.sequence_stop))))
        sdim_list.sort()
        return sdim_list
    
    def add_special_dim(self, index, *reference):
        avouch(len(reference) == 1, TypeError("Only one dimension is acceptable for 'add_special_dim'. "))
        avouch(-self.n_dim <= index < self.n_dim, TypeError(f"Index for 'add_special_dim' should be within the total dimensions: from {-self.n_dim} to {self.n_dim-1}. "))
        if index < 0: index += self.n_dim
        if not isinstance(reference[0], Size): reference = Size(*reference)
        else: reference = reference[0]
        if reference.has_func:
            return self[:index] + func_dim_size(self[index]) + self[index+1:]
        if reference.has_batch:
            return self[:index] + Size({self[index]}) + self[index+1:]
        if reference.has_feature:
            if self.has_feature: avouch(self.feature_start - 1 <= index <= self.feature_stop, TypeError(f"Only dimensions adjacent to current can be converted into features by 'add_special_dim': trying to convert {index}-th dim to feature in {self}. "))
            return self[:index] + Size([self[index]]) + self[index+1:]
        if reference.has_sequence:
            if self.has_sequence: avouch(self.sequence_start - 1 <= index <= self.sequence_stop, TypeError(f"Only dimensions adjacent to current can be converted into sequences by 'add_special_dim': trying to convert {index}-th dim to sequence in {self}. "))
            return self[:index] + Size(repr(str(self[index]))) + self[index+1:]
        return self
    
    def change_special_dim(self, from_dim, *to_dim):
        from_dim = exist_dim(self, from_dim)
        avouch(len(from_dim) == 1, TypeError("Only one 'from_dim' is acceptable for 'change_special_dim'. "))
        return self.add_special_dim(from_dim[0], *to_dim)

    def special_from(self, other, allow_view=False):
        avouch(isinstance(other, (tuple, Size, Tensor)), TypeError(f"Invalid input for Size.special_from: {type(other)}. "))
        if isinstance(other, Tensor): other = other.shape
        if isinstance(other, Size):
            if self.n_dim != other.n_dim:
                if allow_view: return self.view(other)
                raise TypeError(f"Dimension mismatch when inheriting special dimensions: from {other.n_dim} to {self.n_dim}. ")
        self.sz_func_dim = getattr(other, 'sz_func_dim', 0)
        self.sz_batch_dim = getattr(other, 'sz_batch_dim', 0)
        self.sz_feature_dim = getattr(other, 'sz_feature_dim', 0)
        self.sz_sequence_dim = getattr(other, 'sz_sequence_dim', 0)
        return self

    def update_special_from(self, other):
        avouch(isinstance(other, (tuple, Size, Tensor)), TypeError(f"Invalid input for Size.special_from: {type(other)}. "))
        if isinstance(other, Tensor): other = other.shape
        sz_func_dim = getattr(other, 'sz_func_dim', 0)
        if sz_func_dim != 0: self.sz_func_dim = sz_func_dim
        sz_batch_dim = getattr(other, 'sz_batch_dim', 0)
        if sz_batch_dim != 0: self.sz_batch_dim = sz_batch_dim
        sz_feature_dim = getattr(other, 'sz_feature_dim', 0)
        if sz_feature_dim != 0: self.sz_feature_dim = sz_feature_dim
        sz_sequence_dim = getattr(other, 'sz_sequence_dim', 0)
        if sz_sequence_dim != 0: self.sz_sequence_dim = sz_sequence_dim
        return self

    def init_special(self):
        self.sz_func_dim = 0
        self.sz_batch_dim = 0
        self.sz_feature_dim = 0
        self.sz_sequence_dim = 0
        return self
    
    @alias("is_specialdim")
    def is_special_dim(self, i): return self.is_func_dim(i) or self.is_batch_dim(i) or self.is_feature_dim(i) or self.is_sequence_dim(i)

    ## all dimensions:
    @alias("nele")
    @property
    def n_ele(self):
        p = 1
        for i in range_(self.n_dim):
            if self[i] >= 0: p *= self[i]
        return p

    @alias("with_nele")
    def with_n_ele(self, n_ele):
        und = [i for i, x in enumerate(self) if x < 0]
        if len(und) == 0:
            avouch(n_ele == self.n_ele, TypeError(f"Cannot set n_ele={n_ele} for size {self} without undetermined dimensions."))
            return self
        avouch(len(und) == 1, TypeError(f"Cannot set n_ele for size {self} with more than one undetermined dimensions."))
        s_ele = self.n_ele
        avouch(n_ele % s_ele == 0, TypeError(f"Cannot set n_ele={n_ele} for size {self} as it is not a multiplication of current size {s_ele}. "))
        return self[:und[0]] + Size(n_ele // self.n_ele).special_from(self[und[0]:und[0]+1]) + self[und[0]+1:]
    
    def with_dim_size(self, index, size):
        if index < 0: index += self.n_dim
        return self[:index] + Size(size).special_from(self[index:index+1]) + self[index+1:]
    
    def transpose(self, i: int_, j:int_):
        if i == j: return self
        if i > j: i, j = j, i
        if self.is_func_dim(i):
            avouch(None, IndexError("Failure in 'bt.Size.transpose': Cannot move the functional dimension. "))
        if self.is_batch_dim(i):
            avouch(None, IndexError("Failure in 'bt.Size.transpose': Cannot move the batch dimension. "))
        elif self.is_feature_dim(i):
            avouch(self.is_feature_dim(j), IndexError(f"Failure in 'bt.Size.transpose': Cannot move feature dimension {i} out of the feature scope. "))
        elif self.is_space_dim(i):
            avouch(self.is_space_dim(j), IndexError(f"Failure in 'bt.Size.transpose': Cannot move space dimension {i} out of the space scope. "))
        return self[:i] + self[j:j+1] + self[i+1:j] + self[i:i+1] + self[j+1:]

    ## methods:
    @alias("clone")
    def copy(self): return Size(self)

    @alias("raw")
    def tuple(self): return tuple(self)
    
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], tuple): dims = dims[0]
        avouch(sorted(dims) == list(range_(len(self))), TypeError("'permute' needs input dimensions forming a permutation of 0-(n-1). "))
        return sum_([self[d:d+1] for d in dims], Size())    

    @property
    def python_repr(self):
        if self.has_space: output = tuple(self.space)
        else: output = tuple()
        if self.has_sequence:
            sequence = str(list(self[self.sequence_start:self.sequence_stop])).strip('[]')
            if self.sz_sequence_dim > 0: output = (sequence,) + output
            if self.sz_sequence_dim < 0: output = output + (sequence,)
        if self.has_feature:
            feature = list(self[self.feature_start: self.feature_stop])
            if self.sz_feature_dim > 0: output = (feature,) + output
            if self.sz_feature_dim < 0: output = output + (feature,)
        if self.has_batch:
            batch = {self.n_batch}
            if self.sz_batch_dim > 0: output = (batch,) + output
            if self.sz_batch_dim < 0: output = output + (batch,)
        if self.has_func:
            func = (self.n_func,)
            if self.sz_func_dim > 0: output = (func,) + output
            if self.sz_func_dim < 0: output = output + (func,)
        return output

    @alias("__repr__")
    def __str__(self):
        rep = self.python_repr
        return f"batorch.Size{rep}".replace(',)', ')').replace('Ellipsis', '...')
    
    ## operations:
    def __getitem__(self, k):
        if isinstance(k, int_): return super().__getitem__(k)
        avouch(isinstance(k, slice), TypeError(f"Slicing of 'bt.Size' only takes integers or slices, not {k} of type {type(k)}. "))
        s, e = k.start, k.stop
        if s is None: s = 0
        if e is None: e = self.n_dim
        if s < 0: s += self.n_dim
        if e < 0: e += self.n_dim
        if self.has_func:
            sz_func_dim = self.sz_func_dim if s <= self.func_dim and e > self.func_dim else (max_(min_(e, self.func_dim + 1) - s, 0) if s > self.func_dim else min_(max_(s, self.func_dim) - e, 0))
        else: sz_func_dim = 0
        if self.has_batch:
            sz_batch_dim = self.sz_batch_dim if s <= self.batch_dim and e > self.batch_dim else (max_(min_(e, self.batch_dim + 1) - s, 0) if s > self.batch_dim else min_(max_(s, self.batch_dim) - e, 0))
            # sz_batch_dim = self.non_bat_start - s if s < self.non_bat_start else (self.non_bat_stop - e if self.non_bat_stop < e else 0)
        else: sz_batch_dim = 0
        if self.has_feature:
            sz_feature_dim = self.sz_feature_dim if s <= self.feature_start and e >= self.feature_stop else (max_(min_(e, self.feature_stop) - s, 0) if s > self.feature_start else min_(max_(s, self.feature_start) - e, 0))
        else: sz_feature_dim = 0
        if self.has_sequence:
            sz_sequence_dim = self.sz_sequence_dim if s <= self.sequence_start and e >= self.sequence_stop else (max_(min_(e, self.sequence_stop) - s, 0) if s > self.sequence_start else min_(max_(s, self.sequence_start) - e, 0))
        else: sz_sequence_dim = 0
        return self.__class__.__new_raw__(super().__getitem__(k), sz_func_dim=sz_func_dim, sz_batch_dim=sz_batch_dim, sz_feature_dim=sz_feature_dim, sz_sequence_dim=sz_sequence_dim)
    
    @alias('__iadd__')
    def __add__(self, other):
        avouch(isinstance(other, tuple), TypeError("Summation for 'bt.Size' is inherited from python object 'tuple' to perform concatenation, please use `size <<(>>) 2` to perform element-wise summation (subtraction) to increase (decrease) the size. "))
        if len(other) == 0: return self
        if len(self) == 0: return other
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        # Deal with the func dimension
        if self.sz_func_dim == 0:
            if other.sz_func_dim <= 0 or other.n_func_dim == other.n_dim: sz_func_dim = -other.n_func_dim
            elif self.n_dim == 0: sz_func_dim = other.n_func_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: functional dimension in middle. ")
        elif other.sz_func_dim == 0:
            if self.sz_func_dim >= 0 or self.n_func_dim == self.n_dim: sz_func_dim = self.n_func_dim
            elif other.n_dim == 0: sz_func_dim = self.sz_func_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: functional dimension in middle. ")
        else: raise TypeError(f"Error in concatenating {self} and {other}: conflict in functional dimension. ")
        # Deal with the batch dimension
        if self.sz_batch_dim == 0:
            if other.sz_batch_dim <= 0 or other.n_batch_dim == other.n_dim - other.n_func_dim: sz_batch_dim = -other.n_batch_dim
            elif self.n_dim - self.n_func_dim == 0: sz_batch_dim = other.n_batch_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: batch dimension in middle. ")
        elif other.sz_batch_dim == 0:
            if self.sz_batch_dim >= 0 or self.n_batch_dim == self.n_dim - self.n_func_dim: sz_batch_dim = self.n_batch_dim
            elif other.n_dim - other.n_func_dim == 0: sz_batch_dim = self.sz_batch_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: batch dimension in middle. ")
        else: raise TypeError(f"Error in concatenating {self} and {other}: conflict in batch dimension. ")
        # Deal with the feature dimensions
        if self.sz_feature_dim == 0:
            if other.sz_feature_dim <= 0 or other.n_feature_dim == other.n_dim - other.n_func_dim - other.n_batch_dim: sz_feature_dim = -other.n_feature_dim
            elif self.n_sequence_dim + self.n_space_dim == 0: sz_feature_dim = other.sz_feature_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: feature dimensions in middle. ")
        elif other.sz_feature_dim == 0:
            if self.sz_feature_dim >= 0 or self.n_feature_dim == self.n_dim - self.n_func_dim - self.n_batch_dim: sz_feature_dim = self.n_feature_dim
            elif other.n_sequence_dim + other.n_space_dim == 0: sz_feature_dim = self.sz_feature_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: feature dimensions in middle. ")
        elif self.sz_func_dim >= 0 and self.sz_batch_dim >= 0 and (self.sz_feature_dim < 0 or self.n_func_dim + self.n_batch_dim + self.n_feature_dim == self.n_dim) and \
             other.sz_func_dim <= 0 and other.sz_batch_dim <= 0 and (other.sz_feature_dim > 0 or other.n_func_dim + other.n_batch_dim + other.n_feature_dim == other.n_dim):
            sz_feature_dim = [-1, 1][self.feature_start == self.non_bat_start] * (self.n_feature_dim + other.n_feature_dim)
        # elif other.n_feature_dim == other.n_dim and self.sz_feature_dim < 0:
        #     sz_feature_dim = self.sz_feature_dim - other.n_dim
        # elif self.n_feature_dim == self.n_dim and other.sz_feature_dim > 0:
        #     sz_feature_dim = other.sz_feature_dim + self.n_dim
        else: raise TypeError(f"Error in concatenating {self} and {other}: multiple sets of feature dimensions. ")
        # Deal with the sequence dimensions
        if self.sz_sequence_dim == 0:
            if other.sz_sequence_dim <= 0 or other.n_sequence_dim == other.n_dim - other.n_func_dim - other.n_batch_dim - other.n_feature_dim: sz_sequence_dim = -other.n_sequence_dim
            elif self.n_space_dim == 0: sz_sequence_dim = other.sz_sequence_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: sequence dimensions in middle. ")
        elif other.sz_sequence_dim == 0:
            if self.sz_sequence_dim >= 0 or self.n_sequence_dim == self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_feature_dim: sz_sequence_dim = self.n_sequence_dim
            elif other.n_space_dim == 0: sz_sequence_dim = self.sz_sequence_dim
            else: raise TypeError(f"Error in concatenating {self} and {other}: sequence dimensions in middle. ")
        elif self.sz_func_dim >= 0 and self.sz_batch_dim >= 0 and self.sz_feature_dim >= 0 and (self.sz_sequence_dim < 0 or self.n_space_dim == 0) and \
             other.sz_func_dim <= 0 and other.sz_batch_dim <= 0 and other.sz_feature_dim <= 0 and (other.sz_sequence_dim > 0 or other.n_space_dim == 0):
            sz_sequence_dim = [-1, 1][self.sequence_start == self.seq_spc_start] * (self.n_sequence_dim + other.n_sequence_dim)
        # elif other.n_sequence_dim == other.n_dim and self.sz_sequence_dim < 0:
        #     sz_sequence_dim = self.sz_sequence_dim - other.n_dim
        # elif self.n_sequence_dim == self.n_dim and other.sz_sequence_dim > 0:
        #     sz_sequence_dim = other.sz_sequence_dim + self.n_dim
        else: raise TypeError(f"Error in concatenating {self} and {other}: multiple sets of sequence dimensions. ")
        return self.__class__.__new_raw__(super().__add__(other), sz_func_dim=sz_func_dim, sz_batch_dim=sz_batch_dim, sz_feature_dim=sz_feature_dim, sz_sequence_dim=sz_sequence_dim)
        
    def __radd__(self, other):
        avouch(isinstance(other, tuple), TypeError("Summation for 'bt.Size' is inherited from python object 'tuple' to perform concatenation, please use `size <<(>>) 2` to perform element-wise summation (subtraction) to increase (decrease) the size. "))
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        return other.__add__(self)
    
    @alias('__imul__', '__rmul__')
    def __mul__(self, other):
        avouch(isinstance(other, int_), TypeError("Production for 'bt.Size' is inherited from python object 'tuple' to perform duplication, please use `size **(//) 2` to perform element-wise multiplication (division) to enlarge (shrink) the size. "))
        if self.n_func_dim == self.n_dim:
            avouch(other in (0, 1), TypeError(f"Error in {self} * {other}: multiple functional dimensions. "))
            if other == 0: return self.__class__.__new_raw__(tuple())
            return self
        if self.n_batch_dim == self.n_dim:
            avouch(other in (0, 1), TypeError(f"Error in {self} * {other}: multiple batch dimensions. "))
            if other == 0: return self.__class__.__new_raw__(tuple())
            return self
        if self.n_feature_dim == self.n_dim:
            return self.__class__.__new_raw__(super().__mul__(other), sz_feature_dim=self.n_dim * other)
        if self.n_sequence_dim == self.n_dim:
            return self.__class__.__new_raw__(super().__mul__(other), sz_sequence_dim=self.n_dim * other)
        if self.n_space_dim > 0:
            return self.with_space(self.space.tuple() * other)
        if self.n_sequence_dim > 0:
            return self.with_sequence(self.sequence.tuple() * other)
        avouch(self.n_feature_dim > 0, RuntimeError(f"Size {self} encounters an inner problem: n_space_dim + n_sequence_dim + n_feature_dim + n_batch_dim + n_func_dim != n_dim. "))
        return self.with_feature(self.feature.tuple() * other)
    
    ## element-wise operations:
    @staticmethod
    def __op__(self, other, *, operation, identity):
        avouch(isinstance(self, Size), RuntimeError("Inner problem: if 'bt.Size.__op__' is not called manually, please contact the developers with Error Code: B526"))
        avouch(isinstance(other, (num_, tuple)), TypeError(f"Element-wise operations are only used for numbers or tuples, not {type(other)}."))
        op = lambda x, y: (max_(int_(operation(x, y)), 0) if x >= 0 else -1) if identity == 0 or y >= 0 else -1
        if isinstance(other, num_): return self.with_space(tuple(op(x, other) for x in self.space))
        other_func = identity
        other_batch = identity
        other_feature = (identity,)
        other_sequence = (identity,)
        other_space = (identity,)
        if isinstance(other, Size):
            if other.has_func: other_func = other.n_func
            if other.has_batch: other_batch = other.n_batch
            if other.has_feature: other_feature = other.feature
            if other.has_sequence: other_sequence = other.sequence
            if other.has_space: other_space = other.space
        elif isinstance(other, tuple): other_space = other
        else: raise TypeError(f"Cannot perform element-wise operation between types {type(self)} and {type(other)}. ")
        self_feature = tuple()
        self_sequence = tuple()
        self_space = tuple()
        if self.has_feature: self_feature = self.feature
        if self.has_sequence: self_sequence = self.sequence
        if self.has_space: self_space = self.space
        if len(other_feature) == 1: other_feature *= self.n_feature_dim
        elif len(self_feature) == 1: self_feature *= len(other_feature)
        if len(other_sequence) == 1: other_sequence *= self.n_sequence_dim
        elif len(self_sequence) == 1: self_sequence *= len(other_sequence)
        if len(other_space) == 1: other_space *= self.n_space_dim
        elif len(self_space) == 1: self_space *= len(other_space)
        avouch(isinstance(other_func, num_), TypeError(f"Invalid operation between {self} and {other}: conflict in functional dimension. "))
        avouch(isinstance(other_batch, num_), TypeError(f"Invalid operation between {self} and {other}: conflict in batch dimension. "))
        avouch(isinstance(other_feature, tuple) and len(other_feature) == self.n_feature_dim, TypeError(f"Invalid operation between {self} and {other}: conflict in feature size. "))
        avouch(isinstance(other_sequence, tuple) and len(other_sequence) == self.n_sequence_dim, TypeError(f"Invalid operation between {self} and {other}: conflict in sequence size. "))
        avouch(isinstance(other_space, tuple) and len(other_space) == self.n_space_dim, TypeError(f"Invalid operation between {self} and {other}: conflict in space size. "))
        if self.has_func: self = self.with_func(op(self.n_func, other_func))
        if self.has_batch: self = self.with_batch(op(self.n_batch, other_batch))
        if self.has_feature: self = self.with_feature(tuple(op(x, y) for x, y in zip(self_feature, other_feature)))
        if self.has_sequence: self = self.with_sequence(tuple(op(x, y) for x, y in zip(self_sequence, other_sequence)))
        if self.has_space: self = self.with_space(tuple(op(x, y) for x, y in zip(self_space, other_space)))
        return self

    @alias('__ilshift__', '__rlshift__')
    def __lshift__(self, other): return Size.__op__(self, other, operation=lambda x, y: x + y, identity=0)
    @alias('__irshift__')
    def __rshift__(self, other): return Size.__op__(self, other, operation=lambda x, y: x - y, identity=0)
    def __rrshift__(self, other): return Size.__op__(self, other, operation=lambda x, y: y - x, identity=0)
    @alias('__ipow__', '__rpow__')
    def __pow__(self, other): return Size.__op__(self, other, operation=lambda x, y: x * y, identity=1)
    @alias('__ifloordiv__')
    def __floordiv__(self, other): return Size.__op__(self, other, operation=lambda x, y: x // y, identity=1)
    def __rfloordiv__(self, other): return Size.__op__(other, self, operation=lambda x, y: y // x, identity=1)
    
    def __xor__(self, other):
        """
        A ^ B returns A_ and B_ of the same number of dimensions, given that A_ has the same total element to A and B_ has the same total element to B. 
        One can expand to tensors of sizes A and B to A_ and B_ so that pytorch can easily handle calculations. 
        """
        avouch(isinstance(self, Size) and isinstance(other, tuple), TypeError("xor for bt.Size only accept two tuples."))
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        # Deal with func dimensions
        swap = False # swap to ensure that variable 'self' has more and recover when done
        if not self.has_func and other.has_func: self, other = other, self; swap = True
        if self.has_func and not other.has_func:
            if self.sz_func_dim > 0: other = Size(1).with_func_dim(True) + other
            else: other = other + Size(1).with_func_dim(True)
            other.sz_func_dim = self.sz_func_dim
        if swap: self, other = other, self
        if self.has_func and other.has_func:
            avouch(self.sz_func_dim * other.sz_func_dim > 0 or self.n_func_dim == self.n_dim or other.n_func_dim == other.n_dim,
                   TypeError(f"Conflict occurred in unifying sizes {self} and {other}: mismatched order between functional dimension and others. "))
        # Deal with batch dimensions
        swap = False # swap to ensure that variable 'self' has more and recover when done
        if not self.has_batch and other.has_batch: self, other = other, self; swap = True
        if self.has_batch and not other.has_batch:
            if self.sz_batch_dim > 0: other = other[:other.size_start] + Size({1}) + other[other.size_start:]
            else: other = other[:other.size_stop] + Size({1}) + other[other.size_stop:]
            other.sz_batch_dim = self.sz_batch_dim
        if swap: self, other = other, self
        if self.has_batch and other.has_batch:
            avouch(self.sz_batch_dim * other.sz_batch_dim > 0 or self.n_batch_dim == self.n_dim - self.n_func_dim or other.n_batch_dim == other.n_dim - other.n_func_dim,
                   TypeError(f"Conflict occurred in unifying sizes {self} and {other}: mismatched order between batch dimension and others. "))
        # Deal with feature dimensions
        swap = False # swap to ensure that variable 'self' has more and recover when done
        if not self.has_feature and other.has_feature: self, other = other, self; swap = True
        if self.has_feature and not other.has_feature:
            if self.sz_feature_dim > 0: other = other[:other.non_bat_start] + Size([1] * self.n_feature_dim) + other[other.non_bat_start:]
            else: other = other[:other.non_bat_stop] + Size([1] * self.n_feature_dim) + other[other.non_bat_stop:]
            other.sz_feature_dim = self.sz_feature_dim
        if swap: self, other = other, self
        if self.has_feature and other.has_feature:
            avouch(self.sz_feature_dim * other.sz_feature_dim > 0 or self.n_feature_dim == self.n_dim - self.n_func_dim - self.n_batch_dim or other.n_feature_dim == other.n_dim - other.n_func_dim - other.n_batch_dim,
                   TypeError(f"Conflict occurred in unifying sizes {self} and {other}: mismatched order between feature dimensions and sequence/space dimensions. "))
            if self.n_feature_dim > other.n_feature_dim: other = other.with_feature((1,) * (self.n_feature_dim - other.n_feature_dim) + other.feature.tuple())
            else: self = self.with_feature((1,) * (other.n_feature_dim - self.n_feature_dim) + self.feature.tuple())
        # Deal with sequence dimensions
        swap = False # swap to ensure that variable 'self' has more and recover when done
        if not self.has_sequence and other.has_sequence: self, other = other, self; swap = True
        if self.has_sequence and not other.has_sequence:
            if self.sz_sequence_dim > 0: other = other[:other.seq_spc_start] + Size(('1',) * self.n_sequence_dim) + other[other.seq_spc_start:]
            else: other = other[:other.seq_spc_stop] + Size(('1',) * self.n_sequence_dim) + other[other.seq_spc_stop:]
            other.sz_sequence_dim = self.sz_sequence_dim
        if swap: self, other = other, self
        if self.has_sequence and other.has_sequence:
            avouch(self.sz_sequence_dim * other.sz_sequence_dim > 0 or self.n_sequence_dim == self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_feature_dim or other.n_sequence_dim == other.n_dim - other.n_func_dim - other.n_batch_dim - self.n_feature_dim,
                   TypeError(f"Conflict occurred in unifying sizes {self} and {other}: mismatched order between sequence dimensions and space dimensions. "))
            if self.n_sequence_dim > other.n_sequence_dim: other = other.with_sequence((1,) * (self.n_sequence_dim - other.n_sequence_dim) + other.sequence.tuple())
            else: self = self.with_sequence((1,) * (other.n_sequence_dim - self.n_sequence_dim) + self.sequence.tuple())
        # Deal with space dimensions
        swap = False # swap to ensure that variable 'self' has more and recover when done
        if not self.has_space and other.has_space: self, other = other, self; swap = True
        if self.has_space and not other.has_space:
            if self.sz_sequence_dim * other.sz_sequence_dim < 0: other.sz_sequence_dim = self.sz_sequence_dim
            elif other.sz_sequence_dim == 0:
                if self.sz_feature_dim * other.sz_feature_dim < 0: other.sz_feature_dim = self.sz_feature_dim
                elif other.sz_feature_dim == 0:
                    if self.sz_batch_dim * other.sz_batch_dim < 0: other.sz_batch_dim = self.sz_batch_dim
                    elif other.sz_batch_dim == 0:
                        if self.sz_func_dim * other.sz_func_dim < 0:
                            other.sz_func_dim = self.sz_func_dim
            other = other[:other.space_start] + Size((1,) * self.n_space_dim) + other[other.space_start:]
        if swap: self, other = other, self
        if self.has_space and other.has_space:
            if self.n_space_dim > other.n_space_dim: other = other.with_space((1,) * (self.n_space_dim - other.n_space_dim) + other.space.tuple())
            else: self = self.with_space((1,) * (other.n_space_dim - self.n_space_dim) + self.space.tuple())
        return self, other
    
func_dim_size = lambda i: Size.__new_raw__((i,), sz_func_dim=1)
func_dim = func_dim_size(1)
    
class FakeSize(tuple):
    def __new__(cls, raw_tuple, sz_func_dim = 0, sz_batch_dim = 0, sz_feature_dim = 0, sz_sequence_dim = 0):
        """
        Create a FakeSize without n_dim and checks involving n_dim and special-dim conflicts. 
        THIS IS PRIVATE FOR BaTorch 2.0, please donot use this if you are not familiar with is. 
        This is designed in the first place for the dimension manipulations with a tuple of 
            integers is provided along with special dimension information. 
        
        Examples::
            >>> bt.Size(2,3,4).special_from(bt.FakeSize((10, 10), sz_batch_dim=1))
            batorch.Size({2}, 3, 4)
        """
        if raw_tuple is None: return None
        if isinstance(raw_tuple, Size):
            self = super().__new__(cls, raw_tuple.tuple())
            self.sz_func_dim = raw_tuple.sz_func_dim
            self.sz_batch_dim = raw_tuple.sz_batch_dim
            self.sz_feature_dim = raw_tuple.sz_feature_dim
            self.sz_sequence_dim = raw_tuple.sz_sequence_dim
            return self
        self = super().__new__(cls, raw_tuple)
        self.sz_func_dim = sz_func_dim
        self.sz_batch_dim = sz_batch_dim
        self.sz_feature_dim = sz_feature_dim
        self.sz_sequence_dim = sz_sequence_dim
        return self
    def __repr__(self):
        return 'FakeSize' + super().__repr__().rstrip(',)') + f", sz_func_dim={self.sz_func_dim}, sz_batch_dim={self.sz_batch_dim}, sz_feature_dim={self.sz_feature_dim}, sz_sequence_dim={self.sz_sequence_dim})"
    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], int_): return super().__getitem__(*args)
        return FakeSize(super().__getitem__(*args), sz_func_dim = self.sz_func_dim, sz_batch_dim = self.sz_batch_dim, sz_feature_dim = self.sz_feature_dim, sz_sequence_dim = self.sz_sequence_dim)
    @alias('__iadd__')
    def __add__(self, other):
        return FakeSize(super().__add__(tuple(other)), sz_func_dim = self.sz_func_dim, sz_batch_dim = self.sz_batch_dim, sz_feature_dim = self.sz_feature_dim, sz_sequence_dim = self.sz_sequence_dim)
    def __radd__(self, other):
        return FakeSize(tuple(other) + tuple(self), 
            sz_func_dim = getattr(other, 'sz_func_dim', self.sz_func_dim), 
            sz_batch_dim = getattr(other, 'sz_batch_dim', self.sz_batch_dim), 
            sz_feature_dim = getattr(other, 'sz_feature_dim', self.sz_feature_dim), 
            sz_sequence_dim = getattr(other, 'sz_sequence_dim', self.sz_sequence_dim)
        )

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
        return self
    
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
            return torch.Tensor.__getitem__(self, indices).as_subclass(Tensor).special_from(new_shape).grad_fn_name_("indexing")
            # return Tensor._make_subclass(Tensor, super().__getitem__(indices).as_subclass(torch.Tensor), ref_size=new_shape).grad_fn_name_("indexing")
    
    def __iter__(self):
        for i in range_(self.size(0)):
            yield self[i]
        
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
            output = torch_super(self, 'rename')(*args, **kwargs).as_subclass(Tensor).special_from(self)
        for i, n in enumerate(output.names):
            if n is None: continue
            if 'func' in n.lower(): output.add_special_dim(i, func_dim)
            if 'batch' in n.lower(): output.add_special_dim(i, {})
            elif 'channel' in n.lower() or 'feature' in n.lower(): output.add_special_dim(i, [])
            elif 'time' in n.lower() or 'series' in n.lower() or 'sequence' in n.lower(): output.add_special_dim(i, '')
        return output.grad_fn_name_('rename')

    def refine_names(self, *args):
        with torch._C.DisableTorchFunction():
            output = torch_super(self, 'refine_names')(*args).as_subclass(Tensor).special_from(self)
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
            return torch_super(self, 'type')(torch_dt).as_subclass(Tensor).special_from(self).grad_fn_name_('astype')
        # if isinstance(dt, str): return super().type(dt.replace('bt.', 'torch.')).as_subclass(self.__class__)
        # if hasattr(dt, 'dtype'): dt = dt.dtype
        # if isinstance(dt, torch.dtype): return super().type(dt).as_subclass(self.__class__)
        # import numpy as np
        # dt_name = np.dtype(dt).name
        # dtype_map = {'uint16': "int32", 'uint32': "int64", 'uint64': "int64"}
        # torch_dt = getattr(torch, dtype_map.get(dt_name, dt_name), None)
        # avouch(torch_dt is not None, f"Invalid dtype {dt}: {dt_name} cannot be converted into torch dtype.")
        # return super().type(torch_dt).as_subclass(self.__class__)

    def type(self, dt=None):
        with torch._C.DisableTorchFunction():
            if dt is None: return torch_super(self, 'type')().replace("torch.", "batorch.")
            else: return self.astype(dt)

    def __getattribute__(self, key):
        with torch._C.DisableTorchFunction():
            g = super(Tensor, self).__getattribute__(key)
        if key in ('grad', 'real', 'imag'):
            if not isinstance(g, torch.Tensor): return g
            return g.as_subclass(Tensor).special_from(self)
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

    basic_vars = list(locals().keys()) + ['basic_vars']
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
            return obj.as_subclass(Tensor).special_from(ref_shape)
        else:
            ref_shape = broadcast(self_shape, condition_shape, equals_shape, with_size_updates=True)
            self = self.view(ref_shape.updated_sizes[0])
            condition = condition.view(ref_shape.updated_sizes[1])
            equals = equals.view(ref_shape.updated_sizes[2])
            with torch._C.DisableTorchFunction():
                obj = torch_super(self, 'where')(~condition, equals)
            return obj.as_subclass(Tensor).special_from(ref_shape)
    def down_scale(self: 'Tensor', factor=1):
        return self[(slice(None),) * self.space_start + (slice(None, None, factor),) * self.n_space_dim] # suppress: special_from
    def up_scale(self: 'Tensor', factor=1):
        for d in range_(*self.space_range):
            self = self.amplify(factor, d)
        return self # suppress: special_from
    additional_methods = list(set(locals()) - set(basic_vars))
    
    for is_biway, f in [(False, f) for f in additional_methods] + [(True, f) for f in biway_functions]:
        func_codes = get_updated_code(eval(f), mode='method', ignore_args=['out'] if is_biway else [])
        execblock(func_codes)
        if hasattr(torch.Tensor, f):
            torch_doc = f"The document of the original function is:\n{getattr(torch.Tensor, f).__doc__}"
            for line in torch_doc.split('\n'):
                if line.startswith("See :func:"):
                    torch_doc += f"\nwhich is:\n{eval(line.split('See :func:')[-1].strip().strip('`.')).__doc__}"
                    break
        else: torch_doc = ''
        eval(f).__doc__ = f"""Automatically inheritted method from 'torch.Tensor.{f}'. The automatically generated codes are as follows:
        {add_lineno(func_codes)}
        \n{torch_doc}
        """
    else: # aliases (useless 'else' scope for a new section)
        concatenate = cat
        inv = inverse
        tr = trace
        matrix_power = matpow
        matrix_exp = matexp
        matrix_log = matlog
        matrix_rank = rank
        matrix_norm = matnorm
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        try:
            if Tensor in types and cls != Tensor: return NotImplemented
            
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
            print(f"In function {func.__qualname__}:")
            raise e

def expand(self: Tensor, *sizes: Size): return self.expand(*sizes)
def expand_as(self: Tensor, other: Tensor): return self.expand_as(other)
def expand_to(self: Tensor, *target, assign_to_dims: exist_dim=None, dims_allow_mismatch: exist_dim=None): return self.expand_to(*target, assign_to_dims=assign_to_dims, dims_allow_mismatch=dims_allow_mismatch)

basic_locals = list(locals().keys()) + ['basic_locals']
def complex(real: Tensor, imag: Tensor, *, out=None): ...
def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False): ...
def as_tensor(data: Tensor, dtype=None, device=None): ...
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
def empty_like(input: Tensor, *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def full_like(input: Tensor, fill_value=0, *, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def ones_like(input: Tensor, *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def zeros_like(input: Tensor, *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def tensor_like(input, target: Tensor, *, dtype=None, device=None, requires_grad=None):
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
def rand(*size: Size, generator=None, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, pin_memory=False): ...
@collect_memory
def randn(*size: Size, out=None, dtype=None, layout=torch.strided, device=_device.main_device, requires_grad=False, pin_memory=False): ...
@collect_memory
def rand_like(input: Tensor, *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
@collect_memory
def randn_like(input: Tensor, *, dtype=None, layout=None, device=_device.main_device, requires_grad=False, memory_format=torch.preserve_format): ...
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
        else: return torch.cat(tensors, dim, out=out).as_subclass(Tensor).special_from(bt_tensors[0])

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
        else: return torch.stack(tensors, dim[0], out=out).as_subclass(Tensor).special_from(dim)

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
def eye_like(input: Tensor, dim=None):
    """
    create identity matrix from shape of `input` in (the first available condition):
    (1) feature dimensions if size has at least one; 
    (2) space dimensions if size has at least one; 
    (3) sequence dimensions if size has at least one. 
    """
    return eye(input_shape, dim=dim, dtype=input.dtype, device=input.device, layout=input.layout)

additional_functions = [f for f in locals() if f not in basic_locals]

for f in additional_functions + biway_functions:
    func_codes = get_updated_code(eval(f))
    execblock(func_codes)
    if hasattr(torch, f): torch_doc = f"The document of the original function is:\n{getattr(torch, f).__doc__}"
    else: torch_doc = ''
    eval(f).__doc__ = f"""Automatically inheritted package function from 'torch'. The automatically generated codes are as follows:
        {add_lineno(func_codes)}
        \n{torch_doc}
    """
else: # aliases (useless 'else' scope for a new section)
    concatenate = cat
    inv = inverse
    tr = trace
    matrix_power = matpow
    matrix_exp = matexp
    matrix_log = matlog
    matrix_rank = rank
    matrix_norm = matnorm

def to_bttensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
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
