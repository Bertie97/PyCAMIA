
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch.",
    requires = "torch"
)

__all__ = """
    CPU
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
from functools import wraps
from typing import Generator
from .device import GB, CPU, AutoDevice, FixedDevice

with __info__:
    import torch
    from pycamia import ByteSize, SPrint
    from pycamia import touch, avouch, alias, execblock
    from pycamia import get_alphas, arg_tuple, max_argmax
    from pycamia import argmax as _argmax, item, to_list
    from pyoverload import array, iterable

INT = builtins.int
MIN = builtins.min
MAX = builtins.max
ANY = builtins.any
ALL = builtins.all
SUM = builtins.sum
RANGE = builtins.range
FLOAT = builtins.float
NUM = (INT, FLOAT)

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
avouch(ALL(x in new_dim_methods for x in new_dim_inherit_methods))

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
def turn_off_autodevice(): global _device; _device = CPU

def get_cpu_memory_used():
    global _total_cpu_memory_used
    return ByteSize(_total_cpu_memory_used)

def get_gpu_memory_used():
    global _total_gpu_memory_used
    return ByteSize(_total_gpu_memory_used)

def kwget(kwargs, key, default):
    if kwargs is None: return default
    else: return kwargs.get(key, default)
    
def dim_method_wrapper(func):
    if hasattr(func, '__wrapped__'): name = func.__wrapped__.__name__
    else: name = func.__name__
    if name not in new_dim_methods and name not in old_dim_methods: return func
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        single = False
        domain = args; key = None
        if 'dim' in kwargs: domain = kwargs['dim']; key = 'dim'
        elif 'dims' in kwargs: domain = kwargs['dims']; key = 'dims'
        if not isinstance(domain, tuple): domain = (domain,); single = True
        pre_args = []
        post_args = []
        if len(domain) > 1 and name in one_dim_methods_last:
            *pre_args, d = domain
            domain = (d,); single = True
        if len(domain) > 1 and name in one_dim_methods_first:
            d, *post_args = domain
            domain = (d,); single = True
        
        new_domain = []
        if name in new_dim_methods:
            ibatch = ichannel = None
            for i, d in enumerate(domain):
                cls = None
                if isinstance(d, list) and len(d) == 0: ibatch = 0; d = ibatch; cls = list
                elif isinstance(d, dict) and len(d) == 0: ichannel = 1 if self.batch_dim == 0 else 0; d = ichannel; cls = set
                elif isinstance(d, list) and len(d) == 1: ibatch = d[0]; d = ibatch; cls = list
                elif isinstance(d, set) and len(d) == 1: ichannel = list(d)[0]; d = ichannel; cls = set
                if isinstance(d, INT):
                    l, u = - self.ndim - 1, self.ndim
                    if not l <= d <= u:
                        raise IndexError(f"Dimension out of range (expected to be in range of [{l}, {u}], but got {d})")
                    d += self.ndim + i + 1 if d < 0 and 'squeeze' not in name else 0
                if name in new_dim_inherit_methods and cls is not None: d = cls([d])
                new_domain.append(d)
        elif name in old_dim_methods:
            for d in domain:
                if isinstance(d, list) and len(d) <= 1: d = self.batch_dimension
                elif isinstance(d, dict) and len(d) == 0 or isinstance(d, set) and len(d) == 1:
                    d = self.channel_dimension
                if isinstance(d, INT): d += self.ndim if d < 0 and 'squeeze' not in name else 0
                new_domain.append(d)

        domain = tuple(new_domain)
        if key is None: args = domain
        else: kwargs[key] = domain[0] if single else domain

        if name in new_dim_methods and name not in new_dim_inherit_methods:
            if ibatch is not None: kwargs['batch_dim'] = ibatch
            if ichannel is not None: kwargs['channel_dim'] = ichannel

        args = tuple(pre_args) + args + tuple(post_args)
        return func(self, *args, **kwargs)
    return wrapper

class Size(tuple):

    NegSizeError = TypeError("Size cannot have negative values except -1 indicating an arbitrary number. ")

    def __new__(cls, *args, **kwargs):

        if len(args) == 1:
            arg = args[0]
            if hasattr(arg, 'shape'): arg = arg.shape
            if isinstance(arg, (INT, set)): pass
            elif isinstance(arg, Size): args = arg.python_repr
            elif isinstance(arg, tuple): args = arg
            elif isinstance(arg, Generator):
                self = super().__new__(cls, arg)
                self.ndim = len(tuple(arg))
                return self.set_special_(kwargs.get('batch_dim', self.ndim), kwargs.get('channel_dim', self.ndim))
            elif isinstance(arg, list):
                l = len(arg)
                if l != 1:
                    self = super().__new__(cls, arg)
                    self.ndim = l
                    return self.set_special_(kwargs.get('batch_dim', self.ndim), kwargs.get('channel_dim', self.ndim))
            elif isinstance(arg, Size):
                self = super().__new__(cls, arg.tuple())
                self.ndim = arg.ndim
                self._special = arg._special
                self._batch_first = arg._batch_first
                return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))
            else:
                raise TypeError("'Size' object only takes tensors, tuples, lists and generators as initialization. ")

        # args is now a tuple of * or [*], {*} where * is an integer.
        ndim = len(args)
        batch_dim = ndim
        channel_dim = ndim
        raw_args = []
        for i, a in enumerate(args):

            if isinstance(a, list):
                if batch_dim == ndim:
                    batch_dim = i
                else:
                    raise TypeError("Only one batch dimension is allowed.")
                raw_args.append(a[0] if len(a) > 0 else -1)

            elif isinstance(a, (set, dict)):
                if channel_dim == ndim:
                    channel_dim = i
                else:
                    raise TypeError("Only one channel dimension is allowed.")
                raw_args.append(list(a)[0] if len(a) > 0 else -1)

            else: raw_args.append(a)

        self = super().__new__(cls, raw_args)
        self.ndim = ndim
        return self.set_special_(kwargs.get('batch_dim', batch_dim), kwargs.get('channel_dim', channel_dim))

    def tuple(self): return super().__new__(tuple, self)

    @alias("b_dim", "batch_dim")
    @property
    def batch_dimension(self): return self._batch_dimension if self._batch_dimension < self.ndim else None

    @alias("b_dim", "batch_dim")
    @batch_dimension.setter
    def batch_dimension(self, batch_dim): return self.batch_dimension_(batch_dim)

    @property
    def _batch_dimension(self): return self._special[0 if self._batch_first else 1]

    @_batch_dimension.setter
    def _batch_dimension(self, batch_dim):
        self.set_special_(batch_dim, None)

    @alias("b_dim_", "batch_dim_", "with_batchdim")
    def batch_dimension_(self, value):
        if value is None: value = self.ndim
        self._batch_dimension = value
        return self

    @alias("n_batch", "nbatch")
    @property
    def batch_size(self):
        batch_dim = self._batch_dimension
        if batch_dim == self.ndim:
            raise ValueError("There is no batch dimension provided. ")
        return self[batch_dim]

    @alias("c_dim", "channel_dim")
    @property
    def channel_dimension(self): return self._channel_dimension if self._channel_dimension < self.ndim else None

    @alias("c_dim", "channel_dim")
    @channel_dimension.setter
    def channel_dimension(self, channel_dim): return self.channel_dimension_(channel_dim)

    @property
    def _channel_dimension(self): return self._special[1 if self._batch_first else 0]

    @_channel_dimension.setter
    def _channel_dimension(self, channel_dim):
        self.set_special_(None, channel_dim)

    @alias("c_dim_", "channel_dim_", "with_channeldim")
    def channel_dimension_(self, value):
        if value is None: value = self.ndim
        self._channel_dimension = value
        return self

    @alias("n_channel", "nchannel")
    @property
    def channel_size(self):
        channel_dim = self._channel_dimension
        if channel_dim is None:
            raise ValueError("There is no channel dimension provided. ")
        return self[channel_dim]

    @property
    def special(self): return [x for x in self._special if x < self.ndim]

    def special_from_(self, other=None):

        if isinstance(other, Size) or isinstance(other, Tensor) and other.init:
            self._special = [self.ndim if x == other.ndim else x for x in other._special]
            self._batch_first = other._batch_first
        else:
            self._special = [self.ndim, self.ndim]
            self._batch_first = True

        return self

    def add_special_from_(self, other=None):

        if isinstance(other, Size):

            batch_dim = None
            channel_dim = None
            doit = False

            if self._batch_dimension == self.ndim:
                batch_dim = other._batch_dimension
                if batch_dim != self.ndim:
                    doit = True
            if self._channel_dimension == self.ndim:
                channel_dim = other._channel_dimension
                if channel_dim != self.ndim:
                    doit = True
            if doit: self.set_special_(batch_dim, channel_dim)

        return self

    def set_special_(self, batch_dim=None, channel_dim=None):
        """ Set special dimensions for `bt.Size`
        
        Args:
            batch_dim (None or int): the batch dimension, 0 <= batch_dim < n_dim;
                'batch_dim = None' means skip this assignment.
            channel_dim (None or int): the channel dimension, 0 <= channel_dim < n_dim;
                'channel_dim = None' means skip this assignment.
        """

        if batch_dim is not None:
            if not isinstance(batch_dim, INT): batch_dim = self.ndim
            if batch_dim < 0: batch_dim = batch_dim + self.ndim
            if not 0 <= batch_dim <= self.ndim:
                raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.ndim}. ")
        else:
            batch_dim = MIN(self._batch_dimension, self.ndim)

        if channel_dim is not None:
            if not isinstance(channel_dim, INT): channel_dim = self.ndim
            if channel_dim < 0: channel_dim = channel_dim + self.ndim
            if not 0 <= channel_dim <= self.ndim:
                raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.ndim}. ")
        else:
            channel_dim = MIN(self._channel_dimension, self.ndim)

        if batch_dim is None and channel_dim is None: return self

        if batch_dim < channel_dim:
            self._batch_first = True
            self._special = [batch_dim, channel_dim]
        elif channel_dim < batch_dim:
            self._batch_first = False
            self._special = [channel_dim, batch_dim]
        elif batch_dim < self.ndim:
            raise ValueError(f"special dimensions can not be the same: {batch_dim} and {channel_dim}. ")
        else:
            self._batch_first = True
            self._special = [channel_dim, channel_dim]

        return self

    def insert_special_to_tuple(self, target, value):
        s = self._special
        t = tuple(target)
        res = t[:s[0]]
        if s[0] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[0]:s[1]-1]
        if s[1] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[1]-1:]
        return Size(res).special_from_(self)

    def replace_special(self, value):
        s = self._special
        t = tuple(self)
        res = t[:s[0]]
        if s[0] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[0]+1:s[1]]
        if s[1] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[1]+1:]
        return Size(res).special_from_(self)
    
    def with_dimsize(self, k, dim):
        if isinstance(dim, list) and len(dim) in {0, 1}: dim = self.batch_dimension
        elif isinstance(dim, dict) and len(dim) == 0: dim = self.channel_dimension
        elif isinstance(dim, set) and len(dim) == 1: dim = self.channel_dimension
        elif isinstance(dim, INT) and dim < 0: dim += self.n_dim

        return self[:dim] + Size([k] if dim == self.batch_dimension else ({k} if dim == self.channel_dimension else k)) + self[dim+1:]

    @property
    def space(self):
        s = self._special
        t = tuple(self)
        return t[:s[0]] + t[s[0]+1:s[1]] + t[s[1]+1:]

    def reset_space(self, *p):
        p = arg_tuple(p)
        s = self._special
        if s[0] == self.n_dim: return Size(p)
        t = tuple(self)
        return Size(p[:s[0]] + t[s[0]:s[0]+1] + p[s[0]:s[1]-1] + t[s[1]:s[1]+1] + p[s[1]-1:]).special_from_(self)

    @property
    def nele(self):
        p = 1
        for i in self:
            if i == -1: return -1
            p *= i
        return p

    def __len__(self): return self.ndim

    @property
    def n_dim(self): return self.ndim

    @alias('nspace')
    @property
    def n_space(self): return self.ndim - (self._special[0] < self.ndim) - (self._special[1] < self.ndim)

    @alias('nspecial')
    @property
    def n_special(self): return (self._special[0] < self.ndim) + (self._special[1] < self.ndim)

    @property
    def has_batch(self): return self._special[0 if self._batch_first else 1] < self.ndim

    @property
    def has_channel(self): return self._special[1 if self._batch_first else 0] < self.ndim

    @property
    def has_special(self): return self._special != [self.ndim, self.ndim]

    def remove_special_(self):
        self._special = [self.ndim, self.ndim]
        self._batch_first = True
        return self

    def copy(self): return Size(self)

    def __add__(self, other: tuple):
        if not isinstance(other, tuple):
            raise TypeError("Only Size+tuple is available for Size as a python tuple, "
                            "please use size << 1 to increase the size numerically. ")
        if not isinstance(other, Size):
            return Size(tuple(self) + other).special_from_(self)

        ndim = self.ndim + other.ndim
        batch_dim = channel_dim = ndim
        if self.has_batch: batch_dim = self._batch_dimension
        if other.has_batch:
            if batch_dim < ndim: raise TypeError("Batch dimension conflict in addition. ")
            batch_dim = self.ndim + other._batch_dimension
        if self.has_channel: channel_dim = self._channel_dimension
        if other.has_channel:
            if channel_dim < ndim: raise TypeError("Channel dimension conflict in addition. ")
            channel_dim = self.ndim + other._channel_dimension

        return Size(tuple(self) + tuple(other), batch_dim=batch_dim, channel_dim=channel_dim)

    def __radd__(self, other: tuple):
        if not isinstance(other, tuple):
            raise TypeError("Only Size+tuple is available for Size as a python tuple, "
                            "please use size << 1 to increase the size numerically. ")
        if not isinstance(other, Size):
            other = Size(other)
        return other + self
    __iadd__ = __add__

    def __mul__(self, value: INT):
        if not isinstance(value, INT):
            raise TypeError("Only Size*int is available for Size as a python tuple, "
                            "please use size ** 1 to do the multiply numerically. ")
        return Size(tuple(self) * value).special_from_(self)
    __imul__ = __rmul__ = __mul__

    @staticmethod
    def __op__(a: (INT, tuple), b: (INT, tuple), *, op):

        def getvalue(x, y, r):
            if x == -2: return y
            if y == -2: return x
            if x == -1 or y == -1: return -1
            if r: z = op(y, x)
            else: z = op(x, y)
            if z < 0: raise Size.NegSizeError
            return INT(z)

        LengthError = TypeError("Operation only apply for Sizes of the same length, "
                                "please consider to identify the batch/channel dimension or use + to concatenate. ")

        rev = False
        if isinstance(a, NUM):
            a, b = b, a
            rev = True
        if isinstance(a, NUM): raise TypeError("'__op__' for Size do not take two numbers. ")
        if isinstance(b, NUM): return Size(getvalue(x, b, rev) for x in a).special_from_(a)
        if isinstance(a, tuple):
            a, b = b, a
            rev = True
        if isinstance(a, tuple):
            if len(a) == len(b): return Size(getvalue(x, y, rev) for x, y in zip(a, b))
            raise LengthError
        if isinstance(b, tuple):
            if len(a) == len(b): return Size(getvalue(x, y, rev) for x, y in zip(a, b)).special_from_(a)
            if a.space == len(b): b = a.insert_special_to_tuple(b, -2)
        
        # Now deal with two Sizes
        if a.ndim == b.nspace:
            a, b = b, a
            rev = True
        if a.ndim == b.nspace:
            if len(a) == len(b): return Size(getvalue(x, y, rev) for x, y in zip(a, b))
            raise LengthError
        if a.nspace == b.ndim: b = a.insert_special_to_tuple(b, -2)
        if a.ndim == b.ndim:
            if a._special == b._special: return Size(getvalue(x, y, rev) for x, y in zip(a, b)).add_special_from_(a).add_special_from_(b)
            raise TypeError("Only Sizes with same batch/channel dimensions can be operated. ")
        else: raise LengthError

    def __lshift__(self, other: (INT, tuple)): return self.__op__(self, other, op=lambda x, y: x + y)
    __ilshift__ = __rlshift__ = __lshift__

    def __rshift__(self, other: (INT, tuple)): return Size.__op__(self, other, op=lambda x, y: x - y)

    def __rrshift__(self, other: (INT, tuple)): return Size.__op__(other, self, op=lambda x, y: x - y)
    __irshift__ = __rshift__

    def __pow__(self, other: (INT, tuple)): return Size.__op__(self, other, op=lambda x, y: x * y)
    __ipow__ = __rpow__ = __pow__

    def __floordiv__(self, other: (INT, tuple)): return Size.__op__(self, other, op=lambda x, y: x // y)

    def __rfloordiv__(self, other: (INT, tuple)): return Size.__op__(other, self, op=lambda x, y: x // y)
    __ifloordiv__ = __floordiv__

    def __xor__(self, other: tuple):

        if not isinstance(other, Size): other = Size(other)

        if self.ndim == 0: return (Size(1) * other.ndim).special_from_(other), other
        elif other.ndim == 0: return self, (Size(1) * self.ndim).special_from_(self)

        if self.ndim == other.ndim:
            if self.nspecial == 0: return self.special_from_(other), other
            elif other.nspecial == 0: return self, other.special_from_(self)

        if self.nspace == other.nspace:
            if self.nspecial == 0: self = other.insert_special_to_tuple(self, 1)
            elif other.nspecial == 0: other = self.insert_special_to_tuple(other, 1)

        if self.nspecial == other.nspecial:
            if self._batch_first != other._batch_first:
                raise RuntimeError(f"Sizes {self} and {other} with opposite batch-channel order cannot be expand together.")

            self_len1 = self._special[0]
            other_len1 = other._special[0]
            self_len2 = self._special[1] - self._special[0] - 1
            other_len2 = other._special[1] - other._special[0] - 1
            self_len3 = self.ndim - self._special[1] - 1
            other_len3 = other.ndim - other._special[1] - 1
            len1 = MAX(self_len1, other_len1)
            len2 = MAX(self_len2, other_len2)
            len3 = MAX(self_len3, other_len3)
            tup_self = self.tuple()
            tup_other = other.tuple()
            exp_self = (
                (1,) * (len1 - self_len1) + tup_self[:self_len1+1] + 
                (1,) * (len2 - self_len2) + tup_self[self_len1+1:self_len1+self_len2+2] + 
                (1,) * (len3 - self_len3) + tup_self[self_len1+self_len2+2:]
            )
            exp_other = (
                (1,) * (len1 - other_len1) + tup_other[:other_len1+1] + 
                (1,) * (len2 - other_len2) + tup_other[other_len1+1:other_len1+other_len2+2] + 
                (1,) * (len3 - other_len3) + tup_other[other_len1+other_len2+2:]
            )
            if self._batch_first: batch_dim, channel_dim = len1, len1 + len2 + 1
            else: batch_dim, channel_dim = len1 + len2 + 1, len1
            exp_self = Size(exp_self, batch_dim=batch_dim, channel_dim=channel_dim)
            exp_other = Size(exp_other, batch_dim=batch_dim, channel_dim=channel_dim)
            return exp_self, exp_other

        if self.has_batch and other.has_batch:
            lp_self, lp_other = self[:self._batch_dimension] ^ other[:other._batch_dimension]
            rp_self, rp_other = self[self._batch_dimension+1:] ^ other[other._batch_dimension+1:]
            return (
                lp_self + Size([self.batch_size]) + rp_self,
                lp_other + Size([other.batch_size]) + rp_other
            )

        if self.has_channel and other.has_channel:
            lp_self, lp_other = self[:self._channel_dimension] ^ other[:other._channel_dimension]
            rp_self, rp_other = self[self._channel_dimension+1:] ^ other[other._channel_dimension+1:]
            return (
                lp_self + Size({self.channel_size}) + rp_self,
                lp_other + Size({other.channel_size}) + rp_other
            )
        
        tself = tuple(self)
        tother = tuple(other)
        if ALL([a == b for a, b in zip(tself, tother)]):
            ext_self = tuple()
            ext_other = tuple()
            if len(tself) >= len(tother): ext_other = (1,) * (len(tself) - len(tother))
            else: ext_self = (1,) * (len(tother) - len(tself))
            return (self + ext_self).add_special_from_(other), (other + ext_other).add_special_from_(self)

        if ALL([a == b for a, b in zip(tself[::-1], tother[::-1])]):
            ext_self = tuple()
            ext_other = tuple()
            if len(tself) >= len(tother): ext_other = (1,) * (len(tself) - len(tother))
            else: ext_self = (1,) * (len(tother) - len(tself))
            return (ext_self + self).add_special_from_(other), (ext_other + other).add_special_from_(self)

        raise RuntimeError(f"Unexpected error occurs in sizes expanding of {self}, {other}. Please confirm that they can be expanded together. Contact the developers for more information (Error Code: B241). ")
    __ixor__ = __rxor__ = __xor__

    def __getitem__(self, k):
        if isinstance(k, INT): return super().__getitem__(k)
        return Size(*self.python_repr[k])

    @property
    def python_repr(self):
        args = list(self)
        batch_dim = self._batch_dimension
        if batch_dim < self.ndim: args[batch_dim] = [args[batch_dim]]
        channel_dim = self._channel_dimension
        if channel_dim < self.ndim: args[channel_dim] = {args[channel_dim]}
        return tuple(args)

    def __str__(self):
        rep = self.python_repr
        if len(rep) == 1: rep = str(rep).replace(',', '')
        return f"batorch.Size{rep}"
    __repr__ = __str__

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

class Tensor(torch.Tensor):

    init = False

    @staticmethod
    def _make_subclass(cls, data, requires_grad=False, device=None):
        avouch(isinstance(device, AutoDevice), "Please use `AutoDevice` in batorch.Tensor._make_subclass. Please contact the developers if you did not use Tensor._make_subclass directly (Error Code: B242). ")
        data = device(data)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        if isinstance(data, Tensor): return self.special_from_(data)
        return self

    @collect_memory
    def __new__(cls, *args, **kwargs):
        
        batch_dim = channel_dim = None
        for k in kwargs:
            if 'dim' in k:
                if k.startswith('b'): batch_dim = kwargs[k]
                if k.startswith('c'): channel_dim = kwargs[k]
        dev = kwargs.get('device', _device)
        if isinstance(dev, AutoDevice): dev = dev.device
        cpu = torch.device('cpu')

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, INT): pass
            elif isinstance(arg, torch.Tensor):
                rg = kwargs.get('requires_grad', getattr(arg, 'requires_grad', False))
                self = super()._make_subclass(cls, arg, rg)
                self = self.to(dev)
                return self.set_special_(batch_dim, channel_dim)
            elif hasattr(arg, '__tensor__'): return arg.__tensor__()
            elif hasattr(arg, 'shape') or isinstance(arg, list):
                rg = kwargs.get('requires_grad', getattr(arg, 'requires_grad', False))
                self = super().__new__(cls, arg)
                avouch(self.device == cpu)
                self.requires_grad = rg
                self.special_from_(arg)
                return self.set_special_(batch_dim, channel_dim)
            elif isinstance(arg, tuple): args = arg
            else: raise TypeError(f"Unrecognized initialization of 'Tensor': {arg}. ")

        # args is a tuple of integers (or a Size)
        kwargs.update(dict(device=dev))
        rg = kwargs.pop('requires_grad', False)
        dev = kwargs.pop('device', cpu)
        self = super().__new__(cls, *args, **kwargs)
        if dev != cpu: self = self.to(dev)
        self.special_from_(args)
        return self.set_special_(batch_dim, channel_dim).requires_grad_(rg)

    # def __del__(self):
    #     # Release record
    #     if self.device == CPU.main_device:
    #         global _total_cpu_memory_used
    #         _total_cpu_memory_used -= self.byte_size()
    #     else:
    #         global _total_gpu_memory_used
    #         _total_gpu_memory_used -= self.byte_size()
    #     # Delete self
        
    @property
    def ref_count(self):
        return sys.getrefcount(self) - 3

    def requires_grad_(self, mode=True):
        self.requires_grad = mode
        return self

    def refine_names(self, *args, kwargs):
        self.has_names = True
        return super().refine_names(*args, **kwargs)

    @property
    def ishape(self): return super().shape

    @property
    def shape(self):
        shape = Size(super().shape)
        shape.special_from_(self)
        if self.has_names():
            if not shape.has_batch:
                isbatch = [('batch' in x.lower()) if x else x for x in self.names]
                if ANY(isbatch): shape._batch_dimension = isbatch.index(True)
            if not shape.has_channel:
                ischannel = [('channel' in x.lower()) if x else x for x in self.names]
                if ANY(ischannel): shape._channel_dimension = ischannel.index(True)
        return shape

    def rename(self, *args, **kwargs):
        ibatch = ichannel = None
        for i, x in enumerate(args):
            if not isinstance(x, str): continue
            elif 'batch' in x.lower():
                if ibatch is not None:
                    raise TypeError("Multiple batch dimensions not supported. ")
                ibatch = i
            elif 'channel' in x.lower():
                if ichannel is not None:
                    raise TypeError("Multiple channel dimensions not supported. ")
                ichannel = i
            if ibatch is not None and ichannel is not None: break
        return super().rename(*args, **kwargs)

    def refine_names(self, *args):
        ibatch = ichannel = None
        for i, x in enumerate(args):
            if not isinstance(x, str): continue
            elif 'batch' in x.lower():
                if ibatch is not None:
                    raise TypeError("Multiple batch dimensions not supported. ")
                ibatch = i
            elif 'channel' in x.lower():
                if ichannel is not None:
                    raise TypeError("Multiple channel dimensions not supported. ")
                ichannel = i
            if ibatch is not None and ichannel is not None: break
        return super().refine_names(*args)

    @alias("b_dim", "batch_dim")
    @property
    def batch_dimension(self): return self._batch_dimension if self._batch_dimension < self.ndim else None

    @alias("b_dim", "batch_dim")
    @batch_dimension.setter
    def batch_dimension(self, batch_dim): return self.batch_dimension_(batch_dim)

    @property
    def _batch_dimension(self): return self._special[0 if self._batch_first else 1]

    @_batch_dimension.setter
    def _batch_dimension(self, batch_dim):
        self.set_special_(batch_dim, None)

    @alias("b_dim_", "batch_dim_", "with_batchdim")
    def batch_dimension_(self, value):
        if value is None: value = self.ndim
        self._batch_dimension = value
        return self

    @alias("n_batch", "nbatch")
    @property
    def batch_size(self): return self.shape.batch_size

    @alias("c_dim", "channel_dim")
    @property
    def channel_dimension(self): return self._channel_dimension if self._channel_dimension < self.ndim else None

    @alias("c_dim", "channel_dim")
    @channel_dimension.setter
    def channel_dimension(self, channel_dim): return self.channel_dimension_(channel_dim)

    @property
    def _channel_dimension(self): return self._special[1 if self._batch_first else 0]

    @_channel_dimension.setter
    def _channel_dimension(self, channel_dim):
        self.set_special_(None, channel_dim)

    @alias("c_dim_", "channel_dim_", "with_channeldim")
    def channel_dimension_(self, value):
        if value is None: value = self.ndim
        self._channel_dimension = value
        return self


    @alias("n_channel", "nchannel")
    @property
    def channel_size(self): return self.shape.channel_size

    @property
    def space(self):
        s = self._special
        t = tuple(self.ishape)
        return t[:s[0]] + t[s[0]+1:s[1]] + t[s[1]+1:]

    @alias('nele')
    @property
    def n_ele(self): return super().numel()
    def numel(self): return super().numel()

    @property
    def n_dim(self): return self.ndim

    @alias('nspace')
    @property
    def n_space(self): return self.ndim if not self.init else (self.ndim - (self._special[0] < self.ndim) - (self._special[1] < self.ndim))

    @alias('nspecial')
    @property
    def n_special(self): return 0 if not self.init else ((self._special[0] < self.ndim) + (self._special[1] < self.ndim))

    @property
    def has_batch(self): return self.init and self._special[0 if self._batch_first else 1] < self.ndim

    @property
    def has_channel(self): return self.init and self._special[1 if self._batch_first else 0] < self.ndim

    @property
    def has_special(self): return self.init and self._special != [self.ndim, self.ndim]

    @property
    def special(self): return None if not self.init else [x for x in self._special if x < self.ndim]

    def special_from_(self, other=None):

        if isinstance(other, Size) or isinstance(other, Tensor) and other.init:
            self._special = [self.ndim if x == other.ndim else x for x in other._special]
            self._batch_first = other._batch_first
        else:
            self._special = [self.ndim, self.ndim]
            self._batch_first = True

        self.init = True
        return self

    def add_special_from_(self, other=None):

        if isinstance(other, Size):

            doit = False
            batch_dim = None
            channel_dim = None

            if self._batch_dimension == self.ndim:
                batch_dim = other._batch_dimension
                if batch_dim != self.ndim:
                    doit = True
            if self._channel_dimension == self.ndim:
                channel_dim = other._channel_dimension
                if channel_dim != self.ndim:
                    doit = True
            if doit: self.set_special_(batch_dim, channel_dim)

        self.init = True
        return self

    def set_special_(self, batch_dim=None, channel_dim=None, *, special=[], bf=True):
        """ Set special dimensions for `bt.Tensor`
        
        Args:
            batch_dim (None or int): the batch dimension, 0 <= batch_dim < n_dim;
                'batch_dim = None' means skip this assignment.
            channel_dim (None or int): the channel dimension, 0 <= channel_dim < n_dim;
                'channel_dim = None' means skip this assignment.
        """

        if batch_dim is None and channel_dim is None:
            if not special:
                if not self.init: return self.special_from_()
                else: return self
            a, b = special
            if not isinstance(a, INT): a = self.ndim
            if not isinstance(b, INT): b = self.ndim
            if a < 0: a += self.ndim
            if b < 0: b += self.ndim
            if not 0 <= a <= self.ndim or not 0 <= b <= self.ndim:
                raise TypeError(f"Special dimension should be a dimension index which is smaller than {self.ndim}, not {a} and {b}. ")
            if not self.init: self._batch_first = bf
            if a < b:
                self._special = [a, b]
            else:
                self._special = [b, a]
                self._batch_first = not self._batch_first
            self.init = True
            return self

        if batch_dim is not None:
            if not isinstance(batch_dim, INT): batch_dim = self.ndim
            if batch_dim < 0: batch_dim = batch_dim + self.ndim
            if not 0 <= batch_dim <= self.ndim:
                raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.ndim}. ")
        elif self.init:
            batch_dim = MIN(self._batch_dimension, self.ndim)
        else:
            batch_dim = self.ndim

        if channel_dim is not None:
            if not isinstance(channel_dim, INT): channel_dim = self.ndim
            if channel_dim < 0: channel_dim = channel_dim + self.ndim
            if not 0 <= channel_dim <= self.ndim:
                raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.ndim}. ")
        elif self.init:
            channel_dim = MIN(self._channel_dimension, self.ndim)
        else:
            channel_dim = self.ndim

        if batch_dim < channel_dim:
            self._batch_first = True
            self._special = [batch_dim, channel_dim]
        elif channel_dim < batch_dim:
            self._batch_first = False
            self._special = [channel_dim, batch_dim]
        elif batch_dim < self.ndim:
            raise ValueError(f"special dimensions can not be the same: {batch_dim} and {channel_dim}. ")
        else:
            self._batch_first = True
            self._special = [channel_dim, channel_dim]

        self.init = True
        return self

    def remove_special_(self):
        self._special = [self.ndim, self.ndim]
        self._batch_first = True
        return self

    def remove_batch_(self):
        return self.set_special_(batch_dim=self.ndim)

    def remove_channel_(self):
        return self.set_special_(channel_dim=self.ndim)

    def normalize_(self):
        m, M = self.min(), self.max()
        if m == M:
            if M >= 1: return self.one_()
            if m <= 0: return self.zero_()
            return self
        self.sub_(m)
        self.div_(M-m)
        return self

    def normalize(self):
        m, M = self.min(), self.max()
        if m == M:
            if M >= 1: return ones_like(self)
            if m <= 0: return zeros_like(self)
            return self
        return (self - m) / (M - m)

    def tensor(self): return super()._make_subclass(torch.Tensor, self, self.requires_grad)
    
    @alias('extend')
    def append(self, value):
        avouch(self.n_dim == 1, "Only 1-dimensional tensor can use 'append' to concat. ")
        tensor_value = tensor(value) if not isinstance(value, torch.Tensor) else (value.as_subclass(Tensor) if not isinstance(value, Tensor) else value)
        avouch(tensor_value.n_dim <= 1, "Only scalar or 1-dimensional tensors can by concat using append/extend. ")
        return cat(self, tensor_value, dim=0)

    def one_(self): return self.zero_().add_(1)
    
    def autodevice_(self):
        return _device(self)

    def device_(self, device: AutoDevice):
        return device(self)
    
    def range(self): return stack(self.flatten().min(-1).values, self.flatten().max(-1).values, -1)

    def size(self, *k: (INT, str)):
        if len(k) == 0:
            return self.shape
        i = [(self.names.index(x) if x in self.names else x) if isinstance(x, str) else (
            self.batch_dimension if isinstance(x, list) else (self.channel_dimension if isinstance(x, (dict, set)) else x)) for x in k]
        if None in i:
            return super().size(*i)
        if len(i) == 1:
            return self.ishape[i[0]]
        return tuple(self.ishape[x] for x in i)

    def squeeze(self, *dims: INT, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("squeeze function only accept either argument or positional argument. But both are given. ")
        dims = arg_tuple(dims, no_list=True)
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = tuple(i for i, x in enumerate(self.ishape) if x == 1)[::-1]
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        for d in dim:
            if d is None: continue
            self = super(Tensor, self).squeeze(d)
            if d < 0: d = d + ndim
            if a == d: a = ndim
            if d <= a: a -= 1
            if b == d: b = ndim
            if d <= b: b -= 1
            ndim -= 1
        return self.as_subclass(Tensor).set_special_(special=[a, b], bf=bf)

    def squeeze_(self, *dims: INT, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("squeeze_ function only accept either argument or positional argument. But both are given")
        dims = arg_tuple(dims, no_list=True)
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = tuple(i for i, x in enumerate(self.ishape) if x == 1)
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        for d in dim:
            if d is None: continue
            torch.Tensor.squeeze_(self, d)
            if d < 0: d = d + ndim
            if a == d: a = ndim
            if d <= a: a -= 1
            if b == d: b = ndim
            if d <= b: b -= 1
            ndim -= 1
        self.set_special_(special=[a, b], bf=bf)
        return self

    def unsqueeze(self, *dims: INT, dim=None, batch_dim=None, channel_dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("unsqueeze function only accept either argument or positional argument. But both are given")
        dims = arg_tuple(dims, no_list=True)
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = (SUM([i != self.ndim for i in self._special]),)
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        for d in dim:
            self = super(Tensor, self).unsqueeze(d)
            if 0 <= d <= a or d + ndim < a: a += 1
            if 0 <= d <= b or d + ndim < b: b += 1
            ndim += 1
        return self.as_subclass(Tensor).set_special_(special=[a, b], bf=bf).set_special_(batch_dim=batch_dim, channel_dim=channel_dim)

    def unsqueeze_(self, *dims: INT, dim=None, batch_dim=None, channel_dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("unsqueeze_ function only accept either argument or positional argument. But both are given")
        dims = arg_tuple(dims, no_list=True)
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = (0,)
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        for d in dim:
            torch.Tensor.unsqueeze_(self, d)
            if 0 <= d <= a or d + ndim < a: a += 1
            if 0 <= d <= b or d + ndim < b: b += 1
            ndim += 1
        # In-place operation
        self.set_special_(special=[a, b], bf=bf).set_special_(batch_dim=batch_dim, channel_dim=channel_dim)
        return self

    def expand_to(self, *target, axis: list=None, dim: INT=None):
        with torch._C.DisableTorchFunction():

            if len(target) == 1 and not isinstance(target[0], INT): target = target[0]
            if isinstance(target, torch.Tensor): target = target.shape
            if not isinstance(target, Size): target = Size(target)
            if self.init and self.nspecial == target.nspecial and target._batch_first != self._batch_first:
                if self.nspace == 0 and self.ndim == 2: self = self[::-1]
                else: raise TypeError(f"Batch and channel order not matched for {self.shape} and {target}")

            if isinstance(dim, list) and len(dim) == 0: dim = target.batch_dimension
            elif isinstance(dim, dict) and len(dim) == 0: dim = target.channel_dimension
            elif isinstance(dim, list) and len(dim) == 1: dim = dim[0]
            elif isinstance(dim, set) and len(dim) == 1: dim = list(dim)[0]
            if isinstance(dim, INT) and dim < 0: dim += target.ndim
            if axis is None:
                axis_map = list(RANGE(self.ndim))
                p = 0
                use_special = target.has_special
                for i, s in enumerate(self.shape):
                    if p == dim:
                        axis_map[i] = p
                        p += 1
                        continue
                    if p >= target.ndim + 1 and s not in (1, -1):
                        raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
                    if use_special and i == self._batch_dimension:
                        axis_map[i] = target._batch_dimension
                        p = target._batch_dimension + 1
                    elif use_special and i == self._channel_dimension:
                        axis_map[i] = target._channel_dimension
                        p = target._channel_dimension + 1
                    elif s in (1, -1):
                        axis_map[i] = p
                        p += 1
                    else:
                        while p < target.ndim and target[p] not in (s, -1): p += 1
                        axis_map[i] = p
                        p += 1
                    if p >= target.ndim + 1 and s not in (1, -1):
                        raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
                axis=axis_map
            axis = to_list(axis)
            res = self.unsqueeze_to(target, axis=axis)
            shape = res.shape
            if ANY([s > 0 and x % s != 0 and x >= 0 for i, (x, s) in enumerate(zip(target, shape)) if i != dim]):
                raise TypeError(f"Cannot expand the unsqueezed shape {shape} to {target}. Please use keyword 'axis' to manually align the dimensions, " +
                                "use 'dim' to specify the unmatched dimension or use -1 in target size to free dimensions in the expand. ")
            res = res.repeat(tuple(1 if i == dim else (x // s if x >= 0 and s > 0 else 1) for i, (x, s) in enumerate(zip(target, shape))))

        return res.as_subclass(Tensor).special_from_(target)

    def unsqueeze_to(self, *target, axis: list=None):
        with torch._C.DisableTorchFunction():

            if len(target) == 1 and not isinstance(target[0], INT): target = target[0]
            if isinstance(target, torch.Tensor): target = target.shape
            if not isinstance(target, Size): target = Size(target)
            if self.init and self.nspecial == target.nspecial and target._batch_first != self._batch_first:
                if self.nspace == 0 and self.ndim == 2: self = self[::-1]
                else: raise TypeError(f"Batch and channel order not matched for {self.shape} and {target}")

            if axis is None:
                axis_map = list(RANGE(self.ndim))
                p = 0
                for i, s in enumerate(self.shape):
                    if i == self._batch_dimension:
                        axis_map[i] = target._batch_dimension
                        p = target._batch_dimension + 1
                    elif i == self._channel_dimension:
                        axis_map[i] = target._channel_dimension
                        p = target._channel_dimension + 1
                    elif s in (1, -1):
                        axis_map[i] = p
                        p += 1
                    else:
                        while p < target.ndim and target[p] not in (s, -1): p += 1
                        axis_map[i] = p
                        p += 1
                    if p >= target.ndim  + 1: raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
                axis=axis_map
            axis = to_list(axis)
            for i in RANGE(len(target)):
                if i not in axis:
                    self = self.unsqueeze(i)

        return self.as_subclass(Tensor).special_from_(target)

    def unsqueeze_to_(self, *target, axis: list=None):
        with torch._C.DisableTorchFunction():

            if len(target) == 1 and not isinstance(target[0], INT): target = target[0]
            if isinstance(target, torch.Tensor): target = target.shape
            if not isinstance(target, Size): target = Size(target)
            if self.init and self.nspecial == target.nspecial and target._batch_first != self._batch_first:
                if self.nspace == 0 and self.ndim == 2: self = self[::-1]
                else: raise TypeError(f"Batch and channel order not matched for {self.shape} and {target}")

            if axis is None:
                axis_map = list(RANGE(self.ndim))
                p = 0
                for i, s in enumerate(self.shape):
                    if i == self._batch_dimension:
                        axis_map[i] = target._batch_dimension
                        p = target._batch_dimension + 1
                    elif i == self._channel_dimension:
                        axis_map[i] = target._channel_dimension
                        p = target._channel_dimension + 1
                    elif s in (1, -1):
                        axis_map[i] = p
                        p += 1
                    else:
                        while p < target.ndim and target[p] not in (s, -1): p += 1
                        axis_map[i] = p
                        p += 1
                    if p >= target.ndim  + 1: raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
                axis=axis_map
            axis = to_list(axis)
            for i in RANGE(len(target)):
                if i not in axis:
                    torch.Tensor.unsqueeze_(self, i)

        self.special_from_(target)
        return self

    @alias('multiple')
    def multiply(self, num, dim=0):
        d = item(dim)
        return self.unsqueeze(dim).repeat((1,) * d + (num,) + (1,) * (self.ndim - d))

    @alias('ample')
    def amplify(self, num, dim=0):
        d = item(dim)
        return self.multiply(num, d+1).flatten(d, d + 1).special_from_(self)
    
    def row(self, index):
        dim = [d for d in RANGE(3) if d not in self._special][0]
        return self[(slice(None),) * dim + (index,)]
    
    def column(self, index):
        return self[..., index]

    def repeated(self, num, dim=0):
        d = item(dim)
        return self.repeat((1,) * d + (num,) + (1,) * (self.ndim - d - 1))

    def flip(self, dim=0):
        d = item(dim)
        return self[(slice(None),) * d + (arange(self.size(d)-1, -1, -1),) + (slice(None),) * (self.ndim - d - 1)]
    
    def with_pixelvalue(self, ind, val):
        self[ind] = val
        return self

    def sample(self, number: INT = 1, dim: INT = None, random: bool = True):
        """
        sample(self, numbder: int = 1, dim: int = self.batch_dimension, random: bool = True) -> Tensor

        Sample `number` of slices from a given dimension.
        
        Args:
            number (int, optional): the number of slices. Default: `1`
            dim (int, keyword argument): the dimension to slice or select. Default: batch dimension
            random (bool, optional): whether to randomly pick the slices or not. Default: True
        
        Note:
            Using `sample(num, dim)` for data of size (n_1, n_2, ..., n_r) would result in
                a tensor of size (n_1, n_2, ..., n_{dim-1}, num, n_{dim+1}, ..., n_r)
        
        Example::
            >>> data.shape
            batorch.Size(4, 3, 5, 6)
            >>> data.sample(1, 2, random=False).shape
            batorch.Size(4, 3, 1, 6)
            >>> # The above is equivalant to data[:, :, :1, ...].shape.
        """
        if dim is None: dim = self.batch_dimension
        avouch(dim is not None, "Argument 'dim' needed for sampling Tensors with no batch dimension. ")
        avouch(dim < self.n_dim, "`dim` should be smaller than `n_dim` of a tensor.")
        if number < 1: raise TypeError("Argument 'number' for sampling Tensors can not be smaller than 1. ")
        sample_indices = [slice(None)] * self.dim()
        if self.shape[dim] < number: raise TypeError(f"Too many elements needed to be sampled from dimension {dim}")
        if random:
            import random
            samples = random.sample(RANGE(self.shape[dim]), k = number)
        else: samples = list(RANGE(number))
        
        sample_indices[dim] = samples
        output_tensor = self[tuple(sample_indices)].as_subclass(Tensor).special_from_(self)
        output_tensor.indices = samples
        return output_tensor

        # # Deprecated codes of squeezed results for 1-sample
        # if len(samples) != 1:
        # else:
        #     sample_indices[dim] = samples[0]
        #     output_tensor = self[tuple(sample_indices)].as_subclass(Tensor).set_special_(special=[x - 1 if x > dim else (self.ndim-1 if x == dim else x) for x in self._special], bf=self._batch_first)
        #     output_tensor.indices = samples
        #     return output_tensor

    def pick(self, index: INT = None, dim: INT = None, random: bool = False):
        """
        pick(self, index: int = 0, dim: int = self.batch_dimension, random: bool = False) -> Tensor

        Pick one slice on dimension `dim` for big tensors. 
        
        Args:
            index (int, optional): the slice index to pick. Default: `1`
            dim (int, keyword argument): the dimension to slice or select. Default: batch dimension
            random (bool, optional): whether to randomly pick the slice or not. Default: False
        
        Note:
            Using `pick(index, dim)` for data of size (n_1, n_2, ..., n_r) would result in
                a tensor of size (n_1, n_2, ..., n_{dim-1}, n_{dim+1}, ..., n_r)
        
        Examples::
            >>> data.shape
            batorch.Size(4, 3, 5, 6)
            >>> data.sample(-1, 2, random=False).shape
            batorch.Size(4, 3, 6)
            >>> # The above is equivalant to data[:, :, 4, ...].shape.
        """
        if dim is None: dim = self.batch_dimension
        avouch(dim is not None, "Argument 'dim' needed for sampling Tensors with no batch dimension. ")
        avouch(dim < self.n_dim, "`dim` should be smaller than `n_dim` of a tensor.")
        if random:
            avouch(index is None, "'index' should be None if randomly pick is enabled. Use keyword argument 'dim=xx' to identify the dimension.")
            import random
            index = random.randint(0, self.shape[dim]-1)
        if index >= self.shape[dim]: raise IndexError(f"Index out of range: picking {index} from dimension {dim}, should be 0 <= index < {self.shape[dim]}.")
        if dim < 0: dim += self.ndim
        return self[(slice(None),) * dim + (index,)]

    def split(self, sec=1, dim=None, squeeze=False):
        """
        split(self, sec=1, dim=self.channel_dimension, squeeze=False) -> Tensor
        
        Split the tensor on dimension `dim` by sections each of size `sec`.
        
        Args:
            sec (int or list, optional): the section size. Default: `1`
            dim (int, keyword argument): the dimension to split. Default: channel dimension
            squeeze (bool, optional): whether to squeeze the sections, only effective when `sec` == 1.
        
        Note:
            If `sec` is a list, the sizes of the sections will follow the list.

        Examples::
            >>> data.shape
            batorch.Size([4], {6}, 3, 4, 1)
            >>> [x.shape for x in data.split()]
            [batorch.Size([4], {1}, 3, 4, 1), batorch.Size([4], {1}, 3, 4, 1), batorch.Size([4], {1}, 3, 4, 1), batorch.Size([4], {1}, 3, 4, 1), batorch.Size([4], {1}, 3, 4, 1), batorch.Size([4], {1}, 3, 4, 1)]
            >>> [x.shape for x in data.split(squeeze=True)]
            [batorch.Size([4], 3, 4, 1), batorch.Size([4], 3, 4, 1), batorch.Size([4], 3, 4, 1), batorch.Size([4], 3, 4, 1), batorch.Size([4], 3, 4, 1), batorch.Size([4], 3, 4, 1)]
            >>> [x.shape for x in data.split(dim=2, squeeze=True)]
            [batorch.Size([4], {6}, 4, 1), batorch.Size([4], {6}, 4, 1), batorch.Size([4], {6}, 4, 1)]
            >>> [x.shape for x in data.split(3)]
            [batorch.Size([4], {3}, 3, 4, 1), batorch.Size([4], {3}, 3, 4, 1)]
            >>> [x.shape for x in data.split([1,2,3])]
            [batorch.Size([4], {1}, 3, 4, 1), batorch.Size([4], {2}, 3, 4, 1), batorch.Size([4], {3}, 3, 4, 1)]
        """
        if not isinstance(sec, INT):
            sec = touch(lambda: list(sec))
            avouch(sec is not None, "Argument 'sec' for 'split' should be either integer or list of integers.")
            if len(sec) == 1: dim = sec; sec = 1
        if dim is None: dim = self.channel_dimension
        avouch(dim is not None, "Argument 'dim' needed for sampling Tensors with no channel dimension. ")
        avouch(dim < self.n_dim, "`dim` should be smaller than `n_dim` of a tensor.")
        if sec == 1 or isinstance(sec, (tuple, list)) and ALL(x == 1 for x in sec):
            if squeeze: return tuple(x.as_subclass(Tensor).special_from_(self).squeeze(dim) for x in super().split(sec, dim))
        return tuple(x.as_subclass(Tensor).special_from_(self) for x in super().split(sec, dim))

    def movedim(self, dim1: INT, dim2: INT):
        """
        movedim(self, dim1, dim2) -> Tensor

        move dim1 to dim2(specified in the targeting size)
        data of size (2, 3, 4, 5) can be transform to (2, 4, 5, 3) by data.movedim(1, -1) or data.movedim(1, 3)
        """
        if dim1 < 0: dim1 += self.ndim
        if dim2 < 0: dim2 += self.ndim

        with torch._C.DisableTorchFunction():
            if dim1 == dim2: res = self
            elif dim1 < dim2: res = self.unsqueeze(dim2+1).transpose(dim1, dim2+1).squeeze(dim1)
            else: res = self.unsqueeze(dim2).transpose(dim1+1, dim2).squeeze(dim1+1)

        return res.as_subclass(Tensor).set_special_(special=[dim2 if x == dim1 else (
            x if x > dim2 and x > dim1 or x < dim1 and x < dim2 
            else (x + 1 if dim1 > dim2 else x - 1)) for x in self._special], bf=self._batch_first)

    def movedim_(self, dim1: INT, dim2: INT):
        """
        In-place operation for movedim
        """
        if dim1 < 0: dim1 += self.ndim
        if dim2 < 0: dim2 += self.ndim

        with torch._C.DisableTorchFunction():
            if dim1 == dim2: ...
            elif dim1 < dim2: self.unsqueeze_(dim2+1).transpose_(dim1, dim2+1).squeeze_(dim1)
            else: self.unsqueeze_(dim2).transpose_(dim1+1, dim2).squeeze_(dim1+1)

        self._special = [dim2 if x == dim1 else (
            x if x > dim2 and x > dim1 or x < dim1 and x < dim2 
            else (x + 1 if dim1 > dim2 else x - 1)) for x in self._special]
        if self._special[0] > self._special[1]:
            self._special = self._special[::-1]
            self._batch_first = not self._batch_first
        return self

    @alias("joindims")
    def mergedims(self, *dims: INT, target: INT = None):
        """
        mergedims(self, *source, target) -> Tensor

        merge dims into one dimension: target (the last argument)
        data of size (2, 3, 4, 5) can be transform to (24, 5) with a Cartesian of 3 x 2 x 4 by:
            data.mergedims([1, 0, 2], target=0) / data.mergedims(1, 0, 2, target=0)
        Note that one can only omit the target dimension if no order of dimension is changed. 
            the automatically chosen target is the new position of the last dimension one gives. 
            e.g. data.mergedims(1, 0, 3) result in (4, 30) and it follows a Cartesian of 2 x 3 x 5.
        """
        dims = list(arg_tuple(dims))
        avouch(len(dims) >= 2, f"Please input at least two dims to be merged for method 'mergedims', not {dims}. ")
        new_dims = []
        cls_guess = None
        for i, d in enumerate(dims + [target]):
            if d is None: cls = cls_guess; new_dims.append(d); continue
            cls = None
            if isinstance(d, list) and len(d) == 0: ibatch = self.batch_dim if i < len(dims) else 0; d = ibatch; cls = list
            elif isinstance(d, dict) and len(d) == 0: ichannel = self.channel_dim if i < len(dims) else (1 if self.batch_dim == 0 else 0); d = ichannel; cls = set
            elif isinstance(d, list) and len(d) == 1: ibatch = self.batch_dim if i < len(dims) else d[0]; d = ibatch; cls = list
            elif isinstance(d, set) and len(d) == 1: ichannel = self.channel_dim if i < len(dims) else list(d)[0]; d = ichannel; cls = set
            if isinstance(d, INT):
                ndim = self.ndim if i < len(dims) else self.ndim - len(dims) + 1
                l, u = - ndim - 1, ndim
                if not l <= d <= u:
                    raise IndexError(f"Dimension out of range (expected to be in range of [{l}, {u}], but got {d})")
                d += ndim if d < 0 else 0
            else: raise TypeError(f"Invalid dimension {d} found in mergedims, it also happens when one omit 'target='. ")
            new_dims.append(d)
            tcls = None
            if self.batch_dimension == d: tcls = list
            if self.channel_dimension == d: tcls = set
            if i < len(dims): cls_guess = cls_guess if tcls is None or cls_guess == list else tcls
        *dims, target = new_dims
        if target is None:
            target = dims[-1] - SUM([1 if d < dims[-1] else 0 for d in dims[:-1]])
            dims.sort()
        avouch(target < self.ndim - len(dims) + 1, "Dimension out of range, please make sure that 'target' is the dimension in the resulting shape. ")

        res = self.clone()
        other_dims = [i for i in RANGE(self.ndim) if i not in dims]
        out_dims = other_dims[:target] + dims + other_dims[target:]
        res.permute_(out_dims)
        res = res.flatten(target, target + len(dims) - 1)
        if cls is not None:
            if cls == list: res.batch_dim = target
            elif cls == set: res.channel_dim = target
            else: raise AssertionError(f"Invalid dimension tag of type cls: {cls}. Please contact the developer for more info (Error Code: B243). ")
        return res

    def splitdim(self, source: INT, *size: INT):
        """
        splitdim(self, source, *target_size) -> Tensor

        split one dimension source into multiple dimensions: target
        data of size (2, 4, 5) can be transform to (2, 2, 2, 5) with data.splitdim(1, 2, 2).
        Note that batch representations for source and target are different
            (splitdim([1], [2], 2) means split the batchdim at index 1 into a size of ([2], 2), which is 2x2 with batchdim at index 0).
            One can use -1 for arbitrary size. 
        """
        size = Size(arg_tuple(size))
        avouch(len(size) >= 2, f"Please input an at-least-two-dim-shape to split dimension {source} into in method 'splitdim', not {size}. ")

        new_size = self.shape[:source] + size + self.shape[source + 1:]
        prod = 1
        ni = None
        for i, x in enumerate(size):
            if isinstance(x, list) and len(x) == 1: x = x[0]
            elif isinstance(x, set) and len(x) == 1: x = list(x)[0]
            elif isinstance(x, list) and len(x) == 0: x = -1
            elif isinstance(x, dict) and len(x) == 0: x = -1
            if x != -1: prod *= x; continue
            ni = i
        if ni is not None:
            x = self.shape[source] // prod
            if size.batch_dimension == ni: x = [x]
            elif size.channel_dimension == ni: x = {x}
            new_size = new_size[:source + ni] + Size(x) + new_size[source + ni + 1:]

        return self.view(new_size)

    def tobytes(self): return self.detach().cpu().numpy().tobytes()

    def cat(self, other, dim=0):
        return cat((self, other), dim=dim)

    def stack(self, other, dim=0):
        return stack((self, other), dim=dim)
    
    def inv(self):
        return inv(self)

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple): args = args[0]
        ibatch = ichannel = None
        new_size = []
        for i, x in enumerate(args):
            if isinstance(x, list) and len(x) == 0: ibatch = i; x = -1
            elif isinstance(x, dict) and len(x) == 0: ichannel = i; x = -1
            elif isinstance(x, list) and len(x) == 1: ibatch = i; x = x[0]
            elif isinstance(x, set) and len(x) == 1: ichannel = i; x = list(x)[0]
            new_size.append(x)
        output = super().view(*new_size).as_subclass(Tensor)
        if ibatch is not None: output.with_batchdim(ibatch)
        if ichannel is not None: output.with_channeldim(ichannel)
        if isinstance(args, Size): output.special_from_(args)
        return output
    
    def permute_(self, *dims):
        dims = arg_tuple(dims)
        avouch(len(dims) == self.ndim)
        cur_order = list(RANGE(self.ndim))
        for i in RANGE(len(cur_order)):
            j = cur_order.index(dims[i])
            cur_order[i], cur_order[j] = cur_order[j], cur_order[i]
            self.transpose_(i, j)
        return self
    
    def standard(self):
        """
        return a standard version of the tensor: with batch dim as the first dim, channel as the second. 
        """
        dim_list = list(RANGE(self.ndim))
        order = []
        if self._batch_dimension != self.ndim:
            dim_list.remove(self._batch_dimension)
            order.append(self._batch_dimension)
        if self._channel_dimension != self.ndim:
            dim_list.remove(self._channel_dimension)
            order.append(self._channel_dimension)
        return self.permute(*(order + dim_list))
    
    def standard_(self):
        """
        In-place standard of the tensor: with batch dim as the first dim, channel as the second. 
        """
        dim_list = list(RANGE(self.ndim))
        order = []
        if self._batch_dimension != self.ndim:
            dim_list.remove(self._batch_dimension)
            order.append(self._batch_dimension)
        if self._channel_dimension != self.ndim:
            dim_list.remove(self._channel_dimension)
            order.append(self._channel_dimension)
        self.permute_(*(order + dim_list))
        return self
    
    def quantile(self, *args, **kwargs):
        q = tensor(kwargs.get('q', args[0]))
        dim = kwargs.get('dim', args[1] if len(args) >= 2 else None)
        if dim is None: return super().quantile(*args, **kwargs).as_subclass(Tensor)
        if q.ndim > 0: return super().quantile(*args, **kwargs).as_subclass(Tensor).special_from_(self.unsqueeze(0))
        return super().quantile(*args, **kwargs).as_subclass(Tensor).special_from_(self)
    
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
        if isinstance(dt, str): return super().type(dt.replace('bt.', 'torch.') if 'cuda' in dt else ('torch.' + dt.split('.')[-1])).as_subclass(self.__class__).special_from_(self)
        if hasattr(dt, 'dtype'): dt = dt.dtype
        if isinstance(dt, torch.dtype): return super().type(dt).as_subclass(self.__class__).special_from_(self)
        import numpy as np
        dt_name = np.dtype(dt).name
        dtype_map = {'uint16': "int32", 'uint32': "int64", 'uint64': "int64"}
        torch_dt = getattr(torch, dtype_map.get(dt_name, dt_name), None)
        avouch(torch_dt is not None, f"Invalid dtype {dt}: {dt_name} cannot be converted into torch dtype.")
        return super().type(torch_dt).as_subclass(self.__class__).special_from_(self)
    
    def type(self, dt=None):
        if dt is None: return super().type().replace("torch.", "bt.")
        else: return self.astype(dt)

    def byte_size(self):
        return ByteSize(self.element_size() * self.numel())

    @property
    def T(self: 'Tensor') -> 'Tensor':
        output = self.clone()
        if self.n_space == 1: output = output.unsqueeze_(-1)
        if not self.has_special: return super().T
        s = output._special

        with torch._C.DisableTorchFunction():
            permute_dim = tuple(RANGE(s[0]))[::-1] + (s[0],) + tuple(RANGE(s[0]+1, s[1]))[::-1] + (s[1],) * (s[1] != output.ndim) + tuple(RANGE(s[1]+1, output.ndim))[::-1]
        
        return output.permute_(*permute_dim)

    def t(self) -> 'Tensor':
        return self.T

    def t_(self) -> 'Tensor':
        """
        t_() -> Tensor

        In-place version of :meth:`~Tensor.t`
        """
        if not self.has_special: return super().t_()
        s = self._special
        with torch._C.DisableTorchFunction():
            for i in RANGE(s[0] // 2):
                self.transpose_(i, s[0] - i - 1)
            for i in RANGE((s[1] - s[0] - 1) // 2):
                self.transpose_(s[0] + i + 1, s[1] - i - 1)
            for i in RANGE((self.ndim - s[1] - 1) // 2):
                self.transpose_(s[1] + i + 1, self.ndim - i - 1)
        return self
    
    def int(self): return super().long()
    
    def to(self, anything): return super().to(anything).as_subclass(Tensor).special_from_(self)
    
    def __iter__(self):
        for i in RANGE(self.size(0)):
            yield self[i]
            
    def __repr__(self):
        if self.size(0) > 4:
            class_name = self.__class__.__name__
            batch_dim = f"[{self.n_special}]+" if self.n_special > 0 else ''
            xshape = str(self.shape).split('Size')[-1]
            out = SPrint(f"<{batch_dim}{self.n_space}D Tensor on {self.device}: ")
            out(f"shape={xshape}, type={self.type()}, requires_grad={self.requires_grad}>")
            return out.text.rstrip('\n')
        string = super().__repr__()
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")

    def __getattr__(self, key):
        if not self.init:
            if key == '_special': return [self.ndim, self.ndim]
            elif key == '_batch_first': return True
        if key in __all__:
            obj = eval(key)
            if callable(obj):
                def obj_holder(*args, **kwargs):
                    return obj(self, *args, **kwargs)
                return obj_holder
            else: return obj
        else: raise AttributeError(f"'batorch.tensor' object has no attribute '{key}'")
    
    def __getattribute__(self, key):
        g = super().__getattribute__(key)
        if key == 'grad':
            if not isinstance(g, torch.Tensor): return g
            return g.as_subclass(Tensor).special_from_(self)
        return g
    
    def __setattr__(self, key, value):
        if key == 'grad':
            if isinstance(value, Tensor):
                value = value.as_subclass(torch.Tensor)
        super().__setattr__(key, value)

    def __matmul__(self, other, **kwargs):
        if isinstance(other, torch.Tensor) and self.has_special or isinstance(other, Tensor) and other.has_special:
            # if self.nspace == 0 and self.has_channel: self.with_channeldim(None)
            # if other.nspace == 0 and other.has_channel: other.with_channeldim(None)
            if self.device != other.device:
                if self.device != CPU.device: other = other.to(self.device)
                else: self = self.to(other.device)
            space1 = self.space
            space2 = other.space
            rem = len(space1) < 2 or len(space2) < 2
            if len(space1) < 2: space1 += (1,)
            if len(space2) < 2: space2 += (1,)
            if len(space1) < 2: space1 += (1,)
            if len(space2) < 2: space2 += (1,)
            a, b = Size(space1[:-2]) ^ Size(space1[:-2])
            sa = self.shape.reset_space(a)
            sb = other.shape.reset_space(b)
            with torch._C.DisableTorchFunction():
                res = super(torch.Tensor, self.view(sa + space1[-2:])).__matmul__(other.view(sb + space2[-2:]))
            if rem: res.squeeze_(-1)
            if rem and res.size(-1) == 1: res.squeeze_(-1)
            return res.as_subclass(Tensor).special_from_(sa)
        return super().__matmul__(other, **kwargs).as_subclass(Tensor).special_from_()

    def __op__(self, opname, other, **kwargs):
        if not isinstance(other, torch.Tensor): other = tensor(other, device=self.device)
        elif not isinstance(other, Tensor): other = other.as_subclass(Tensor)
        if self.device != other.device:
            if self.device != CPU.device: other = other.to(self.device)
            else: self = self.to(other.device)
        if self.has_special and len(other.shape) > 0 or other.has_special:
            a, b = self.shape ^ other.shape
            with torch._C.DisableTorchFunction():
                res = getattr(super(Tensor, self.view(a).as_subclass(Tensor)), opname)(other.view(b))
            return res.as_subclass(Tensor).special_from_(a)
        sp = None
        if self.has_special: sp = self
        return getattr(super(), opname)(other, **kwargs).as_subclass(Tensor).special_from_(sp)

    for op in '''
    add iadd radd
    sub isub rsub
    mul imul rmul
    div idiv rdiv
    pow ipow rpow
    mod imod rmod
    truediv itruediv rtruediv
    floordiv ifloordiv rfloordiv
    eq ieq req
    ne ine rne
    or ior ror
    and iand rand
    xor ixor rxor
    lt le gt ge
    '''.split():
        exec(f"def __{op}__(self, *args, **kwargs): return self.__op__('__{op}__', *args, **kwargs)")

    def op_(self, opname, *args, **kwargs):
        args = [
            other.remove_special_().unsqueeze_to_(self) if isinstance(other, Tensor) else (
            Tensor.unsqueeze_to_(Tensor(other), self) if isinstance(other, torch.Tensor) else other)
        for other in args]
        sp = None
        if self.has_special: sp = self
        getattr(torch.Tensor, opname)(self, *args, **kwargs).special_from_(sp)
        return self

    for op in '''
    add sub mul div pow mod truediv floordiv addmm addcmul
    '''.split():
        exec(f"def {op}_(self, *args, **kwargs): return self.op_('{op}_', *args, **kwargs)")
    
    def __minmax__(self, fname, *args, **kwargs):
        if len(args) == 1:
            d = args[0]
            if isinstance(d, list): args = [self.batch_dimension]
            elif isinstance(d, (dict, set)): args = [self.channel_dimension]
        return getattr(super(), fname)(*args, **kwargs)
    for f in "min max median argmin argmax".split():
        exec(f"def {f}(self, *args, **kwargs): return self.__minmax__('{f}', *args, **kwargs)")

    ###### old operation code ######
    #    if isinstance(other, torch.Tensor):
    #        other = Tensor(other)
    #        if self.dim() == other.dim():
    #            return super().__add__(other)
    #        elif self.dim() < other.dim():
    #            return self.expand_as(other).__add__(other)
    #        else:
    #            return super().__add__(other.expand_as(self))
    #    return super().__add__(other)
    #################################

    # def __repr__(self, *args, **kwargs):
    #     string = super().__repr__(*args, **kwargs)
    #     if 'shape=' not in string:
    #         string = string.rstrip(')') + f', shape={self.shape})'
    #     return string.replace("tensor", "Tensor")

    def __str__(self, *args, **kwargs):
        string = super().__str__(*args, **kwargs)
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")

    def __hash__(self): return super().__hash__()
    
    # @property
    # def grad(self):
    #     g = self._grad
    #     if not isinstance(g, torch.Tensor): return g
    #     return g.as_subclass(Tensor).special_from_(self)
    
    # @grad.setter
    # def grad(self, value):
    #     if not isinstance(value, Tensor): self._grad=value; return
    #     self._grad = value.as_subclass(torch.Tensor)

    @staticmethod
    def __torch_function_convert__(ret, cls):
        with torch._C.DisableTorchFunction():
            if isinstance(ret, torch.Tensor):
                ret = ret.as_subclass(cls)
            if isinstance(ret, (tuple, list)):
                # Also handles things like namedtuples
                ret = type(ret)(Tensor.__torch_function_convert__(r, cls) for r in ret)
            return ret

    @staticmethod
    def __torch_function_collect__(r, c):
        if isinstance(r, (tuple, list)):
            for x in r: Tensor.__torch_function_collect__(x, c)
        if isinstance(r, Tensor) and not r.init: c.append(r)

    @staticmethod
    def __torch_function_convert_apply__(ret, apply, cls, force=True):

        with torch._C.DisableTorchFunction():

            if isinstance(ret, Tensor):
                if not ret.init or force:
                    if cls != Tensor: ret = ret.as_subclass(cls)
                    apply(ret)
                    return ret

            if isinstance(ret, torch.Tensor):
                ret = ret.as_subclass(cls)
                return ret

            if isinstance(ret, (tuple, list)):
                # Also handles things like namedtuples
                return type(ret)(Tensor.__torch_function_convert_apply__(r, apply, cls) for r in ret)

            if isinstance(ret, Generator):
                # And things iterator like Generators
                def out():
                    for r in ret:
                        if isinstance(r, torch.Tensor):
                            r = r.as_subclass(cls)
                            apply(r)
                            yield r
                return out()

            return ret

    @classmethod
    def __torch_function_ele_wise_func__(cls, func, types, args=(), kwargs=None):
        """
        FOR: tensor where astype type float long cuda
        __add__ __iadd__ __radd__
        __sub__ __isub__ __rsub__
        __mul__ __imul__ __rmul__
        __div__ __idiv__ __rdiv__
        __pow__ __ipow__ __rpow__
        __mod__ __imod__ __rmod__
        __truediv__ __itruediv__ __rtruediv__
        __floordiv__ __ifloordiv__ __rfloordiv__
        __eq__ __ieq__ __req__
        __ne__ __ine__ __rne__
        __or__ __ior__ __ror__
        __and__ __iand__ __rand__
        __xor__ __ixor__ __rxor__
        __lt__ __le__ __gt__ __ge__
        """
        self = args[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_resizing_func__(cls, func, types, args=(), kwargs=None):
        "FOR: zeros ones empty rand randn"
        dims = args[1:]
        if len(dims) == 1 and isinstance(dims[0], tuple): dims = dims[0]
        if not isinstance(dims, Size): dims = Size(dims)
        args = args[:1] + (tuple(dims),)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_full_func__(cls, func, types, args=(), kwargs=None):
        "FOR: full"
        fill = None
        if isinstance(kwargs, dict) and 'fill_value' in kwargs: fill = kwargs['fill_value']
        if fill: dims = args[1:]
        else: dims, fill = args[1:]
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
        if not isinstance(dims, Size): dims = Size(dims)
        args = args[:1] + (tuple(dims), fill)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_resizing_as_func__(cls, func, types, args=(), kwargs=None):
        "FOR: reshape_as view_as unsqueeze_to expand_to"
        dims = args[-1].shape

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_randint_func__(cls, func, types, args=(), kwargs=None):
        "FOR: randint"
        dims = Size(args[3])
        args = args[:3] + (dims,)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_transpose_func__(cls, func, types, args=(), kwargs=None):
        "FOR: transpose transpose_"
        self = args[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            dim1 = kwget(kwargs, 'dim0', args[1])
            dim2 = kwget(kwargs, 'dim1', args[2])
            a, b = self._special
            if a == dim1: a = dim2
            elif a == dim2: a = dim1
            if b == dim1: b = dim2
            elif b == dim2: b = dim1
            r.set_special_(special=[a, b], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_permute_func__(cls, func, types, args=(), kwargs=None):
        "FOR: permute"
        self = args[0]
        dims = args[1:]
        dims = arg_tuple(dims)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        d = lambda i: dims.index(i) if i in dims else self.ndim
        def apply(r):
            r.set_special_(special=[d(self._special[0]), d(self._special[1])], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_matmul_func__(cls, func, types, args=(), kwargs=None):
        "FOR: mm bmm smm matmul __matmul__"
        self, other = args[:2]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self.shape[:-1] + other.shape[-1:])

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_addmm_func__(cls, func, types, args=(), kwargs=None):
        "FOR: addmm addbmm saddmm"
        _, self, other = args[:3]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self.shape[:-1] + other.shape[-1:])

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_reducing_func__(cls, func, types, args=(), kwargs=None):
        "FOR: cummin cummax cumsum cumprod sum prod min max median mean std argmin argmax"
        func_name = str(func).split(' of ')[0].split()[-1].strip("'")
        mkwargs = kwargs if kwargs is not None else {}
        if len(args) == 1 and 'dim' not in mkwargs and func_name in ('sum', 'prod', 'mean', 'std', 'cumsum', 'cumprod'):
            self = args[0]
            s = self._special
            dims = list(RANGE(s[0])) + list(RANGE(s[0]+1, s[1])) + list(RANGE(s[1]+1, self.ndim))
            args += (dims,)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            self = args[0]
            dims = mkwargs.get('dim', args[1:])
            if mkwargs.get('keepdim', False): r.special_from_(self)
            elif len(dims) > 0:
                dims = arg_tuple(dims)
                if len(dims) == 0: r.special_from_()
                elif isinstance(dims[0], torch.Tensor): r.special_from_(self)
                else: r.set_special_(special=[x - len([d for d in dims if 0 <= d < x or d + self.ndim < x]) if x not in dims else r.ndim for x in self._special], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_expanding_func__(cls, func, types, args=(), kwargs=None):
        "FOR: unsqueeze unsqueeze_ squeeze squeeze_"
        # self = args[0]
        # dims = kwget(kwargs, 'dim', args[1:])
        # dims = arg_tuple(dims, no_list=True)
        # if len(dims) == 0: dims = [i for i, s in enumerate(self.shape) if s == 1]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r): ...
            # a, b = self._special
            # ndim = self.ndim
            # for d in dims:
            #     if 0 <= d <= a or d + ndim < a: a += 1
            #     if 0 <= d <= b or d + ndim < b: b += 1
            #     ndim += 1
            # r.set_special_(special=[a, b], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_flatten_func__(cls, func, types, args=(), kwargs=None):
        "FOR: flatten"
        dims = kwget(kwargs, 'dim', args[1:])
        if isinstance(dims, tuple) and len(dims) <= 0:
            dims = kwget(kwargs, 'start_dim', 0), kwget(kwargs, 'end_dim', -1)
            self = args[0]
            space_dims = [x for x in RANGE(self.ndim) if x not in self._special]
            if len(space_dims) > 0 and space_dims[-1] - space_dims[0] == len(space_dims) - 1:
                dims = space_dims[0], space_dims[-1]
            args = (self,) + dims

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            self = args[0]
            if len(dims) == 1:
                dim = dims[0]
                if dim < 0: dim += self.ndim
                r.set_special_(special=[r.ndim if x >= dim else x for x in self._special], bf=self._batch_first)
            else:
                dim1, dim2 = dims
                if dim1 < 0: dim1 += self.ndim
                if dim2 < 0: dim2 += self.ndim
                r.set_special_(special=[r.ndim if dim1 <= x <= dim2 else 
                                        ((r.ndim if x - dim2 + dim1 > r.ndim else x - dim2 + dim1) if x > dim2 else x) 
                                        for x in self._special], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_getitem_func__(cls, func, types, args=(), kwargs=None):
        "FOR: __getitem__" # __iter__?
        if len(args) == 1: self, indices = args[0], 0
        else: self, indices = args

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

            ndim = self.ndim

            if not isinstance(indices, tuple): indices = (indices,)
            types = [type(x) for x in indices]
            if type(...) in types:
                if types.count(type(...)) > 1:
                    raise TypeError("")
                lp = indices[:types.index(type(...))]
                rp = indices[types.index(type(...))+1:]
            else:
                lp = indices
                rp = tuple()
            offset_rp = ndim - len(rp)

            lm, li = max_argmax([x.ndim if isinstance(x, torch.Tensor) else 0 for x in lp])
            rm, ri = max_argmax([x.ndim if isinstance(x, torch.Tensor) else 0 for x in rp])
            ri = [offset_rp + x for x in ri]
            if lm is not None and lm > 0: 
                if rm is not None and rm > lm: sub_dim = rm; matched_dims = ri; offset_sub = ri[0]; is_rp = True
                elif rm is not None and rm == lm: sub_dim = rm; matched_dims = li + ri; offset_sub = li[0]; is_rp = False
                else: sub_dim = lm; matched_dims = li; offset_sub = li[0]; is_rp = False
            elif rm is not None and rm > 0: sub_dim = rm; matched_dims = ri; offset_sub = ri[0]; is_rp = True
            else: offset_sub = None; matched_dims = []; is_rp = False
            if offset_sub is not None:
                ref_t = rp[ri[0] - offset_rp] if is_rp else lp[li[0]]
                sub_special = ref_t._special
                sub_bf = ref_t._batch_first
            else: offset_sub = 0; sub_dim = 0; sub_special = [sub_dim, sub_dim]; sub_bf = True
            removed_dims = matched_dims + [i for i, x in enumerate(lp) if isinstance(x, INT)] + [i + offset_rp for i, x in enumerate(rp) if isinstance(x, INT)]
    
        def apply(r):
            a, b = self._special
            rdim = r.ndim
            # offset2 = rdim - ndim + len(lp) + len(rp) - len([d for d in lp + rp if isinstance(d, slice)])
            # def notseq(x): return not isinstance(x, slice) and not (isinstance(x, bt.Tensor) and x.ndim == 1)
            # if a < len(lp) and notseq(lp[a]): a = None
            # elif offset <= a < ndim and notseq(rp[a - offset]): a = None
            # if b < len(lp) and notseq(lp[b]): b = None
            # elif offset <= b < ndim and notseq(rp[b - offset]): b = None
            # if a < ndim:
            #     a += offset2 if a > index else 0
            #     a -= len([d for d in RANGE(len(lp)) if (0 <= d < a or d + ndim < a) and not isinstance(lp[d], slice)])
            #     a -= len([d for d in RANGE(offset, ndim) if (0 <= d < a or d + ndim < a) and not isinstance(rp[d-offset], slice)])
            # elif a is not None: a = rdim
            # if b is not None and b < ndim:
            #     b += offset2 if b > index else 0
            #     b -= len([d for d in RANGE(len(lp)) if (0 <= d < b or d + ndim < b) and not isinstance(lp[d], slice)])
            #     b -= len([d for d in RANGE(offset, ndim) if (0 <= d < b or d + ndim < b) and not isinstance(rp[d-offset], slice)])
            # elif b is not None: b = rdim
            if a >= ndim or a in removed_dims and not (sub_dim == 1 and len(matched_dims) == 1): a = rdim
            else:
                to_rm = len([1 for x in removed_dims if x < a])
                if a > offset_sub: a += sub_dim
                a -= to_rm
            if b >= ndim or b in removed_dims and not (sub_dim == 1 and len(matched_dims) == 1): b = rdim
            else:
                to_rm = len([1 for x in removed_dims if x < b])
                if b > offset_sub: b += sub_dim
                b -= to_rm
            bf = self._batch_first
            if a == rdim: a = sub_special[sub_bf != bf] + offset_sub if sub_special[sub_bf != bf] != sub_dim else rdim
            if b == rdim: b = sub_special[sub_bf == bf] + offset_sub if sub_special[sub_bf == bf] != sub_dim else rdim
            if a > b: a, b = b, a; bf = not bf
            r.set_special_(special=[a, b], bf=bf)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_concatenate_func__(cls, func, types, args=(), kwargs=None):
        "FOR: cat concatenate"
        self = args[0]
        if isinstance(self, tuple): self = self[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_stack_func__(cls, func, types, args=(), kwargs=None):
        "FOR: stack"
        self = args[0]
        if isinstance(self, tuple): self = self[0]
        dim = kwget(kwargs, 'dim', args[-1] if isinstance(args[-1], INT) else 0)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(special=[x + 1 if x >= dim else x for x in self._special])

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_default_func__(cls, func, types, args=(), kwargs=None):
        "FOR: all the remainings"
        self = args[0]

        try:
            with torch._C.DisableTorchFunction():
                ret = super().__torch_function__(func, types, args, kwargs)
                if isinstance(ret, type(NotImplemented)): return ret
        except RuntimeError as e:
            if kwargs is None: kwargs = {}
            try:
                return getattr(self.as_subclass(torch.Tensor), func.__name__)(*args[1:], **kwargs)
            except: raise e

        def apply(r):
            r.special_from_(self)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls, force=False)

    def __torch_function_keys__(func):
        return func.__func__.__doc__.split('\n\n')[0].split(':')[-1].strip().split()

    __torch_function_map__ = {k: '__torch_function_ele_wise_func__' for k in __torch_function_keys__(__torch_function_ele_wise_func__)}
    __torch_function_map__.update({k: '__torch_function_resizing_func__' for k in __torch_function_keys__(__torch_function_resizing_func__)})
    __torch_function_map__.update({k: '__torch_function_full_func__' for k in __torch_function_keys__(__torch_function_full_func__)})
    __torch_function_map__.update({k: '__torch_function_resizing_as_func__' for k in __torch_function_keys__(__torch_function_resizing_as_func__)})
    __torch_function_map__.update({k: '__torch_function_randint_func__' for k in __torch_function_keys__(__torch_function_randint_func__)})
    __torch_function_map__.update({k: '__torch_function_transpose_func__' for k in __torch_function_keys__(__torch_function_transpose_func__)})
    __torch_function_map__.update({k: '__torch_function_permute_func__' for k in __torch_function_keys__(__torch_function_permute_func__)})
    __torch_function_map__.update({k: '__torch_function_matmul_func__' for k in __torch_function_keys__(__torch_function_matmul_func__)})
    __torch_function_map__.update({k: '__torch_function_addmm_func__' for k in __torch_function_keys__(__torch_function_addmm_func__)})
    __torch_function_map__.update({k: '__torch_function_reducing_func__' for k in __torch_function_keys__(__torch_function_reducing_func__)})
    __torch_function_map__.update({k: '__torch_function_expanding_func__' for k in __torch_function_keys__(__torch_function_expanding_func__)})
    __torch_function_map__.update({k: '__torch_function_flatten_func__' for k in __torch_function_keys__(__torch_function_flatten_func__)})
    __torch_function_map__.update({k: '__torch_function_getitem_func__' for k in __torch_function_keys__(__torch_function_getitem_func__)})
    __torch_function_map__.update({k: '__torch_function_concatenate_func__' for k in __torch_function_keys__(__torch_function_concatenate_func__)})
    __torch_function_map__.update({k: '__torch_function_stack_func__' for k in __torch_function_keys__(__torch_function_stack_func__)})

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        try:
            if Tensor in types and cls != Tensor: return NotImplemented

            if len(args) == 0: return super().__torch_function__(func, types, args, kwargs)

            sfunc = str(func)
            if sfunc.startswith('<attribute') or sfunc.startswith('<property'):
                return super().__torch_function__(func, types, args, kwargs)

            func_name = sfunc.split(' of ')[0].split(' at ')[0].split()[-1].strip("'").split('.')[-1]
            if func_name in ('__get__', '__set__', '__delete__'):
                return super().__torch_function__(func, types, args, kwargs)

            self = args[0]
            types = tuple(cls if t.__name__ == "Parameter" else t for t in types)
            torch_func_name = Tensor.__torch_function_map__.get(func_name, None)
            if func_name != 'to':
                args = (self,) + tuple(a.to(self.device) if a.__class__.__name__ == "Tensor" and a.device != self.device else a for i, a in enumerate(args[1:]))
            if isinstance(self, Tensor) and self.init and self.has_special: pass
            elif torch_func_name in ('__torch_function_resizing_func__', '__torch_function_full_func__', '__torch_function_resizing_as_func__', '__torch_function_randint_func__'): pass
            else:
                with torch._C.DisableTorchFunction():
                    ret = super().__torch_function__(func, types, args, kwargs)
                def apply(r): r.special_from_()
                return Tensor.__torch_function_convert_apply__(ret, apply, cls)

            if torch_func_name is None: return Tensor.__torch_function_default_func__(func, types, args, kwargs)
            else: return getattr(Tensor, torch_func_name)(func, types, args, kwargs)
            
        except Exception as e:
            print(f"In function {func}:")
            raise e#.with_traceback(None)

    for f in new_dim_methods + old_dim_methods:
        if f in locals(): locals()[f] = dim_method_wrapper(locals()[f])
        else: exec(f"{f} = dim_method_wrapper(torch.Tensor.{f})")

for func in '''
zeros ones empty
rand randn
'''.split():
    __all__.extend([func, func+'_like'])
    execblock(f"""
    def {func}(*args, **kwargs) -> Tensor:
        rg = kwargs.pop('requires_grad', False)

        size= None
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, torch.Tensor):
                return torch.{func}_like(arg, **kwargs).as_subclass(Tensor).special_from_(arg).autodevice_().requires_grad_(rg)
            elif isinstance(arg, tuple):
                if isinstance(arg, Size): size = arg
                else: args = arg
        
        if size is None: size = Size(*args)
        return torch.{func}(size, **kwargs).as_subclass(Tensor).special_from_(size).autodevice_().requires_grad_(rg)

    def {func}_like(tensor, **kwargs) -> Tensor:
        return {func}(tensor, **kwargs)
""")

for func in '''
full
'''.split():
    __all__.extend([func, func+'_like'])
    execblock(f"""
    def {func}(*args, **kwargs) -> Tensor:
        rg = kwargs.pop('requires_grad', False)
        fill = kwargs.pop('fill_value', None)
        
        size= None
        if fill is None:
            arg, fill = args
            if isinstance(arg, torch.Tensor):
                return torch.{func}_like(arg, fill, **kwargs).as_subclass(Tensor).special_from_(arg).autodevice_().requires_grad_(rg)
            elif isinstance(arg, tuple):
                if isinstance(arg, Size): size = arg
                else: args = arg
        
        if size is None: size = Size(*args)
        return torch.{func}(size, fill, **kwargs).as_subclass(Tensor).special_from_(size).autodevice_().requires_grad_(rg)

    def {func}_like(tensor, **kwargs) -> Tensor:
        return {func}(tensor, **kwargs)
""")

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

    def __call__(self, *size, **kwargs) -> Tensor:
        rg = kwargs.pop('requires_grad', False)
        return torch.randint(self.range[0], self.range[1], Size(size), **kwargs).as_subclass(Tensor).special_from_(size).autodevice_().requires_grad_(rg)

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

    def __call__(self, data, **kwargs) -> Tensor:
        rg = kwargs.pop('requires_grad', False)
        return torch.randint_like(data, self.range[0], self.range[1], **kwargs).as_subclass(Tensor).special_from_(data.shape).autodevice_().requires_grad_(rg)

randint = _Randint()
randint_like = _Randint_like()
__all__.extend(["randint", "randint_like"])

__all__.extend(["eye", "eye_as", "eye_like", "cat", "stack", "t", "unsqueeze", "squeeze", "tensor", "tr", "trace"])

def eye(*size: tuple, **kwargs) -> Tensor:
    if len(size) == 1 and isinstance(size, tuple): size = size[0]

    size = Size(size)
    if size.nspace < 1: raise TypeError("Empty size not valid for 'eye'. ")
    if size.nspace == 1: size = size + (size.space[0],)
    if size.nspace > 2: raise TypeError("No more than 2-D is allowed for 'eye'. ")
    n = MIN(*size.space)
    s = [slice(None)] * size.ndim
    for i in RANGE(size.ndim):
        if i not in size.special:
            s[i] = torch.arange(n)
    out = zeros(size, **kwargs)
    out[tuple(s)] = 1
    return out

@alias("eye_as")
def eye_like(tensor: array) -> Tensor:
    return eye(tensor.shape)

def cat(*list_of_tensors, dim=None, **kwargs) -> Tensor:
    if dim is None:
        if len(list_of_tensors) > 1 and not isinstance(list_of_tensors[-1], Tensor):
            dim = list_of_tensors[-1]
            list_of_tensors =list_of_tensors[:-1]
        else: dim = 0
    list_of_tensors = arg_tuple(list_of_tensors)
    if len(list_of_tensors) == 0: return tensor([])
    pivot = list_of_tensors[_argmax([x.ndim for x in list_of_tensors])[0]]
    list_of_tensors = [x.expand_to(pivot, dim=dim) for x in list_of_tensors if x.nele > 0]
    avouch(ALL([SUM([a != b for a, b in zip(x.shape, pivot.shape)]) <= 1 for x in list_of_tensors]), 
           "Tensors can only be concatenated when all of them have a same shape except for one dimension. " + 
           f"Current: {[x.shape for x in list_of_tensors]}")
    ibatch = ichannel = None
    if isinstance(dim, list) and len(dim) == 1: dim = dim[0]; ibatch = dim
    elif isinstance(dim, list) and len(dim) == 0:
        avouch(ALL([x.has_batch for x in list_of_tensors]), "Tensors can only be concatenated along batch dimension when all of them have one. ")
        dim = pivot.batch_dim; ibatch = dim
    elif isinstance(dim, set) and len(dim) == 1: dim = list(dim)[0]; ichannel = dim
    elif isinstance(dim, dict) and len(dim) == 0:
        avouch(ALL([x.has_channel for x in list_of_tensors]), "Tensors can only be concatenated along channel dimension when all of them have one. ")
        dim = pivot.channel_dim; ichannel = dim
    elif isinstance(dim, INT): pass
    else: raise TypeError(f"Not identified dimension for concatenation: {dim}.")
    if dim < 0: dim += list_of_tensors[0].ndim
    self = [t for t in list_of_tensors if isinstance(t, Tensor)]
    if len(list_of_tensors) == 0: return torch.tensor([]).as_subclass(Tensor)
    elif len(self) == 0: return torch.cat(list_of_tensors, dim, **kwargs).as_subclass(Tensor).special_from_().set_special_(batch_dim=ibatch, channel_dim=ichannel)
    return torch.cat(list_of_tensors, dim, **kwargs).as_subclass(Tensor).special_from_(self[0]).set_special_(batch_dim=ibatch, channel_dim=ichannel)

def stack(*list_of_tensors, dim=None, **kwargs) -> Tensor:
    if dim is None:
        if len(list_of_tensors) > 1 and not isinstance(list_of_tensors[-1], torch.Tensor):
            dim = list_of_tensors[-1]
            list_of_tensors = list_of_tensors[:-1]
        else: dim = 0
    list_of_tensors = arg_tuple(list_of_tensors)
    if len(list_of_tensors) == 0: return tensor([])
    pivot = list_of_tensors[_argmax([x.ndim for x in list_of_tensors])[0]]
    if ALL([x.shape == pivot.shape for x in list_of_tensors]): pass
    else:
        list_of_tensors = [x.expand_to(pivot) for x in list_of_tensors if x.nele > 0]
        avouch(ALL([x.shape == pivot.shape for x in list_of_tensors]), 
            "Tensors can only be stacked when all of them have an exact same shape. ")
    ibatch = ichannel = None
    if isinstance(dim, list) and len(dim) == 1: dim = dim[0]; ibatch = dim
    elif isinstance(dim, list) and len(dim) == 0:
        avouch(ALL([not x.has_batch for x in list_of_tensors]), "Tensors can only be stacked along batch dimension when none of them has one. ")
        dim = 0; ibatch = dim
    elif isinstance(dim, set) and len(dim) == 1: dim = list(dim)[0]; ichannel = dim
    elif isinstance(dim, dict) and len(dim) == 0:
        avouch(ALL([not x.has_channel for x in list_of_tensors]), "Tensors can only be stacked along channel dimension when none of them has one. ")
        dim = 1 if pivot.has_batch else 0; ichannel = dim
    elif isinstance(dim, INT): pass
    else: raise TypeError(f"Not identified dimension for concatenation: {dim}.")
    if dim < 0: dim += list_of_tensors[0].ndim + 1
    self = [t for t in list_of_tensors if isinstance(t, Tensor)]
    if len(self) == 0: return torch.stack(list_of_tensors, dim, **kwargs).as_subclass(Tensor).special_from_().set_special_(batch_dim=ibatch, channel_dim=ichannel)
    return torch.stack(list_of_tensors, dim, **kwargs).as_subclass(Tensor).set_special_(special=[x + 1 if x >= dim else x for x in self[0]._special], bf=self[0]._batch_first).set_special_(batch_dim=ibatch, channel_dim=ichannel)

def t(tensor: array) -> Tensor:
    if isinstance(tensor, Tensor): return tensor.T
    else: return torch.t(tensor).as_subclass(Tensor).special_from_(tensor)

def squeeze(tensor, *args, **kwargs) -> Tensor:
    if isinstance(tensor, Tensor): return tensor.squeeze(*args, **kwargs)
    else: return torch.squeeze(tensor, *args, **kwargs).as_subclass(Tensor).special_from_(tensor)

def unsqueeze(tensor, *args, **kwargs) -> Tensor:
    if isinstance(tensor, Tensor): return tensor.unsqueeze(*args, **kwargs)
    else: return torch.unsqueeze(tensor, *args, **kwargs).as_subclass(Tensor).special_from_(tensor)
    
def diag(x) -> Tensor:
    if not isinstance(x, Tensor): x = tensor(x)
    if x.n_space == 1:
        d = item([i for i in RANGE(x.n_dim) if i not in x._special])
        n = x.size(d)
        shape_out = x.shape[:d+1] + (n,) + x.shape[d+1:]
        out = zeros(shape_out).to(x.device).type(x.type())
        arn = arange(n).to(x.device)
        out[(slice(None),)*d + (arn, arn) + (slice(None),)*(x.n_dim-d-1)] = \
            x[(slice(None),)*d + (arn,) + (slice(None),)*(x.n_dim-d-1)]
        return out
    elif x.n_space == 2:
        p, q = [i for i in RANGE(x.n_dim) if i not in x._special]
        shape_out = x.shape[:q] + (1,) + x.shape[q+1:]
        ind = arange(x.size(p)).expand_to(shape_out, axis=p, dim=q)
        return x.gather(q, ind).squeeze(q)
    return torch.diag(x).as_subclass(Tensor).special_from_(x)

@alias("tr")
def trace(x) -> Tensor:
    return diag(x).sum(-1)

def tensor(data, *, device=None, requires_grad=False, **kwargs) -> Tensor:
    batch_dim = channel_dim = None
    for k in kwargs:
        if 'dim' in k:
            if k.startswith('b'): batch_dim = kwargs[k]
            if k.startswith('c'): channel_dim = kwargs[k]
    if device is None: device = _device
    if isinstance(device, AutoDevice): device = device.device
    if isinstance(data, (INT, FLOAT)):
        out = torch.tensor(data, device=device, requires_grad=requires_grad).as_subclass(Tensor).special_from_()
    elif isinstance(data, Tensor):
        out = data.clone()
        if device != out.device: out = out.to(device).requires_grad_(requires_grad)
    elif isinstance(data, torch.Tensor):
        out = data.as_subclass(Tensor).special_from_()
        if device != out.device: out = out.to(device).requires_grad_(requires_grad)
    else: out = Tensor(data, device=device, requires_grad=requires_grad)
    if batch_dim is not None: out.batch_dim = batch_dim
    if channel_dim is not None: out.channel_dim = channel_dim
    return out

def batch_tensor(*args, **kwargs) -> Tensor:
    if len(args) == 1 and not isinstance(args[0], (INT, FLOAT)): args = args[0]
    if not isinstance(args, Tensor): args = tensor(list(args), **kwargs)
    avouch(args.ndim == 1, "batorch.batch_tensor only takes 1D data. ")
    return to_device(args.batch_dimension_(0))

def channel_tensor(*args, **kwargs) -> Tensor:
    if len(args) == 1 and not isinstance(args[0], (INT, FLOAT)): args = args[0]
    if not isinstance(args, Tensor): args = tensor(list(args), **kwargs)
    avouch(args.ndim == 1, "batorch.channel_tensor only takes 1D data. ")
    return to_device(args.channel_dimension_(0))

generating_funcs = """
    range arange linspace
""".split()

stop_set = {"typename"}

for key in dir(torch):
    if key in stop_set: continue
    if key in __all__: continue
    if key in globals(): continue
    if not callable(eval(f"torch.{key}")):
        exec(f"{key} = torch.{key}")
        __all__.append(key)
        continue
    func = key
    __all__.append(func)
    if func in generating_funcs:
        doit = "ret.requires_grad_(rg)"
    else: doit = ''
    execblock(f"""
    def {func}(*args, **kwargs) -> Tensor:
        ref_shape = None
        if len(args) > 0 and isinstance(args[0], Tensor): ref_shape = args[0].shape
        rg = kwargs.pop('requires_grad', False)
        ret = torch.{func}(*args, **kwargs)
        def apply(ret):
            with torch._C.DisableTorchFunction():
                if isinstance(ret, Tensor) and not ret.init or not isinstance(ret, Tensor) and isinstance(ret, torch.Tensor):
                    ref = None
                    if ref_shape is not None and len(ret.shape) == len(ref_shape): ref = ref_shape
                    ret = ret.as_subclass(Tensor).autodevice_()
                    ret = ret.as_subclass(Tensor).special_from_(ref)
                    {doit}
                    return ret

                if not isinstance(ret, torch.Tensor) and isinstance(ret, (tuple, list, iterable)):
                    # Also handles things like namedtuples
                    return type(ret)(apply(r) for r in ret)

                return ret
        return apply(ret)
    """)

inv = eval('inverse')
