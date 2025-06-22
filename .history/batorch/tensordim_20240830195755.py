
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The dimensions for tensors in batch.",
    requires = "torch"
)

__all__ = """
    new_dim        exist_dim
    del_dim        iter_dim       linalg_dim

    Size           FakeSize
    func_dim_size  func_dim
""".split()

# import builtins, re, sys, math
from abc import ABCMeta
# from collections import defaultdict
# from functools import wraps
# from typing import Generator
# from .device import GB, AutoDevice, SleepingDevice

with __info__:
    from pyoverload import null
    from pycamia import avouch, touch, alias
#     import torch
#     import batorch as bt
#     from pyoverload import null, to_torch_dtype, dtype as dtype_, method
#     from pycamia import ByteSize, Version
#     from pycamia import avouch, touch, alias, void
#     from pycamia import execblock, get_num_indent, add_lineno
#     from pycamia import tokenize, token_replace, identity_function
#     from pycamia import get_alphas, arg_extract, max_argmax
#     from pycamia import argmax as argmax_, prod as prod_, item, to_list
#     from pycamia import get_reference_line

num = (int, float)

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
        if isinstance(item, tuple): return all(self.__instancecheck__(x) for x in item)
        if isinstance(item, dict) and len(item) == 0: return True
        if isinstance(item, set) and len(item) == 1 and isinstance(item[0], int): return True
        if isinstance(item, list) and all(isinstance(x, int) for x in item): return True
        if isinstance(item, str) and len(item) == 0: return True
        if isinstance(item, str):
            try:
                item = eval(item)
                if isinstance(item, int): return True
                item = tuple(item)
            except: return False
            if all(isinstance(x, int) for x in item): return True
        if isinstance(item, int): return True
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
                avouch(isinstance(i_func, int), TypeError(f"Invalid new dimension {(i_func,)}: must be integer in format (...,). "))
                avouch(-n_dim-1 <= i_func <= n_dim, IndexError(f"Invalid new dimension {(i_func,)}: dimension out of range, should be in [0, {n_dim}]. "))
                avouch(i_func in (0, -total_n_dim, total_n_dim-1, -1), TypeError("New functional dimension should be the first/last dimension. "))
                int_args.append(i_func)
                sz_func_dim = 1 if i_func == 0 else -1
            elif isinstance(arg, dict) and len(arg) == 0:
                # unspecified batch dimension
                avouch(sz_batch_dim == 0, TypeError("Cannot add new batch dimension for tensor with batch. "))
                int_args.append(max(sz_func_dim, 0))
                sz_batch_dim = 1
            elif isinstance(arg, set) and len(arg) == 1:
                # specified batch dimension
                avouch(sz_batch_dim == 0, TypeError("Cannot add new batch dimension for tensor with batch. "))
                i_batch = arg.pop()
                avouch(isinstance(i_batch, int), TypeError(f"Invalid new dimension {set([i_batch])}: must be integer in {{}}. "))
                avouch(-n_dim-1 <= i_batch <= n_dim, IndexError(f"Invalid new dimension {set([i_batch])}: dimension out of range, should be in [0, {n_dim}]. "))
                if i_batch < 0: i_batch += total_n_dim
                # new batch dimension
                for pm_n in [1, -1]:
                    if i_batch in (max(pm_n * sz_func_dim, 0), n_dim + min(pm_n * sz_func_dim, 0)):
                        sz_func_dim = pm_n * sz_func_dim
                        break
                else: raise TypeError("New batch dimension should be the first/last dimension apart from the functional dimension. ")
                # avouch(i_batch in (max(sz_func_dim, 0), max(sz_func_dim, 0)-total_n_dim, total_n_dim-1+min(sz_func_dim, 0), min(sz_func_dim, 0)-1), TypeError("New batch dimension should be the first/last dimension apart from the functional dimension. "))
                int_args.append(i_batch)
                sz_batch_dim = 1 if i_batch == max(sz_func_dim, 0) else -1
            elif isinstance(arg, list) and len(arg) == 0:
                # unspecified feature dimension
                avouch(sz_feature_dim == 0, TypeError("Cannot add new feature dimension with size 1 for tensor already with feature: multiple choice in placing new dimension (use [*] to identify). "))
                int_args.append(max(sz_func_dim, 0) + max(sz_batch_dim, 0))
                sz_feature_dim = 1
            elif isinstance(arg, list) and len(arg) == 1:
                # specified feature dimension
                i_new_feature = arg[0]
                avouch(isinstance(i_new_feature, int), TypeError(f"Invalid new dimension [{i_new_feature}]: must be integer in []. "))
                avouch(-n_dim-1 <= i_new_feature <= n_dim, IndexError(f"Invalid new dimension [{i_new_feature}]: dimension out of range, should be in [0, {n_dim}]. "))
                if i_new_feature < 0: i_new_feature += total_n_dim
                if sz_feature_dim != 0:
                    start = max(sz_func_dim, 0) + max(sz_batch_dim, 0) if sz_feature_dim > 0 else n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + sz_feature_dim
                    stop = max(sz_func_dim, 0) + max(sz_batch_dim, 0) + sz_feature_dim if sz_feature_dim > 0 else n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0)
                    avouch(start <= i_new_feature <= stop, TypeError("New feature dimension should not be apart from existing feature dimensions. "))
                else:
                    # new feature dimension
                    pms = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    for pm_n, pm_b in pms:
                        if pm_n < 0 and (sz_batch_dim != 0): continue
                        start = max(pm_n * sz_func_dim, 0) + max(pm_b * sz_batch_dim, 0)
                        stop = n_dim + min(pm_n * sz_func_dim, 0) + min(pm_b * sz_batch_dim, 0)
                        if i_new_feature in (start, stop):
                            sz_func_dim = pm_n * sz_func_dim
                            sz_batch_dim = pm_b * sz_batch_dim
                            break
                    else: raise TypeError("New feature dimension should be the first/last dimension apart from the batch dimension. ")
                    # avouch(i_new_feature in (max(sz_func_dim, 0) + max(sz_batch_dim, 0), n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0)), 
                    #        TypeError("New feature dimension should be the first/last dimension apart from the batch dimension. "))
                int_args.append(i_new_feature)
                if sz_feature_dim > 0: sz_feature_dim += 1
                elif sz_feature_dim < 0: sz_feature_dim -= 1
                else: sz_feature_dim = 1 if i_new_feature == max(sz_func_dim, 0) + max(sz_batch_dim, 0) else -1
            elif isinstance(arg, str) and len(arg) == 0:
                # unspecified sequence dimension
                avouch(sz_sequence_dim == 0, TypeError("Cannot add new sequence dimension with size 1 for tensor already with sequence: multiple choice in placing new dimension (use '*' to identify). "))
                int_args.append(n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0))
                sz_sequence_dim = -1
            elif isinstance(arg, str) and ',' not in arg:
                # specified sequence dimension
                i_new_sequence = touch(lambda: eval(arg), default=None)
                i_new_sequence = touch(lambda: i_new_sequence[0], default=i_new_sequence)
                avouch(i_new_sequence is not None, TypeError(f"Invalid new dimension '{arg}': must be integer in ''. "))
                avouch(isinstance(i_new_sequence, int), TypeError(f"Invalid new dimension '{i_new_sequence}': must be integer in ''. "))
                avouch(-n_dim-1 <= i_new_sequence <= n_dim, IndexError(f"Invalid new dimension [{i_new_sequence}]: dimension out of range, should be in [0, {n_dim}]. "))
                if i_new_sequence < 0: i_new_sequence += total_n_dim
                if sz_sequence_dim != 0:
                    start = max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0) if sz_sequence_dim > 0 else n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0) + sz_sequence_dim
                    stop = max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0) + sz_sequence_dim if sz_sequence_dim > 0 else n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0)
                    avouch(start <= i_new_sequence <= stop, TypeError("New sequence dimension should not be apart from existing sequence dimensions. "))
                else:
                    # new sequential dimension
                    pms = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                           (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
                    for pm_n, pm_b, pm_f in pms:
                        if pm_n < 0 and (sz_batch_dim != 0 or sz_feature_dim != 0): continue
                        if pm_b < 0 and (sz_feature_dim != 0): continue
                        start = max(pm_n * sz_func_dim, 0) + max(pm_b * sz_batch_dim, 0) + max(pm_f * sz_feature_dim, 0)
                        stop = n_dim + min(pm_n * sz_func_dim, 0) + min(pm_b * sz_batch_dim, 0) + min(pm_f * sz_feature_dim, 0)
                        if i_new_sequence in (start, stop):
                            sz_func_dim = pm_n * sz_func_dim
                            sz_batch_dim = pm_b * sz_batch_dim
                            sz_feature_dim = pm_f * sz_feature_dim
                            break
                    else: raise TypeError('New sequence dimension should be the first/last dimension apart from the batch and feature dimensions. ')
                    # avouch(i_new_sequence in (max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0), n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0)), 
                    #        TypeError('New sequence dimension should be the first/last dimension apart from the batch and feature dimensions. '))
                int_args.append(i_new_sequence)
                if sz_sequence_dim > 0: sz_sequence_dim += 1
                elif sz_sequence_dim < 0: sz_sequence_dim -= 1
                else: sz_sequence_dim = 1 if i_new_sequence == (max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0)) else -1
            elif isinstance(arg, int):
                # specified spatial dimension
                avouch(isinstance(arg, int), TypeError(f"Invalid new dimension {arg}: must be an integer. "))
                avouch(-n_dim-1 <= arg <= n_dim, IndexError(f"Invalid new dimension [{arg}]: dimension out of range, should be in [0, {n_dim}]. "))
                if arg < 0: arg += total_n_dim
                start = max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0) + max(sz_sequence_dim, 0)
                stop = n_dim + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0) + min(sz_sequence_dim, 0)
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
                        start = max(pm_n * sz_func_dim, 0) + max(pm_b * sz_batch_dim, 0) + max(pm_f * sz_feature_dim, 0) + max(pm_s * sz_sequence_dim, 0)
                        stop = n_dim + min(pm_n * sz_func_dim, 0) + min(pm_b * sz_batch_dim, 0) + min(pm_f * sz_feature_dim, 0) + min(pm_s * sz_sequence_dim, 0)
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
        if isinstance(item, tuple): return all(self.__instancecheck__(x) for x in item)
        if isinstance(item, dict) and len(item) == 0: return True
        if isinstance(item, set) and len(item) == 1 and isinstance(item[0], int): return True
        if isinstance(item, list) and all(isinstance(x, int) for x in item): return True
        if isinstance(item, str) and len(item) == 0: return True
        if isinstance(item, str):
            try: item = eval(item)
            except: return False
            try: item = tuple(item)
            except: item = (item,)
            if all(isinstance(x, int) for x in item): return True
        if isinstance(item, int): return True
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
            if isinstance(arg, int): combined_args.append(arg)
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
            elif isinstance(arg, str) and isinstance(touch(lambda: eval(arg)), (int, tuple)) and isinstance(combined_args[-1], str):
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
                int_args.extend(range(*self.feature_range))
            elif isinstance(arg, list):
                avouch(self.has_feature, TypeError(f"Cannot find feature dimensions in {self_repr}. "))
                avouch(all(-self.n_feature_dim <= a < self.n_feature_dim for a in arg), IndexError(f"Cannot identify feature dimensions {arg}: index out of range (the indices are only counted for feature dimensions). "))
                arg = [a + self.n_feature_dim if a < 0 else a for a in arg]
                int_args.extend(a + self.feature_start for a in arg)
            elif isinstance(arg, str) and len(arg) == 0:
                avouch(self.has_sequence, TypeError(f"Cannot find sequence dimensions in {self_repr}. "))
                int_args.extend(range(*self.sequence_range))
            elif isinstance(arg, str) and arg in self.names: int_args.append(self.names.index(arg))
            elif isinstance(arg, str):
                avouch(self.has_sequence, TypeError(f"Cannot find sequence dimensions in {self_repr}. "))
                try: arg = eval(arg)
                except: raise TypeError(f"Invalid representation for sequence dimensions: '{arg}'. ")
                try: arg = tuple(arg)
                except: arg = (arg,)
                avouch(all(-self.n_sequence_dim <= a < self.n_sequence_dim for a in arg), IndexError(f"Cannot identify sequence dimensions {arg}: index out of range (the indices are only counted for sequence dimensions). "))
                arg = [a + self.n_sequence_dim if a < 0 else a for a in arg]
                int_args.extend(a + self.sequence_start for a in arg)
            elif arg is ...:
                avouch(self.has_space, TypeError(f"Cannot find space dimensions in {self_repr}. "))
                int_args.extend(range(*self.space_range))
            elif arg is None:
                avouch(len(int_args) == 0, TypeError(f"Cannot use 'None' along with other dimensions in {self_repr}. "))
                int_args = list(range(self.n_dim))
                break
            elif isinstance(arg, int):
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
        avouch(cls in (new_dim, exist_dim, del_dim), TypeError(f"Invalid iter_dim for non-dimension class {cls}, should be one of [new_dim, exist_dim, del_dim]. "))
        avouch(isinstance(arg, int) or arg in [..., slice(None)], TypeError(f"Invalid subscript for '{cls.__name__}': {arg}, should be int, ... or : ."))
        this.cls = cls
        this.arg = arg
        if arg == ...: arg_str = '...'
        elif arg == slice(None): arg_str = ':'
        else: arg_str = str(arg)
        this.__name__ = f"{cls.__name__}[{arg_str}]"

    def __call__(this, self, *args):
        dims = this.cls(self, *args)
        if isinstance(this.arg, int):
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
        avouch(len(arg) == 2 and (arg[0] is None or isinstance(arg[0], int)) and (arg[1] is None or isinstance(arg[1], int)), 
               TypeError("'linalg_dim' takes only subscripts of (int, int), indicating the min/max number of dimensions. "))
        ret = lambda *a: linalg_dim(*a, min_n_dim=arg[0], max_n_dim=arg[1])
        ret.__name__ = f"linalg_dim[{arg[0]}, {arg[1]}]"
        return ret

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
        avouch(isinstance(shape, tuple) and all(isinstance(s, num) for s in shape), TypeError(f"Invalid 'shape = {shape}' for bt.Size, should be a tuple. "))
        avouch(sz_func_dim in (0, 1, -1), TypeError(f"Invalid 'sz_func_dim = {sz_func_dim}' for bt.Size, should be 0, 1, or -1. "))
        avouch(sz_batch_dim in (0, 1, -1), TypeError(f"Invalid 'sz_batch_dim = {sz_batch_dim}' for bt.Size, should be 0, 1, or -1. "))
        avouch(isinstance(sz_feature_dim, int), TypeError(f"Invalid 'sz_feature_dim = {sz_feature_dim}' for bt.Size, should be an integer. "))
        avouch(isinstance(sz_sequence_dim, int), TypeError(f"Invalid 'sz_sequence_dim = {sz_sequence_dim}' for bt.Size, should be an integer. "))
        avouch(len(shape) >= abs(sz_func_dim) + abs(sz_batch_dim) + abs(sz_feature_dim) + abs(sz_sequence_dim), TypeError(f"Too many special dimensions for shape of length {len(shape)}: sz_func_dim = {sz_func_dim}, sz_batch_dim = {sz_batch_dim}, sz_feature_dim = {sz_feature_dim}, sz_sequence_dim = {sz_sequence_dim}. "))
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
        if len(shape) > 0 and not all(isinstance(x, num) for x in shape):
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
                           max(sz_func_dim, 0), 
                           max(sz_func_dim, 0) - n_dim, 
                           n_dim - 1 + min(sz_func_dim, 0), 
                           min(sz_func_dim, 0) - 1)
            avouch(batch_dim in batch_cands, TypeError(f"Invalid 'batch_dim = {batch_dim}' for bt.Size, should be None or {batch_cands[1]} (before the space-sequence dimensions), or {batch_cands[-1]} (after the space-sequence dimensions). "))
            sz_batch_dim = 0 if batch_dim is None else (1 if batch_dim in (0, -n_dim) else -1)
        # Deal with the feature dimension(s)
        if sz_feature_dim == 0:
            if n_feature_dim is None:
                channel_cands = (None, 
                                 max(sz_func_dim, 0) + max(sz_batch_dim, 0), 
                                 max(sz_func_dim, 0) + max(sz_batch_dim, 0) - n_dim, 
                                 n_dim - 1 + min(sz_func_dim, 0) + min(sz_batch_dim, 0), 
                                 min(sz_func_dim, 0) + min(sz_batch_dim, 0) - 1)
                avouch(channel_dim in channel_cands, TypeError(f"Invalid 'channel_dim = {channel_dim}' for bt.Size, should be None or {channel_cands[1]} (before the space-sequence dimensions), or {channel_cands[-1]} (after the space-sequence dimensions). "))
                sz_feature_dim = 0 if channel_dim is None else (1 if channel_dim in channel_cands[1:3] else -1)
            elif channel_dim is not None: raise TypeError("Argument 'channel_dim' is conflict to 'n_feature_dim' for bt.Size, they cannot be assigned simultaneously. ")
            else:
                avouch(isinstance(n_feature_dim, int) and n_feature_dim >= 0, TypeError(f"Invalid 'n_feature_dim = {n_feature_dim}' for bt.Size, should be an integer. "))
                sz_feature_dim = n_feature_dim
        # Deal with the sequence dimension(s)
        if sz_sequence_dim == 0:
            if n_sequence_dim is None:
                sequence_cands = (None, 
                                  max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0), 
                                  max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0) - n_dim, 
                                  n_dim - 1 + min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0), 
                                  min(sz_func_dim, 0) + min(sz_batch_dim, 0) + min(sz_feature_dim, 0) - 1)
                avouch(sequence_dim in sequence_cands, TypeError(f"Invalid 'sequence_dim = {sequence_dim}' for bt.Size, should be one of None, {sequence_cands[1]} (before the space dimensions), or {sequence_cands[-1]} (after the space dimensions)"))
                sz_sequence_dim = 0 if sequence_dim is None else (1 if sequence_dim in sequence_cands[1:3] else -1)
            elif sequence_dim is not None: raise TypeError("Argument 'sequence_dim' is conflict to 'n_sequence_dim' for bt.Size, they cannot be assigned simultaneously. ")
            else:
                avouch(isinstance(n_sequence_dim, int) and n_sequence_dim >= 0, TypeError(f"Invalid 'n_sequence_dim = {n_sequence_dim}' for bt.Size, should be an integer. "))
                sz_sequence_dim = - n_sequence_dim
        # Check dimension consistency
        avouch(abs(sz_func_dim) + abs(sz_batch_dim) + abs(sz_feature_dim) + abs(sz_sequence_dim) <= n_dim, TypeError(f"Too many special dimensions for shape of length {n_dim}. "))
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
                avouch(len(s) == 1 and isinstance(s[0], num) or len(s) == 0, TypeError(f"Invalid 'shape = {shape}' for bt.Size (only (x,) is available for functional dimension)."))
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
                avouch(i in (max(sz_func_dim, 0), len(shape) - 1 + min(sz_func_dim, 0)), TypeError(f"Invalid 'shape = {shape}' for bt.Size (batch dimension can only be the first/last dimension apart from the functional dimension)."))
                if len(s) == 0: raw_size.append(-1)
                else: x = s.pop(); raw_size.append(x); shape[i].add(x)
                if i == max(sz_func_dim, 0):
                    sz_batch_dim = 1
                else:
                    sz_batch_dim = -1
                    ends_with_batch = True
                cum_i += 1
            elif isinstance(s, list): # feature dimensions
                avouch(sz_feature_dim <= 0 or isinstance(shape[i-1], (tuple, dict, set, list)), 
                       TypeError(f"Invalid 'shape = {shape}' for bt.Size (feature dimensions should be neighboring dimensions)."))
                avouch(all([isinstance(y, num) for y in s]), TypeError(f"Invalid 'shape = {shape}' for bt.Size (representation for feature dimensions should be a list of integers)."))
                avouch(not ends_with_batch and not ends_with_func, TypeError(f"Invalid 'shape = {shape}' for bt.Size (batch dimension can only be the first/last dimension apart from the functional dimension)."))
                if len(s) == 0: raw_size.append(-1); len_feat = 1
                else: raw_size.extend(s); len_feat = len(s)
                if sz_feature_dim == 0:
                    if cum_i == max(sz_func_dim, 0) + max(sz_batch_dim, 0): sz_feature_dim = len_feat
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
                avouch(all([isinstance(y, num) for y in s_val]), TypeError(f"Invalid 'shape = {shape}' for bt.Size (representation for sequence dimensions should be a list of integers)."))
                raw_size.extend(s_val); len_sqs = len(s_val)
                if sz_sequence_dim == 0:
                    ends_with_sequence = cum_i > max(sz_func_dim, 0) + max(sz_batch_dim, 0) + max(sz_feature_dim, 0)
                    sz_sequence_dim = -len_sqs
                elif sz_sequence_dim > 0: sz_sequence_dim += len_sqs
                else: sz_sequence_dim -= len_sqs
                cum_i += len_sqs
            elif isinstance(s, num):
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
            if touch(lambda: int(args[0])) is not None:
                kwargs['sz_sequence_dim'] = 1
                return cls.__new_tuple__((int(args[0]),), **kwargs)
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
    def n_func_dim(self): return abs(self.sz_func_dim)
    
    @alias("is_funcdim")
    def is_func_dim(self, i):
        avouch(isinstance(i, int), TypeError(f"Invalid call 'is_func_dim({i})': the argument is not an integer. "))
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
        avouch(ifunc is None or isinstance(ifunc, bool) or ifunc in (0, -self.n_dim, self.n_dim-1, -1), TypeError("'bt.Size.with_func_dim' only takes input bool or integer 0, -1."))
        if ifunc or isinstance(ifunc, int):
            avouch(self.n_batch_dim + self.n_feature_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set func_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
            self.sz_func_dim = 1 if ifunc in (0, -self.n_dim, True) else -1
        else: self.sz_func_dim = 0
        return self

    @alias("sz_func_dim_", "with_szfuncdim")
    def with_sz_func_dim(self, nfcdim):
        avouch(nfcdim is None or isinstance(nfcdim, int), TypeError("'bt.Size.with_sz_func_dim' only takes input of an integer."))
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
        avouch(isinstance(n_func, int), TypeError("Func size should be an integer. "))
        if self.sz_func_dim >= 0:
            return func_dim_size(n_func) + self[self.n_func_dim:]
        else: return self[:-self.n_func_dim] + func_dim_size(n_func)

    @property
    def size_start(self): return max(self.sz_func_dim, 0)
    @property
    def size_stop(self): return self.n_dim + min(self.sz_func_dim, 0)

    ## batch dimension:
    @property
    def has_batch(self): return self.sz_batch_dim != 0

    @alias("nbatchdim")
    @property
    def n_batch_dim(self): return abs(self.sz_batch_dim)
    
    @alias("is_batchdim")
    def is_batch_dim(self, i):
        avouch(isinstance(i, int), TypeError(f"Invalid call 'is_batch_dim({i})': the argument is not an integer. "))
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
        avouch(ibatch is None or isinstance(ibatch, bool) or ibatch in (self.size_start, self.size_start-self.n_dim, self.size_stop-1, self.size_stop-1-self.n_dim), TypeError(f"'bt.Size.with_batch_dim' only takes input bool or integers {self.size_start}, {self.size_stop-1}."))
        if ibatch or isinstance(ibatch, int):
            avouch(self.n_func_dim + self.n_feature_dim + self.n_sequence_dim < self.n_dim, TypeError(f"Cannot set batch_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
            self.sz_batch_dim = 1 if ibatch in (self.size_start, self.size_start-self.n_dim, True) else -1
        else: self.sz_batch_dim = 0
        return self

    @alias("sz_batch_dim_", "with_szbatchdim")
    def with_sz_batch_dim(self, nbdim):
        avouch(nbdim is None or isinstance(nbdim, int), TypeError("'bt.Size.with_sz_batch_dim' only takes input of an integer."))
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
        avouch(isinstance(n_batch, int), TypeError("Batch size should be an integer. "))
        if self.sz_batch_dim > 0:
            return self[:self.size_start] + Size({n_batch}) + self[self.size_start + self.n_batch_dim:]
        else: return self[:self.size_stop-self.n_batch_dim] + Size({n_batch}) + self[self.size_stop:]

    @property
    def non_bat_start(self): return max(self.sz_func_dim, 0) + max(self.sz_batch_dim, 0)
    @property
    def non_bat_stop(self): return self.n_dim + min(self.sz_func_dim, 0) + min(self.sz_batch_dim, 0)

    ## channel dimension: 
    @property
    def has_channel(self): return self.sz_feature_dim in (1, -1)

    @alias("nchanneldim")
    @property
    def n_channel_dim(self): return int(self.has_channel)

    @alias("is_channeldim")
    def is_channel_dim(self, i):
        avouch(isinstance(i, int), TypeError(f"Invalid call 'is_channel_dim({i})': the argument is not an integer. "))
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
        avouch(isinstance(n_channel, int), TypeError("Channel size should be an integer. "))
        if self.sz_feature_dim > 0:
            return self[:self.non_bat_start] + Size([n_channel]) + self[self.non_bat_start + self.n_feature_dim:]
        else: return self[:self.non_bat_stop - self.n_feature_dim] + Size([n_channel]) + self[self.non_bat_stop:]

    ## feature dimensions:
    @property
    def has_feature(self): return self.n_feature_dim != 0

    @alias("is_featuredim")
    def is_feature_dim(self, i):
        avouch(isinstance(i, int), TypeError(f"Invalid call 'is_feature_dim({i})': the argument is not an integer. "))
        if not self.has_feature: return False
        if i < 0: i += self.n_dim
        return self.feature_start <= i < self.feature_stop

    @property
    def n_feature_dim(self): return abs(self.sz_feature_dim)

    @alias("nfeaturedim")
    @n_feature_dim.setter
    def n_feature_dim(self, n): return self.with_n_feature_dim(n)

    @alias("sz_feature_dim_", "with_szfeaturedim")
    def with_sz_feature_dim(self, nfdim):
        avouch(nfdim is None or isinstance(nfdim, int), TypeError("'bt.Size.with_sz_feature_dim' only takes input of an integer."))
        if nfdim is None: self.sz_feature_dim = 0
        else:
            avouch(self.n_func_dim + self.n_batch_dim + abs(nfdim) + self.n_sequence_dim <= self.n_dim, TypeError(f"Cannot assign {abs(nfdim)} features in size {self} with non-special dimension of {self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_sequence_dim}, or there will be conflict. "))
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
        avouch(dim is None or isinstance(dim, int), TypeError(f"Feature start should be an integer. "))
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
        avouch(dim is None or isinstance(dim, int), TypeError(f"Feature stop should be an integer. "))
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
        for i in range(*self.feature_range): p *= self[i]
        return p

    @alias("feature_size")
    @property
    def feature(self):
        return self[self.feature_start: self.feature_stop]

    def with_feature(self, *size):
        if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
        avouch(all(isinstance(x, int) for x in size), TypeError("feature size should be a tuple of integers. "))
        if not self.has_feature: start = self.non_bat_start; stop = self.non_bat_start
        else: start = self.feature_start; stop = self.feature_stop
        # avouch(len(size) == self.n_feature_dim, f"Cannot substitute feature in {self} by {size} as their dimensions are not the same.")
        return self[:start] + Size(size, sz_feature_dim=len(size)) + self[stop:]
        
    @property
    def seq_spc_start(self): return max(self.sz_func_dim, 0) + max(self.sz_batch_dim, 0) + max(self.sz_feature_dim, 0)
    @property
    def seq_spc_stop(self): return self.n_dim + min(self.sz_func_dim, 0) + min(self.sz_batch_dim, 0) + min(self.sz_feature_dim, 0)

    ## sequence dimensions:
    @alias("has_time", "has_series")
    @property
    def has_sequence(self): return self.sz_sequence_dim != 0

    @alias("is_timedim", "is_seriesdim", "is_sequencedim")
    @alias("is_time_dim", "is_series_dim")
    def is_sequence_dim(self, i):
        avouch(isinstance(i, int), TypeError(f"Invalid call 'is_sequence_dim({i})': the argument is not an integer. "))
        if not self.has_sequence: return False
        if i < 0: i += self.n_dim
        return self.sequence_start <= i < self.sequence_stop

    @property
    def n_sequence_dim(self): return abs(self.sz_sequence_dim)

    @alias("ntimedim", "nseriesdim", "nsequencedim")
    @alias("n_time_dim", "n_series_dim")
    @n_sequence_dim.setter
    def n_sequence_dim(self, n): return self.with_n_sequence_dim(n)

    @alias("sz_time_dim_", "sz_series_dim_", "sz_sequence_dim_")
    @alias("with_sztimedim", "with_szseriesdim", "with_szsequencedim")
    @alias("with_sz_time_dim", "with_sz_series_dim")
    def with_sz_sequence_dim(self, nsdim):
        avouch(nsdim is None or isinstance(nsdim, int), TypeError("'bt.Size.with_sz_sequence_dim' only takes input of an integer."))
        if nsdim is None: self.sz_sequence_dim = 0
        else:
            avouch(self.n_func_dim + self.n_batch_dim + self.n_feature_dim + abs(nsdim) <= self.n_dim, TypeError(f"Cannot assign {abs(nsdim)} sequence dimensions in size {self} with non-special dimension of {self.n_dim - self.n_func_dim - self.n_batch_dim - self.n_feature_dim}, or there will be conflict. "))
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
        avouch(dim is None or isinstance(dim, (bool, int)), TypeError("'bt.Size.with_sequence_dim' only takes integer or bool."))
        if isinstance(dim, bool): dim = -1 if dim else None
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
        avouch(dim is None or isinstance(dim, int), TypeError(f"Sequence start should be an integer. "))
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
        avouch(dim is None or isinstance(dim, int), TypeError(f"Sequence stop should be an integer. "))
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
        for i in range(*self.sequence_range): p *= self[i]
        return p

    @alias("time_size", "series_size", "sequence_size")
    @alias("time", "series")
    @property
    def sequence(self):
        return self[self.sequence_start:self.sequence_stop]

    @alias("with_time", "with_series")
    def with_sequence(self, *size):
        if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
        avouch(all(isinstance(x, int) for x in size), TypeError("sequence size should be a tuple of integers. "))
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
        return max(self.sz_func_dim, 0) + max(self.sz_batch_dim, 0) + max(self.sz_feature_dim, 0) + max(self.sz_sequence_dim, 0)
    
    @property
    def space_stop(self):
        return self.n_dim + min(self.sz_func_dim, 0) + min(self.sz_batch_dim, 0) + min(self.sz_feature_dim, 0) + min(self.sz_sequence_dim, 0)

    @property
    def space_range(self):
        return (self.space_start, self.space_stop)

    @alias("nspace")
    @property
    def n_space(self):
        avouch(self.has_space, TypeError(f"Cannot get space dimensions from size {self}."))
        p = 1
        for i in range(*self.space_range): p *= self[i]
        return p

    @alias("space_size")
    @property
    def space(self):
        return self[self.space_start:self.space_stop]

    def with_space(self, *size):
        if len(size) == 1 and isinstance(size[0], tuple): size = size[0]
        avouch(all(isinstance(x, int) for x in size), TypeError("space size should be a tuple of integers. "))
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
                     ([] if self.sz_feature_dim == 0 else list(range(self.feature_start, self.feature_stop))) + 
                     ([] if self.sz_sequence_dim == 0 else list(range(self.sequence_start, self.sequence_stop))))
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
        for i in range(self.n_dim):
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
    
    def transpose(self, i: int, j:int):
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
        avouch(sorted(dims) == list(range(len(self))), TypeError("'permute' needs input dimensions forming a permutation of 0-(n-1). "))
        return sum([self[d:d+1] for d in dims], Size())    

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
        if isinstance(k, int): return super().__getitem__(k)
        avouch(isinstance(k, slice), TypeError(f"Slicing of 'bt.Size' only takes integers or slices, not {k} of type {type(k)}. "))
        s, e = k.start, k.stop
        if s is None: s = 0
        if e is None: e = self.n_dim
        if s < 0: s += self.n_dim
        if e < 0: e += self.n_dim
        if self.has_func:
            sz_func_dim = self.sz_func_dim if s <= self.func_dim and e > self.func_dim else (max(min(e, self.func_dim + 1) - s, 0) if s > self.func_dim else min(max(s, self.func_dim) - e, 0))
        else: sz_func_dim = 0
        if self.has_batch:
            sz_batch_dim = self.sz_batch_dim if s <= self.batch_dim and e > self.batch_dim else (max(min(e, self.batch_dim + 1) - s, 0) if s > self.batch_dim else min(max(s, self.batch_dim) - e, 0))
            # sz_batch_dim = self.non_bat_start - s if s < self.non_bat_start else (self.non_bat_stop - e if self.non_bat_stop < e else 0)
        else: sz_batch_dim = 0
        if self.has_feature:
            sz_feature_dim = self.sz_feature_dim if s <= self.feature_start and e >= self.feature_stop else (max(min(e, self.feature_stop) - s, 0) if s > self.feature_start else min(max(s, self.feature_start) - e, 0))
        else: sz_feature_dim = 0
        if self.has_sequence:
            sz_sequence_dim = self.sz_sequence_dim if s <= self.sequence_start and e >= self.sequence_stop else (max(min(e, self.sequence_stop) - s, 0) if s > self.sequence_start else min(max(s, self.sequence_start) - e, 0))
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
        avouch(isinstance(other, int), TypeError("Production for 'bt.Size' is inherited from python object 'tuple' to perform duplication, please use `size **(//) 2` to perform element-wise multiplication (division) to enlarge (shrink) the size. "))
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
        avouch(isinstance(other, (num, tuple)), TypeError(f"Element-wise operations are only used for numbers or tuples, not {type(other)}."))
        op = lambda x, y: (max(int(operation(x, y)), 0) if x >= 0 else -1) if identity == 0 or y >= 0 else -1
        if isinstance(other, num): return self.with_space(tuple(op(x, other) for x in self.space))
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
        avouch(isinstance(other_func, num), TypeError(f"Invalid operation between {self} and {other}: conflict in functional dimension. "))
        avouch(isinstance(other_batch, num), TypeError(f"Invalid operation between {self} and {other}: conflict in batch dimension. "))
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
        if len(args) == 1 and isinstance(args[0], int): return super().__getitem__(*args)
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
