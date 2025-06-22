
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The dimensions manipulations for tensors in batch.",
    create = "2025-01",
    requires = "torch"
)

__all__ = """
    FuncDim        BatchDim       SequeDim       FeatDim
    SpaceDim       SpecDim        AllDim
    SpecialDimensions             MajorSpecialDimensions        empty_special_dims
    new_dim        exist_dim
    del_dim        iter_dim       linalg_dim

    Size           FakeSize
""".split()

from abc import ABCMeta
from typing import Generator

with __info__:
    import torch
    from pyoverload import null
    from pycamia import avouch, touch, alias, Warn
    from pycamia import prod

size_eletype = (int, float)

"""
Dimensions: There are five types of dimensions for batorch tensors. 
1. functional dimension. It is a systemetic dimension nested as the covering layer which is commonly the 0-th dimension. 
    It can be used as the dimension representing coordinates, or as the temporal dimension for sequential problems with fixed timeline. 
2. batch dimension. It is the dimension for paralleled computing, which is designed for mini-batch methods. 
4. sequence dimension. It is the dimension(s) representing sequences, both spatial and temporal. There can be multiple sequence dimensions. 
3. feature dimension. It is the dimension(s) representing features, which is also the channel dimension in image processing. There can be multiple feature dimensions. 

Dimension Indicators: there are two types of dimension indicators for batorch functions. 
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

########################################
##  Define Dimensions Here            ##
########################################

class Dimension:
    """ Properties (* represents the property that must exist):
    *name(str)          The name of dimension. 
    *abbr(str)          The abbreviation. 
    *unique(bool)       Whether the dimension should be unique. 
    last(bool)          Whether the dimension is the last dimensions or not by default. Defaults to False. 
                        If it is True, the variable > 0 means the last dimensions. 
    *get_value(method)  A method that takes an expression and returns the values if it represents the dimension (in tuple). It should return None otherwise. 
    """
    # Default values
    last = False
    # Slots
    __slots__ = ['name', 'abbr', 'unique']
    
    @classmethod
    @property
    def full_name(self): return self.name + " dimension" + ('' if self.unique else 's')
    
    @classmethod
    @property
    def var(self): return self.name + "_dim"
    
    @classmethod
    def named_vars(self, format: str = '{name_dim}'):
        candidates = [(self.abbr, self.abbr + "_dim"), (self.name, self.name + "_dim")]
        var_names = [format.format(name=name, name_dim=name_dim) for name, name_dim in candidates]
        return tuple(set(var_names))
    
    @classmethod
    def first_in(cls, special_dims: dict):
        return cls.last and special_dims.get(cls.var, 0) < 0 or not cls.last and special_dims.get(cls.var, 0) > 0
    
    @classmethod
    def last_in(cls, special_dims: dict):
        return cls.last and special_dims.get(cls.var, 0) > 0 or not cls.last and special_dims.get(cls.var, 0) < 0
    
    @classmethod
    def var_first(cls, number: int):
        return -number if cls.last else number
    
    @classmethod
    def var_last(cls, number: int):
        return number if cls.last else -number
    
    def __init__(self, *index): self.index = index
    
    @classmethod
    def value(cls, expr, default=1):
        if not isinstance(default, tuple): default = default,
        if isinstance(expr, type) and expr == cls: return default
        if isinstance(expr, cls): return expr.index
        value = cls.get_value(expr, default=default)
        return value

class FuncDim(Dimension):
    name    = "function"
    abbr    = 'func'
    unique  = True
    
    @classmethod
    def get_value(cls, expr, default=1):
        # expr: N/A, (...,), (..., ...)
        if isinstance(expr, tuple): return expr
    
    @classmethod
    def repr(cls, value: tuple): return value

class BatchDim(Dimension):
    name    = "batch"
    abbr    = 'batch'
    unique  = True
    
    @classmethod
    def get_value(cls, expr, default=1):
        # expr: {}, {...}, {..., ...}
        if isinstance(expr, dict) and len(expr) == 0: return default
        elif isinstance(expr, set): return tuple(expr)
    
    @classmethod
    def repr(cls, value: tuple): return set(value)

class SequeDim(Dimension):
    name    = "sequence"
    abbr    = 'seque'
    unique  = False
    
    @classmethod
    def get_value(cls, expr, default=1):
        # expr: '', '.', '..., ...'
        if isinstance(expr, str):
            if len(expr) == 0: return default
            elif ',' not in expr: return eval(expr),
            else: return eval(expr)
    
    @classmethod
    def repr(cls, value: tuple): return str(value).strip("(,)")

class TimeDim(SequeDim):
    name    = "time"
    abbr    = 'time'
    unique  = True
    var     = SequeDim.var

class FeatDim(Dimension):
    name    = "feature"
    abbr    = 'feat'
    unique  = False
    
    @classmethod
    def get_value(cls, expr, default=1):
        # expr: [], [...], [..., ...]
        if isinstance(expr, list):
            if len(expr) == 0: return default
            else: return tuple(expr)
    
    @classmethod
    def repr(cls, value: tuple): return list(value)

class ChanDim(FeatDim):
    name    = "channel"
    abbr    = 'chan'
    unique  = True
    var     = FeatDim.var

class SpaceDim(Dimension):
    name    = "space"
    abbr    = 'space'
    unique  = False
    
    @classmethod
    def get_value(cls, expr, default=None):
        # expr: ..., integers
        if isinstance(expr, size_eletype): return expr,
        elif expr == ...: return default

class SpecDim(Dimension):
    name    = "special"
    abbr    = 'spec'
    unique  = False

class AllDim(Dimension):
    name    = "all"
    abbr    = 'all'
    unique  = False
    
    @classmethod
    def get_value(cls, expr, default=None):
        # expr: integers
        if isinstance(expr, size_eletype): return expr,
        if expr is None: return default

########################################
##  Lists For Special Dimensions      ##
########################################

# Note that the corresponding unique dimension should be placed before multiple ones. 
SpecialDimensions = [FuncDim, BatchDim, TimeDim, SequeDim, ChanDim, FeatDim]
MajorSpecialDimensions = [d for d in SpecialDimensions if d.__mro__[1] == Dimension]
empty_special_dims = lambda: {vn: 0 for vn in {sd.var for sd in SpecialDimensions}}

########################################
##  Define Size Type Accordingly      ##
########################################

class Size(tuple):
    
    @classmethod
    def update_special_dims(cls, special_dims=None, **update_special_dims):
        if special_dims is None: special_dims = empty_special_dims()
        for dim in SpecialDimensions:
            args = {vn: update_special_dims[vn] for vn in update_special_dims if dim.abbr in vn}
            if len(args) == 0: continue
            elif len(args) > 1: raise TypeError(f"More than one value specified for '{dim.var}': {args}. ")
            if special_dims[dim.var] != 0: raise TypeError(f"More than one value specified for '{dim.var}': in {update_special_dims} with initial values {special_dims}. ")
            key, value = list(args.items())[0]
            avouch(isinstance(value, int), TypeError(f"Invalid '{key} = {value}' for bt.Size, should be an integer. "))
            if isinstance(value, int):
                if dim.unique: avouch(value in (0, 1, -1), TypeError(f"Invalid '{key} = {value}' for bt.Size, should be 0, 1, or -1 for unique dimension. "))
            elif isinstance(value, bool):
                avouch(dim.unique, TypeError(f"Invalid '{key} = {value}' for bt.Size, cannot use bool value for non-unique dimension. "))
                value = int(value)
            else: raise TypeError(f"Invalid '{key} = {value}' for bt.Size, should be an integer (or bool). ")
            special_dims[dim.var] = value
        return special_dims
    
    @classmethod
    def __new_raw__(cls, shape, **special_dims):
        """
        The raw construction function defined by the inner parameters.

        Args:
            shape (tuple of ints): The raw tuple structure. 
            special_dims (kwargs): The dict containing the following arguments:
                function_dimension (int, optional): An inner parameter for functional dimension, it can only be 0, 1, or -1. Defaults to 0.
                batch_dimension (int, optional): An inner parameter for batch dimension, it can only be 0, 1, or -1. Defaults to 0.
                sequence(time)_dimension (int, optional): An inner parameter for sequence dimensions, being positive when they are in front of the feature-space dimensions. Defaults to 0.
                feature(channel)_dimension (int, optional): An inner parameter for feature dimensions, being positive when they are in front of the space dimensions. Defaults to 0.
        """
        avouch(isinstance(shape, tuple) and all(isinstance(s, size_eletype) for s in shape), TypeError(f"Invalid 'shape = {shape}' for bt.Size, should be a tuple. "))
        
        result_special_dims = cls.update_special_dims(**special_dims)
        avouch(len(shape) >= sum(abs(n) for n in result_special_dims.values()), 
               TypeError(f"Too many special dimensions for shape of length {len(shape)}: {result_special_dims}. "))
        
        self = super().__new__(cls, shape)
        self.special_dims = result_special_dims
        return self

    @classmethod
    def __new_size__(cls, size, **update_special_dims):
        """The construction function for a bt.Size object. """
        avouch(isinstance(size, Size), TypeError(f"Invalid 'size = {size}' for bt.Size, should be a bt.Size object. "))
        result_special_dims = size.special_dims.copy()
        result_special_dims.update(update_special_dims)
        return cls.__new_raw__(tuple(size), **result_special_dims)
    
    @classmethod
    def __new_tuple__(cls, shape, **special_dims):
        """
        The construction function for a tuple with readable parameters. 

        Args:
            shape (tuple of ints): The raw tuple structure. 
            special_dims (kwargs): The dict containing the following arguments:
                [Unique Dimensions]:
                func_dim (bool, optional): The index of the functional dimension, having a domain of 0 or n_dim - 1, the first or last dimension. Defaults to None.
                batch_dim (bool, optional): The index of the batch dimension, having a domain of 0 or n_dim - 1, the first or last dimension. Defaults to None.
                chan_dim (bool, optional): The index of the channel dimension, being the first or last dimension apart from the batch dimension. Defaults to None.
                time_dim (bool, optional): The index of the time dimension, having the first or last dimension apart from the batch and channel dimension. Defaults to None.
                [Multiple Dimensions]:
                seque_dim (int, optional): The number of sequence dimensions, being positive when they are in front of the feature-space dimensions. Defaults to 0.
                feat_dim (int, optional): The number of feature dimensions, being positive when they are in front of the space dimensions. Defaults to 0.
        """
        raw_shape = cls.__new_repr__(shape)
        
        result_special_dims = cls.update_special_dims(raw_shape.special_dims, **special_dims)
        return cls.__new_raw__(tuple(raw_shape), **result_special_dims)
    
    @classmethod
    def __new_repr__(cls, shape):
        """
        The constructor using python representations. Including:
        1. (n_func,) for functional dimension, 
        2. {n_batch} for batch dimension, 
        4. 'n_sequence' for sequence dimensions, 
        3. [n_feature] for feature dimensions, 
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
        self = super().__new__(cls)
        self.init_special()
        
        for i, element in enumerate(shape):
            for dim in MajorSpecialDimensions + [SpaceDim]:
                value = dim.value(element, default=-1)
                if value is None: continue
                self += cls.__new_raw__(value, **{dim.var: len(value)})
                break
            else: raise TypeError(f"Unrecognized dimension representation: {element}. ")
        return self

    def __new__(cls, *args, **kwargs):
        """
        The construction function for 'bt.Size'. 

        Usages:
            bt.Size(shape: torch.Tensor/bt.Tensor/bt.Size/generator/tuple/str, batch_dim=False, n_sequence_dim=None, n_feature_dim=None)
            bt.Size(*shape: python_repr[int, dict[0], set[1], list[], str], batch_dim=False, n_sequence_dim=None, n_feature_dim=None)
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
        # Initialize subject
        if len(args) == 1 and hasattr(args[0], 'shape'): args = (args[0].shape,)
        if len(args) == 1 and isinstance(args[0], Generator): self = cls.__new_tuple__(tuple(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args[0], FakeSize): self = cls.__new_raw__(tuple(args[0]), **kwargs).special_from(args[0])
        elif len(args) == 1 and isinstance(args[0], Size): self = cls.__new_size__(args[0], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple): self = cls.__new_tuple__(args[0], **kwargs)
        elif len(args) == 1 and isinstance(args[0], str):
            if args[0] == '':
                kwargs[SequeDim.var] = 1
                self = cls.__new_tuple__((-1,), **kwargs)
            elif touch(lambda: int(args[0])) is not None:
                kwargs[SequeDim.var] = 1
                self = cls.__new_tuple__((int(args[0]),), **kwargs)
            else:
                self = cls.__new_tuple__(eval(args[0]), **kwargs)
                if self.n_special_dim > 0 or args[0].startswith('('): ...
                self = self.with_sequence_dim(self.n_dim)
        else: self = cls.__new_tuple__(args, **kwargs)
        
        return self

    def __init__(self, *args, **kwargs): ...
    
    def init_special(self):
        self.special_dims = empty_special_dims()
        return self
    
    def special_from(self, other):
        other = getattr(other, 'shape', other)
        avouch(isinstance(other, tuple), TypeError(f"Invalid input for Size.special_from: {type(other)}. "))
        if isinstance(other, Size):
            avouch(self.n_dim == other.n_dim, TypeError(f"Dimension mismatch when inheriting special dimensions: from {other.n_dim} to {self.n_dim}. "))
        if not hasattr(self, 'special_dims'): self.init_special()
        self.special_dims.update(getattr(other, 'special_dims', {}))
        return self
    
    def update_special_from(self, other):
        other = getattr(other, 'shape', other)
        avouch(isinstance(other, tuple) or is_tensor, TypeError(f"Invalid input for Size.special_from: {type(other)}. "))
        self.special_dims.update({k: v for k, v in getattr(other, 'special_dims', {}).items() if v != 0})
        return self
    
    @alias('ndim')
    @property
    def n_dim(self): return len(self)
    
    def get_dim_type(self, index):
        for dim in MajorSpecialDimensions:
            i_start, i_stop = self.get_dim_range(dim)
            if i_start is None: continue
            if i_start <= index < i_stop: return dim
        return SpaceDim
    
    def get_dim_start(self, dim):
        if not getattr(self, f"has_{dim.name}"): return
        i_start, i_stop = getattr(self, f"{dim.name}_domain")
        if dim == SpaceDim: return i_start
        if dim.first_in(self.special_dims): return i_start
        return i_stop - abs(self.special_dims[dim.var])
    
    def get_dim_stop(self, dim):
        if not getattr(self, f"has_{dim.name}"): return
        i_start, i_stop = getattr(self, f"{dim.name}_domain")
        if dim == SpaceDim: return i_stop
        if dim.last_in(self.special_dims): return i_stop
        return i_start + abs(self.special_dims[dim.var])
    
    def get_dim_range(self, dim):
        return self.get_dim_start(dim), self.get_dim_stop(dim)
    
    def get_dim_domain(self, dim):
        n_first_dim = 0
        n_last_dim = 0
        for current_dim in SpecialDimensions:
            if current_dim == dim: break
            if current_dim in MajorSpecialDimensions:
                if current_dim.first_in(self.special_dims):
                    n_first_dim += abs(self.special_dims[current_dim.var])
                else: n_last_dim += abs(self.special_dims[current_dim.var])
        return n_first_dim, self.n_dim - n_last_dim
    
    def has_dim(self, dim):
        if dim == SpecDim: return self.n_special_dim != 0
        if dim == SpaceDim: return self.n_special_dim < self.n_dim
        if dim.unique: return self.special_dims[dim.var] in (1, -1)
        return self.special_dims[dim.var] != 0
    
    def is_of_dim(self, i, dim, name=""):
        if not name: name = f"is_{dim.var}"
        avouch(isinstance(i, int), TypeError(f"Invalid call '{name}({i})': the argument is not an integer. "))
        avouch(-self.n_dim <= i < self.n_dim, IndexError(f"Index out of bound: {i} should be in [{-self.n_dim}, {self.n_dim-1}]. "))
        
        if dim == SpecDim: return not self.is_of_dim(i, SpaceDim)
        if not getattr(self, f"has_{dim.name}"): return False
        if i < 0: i += self.n_dim
        start, stop = getattr(self, f"{dim.abbr}_range")
        if start is None: return False
        return start <= i < stop
    
    def get_i_dim(self, dim):
        if not getattr(self, f"has_{dim.name}"): return
        i_start, i_stop = getattr(self, f"{dim.name}_domain")
        if dim.first_in(self.special_dims): return i_start
        return i_stop - 1
    
    def set_i_dim(self, value, dim, name=""):
        if not name: name = f"with_{dim.var}"
        if value is None: value = 0
        elif isinstance(value, int):
            i_start, i_stop = getattr(self, f"{dim.name}_domain")
            if value == i_start: value = 1
            elif value == i_stop - 1: value = -1
            else: raise TypeError(f"Invalid value={value} for {dim.full_name}, which should be either {i_start} or { i_stop - 1}. ")
        elif isinstance(value, bool): value = int(value)
        else: raise TypeError(f"'bt.Size.{name}' only takes input bool or integer. ")
        
        self.special_dims[dim.var] = value
        return self
    
    def get_n_dim(self, dim):
        if dim == SpecDim: return sum(abs(self.special_dims[d.var]) for d in MajorSpecialDimensions)
        if dim == SpaceDim: return self.n_dim - self.n_special_dim
        return abs(self.special_dims[dim.var])
    
    def set_n_dim(self, value, dim):
        if value is None: value = 0
        self.special_dims[dim.var] = value
        return self
    
    def get_dim_size(self, dim, use_tuple=True):
        start, stop = getattr(self, f"{dim.name}_range")
        if start is None:
            if use_tuple: return Size()
            else: raise TypeError(f"Cannot get the {dim.full_name} when it does not exist. ")
        if use_tuple: return self[start:stop]
        if dim.unique: return self[start]
        return prod(self[start:stop])
    
    def set_dim_size(self, value, dim, place_major=True):
        if isinstance(value, int): value = (value,)
        start, stop = getattr(self, f"{dim.abbr}_range")
        if start is None:
            # if place_major is None: raise TypeError(f"Cannot assign dimension size when {dim.full_name} is unavailable. ")
            if place_major and dim.last or not place_major and not dim.last:
                _, stop = getattr(self, f"{dim.name}_domain"); start = stop
            else: start, _ = getattr(self, f"{dim.name}_domain"); stop = start
        return self[:start] + self.__class__.__new_raw__(value, **{dim.var: len(value)}) + self[stop:]
    
    @alias("nspecial")
    @property
    def n_special(self):
        return prod([self[sd] for sd in self.i_special_dims])
    
    def get_n_ele(self, omit_undetermined=False):
        p = 1
        undetermined = False
        for x in self:
            if x < 0:
                if not omit_undetermined: undetermined = True;
                continue
            p *= x
        if undetermined: Warn(f"Counting the number of elements with undetermined dimension, resulting in {p}N. ")
        return p
    
    @alias("with_nele")
    def with_n_ele(self, n_ele):
        undetermined = [i for i, x in enumerate(self) if x < 0]
        if len(undetermined) == 0:
            avouch(n_ele == self.n_ele, TypeError(f"Cannot set n_ele={n_ele} for size {self} without undetermined dimensions."))
            return self
        avouch(len(undetermined) == 1, TypeError(f"Cannot set n_ele for size {self} with more than one undetermined dimensions."))
        s_ele = self.get_n_ele(omit_undetermined=True)
        avouch(n_ele % s_ele == 0, TypeError(f"Cannot set n_ele={n_ele} for size {self} as it is not a multiplication of current size {s_ele}. "))
        return self[:undetermined[0]] + Size(n_ele // s_ele).special_from(self[undetermined[0]:undetermined[0]+1]) + self[undetermined[0]+1:]
    
    n_ele = alias("nele")(property(get_n_ele).setter(with_n_ele))
    
    def with_dim_size(self, index, size):
        if index < 0: index += self.n_dim
        return self[:index] + Size(size).special_from(self[index:index+1]) + self[index+1:]
    
    @property
    def i_special_dims(self):
        sdim_list = []
        for dim in MajorSpecialDimensions:
            if not getattr(self, f"has_{dim.name}"): continue
            sdim_list.extend(list(range(*getattr(self, f"{dim.name}_range"))))
        sdim_list.sort()
        return sdim_list
    
    def add_special_dim(self, index, reference):
        if not isinstance(reference, Size): reference = Size(reference)
        avouch(-self.n_dim <= index < self.n_dim, TypeError(f"Index for 'add_special_dim' should be within the total dimensions: from {-self.n_dim} to {self.n_dim-1}. "))
        avouch(reference.n_dim == 1, TypeError(f"The reference for 'add_special_dim' should be of 1 dimensional. "))
        if index < 0: index += self.n_dim
        for dim in MajorSpecialDimensions:
            if getattr(reference, f"has_{dim.name}"):
                start, stop = getattr(self, f"{dim.name}_range")
                if dim.unique: avouch(start is None or index == start, TypeError(f"Cannot add unique dimension when one of them already exists: trying to convert {index}-th dim to {dim.name} in {self}. "))
                elif start is not None: avouch(start-1 <= index <= stop, TypeError(f"Only dimensions adjacent to current can be converted into {dim.name} by 'add_special_dim': trying to convert {index}-th dim to feature in {self}. "))
                return self[:index] + self.__class__.__new_raw__((self[index],), **{dim.var: 1}) + self[index+1:]
        return self
    
    def change_special_dim(self, from_dim, to_dim):
        from_dim = exist_dim(self, from_dim)
        avouch(len(from_dim) == 1, TypeError("Only one 'from_dim' is acceptable for 'change_special_dim'. "))
        return self.add_special_dim(from_dim[0], Size(to_dim))
    
    ## methods:
    def transpose(self, i: int, j:int):
        if i == j: return self
        if i > j: i, j = j, i
        if i < 0: i += self.n_dim
        if j < 0: j += self.n_dim
        return self[:i] + self[j:j+1] + self[i+1:j] + self[i:i+1] + self[j+1:]
    
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], tuple): dims = dims[0]
        avouch(sorted(dims) == list(range(len(self))), TypeError("'permute' needs input dimensions forming a permutation of 0~(n-1). "))
        return sum([self[d:d+1] for d in dims], Size())

    @alias("clone")
    def copy(self): return Size(self)

    @alias("raw")
    def tuple(self): return tuple(self)
    
    @alias('__iadd__')
    def __add__(self, other):
        avouch(isinstance(other, tuple), TypeError("Summation for 'bt.Size' is inherited from python object 'tuple' to perform concatenation, please use `size <<(>>) 2` to perform element-wise summation (subtraction) to increase (decrease) the size. "))
        if len(other) == 0: return self
        if len(self) == 0: return other
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        
        special_dims = empty_special_dims()
        self_outer_dim = 0
        other_outer_dim = 0
        for dim in MajorSpecialDimensions:
            n_self_dim = abs(self.special_dims[dim.var])
            n_other_dim = abs(other.special_dims[dim.var])
            if n_self_dim == 0 and n_other_dim == 0: ...
            elif n_other_dim == 0:
                # Use the dimension in self, note that it becomes first dimensions. 
                if dim.first_in(self.special_dims):
                    # self(special, *) + other(*)
                    special_dims[dim.var] = self.special_dims[dim.var]
                elif n_self_dim == self.n_dim - self_outer_dim:
                    # self(outer, special) + other(*)
                    special_dims[dim.var] = dim.var_first(n_self_dim)
                elif other.n_dim == other_outer_dim:
                    # self(*, special) + other(outer)
                    special_dims[dim.var] = dim.var_last(n_self_dim)
                else: raise TypeError(f"Error in concatenating {self} and {other}: {dim.full_name} in middle. ")
            elif n_self_dim == 0:
                # Use the dimension in other, note that it becomes last dimensions. 
                if dim.last_in(other.special_dims):
                    # self(*) + other(*, special)
                    special_dims[dim.var] = other.special_dims[dim.var]
                elif n_other_dim == other.n_dim - other_outer_dim:
                    # self(*) + other(special, outer)
                    special_dims[dim.var] = dim.var_last(n_other_dim)
                elif self.n_dim == self_outer_dim:
                    # self(outer) + other(special, *)
                    special_dims[dim.var] = dim.var_first(n_other_dim)
                else: raise TypeError(f"Error in concatenating {self} and {other}: {dim.full_name} in middle. ")
            elif dim.unique: raise TypeError(f"Error in concatenating {self} and {other}: conflict in {dim.full_name}. ")
            elif (dim.last_in(self.special_dims) or self_outer_dim + n_self_dim == self.n_dim) \
             and (dim.first_in(other.special_dims) or other_outer_dim + n_other_dim == other.n_dim):
                special_dims[dim.var] = dim.var_first(n_self_dim + n_other_dim)
                if self_outer_dim + n_self_dim < self.n_dim:
                    special_dims[dim.var] = -special_dims[dim.var]
            else: raise TypeError(f"Error in concatenating {self} and {other}: multiple sets of {dim.full_name}. ")
            self_outer_dim += n_self_dim
            other_outer_dim += n_other_dim
        
        return self.__class__.__new_raw__(super().__add__(other), **special_dims)
    
    def __radd__(self, other):
        avouch(isinstance(other, tuple), TypeError("Summation for 'bt.Size' is inherited from python object 'tuple' to perform concatenation, please use `size <<(>>) 2` to perform element-wise summation (subtraction) to increase (decrease) the size. "))
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        return other.__add__(self)
    
    @alias('__imul__', '__rmul__')
    def __mul__(self, other):
        avouch(isinstance(other, int), TypeError("Production for 'bt.Size' is inherited from python object 'tuple' to perform duplication, please use `size **(//) 2` to perform element-wise multiplication (division) to enlarge (shrink) the size. "))
        
        for dim in [SpaceDim] + MajorSpecialDimensions[::-1]:
            if getattr(self, f"has_{dim.name}"):
                if dim.unique: return getattr(self, f"with_{dim.name}")((getattr(self, f"n_{dim.name}"), ) * other)
                return getattr(self, f"with_{dim.name}")(getattr(self, dim.name).tuple() * other)
        return self
    
    ## element-wise operations:
    @staticmethod
    def __op__(self, other, *, operation, identity):
        avouch(isinstance(self, Size), RuntimeError("Inner problem: if 'bt.Size.__op__' is not called by user, please contact the developers with Error Code: B526"))
        avouch(isinstance(other, (size_eletype, tuple)), TypeError(f"Element-wise operations are only used for numbers or tuples, not {type(other)}."))
        op = lambda x, y: (max(int(operation(x, y)), 0) if x >= 0 else -1) if identity == 0 or y >= 0 else -1
        if isinstance(other, size_eletype): return self.with_space(tuple(op(x, other) for x in self.space))
        
        if isinstance(other, Size): ...
        elif isinstance(other, tuple): other = self.__class__.__new_raw__(other)
        else: raise TypeError(f"Cannot perform element-wise operation between types {type(self)} and {type(other)}. ")
        
        for dim in MajorSpecialDimensions + [SpaceDim]:
            if not getattr(self, f"has_{dim.name}"): continue
            if not getattr(other, f"has_{dim.name}"): continue
            if dim.unique: self = getattr(self, f"with_n_{dim.name}")(op(getattr(self, f"n_{dim.name}"), getattr(other, f"n_{dim.name}")))
            else:
                other_value = getattr(other, f"{dim.name}").tuple()
                if len(other_value) == 1: other_value *= len(getattr(self, f"{dim.name}"))
                self = getattr(self, f"with_{dim.name}")(op(x, y) for x, y in zip(getattr(self, f"{dim.name}"), other_value))
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
        input_self, input_other = self, other
        raw_self, raw_other = self.tuple(), other.tuple()
        
        cum_self = 0
        cum_other = 0
        res_self_left = []
        res_self_right = []
        res_other_left = []
        res_other_right = []
        special_dims = empty_special_dims()
        for dim in MajorSpecialDimensions + [SpaceDim]:
            if dim != SpaceDim:
                var_self = self.special_dims[dim.var]
                var_other = other.special_dims[dim.var]
            else:
                var_self = self.n_space_dim
                var_other = other.n_space_dim
            
            cum_self += abs(var_self)
            cum_other += abs(var_other)
            
            if dim == SpaceDim: ...
            elif var_self * var_other >= 0: ...
            elif self.n_dim == cum_self: var_self = -var_self
            elif other.n_dim == cum_other: var_other = -var_other
            else: raise TypeError(f"Conflict occurred in unifying sizes {input_self} and {input_other}: mismatched order between {dim.full_name} and other dimensions. ")
            
            if var_self >= 0 and var_other >= 0:
                dim_len = max(var_self, var_other)
                size_self = (1,) * (dim_len - var_self)
                size_other = (1,) * (dim_len - var_other)
            else: # both negative
                dim_len = min(var_self, var_other)
                size_self = (1,) * (var_self - dim_len)
                size_other = (1,) * (var_other - dim_len)
            if var_self != 0: size_self += raw_self[slice(*self.get_dim_range(dim))]
            if var_other != 0: size_other += raw_other[slice(*other.get_dim_range(dim))]
            if dim.first_in({dim.var: dim_len}):
                res_self_left.append(size_self)
                res_other_left.append(size_other)
            else:
                res_self_right.append(size_self)
                res_other_right.append(size_other)
            special_dims[dim.var] = dim_len
        
        return (self.__class__.__new_raw__(sum(res_self_left + res_self_right, tuple()), **special_dims), 
                self.__class__.__new_raw__(sum(res_other_left + res_other_right, tuple()), **special_dims))

    @property
    def python_repr(self):
        left = []; i_left = 0
        right = []; i_right = len(self)
        for dim in MajorSpecialDimensions:
            num_dim = self.special_dims[dim.var]
            if num_dim > 0:
                left.append(dim.repr(super().__getitem__(slice(i_left, i_left + num_dim))))
                i_left += num_dim
            elif num_dim < 0:
                right.append(dim.repr(super().__getitem__(slice(i_right + num_dim, i_right))))
                i_right += num_dim
        if i_left < i_right: left.extend(super().__getitem__(slice(i_left, i_right)))
        return tuple(left) + tuple(right[::-1])

    @alias("__repr__")
    def __str__(self):
        rep = self.python_repr
        return f"batorch.Size{rep}".replace(',)', ')').replace('Ellipsis', '...')
    
    ## operations:
    def __getitem__(self, k):
        if isinstance(k, int): return super().__getitem__(k)
        avouch(isinstance(k, slice), TypeError(f"Slicing of 'bt.Size' only takes integers or slices, not {k} of type {type(k)}. "))
        start, stop = k.start, k.stop
        if start is None: start = 0
        if stop is None: stop = self.n_dim
        if start < 0: start += self.n_dim
        if stop < 0: stop += self.n_dim
        
        special_dims = empty_special_dims()
        for dim in MajorSpecialDimensions:
            if not getattr(self, f"has_{dim.name}"): continue
            dim_start, dim_stop = getattr(self, f"{dim.name}_range")
            if start <= dim_start and stop >= dim_stop: special_dims[dim.var] = self.special_dims[dim.var]
            elif start > dim_start: special_dims[dim.var] = max(min(stop, dim_stop) - start, 0)
            else: special_dims[dim.var] = min(max(start, dim_start) - stop, 0)
        
        return self.__class__.__new_raw__(super().__getitem__(k), **special_dims)
    
    def __eq__(self, other):
        if not isinstance(other, tuple): return False
        if not isinstance(other, Size): return super().__eq__(other)
        return super().__eq__(other) and all(abs(self.special_dims[dim.var]) == abs(other.special_dims[dim.var]) for dim in MajorSpecialDimensions)

# Construct related properties
for dim in SpecialDimensions + [SpaceDim, SpecDim]:
    
    for name in dim.named_vars("has_{name}") + dim.named_vars("has_{name_dim}"):
        def has_dim(s, dim=dim): return s.has_dim(dim)
        setattr(Size, name, property(has_dim))
    
    if not dim in (SpecDim,):
        
        for name in dim.named_vars("{name}_start"):
            def get_dim_start(s, dim=dim): return s.get_dim_start(dim)
            setattr(Size, name, property(get_dim_start))

        for name in dim.named_vars("{name}_stop"):
            def get_dim_stop(s, dim=dim): return s.get_dim_stop(dim)
            setattr(Size, name, property(get_dim_stop))
        
        for name in dim.named_vars("{name}_range"):
            def get_dim_range(s, dim=dim): return s.get_dim_range(dim)
            setattr(Size, name, property(get_dim_range))

        for name in dim.named_vars("{name}_domain"):
            def get_dim_domain(s, dim=dim): return s.get_dim_domain(dim)
            setattr(Size, name, property(get_dim_domain))
    
    for name in dim.named_vars("is_{name_dim}"):
        def is_of_dim(s, v, dim=dim, name=name): return s.is_of_dim(v, dim, name)
        setattr(Size, name, is_of_dim)
    
    # Set-Get Unique Dimension
    if not dim in (SpecDim, SpaceDim) and dim.unique:
        
        for name in dim.named_vars("with_i_{name_dim}"):
            def set_i_dim(s, v, dim=dim, name=name): return s.set_i_dim(dim, name)
            setattr(Size, name, set_i_dim)
        
        for name in dim.named_vars("i_{name_dim}") + dim.named_vars("{name_dim}"):
            def set_i_dim(s, v, dim=dim, name=name): return s.set_i_dim(dim, name)
            def get_i_dim(s, dim=dim): return s.get_i_dim(dim)
            setattr(Size, name, property(get_i_dim).setter(set_i_dim))
    
    # Set-Get Multiple Dimensions
    if not dim in (SpecDim, SpaceDim):
        
        for name in dim.named_vars("with_{name_dim}") + dim.named_vars("with_prop_{name_dim}") + (dim.named_vars("with_n_{name_dim}") if not dim.unique else tuple()):
            def set_n_dim(s, v, dim=dim): return s.set_n_dim(v, dim)
            setattr(Size, name, set_n_dim)
    
    for name in dim.named_vars("n_{name_dim}"):
        def set_n_dim(s, v, dim=dim): return s.set_n_dim(v, dim)
        def get_n_dim(s, dim=dim): return s.get_n_dim(dim)
        if not dim in (SpecDim, SpaceDim):
            setattr(Size, name, property(get_n_dim).setter(set_n_dim))
        else: setattr(Size, name, property(get_n_dim))
    
    # Set-Get Dimension Size / Tuple
    if dim not in (SpecDim,):
        
        for name in (dim.named_vars("with_n_{name}") if dim.unique else tuple()) + dim.named_vars("with_{name}"):
            def set_dim_size(s, v, dim=dim): return s.set_dim_size(v, dim)
            setattr(Size, name, set_dim_size)
        
        for name in dim.named_vars("n_{name}") + dim.named_vars("n{name}") + dim.named_vars("{name}_size"):
            def set_dim_size(s, v, dim=dim): return s.set_dim_size(v, dim)
            def get_dim_size(s, dim=dim): return s.get_dim_size(dim, use_tuple=False)
            setattr(Size, name, property(get_dim_size).setter(set_dim_size))
        
        if not dim.unique:
            for name in dim.named_vars("{name}"):
                def set_dim_size(s, v, dim=dim): return s.set_dim_size(v, dim)
                def get_dim_size(s, dim=dim): return s.get_dim_size(dim, use_tuple=True)
                setattr(Size, name, property(get_dim_size).setter(set_dim_size))

############################################################
##  Define a Fake Size to Inherit the Properties          ##
############################################################

class FakeSize(tuple):
    def __new__(cls, raw_tuple, **special_dims):
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
        self = super().__new__(cls, tuple(raw_tuple))
        if isinstance(raw_tuple, Size):
            result_special_dims = raw_tuple.special_dims
        else: result_special_dims = empty_special_dims()
        result_special_dims.update(special_dims)
        self.special_dims = result_special_dims
        return self
    def __repr__(self):
        return 'FakeSize' + super().__repr__().rstrip(',)') + f", special_dims={self.special_dims})"
    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], int): return super().__getitem__(*args)
        return FakeSize(super().__getitem__(*args), **self.special_dims)
    @alias('__iadd__')
    def __add__(self, other):
        return FakeSize(super().__add__(tuple(other)), **self.special_dims)
    def __radd__(self, other):
        return FakeSize(tuple(other) + tuple(self), getattr(other, 'special_dims', self.special_dims))


class new_dim_meta(ABCMeta):

    def __instancecheck__(self, item):
        if isinstance(item, tuple): return all(self.__instancecheck__(x) for x in item)
        for dim in MajorSpecialDimensions + [SpaceDim]:
            if dim.value(item) is not None: break
        else: return False
        return True

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
        if len(args) == 0: args = [(dim,) for dim in MajorSpecialDimensions[1:] if self.special_dims[dim.var] == 0][0]
        if isinstance(args, FakeSize): return FakeSize(tuple((x + self.n_dim) if x < 0 else x for x in args), **args.special_dims)
        if isinstance(args, Size): args = args.python_repr
        if not (hasattr(self, 'n_dim') and hasattr(self, 'special_dims')):
            # Failed to get the special dimensions, ready to throw error. 
            if isinstance(self, torch.Tensor): self = self.as_subclass(torch.Tensor)
            else: self = tuple(self)
            raise AttributeError(f"Cannot get special dimension from {self}. Possible reasons are:\n(1) The input object is not bt.Tensor/Size. \n(2) Special dimensions are lost during unreimplemented torch functions. ")
        self = getattr(self, 'shape', self)
        
        int_args = tuple()
        for arg in args:
            for dim in MajorSpecialDimensions + [SpaceDim]:
                start, stop = getattr(self, f"{dim.name}_range")
                if start is None: start, stop = getattr(self, f"{dim.name}_domain")
                default_index = stop if dim.last else start
                indices = dim.value(arg, default = default_index)
                if indices is None: continue
                else: indices = tuple(i + self.n_dim + 1 if i < 0 else i for i in indices)
                if dim.unique: avouch(self.special_dims[dim.var] == 0, f"Cannot add new {dim.full_name} for tensor already with one. ")
                for index in indices:
                    self = self[:index] + Size(dim(1)) + self[index:]
                int_args += indices
                break
            else: raise TypeError(f"Cannot recognize {arg} for {self}. ")
        return FakeSize(int_args, **self.special_dims)
    
    @classmethod
    def __class_getitem__(cls, arg):
        return iter_dim(cls, arg)

class exist_dim_meta(ABCMeta):

    def __instancecheck__(self, item):
        if isinstance(item, tuple): return all(self.__instancecheck__(x) for x in item)
        for dim in MajorSpecialDimensions + [SpaceDim, AllDim]:
            if dim.value(item) is not None: break
        else: return False
        return True

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
        if isinstance(args, FakeSize): return FakeSize(tuple((x + self.n_dim) if x < 0 else x for x in args), **args.special_dims)
        if isinstance(args, Size): args = args.python_repr
        if not (hasattr(self, 'n_dim') and hasattr(self, 'special_dims')):
            # Failed to get the special dimensions, ready to throw error. 
            if isinstance(self, torch.Tensor): self = self.as_subclass(torch.Tensor)
            else: self = tuple(self)
            raise AttributeError(f"Cannot get special dimension from {self}. Possible reasons are:\n(1) The input object is not bt.Tensor/Size. \n(2) Special dimensions are lost during unreimplemented torch functions. ")
        self = getattr(self, 'shape', self)
        del_simulator = self
        
        int_args = tuple()
        for arg in args:
            for dim in MajorSpecialDimensions + [AllDim, SpaceDim]:
                if dim == AllDim: start, stop = 0, self.n_dim
                else: start, stop = getattr(self, f"{dim.name}_range")
                if start is None: continue
                indices = dim.value(arg, default = None)
                if indices is None: continue
                if indices == (None,): indices = tuple(range(start, stop))
                else: indices = tuple(stop+i if i < 0 else start+i for i in indices)
                for index in indices:
                    del_simulator = del_simulator[:index] + del_simulator[index+1:]
                int_args += indices
                break
            else: raise TypeError(f"Cannot find dimension '{arg}' in {self}. ")
        result = FakeSize(int_args, **self.special_dims)
        result.del_special_dims = del_simulator.special_dims
        return result
    
    @classmethod
    def __class_getitem__(cls, arg):
        return iter_dim(cls, arg)

class del_dim(exist_dim):
    def __new__(this, self, *args):
        result = super().__new__(this, self, *args)
        if hasattr(result, 'del_special_dims'):
            result.special_dims = result.del_special_dims
        return result
    
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
            else: raise TypeError(f"Invalid size {input.shape} for linalg_dim: at least {min_n_dim} non-unique dimension needed. ")
        else: dim = exist_dim(input, *dim)
        if max_n_dim is not None and max_n_dim > 0 and len(dim) > max_n_dim: dim = dim[-max_n_dim:]
        dim.special_dims = empty_special_dims()
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
