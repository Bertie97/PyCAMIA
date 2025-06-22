
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The dimensions manipulations for tensors in batch.",
    create = "2025-01",
    requires = "torch"
)

__all__ = """
    FuncDim        BatchDim       SequeDim       FeatDim        SpaceDim
    new_dim        exist_dim
    del_dim        iter_dim       linalg_dim

    Size           FakeSize
""".split()

from abc import ABCMeta
from typing import Generator

with __info__:
    import torch
    from pyoverload import null
    from pycamia import avouch, touch, alias

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
    uname(str)          The name when the dimension is mentioned uniquely. 
    uabbr(str)          The abbreviation for unique_name
    last(bool)          Whether the dimension is the last dimensions or not by default. Defaults to False. 
                        If it is True, the variable > 0 means the last dimensions. 
    *get_value(method)  A method that takes an expression and returns the values if it represents the dimension (in tuple). It should return None otherwise. 
    """
    @classmethod
    @property
    def full_name(self): return self.name + " dimension"
    
    @classmethod
    @property
    def var(self): return self.name + "_dimension"
    
    @classmethod
    @property
    def abbr_var(self): return self.abbr + "_dim"
    
    @classmethod
    @property
    def uvar(self): return getattr(self, 'uname', '') + "_dimension"
    
    @classmethod
    @property
    def abbr_uvar(self): return getattr(self, 'uabbr', '') + "_dim"
    
    @classmethod
    def is_first(cls, special_dims):
        return getattr(cls, 'last', False) and special_dims.get(cls.var, 0) < 0 or not getattr(cls, 'last', False) and special_dims.get(cls.var, 0) > 0
    
    @classmethod
    def is_last(cls, special_dims):
        return getattr(cls, 'last', False) and special_dims.get(cls.var, 0) > 0 or not getattr(cls, 'last', False) and special_dims.get(cls.var, 0) < 0

class FuncDim(Dimension):
    name    = "functional"
    abbr    = 'func'
    unique  = True
    
    @classmethod
    def get_value(cls, expr):
        # expr: None, (...,), (..., ...)
        if expr is None: return 1,
        elif isinstance(expr, tuple): return expr
    
    @classmethod
    def repr(cls, value: tuple): return value

class BatchDim(Dimension):
    name    = "batch"
    abbr    = 'batch'
    unique  = True
    
    @classmethod
    def get_value(cls, expr):
        # expr: {}, {...}, {..., ...}
        if isinstance(expr, dict) and len(expr) == 0: return 1,
        elif isinstance(expr, set): return tuple(expr)
    
    @classmethod
    def repr(cls, value: tuple): return set(value)

class SequeDim(Dimension):
    name    = "sequence"
    abbr    = 'seque'
    unique  = False
    uname   = "sequence"
    uabbr   = 'seque'
    
    @classmethod
    def get_value(cls, expr):
        # expr: '', '.', '..., ...'
        if isinstance(expr, str):
            if len(expr) == 0: return 1,
            elif ',' not in expr: return eval(expr),
            else: return eval(expr)
    
    @classmethod
    def repr(cls, value: tuple): return str(value).strip("(,)")

class FeatDim(Dimension):
    name    = "feature"
    abbr    = 'feat'
    unique  = False
    uname   = "channel"
    uabbr   = 'chan'
    
    @classmethod
    def get_value(cls, expr):
        # expr: [], [...], [..., ...]
        if isinstance(expr, list):
            if len(expr) == 0: return 1,
            else: return tuple(expr)
    
    @classmethod
    def repr(cls, value: tuple): return list(value)

class SpaceDim(Dimension):
    name    = "space"
    abbr    = 'space'
    unique  = False
    
    @classmethod
    def get_value(cls, expr):
        # expr: integers
        if isinstance(expr, size_eletype): return expr,

SpecialDimensions = [FuncDim, BatchDim, SequeDim, FeatDim]
Dimensions = [FuncDim, BatchDim, SequeDim, FeatDim, SpaceDim]

class FakeSize: ...


########################################
##  Define Size Type Accordingly      ##
########################################

class Size(tuple):
    
    @classmethod
    def __new_raw__(cls, shape, **special_dims):
        """
        The raw construction function defined by the inner parameters.

        Args:
            shape (tuple of ints): The raw tuple structure. 
            special_dims (kwargs): The dict containing the following arguments:
                functional_dimension (int, optional): An inner parameter for functional dimension, it can only be 0, 1, or -1. Defaults to 0.
                batch_dimension (int, optional): An inner parameter for batch dimension, it can only be 0, 1, or -1. Defaults to 0.
                sequence_dimension (int, optional): An inner parameter for sequence dimensions, being positive when they are in front of the feature-space dimensions. Defaults to 0.
                feature_dimension (int, optional): An inner parameter for feature dimensions, being positive when they are in front of the space dimensions. Defaults to 0.
        """
        avouch(isinstance(shape, tuple) and all(isinstance(s, size_eletype) for s in shape), TypeError(f"Invalid 'shape = {shape}' for bt.Size, should be a tuple. "))
        
        result_special_dims = {}
        for dim in SpecialDimensions:
            args = {vn: special_dims[vn] for vn in [dim.var, dim.abbr_var] if vn in special_dims}
            if len(args) == 0: result_special_dims[dim.var] = 0; continue
            elif len(args) > 1: raise TypeError(f"More than one value specified for '{dim.var}': {args}. ".replace('{', '').replace('}', ''))
            key, value = list(args.items())[0]
            avouch(isinstance(value, int), TypeError(f"Invalid '{key} = {value}' for bt.Size, should be an integer. "))
            if dim.unique: avouch(value in (0, 1, -1), TypeError(f"Invalid '{key} = {value}' for bt.Size, should be 0, 1, or -1 for unique dimension. "))
            result_special_dims[dim.var] = value
        avouch(len(shape) >= sum(abs(n) for n in result_special_dims.values()), 
               TypeError(f"Too many special dimensions for shape of length {len(shape)}: {result_special_dims}. "))
        
        self = super().__new__(cls, shape)
        self.special_dims = result_special_dims
        return self

    @classmethod
    def __new_size__(cls, size, **update_special_dims):
        """The construction function for a bt.Size object. """
        avouch(isinstance(size, Size), TypeError(f"Invalid 'size = {size}' for bt.Size, should be a bt.Size object. "))
        for dim in SpecialDimensions:
            update_special_dims.setdefault(dim.var, size.special_dims[dim.var])
        return cls.__new_raw__(tuple(size), **update_special_dims)
    
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
                seque_dim (bool, optional): The index of the sequence dimension, having the first or last dimension apart from the batch and channel dimension. Defaults to None.
                [Multiple Dimensions]:
                seque_dim (int, optional): The number of sequence dimensions, being positive when they are in front of the feature-space dimensions. Defaults to 0.
                feat_dim (int, optional): The number of feature dimensions, being positive when they are in front of the space dimensions. Defaults to 0.
        """
        raw_shape = cls.__new_repr__(shape)
        
        result_special_dims = raw_shape.special_dims
        for dim in SpecialDimensions:
            args = {vn: special_dims[vn] for vn in [dim.var, dim.abbr_var, dim.uvar, dim.abbr_uvar] if vn in special_dims}
            if len(args) == 0: continue
            elif len(args) > 1: raise TypeError(f"More than one value specified for '{dim.var}': {args}. ".replace('{', '').replace('}', ''))
            key, value = list(args.items())[0]
            if isinstance(value, int): ...
            elif isinstance(value, bool):
                avouch(dim.unique or key in (dim.uvar, dim.abbr_uvar), TypeError(f"Invalid '{key} = {value}' for bt.Size, cannot use bool value for non-unique dimension. "))
                value = int(value)
            else: raise TypeError(f"Invalid '{key} = {value}' for bt.Size, should be an integer (or bool). ")
            if value != 0:
                prev_value = result_special_dims.get(dim.var, 0)
                if prev_value == 0: result_special_dims[dim.var] = value
                else: raise TypeError(f"Cannot specify dimension by '{key}={value}': representation implies {dim.var}={prev_value}. ")
        
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
        special_dims = {dim.var: 0 for dim in SpecialDimensions}
        self = super().__new__(cls)
        self.init_special()
        
        for i, element in enumerate(shape):
            for dim in Dimensions:
                value = dim.get_value(element)
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
        if len(args) == 1 and hasattr(args[0], 'shape'): args = (args[0].shape,)
        if len(args) == 1 and isinstance(args[0], Generator): return cls.__new_tuple__(tuple(args[0]), **kwargs)
        if len(args) == 1 and isinstance(args[0], FakeSize): return cls.__new_raw__(tuple(args[0]), **kwargs).special_from(args[0])
        if len(args) == 1 and isinstance(args[0], Size): return cls.__new_size__(args[0], **kwargs)
        if len(args) == 1 and isinstance(args[0], tuple): return cls.__new_tuple__(args[0], **kwargs)
        if len(args) == 1 and isinstance(args[0], str):
            if args[0] == '':
                kwargs[SequeDim.var] = 1
                return cls.__new_tuple__((-1,), **kwargs)
            if touch(lambda: int(args[0])) is not None:
                kwargs[SequeDim.var] = 1
                return cls.__new_tuple__((int(args[0]),), **kwargs)
            self = cls.__new_tuple__(eval(args[0]), **kwargs)
            if self.n_special_dim > 0 or args[0].startswith('('): return self
            return self.sz_sequence_dim_(self.n_dim)
        return cls.__new_tuple__(args, **kwargs)

    def __init__(self, *args, **kwargs): ...
    
    def init_special(self):
        self.special_dims = {dim.var: 0 for dim in SpecialDimensions}
        return self
    
    def special_from(self, other, allow_view=False):
        is_tensor = any(str(c).split("'")[1] == "batorch.tensor.Tensor" for c in other.__class__.__mro__)
        avouch(isinstance(other, tuple) or is_tensor, TypeError(f"Invalid input for Size.special_from: {type(other)}. "))
        if is_tensor: other = getattr(other, 'shape')
        if isinstance(other, Size):
            if self.n_dim != other.n_dim:
                if allow_view: return self.view(other)
                raise TypeError(f"Dimension mismatch when inheriting special dimensions: from {other.n_dim} to {self.n_dim}. ")
        for dim in SpecialDimensions:
            self.special_dims[dim.var] = other.special_dims.get(dim.var, 0)
        return self
    
    def update_special_from(self, other):
        is_tensor = any(str(c).split("'")[1] == "batorch.tensor.Tensor" for c in other.__class__.__mro__)
        avouch(isinstance(other, tuple) or is_tensor, TypeError(f"Invalid input for Size.special_from: {type(other)}. "))
        if is_tensor: other = getattr(other, 'shape', (1,))
        for dim in SpecialDimensions:
            value = other.special_dims.get(dim.var, 0)
            if value != 0: self.special_dims[dim.var] = value
        return self
    
    @alias('ndim')
    @property
    def n_dim(self): return len(self)
    
    def __getattr__(self, name):
        n_first_dim = 0
        n_last_dim = 0
        for dim in SpecialDimensions:
            has_uname = hasattr(dim, 'uabbr')
            if name in (f"has_{dim.abbr}", f"has_{dim.name}", f"has_{dim.var}"):
                # has_{name}
                return self.special_dims[dim.var] != 0
            if has_uname and name in (f"has_{dim.uabbr}", f"has_{dim.uname}", f"has_{dim.uvar}", f"has_{dim.abbr_uvar}"):
                # has_{name} (unique)
                return self.special_dims[dim.var] == 1
            elif name in (f"n_{dim.abbr_var}", f"n_{dim.var}"):
                # n_{name}_dim
                return abs(self.special_dims[dim.var])
            elif name in (f"{dim.abbr}_start", f"{dim.name}_start"):
                # {name}_start
                return n_first_dim if dim.is_first(self.special_dims) else (self.n_dim - n_last_dim - getattr(self, f"n_{dim.abbr_var}"))
            elif name in (f"{dim.abbr}_start", f"{dim.name}_stop"):
                # {name}_stop
                return (self.n_dim - n_last_dim) if dim.is_last(self.special_dims) else (n_first_dim + getattr(self, f"n_{dim.abbr_var}"))
            elif (dim.unique and name in (f"is_{dim.abbr_var}", f"is_{dim.var}") or 
                  not dim.unique and name in (f"is_{dim.abbr_var}s", f"is_{dim.var}s") or 
                  has_uname and name in (f"is_{dim.abbr_uvar}", f"is_{dim.uvar}")):
                # is_{name}_dim
                do_unique = dim.unique or name in (f"is_{dim.abbr_uvar}", f"is_{dim.uvar}")
                def judge_func(i):
                    avouch(isinstance(i, int), TypeError(f"Invalid call 'name({i})': the argument is not an integer. "))
                    if self.special_dims[dim.var] == 0: return False
                    if i < 0: i += self.n_dim
                    in_range = getattr(self, f"{dim.abbr}_start") <= i < getattr(self, f"{dim.abbr}_stop")
                    if do_unique: return abs(self.special_dims[dim.var]) == 1 and in_range
                    return in_range
                return judge_func
            elif (dim.unique and name in (f"with_{dim.abbr_var}", f"with_{dim.var}") or 
                  not dim.unique and name in (f"with_{dim.abbr_var}s", f"with_{dim.var}s") or
                  has_uname and name in (f"with_{dim.abbr_uvar}", f"with_{dim.uvar}")):
                # with_{name}_dim
                do_unique = dim.unique or name in (f"with_{dim.abbr_uvar}", f"with_{dim.uvar}")
                def set_dim_func(ifunc):
                    avouch(ifunc is None or isinstance(ifunc, bool) or ifunc in (0, -self.n_dim, self.n_dim-1, -1), TypeError("'bt.Size.with_func_dim' only takes input bool or integer 0, -1."))
                    if ifunc or isinstance(ifunc, int):
                        avouch(self.n_batch_dim + self.n_sequence_dim + self.n_feature_dim < self.n_dim, TypeError(f"Cannot set func_dim for size {self} of non-special dimension 0{' (scalar)' if self.n_dim == 0 else ''}."))
                        self.sz_func_dim = 1 if ifunc in (0, -self.n_dim, True) else -1
                    else: self.sz_func_dim = 0
                return set_dim_func
            elif not dim.unique and name in (f"with_{dim.abbr_var}s", f"with_{dim.abbr_uvar}" if hasattr(dim, 'abbr_uvar') else ''):
                # with_{name}_dim
                
            elif dim.unique and name in (f"{dim.abbr}_dim", f"{dim.uabbr}_dim"):
                # {name}_dim
                if self.special_dims[dim.var] == 0: return
                elif dim.is_first(self.special_dims): return getattr(self, f"{dim.abbr}_start")
                elif dim.is_last(self.special_dims): return getattr(self, f"{dim.abbr}_stop")
            if dim.is_first(self.special_dims):
                n_first_dim += abs(self.special_dims[dim.var])
            else: n_last_dim += abs(self.special_dims[dim.var])
    
    @alias('__iadd__')
    def __add__(self, other):
        avouch(isinstance(other, tuple), TypeError("Summation for 'bt.Size' is inherited from python object 'tuple' to perform concatenation, please use `size <<(>>) 2` to perform element-wise summation (subtraction) to increase (decrease) the size. "))
        if len(other) == 0: return self
        if len(self) == 0: return other
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        
        special_dims = {}
        self_outer_dim = 0
        other_outer_dim = 0
        for dim in SpecialDimensions:
            if self.special_dims[dim.var] == 0 and other.special_dims[dim.var] == 0:
                special_dims[dim.var] = 0
            elif self.special_dims[dim.var] == 0:
                # Use the dimension in other, note that it becomes last dimensions. 
                if dim.is_last(other.special_dims):
                    # self(*) + other(*, special)
                    special_dims[dim.var] = other.special_dims[dim.var]
                elif abs(other.special_dims[dim.var]) == other.n_dim - other_outer_dim:
                    # self(*) + other(special, outer)
                    special_dims[dim.var] = (+1 if getattr(dim, 'last', False) else -1) * (other.n_dim - other_outer_dim)
                elif self.n_dim == self_outer_dim:
                    # self(outer) + other(special, *)
                    special_dims[dim.var] = (-1 if getattr(dim, 'last', False) else +1) * abs(other.special_dims[dim.var])
                else: raise TypeError(f"Error in concatenating {self} and {other}: {dim.full_name} in middle. ")
            elif other.special_dims[dim.var] == 0:
                # Use the dimension in other, note that it becomes last dimensions. 
                if dim.is_first(self.special_dims):
                    # self(special, *) + other(*)
                    special_dims[dim.var] = self.special_dims[dim.var]
                elif abs(self.special_dims[dim.var]) == self.n_dim - self_outer_dim:
                    # self(outer, special) + other(*)
                    special_dims[dim.var] = (-1 if getattr(dim, 'last', False) else +1) * (self.n_dim - self_outer_dim)
                elif other.n_dim == other_outer_dim:
                    # self(*, special) + other(outer)
                    special_dims[dim.var] = (+1 if getattr(dim, 'last', False) else -1) * abs(self.special_dims[dim.var])
                else: raise TypeError(f"Error in concatenating {self} and {other}: {dim.full_name} in middle. ")
            elif dim.unique: raise TypeError(f"Error in concatenating {self} and {other}: conflict in {dim.full_name}. ")
            elif (dim.is_last(self.special_dims) or self_outer_dim + abs(self.special_dims[dim.var]) == self.n_dim) \
             and (dim.is_first(other.special_dims) or other_outer_dim + abs(other.special_dims[dim.var]) == other.n_dim):
                special_dims[dim.var] = (-1 if getattr(dim, 'last', False) else +1) * (abs(self.special_dims[dim.var]) + abs(other.special_dims[dim.var]))
                if self_outer_dim + abs(self.special_dims[dim.var]) < self.n_dim:
                    special_dims[dim.var] = -special_dims[dim.var]
            else: raise TypeError(f"Error in concatenating {self} and {other}: multiple sets of {dim.full_name}s. ")
            self_outer_dim += abs(self.special_dims[dim.var])
            other_outer_dim += abs(other.special_dims[dim.var])
        
        return self.__class__.__new_raw__(super().__add__(other), **special_dims)
    
    def __radd__(self, other):
        avouch(isinstance(other, tuple), TypeError("Summation for 'bt.Size' is inherited from python object 'tuple' to perform concatenation, please use `size <<(>>) 2` to perform element-wise summation (subtraction) to increase (decrease) the size. "))
        if not isinstance(other, Size): other = self.__class__.__new_raw__(other)
        return other.__add__(self)

    @property
    def python_repr(self):
        left = []; i_left = 0
        right = []; i_right = len(self)
        for dim in SpecialDimensions:
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
