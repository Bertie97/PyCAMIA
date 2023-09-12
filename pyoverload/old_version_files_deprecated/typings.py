
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "pyoverload",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Types for typehint and overload. ",
    requires = ""
)

__all__ = """
    ArrayType
    ScalarType
    avoid
    union
    tag
    class_satisfies
    instance_satisfies
    
    object
    bool
    int
    long
    float
    double
    real
    complex
    scalar
    bool_scalar
    int_scalar
    long_scalar
    float_scalar
    double_scalar
    real_scalar
    complex_scalar
    property
    dtype
    type
    class_type
    super
    null
    
    callable
    functional
    lambda_func
    method
    function
    builtin_function_or_method
    method_descriptor
    method_wrapper
    classmethod
    staticmethod
    
    str
    bytearray
    bytes
    
    array
    iterable
    sequence
    list
    tuple
    dict
    set
    reversed
    frozenset
    
    zip
    enumerate
""".split()

import random, copy, builtins
from abc import ABCMeta

def _touch(f):
    try: return f()
    except Exception: ...

### >>> type generators <<< ###

class ArrayType(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, origin, args):
        if isinstance(args, (builtins.type, builtins.int)): args = (args,)

        get_name = lambda x: getattr(x, '__name__',
            f"{getattr(x.start, '__name__', x.start)}:{getattr(x.stop, '__name__', x.stop)}"
            if isinstance(x, builtins.slice) else builtins.str(x)
        ).split('.')[-1]
        if isinstance(args, builtins.list):
            arg_names = ', '.join(get_name(a) for a in args)
            type_name = f"{get_name(origin)}[{len(args)}]:[{arg_names}]"
        elif isinstance(args, builtins.tuple):
            type_names = ': '.join(get_name(t) for t in args if isinstance(t, builtins.type))
            size = builtins.tuple(s for s in args if not isinstance(s, builtins.type))
            if len(size) == 1: size = size[0]
            type_name = f"{get_name(origin)}[{type_names}, {size}]"

        if isinstance(args, builtins.list):
            return super().__new__(cls, type_name, (origin,), 
                builtins.dict(origin=origin, detail_mode='elements', element_types=args))
        if not isinstance(args, (builtins.tuple, builtins.type, builtins.int)):
            raise TypeError(f"Invalid ArrayType: {type_name}; ArrayType subscript only take list, tuple, type, or int. ")
        base_type = [builtins.object]
        base_size = -1
        if isinstance(args, builtins.tuple):
            base_type = []
            for arg in args:
                if isinstance(arg, builtins.type):
                    base_type.append(arg)
                elif isinstance(arg, builtins.tuple) and all(isinstance(a, builtins.type) or isinstance(a, dtype) for a in arg):
                    base_type.append(arg)
                elif isinstance(arg, builtins.slice):
                    if arg.step is not None:
                        raise TypeError(f"Invalid ArrayType: {type_name}; two colons in ArrayType subscript.")
                    base_type.append(arg.start)
                    base_type.append(arg.stop)
                else: break
            if len(base_type) <= 1: ...
            elif len(base_type) == 2 and issubclass(origin, builtins.dict): ...
            elif issubclass(origin, builtins.zip): ...
            else:
                raise TypeError(f"Invalid ArrayType: {type_name}; at most 2 types for dict key-value pair in subscript of ArrayType, one should use tuple of types to include multiple probabilities or use [(...),] if no size is given. ")
            base_size = args[len(base_type):]
            if len(base_size) == 1: base_size = base_size[0]
            if len(base_type) == 0: base_type = [builtins.object]
        return super().__new__(cls, type_name, (builtins.object,), 
            builtins.dict(origin=origin, detail_mode='cells', base_type=base_type, base_size=base_size))
    
    def __init__(self, origin, args): ...

    @property
    def __name__(self): return builtins.str(self)

    def __str__(self):
        get_name = lambda x: getattr(x, '__name__',
            f"{getattr(x.start, '__name__', x.start)}:{getattr(x.stop, '__name__', x.stop)}"
            if isinstance(x, builtins.slice) else builtins.str(x)
        ).split('.')[-1]
        if self.detail_mode == 'elements':
            arg_names = ', '.join(get_name(a) for a in self.element_types)
            return f"{get_name(self.origin)}[{len(self.element_types)}]:[{arg_names}]"
        elif self.detail_mode == 'cells':
            type_names = ': '.join(get_name(t) for t in self.base_type)
            return f"{get_name(self.origin)}[{type_names}, {self.base_size}]"
        else: raise RuntimeError(f"Unrecognized detail_mode {self.detail_mode}, please contact the developer for more information (Error Code: H122).")
        
    def __instancecheck__(self, x):
        if not isinstance(x, self.origin): return False
        if self.detail_mode == 'elements':
            y = builtins.list(x)
            if len(y) != len(self.element_types): return False
            return all(isinstance(u, t) for u, t in builtins.zip(y, self.element_types))
        elif self.detail_mode == 'cells':
            n_max_ele = 100
            if issubclass(self.origin, builtins.dict):
                if not isinstance(self.base_size, builtins.int): raise TypeError(f"in {self}: dict size should be an integer")
                if self.base_size >= 0 and len(x) != self.base_size: return False
                if len(self.base_type) == 1: self.base_type = [builtins.object, self.base_type[0]]
                if len(x) > n_max_ele: n_keys = n_max_ele
                else: n_keys = len(x)
                return all(isinstance(k, self.base_type[0]) and isinstance(x[k], self.base_type[1]) for k, _ in builtins.zip(x, builtins.range(n_max_ele)))
            if issubclass(self.origin, builtins.zip):
                if not isinstance(self.base_size, builtins.int): raise TypeError(f"in {self}: zip size should be an integer")
                y = copy.deepcopy(x)
                l = 0
                while True if self.base_size >= 0 else n_max_ele:
                    try: sample = next(y)
                    except StopIteration: break
                    if not all(isinstance(u, t) for u, t in builtins.zip(sample, self.base_type)): return False
                    l += 1
                if self.base_size >= 0 and self.base_size != l: return False
                return True
            if issubclass(self.origin, builtins.enumerate):
                if not isinstance(self.base_size, builtins.int): raise TypeError(f"in {self}: enumerate size should be an integer")
                y = copy.deepcopy(x)
                l = 0
                while True if self.base_size >= 0 else n_max_ele:
                    try: _, sample = next(y)
                    except StopIteration: break
                    if not isinstance(sample, self.base_type): return False
                    l += 1
                if self.base_size >= 0 and self.base_size != l: return False
                return True
            if hasattr(x, 'shape'):
                if self.base_size != -1:
                    if not isinstance(self.base_size, builtins.tuple): self.base_size = (self.base_size,)
                    if not (builtins.type(x.shape)(self.base_size) == x.shape or _touch(lambda: self.base_size == builtins.tuple(x.shape))): return False
                _dtype = getattr(x, 'dtype', None)
                if _dtype is None: raise TypeError(f"Cannot find 'dtype' in examination isinstance({x}, {self}), please contact the developer for more information (Error Code: H123).")
                if len(self.base_type) > 1: raise TypeError(f"{x.__class__.__name__} object can only have one dtype, instead of {len(self.base_type)}: {self.base_type}")
                return issubclass(dtype(_dtype), self.base_type[0])
                # return self.base_type[0] == builtins.object or dtype == self.base_type[0] or self.base_type[0].__name__.split('.')[-1] in builtins.str(dtype)
            if not isinstance(self.base_size, builtins.int): raise TypeError(f"in {self}: array size should be an integer")
            if len(self.base_type) > 1: raise TypeError(f"{x.__class__.__name__} object can only have one type, instead of {len(self.base_type)}: {self.base_type}")
            x = builtins.list(x)
            if self.base_size >= 0 and len(x) != self.base_size: return False
            return all(isinstance(u, self.base_type[0]) for u in x)
        else: raise RuntimeError(f"Unrecognized detail_mode {self.detail_mode}, please contact the developer for more information (Error Code: H122).")
    
    def __subclasscheck__(self, t):
        return issubclass(t, self.origin)
    
class ScalarType(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, *base_types):
        if len(base_types) == 1 and isinstance(base_types[0], builtins.tuple): base_types = base_types[0]
        name = f"Scalar[{', '.join(getattr(t, '__name__', builtins.str(t)) for t in base_types)}]"
        return super().__new__(cls, name, (builtins.object,), builtins.dict(base_types = base_types))
        
    def __init__(self, *base_types): ...

    @property
    def __name__(self): return builtins.str(self)

    def __str__(self): return f"Scalar[{', '.join(getattr(t, '__name__', builtins.str(t)) for t in self.base_types)}]"
    
    def __instancecheck__(self, x):
        if len(getattr(x, 'shape', [])) > 0: return False
        if len(self.base_types) == 0: return True
        for t in self.base_types:
            if isinstance(t, builtins.type):
                if isinstance(x, t): return True
            if isinstance(x, dtype(t)): return True
        return False

class avoid(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, origin):
        type_name = '!' + origin.__name__.split('.')[-1]
        return super().__new__(cls, type_name, (builtins.object,), builtins.dict(origin=origin))
    
    def __init__(self, origin): ...
    
    def __instancecheck__(self, x):
        return not isinstance(x, self.origin)
    def __subclasscheck__(self, t):
        return not issubclass(t, self.origin)
    
    def __str__(self): return self.__name__

class union(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, *types):
        types = tuple(t if isinstance(t, builtins.type) else tag(str(t)) for t in types)
        type_name = 'union[' + ', '.join(t.__name__.split('.')[-1] for t in types) + ']'
        return super().__new__(cls, type_name, (builtins.object,), builtins.dict(types=types))
    
    def __init__(self, *types): ...
    
    def __instancecheck__(self, x):
        return isinstance(x, self.types)
    def __subclasscheck__(self, t):
        return issubclass(t, self.types)
    
    def __str__(self): return self.__name__

class tag(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, text_tag):
        type_name = f"tag[{text_tag}]"
        return super().__new__(cls, type_name, (builtins.object,), builtins.dict(tag=text_tag))

    def __init__(self, text_tag): ...
    
    def __instancecheck__(self, x):
        return any(self.tag in getattr(t, '__qualname__', str(t)).split('.') for t in x.__class__.__mro__)
    def __subclasscheck__(self, t):
        return isinstance(t, builtins.type) and any(self.tag in getattr(it, '__qualname__', str(it)).split('.') for it in t.__mro__)
    
    def __str__(self): return self.__name__

class class_satisfies(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, is_subclass, is_instance=None):
        type_name = f"check_class_by_[{is_subclass.__name__}]"
        if is_instance is None: is_instance = lambda x: is_subclass(x.__class__)
        return super().__new__(cls, type_name, (builtins.object,), builtins.dict(subclasscheck=is_subclass, instancecheck=is_instance))

    def __init__(self, is_subclass, is_instance=None): ...
    def __call__(self, x): return self.mrois_subclass(x)
    
    def __instancecheck__(self, x):
        return self.instancecheck(x)
    def __subclasscheck__(self, t):
        return self.subclasscheck(t)
    
    def __str__(self): return self.__name__

class instance_satisfies(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, is_instance):
        type_name = f"check_instance_by_[{is_instance.__name__}]"
        return super().__new__(cls, type_name, (builtins.object,), builtins.dict(instancecheck=is_instance))

    def __init__(self, is_instance): ...
    def __call__(self, x): return self.instancecheck(x)
    
    def __instancecheck__(self, x):
        return self.instancecheck(x)
    def __subclasscheck__(self, t):
        return builtins.NotImplementedError
    
    def __str__(self): return self.__name__

### >>> basic element objects <<< ###

class object(builtins.object, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'object' is not regarded iterable")
        return ArrayType(iterable, (builtins.object, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.object in t.__mro__

class bool(builtins.int, metaclass=ABCMeta):
    
    @builtins.staticmethod
    def __new__(cls, arg=None):
        return super().__new__(cls, builtins.bool(arg))
    
    def __and__(self, y): return builtins.bool(self) and y
    def __rand__(self, x): return x and builtins.bool(self)
    def __or__(self, y): return builtins.bool(self) or y
    def __ror__(self, x): return x or builtins.bool(self)
    def __xor__(self, y): return builtins.bool(self).__xor__(y)
    def __rxor__(self, x): return builtins.bool(self).__rxor__(x)
    
    def __repr__(self): return builtins.bool(self).__repr__()
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'bool' is not iterable")
        return ArrayType(iterable, (builtins.bool, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.bool in t.__mro__

class int(builtins.int, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_and__(cls, other): print("in it!!")
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'int' is not iterable")
        return ArrayType(iterable, (builtins.int, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.int in t.__mro__

class long(builtins.int, metaclass=ABCMeta):
    
    __dtype__ = 'int64'
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'int' is not iterable")
        return ArrayType(iterable, (long, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.int in t.__mro__

class float(builtins.float, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'float' is not iterable")
        return ArrayType(iterable, (builtins.float, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.float in t.__mro__

class double(builtins.float, metaclass=ABCMeta):
    
    __dtype__ = 'float64'
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'float' is not iterable")
        return ArrayType(iterable, (double, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.float in t.__mro__

class real(builtins.property, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'real' is not iterable")
        return ArrayType(iterable, (real, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.int in t.__mro__ or builtins.float in t.__mro__

class complex(builtins.complex, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'complex' is not iterable")
        return ArrayType(iterable, (builtins.complex, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.complex in t.__mro__

scalar = ScalarType()
bool_scalar = ScalarType(bool)
int_scalar = ScalarType(int)
long_scalar = ScalarType(long)
float_scalar = ScalarType(float)
double_scalar = ScalarType(double)
real_scalar = ScalarType(real)
complex_scalar = ScalarType(complex)

class property(builtins.property, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'property' is not iterable")
        return ArrayType(iterable, (builtins.property, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.property in t.__mro__

class dtype(metaclass=ABCMeta):

    @builtins.staticmethod
    def __new__(cls, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, builtins.str):
                module, *_, _dtype = arg.split('.')[0] if '.' in arg else [None, arg]
            elif isinstance(arg, builtins.type):
                module = getattr(arg, '__module__', None)
                _dtype = arg
                if any(getattr(t, '__module__', '').startswith('pyoverload') for t in arg.__mro__):
                    module = None
                    _dtype = getattr(arg, '__dtype__', arg)
                if not isinstance(_dtype, str) and any(getattr(t, '__module__', '').startswith('builtins') for t in arg.__mro__):
                    module = None
                    _dtype = arg.__name__.split('.')[-1]
            elif hasattr(arg, 'dtype') and isinstance(arg.dtype, dtype):
                module = getattr(arg.__class__, '__module__', None)
                _dtype = getattr(arg, 'dtype')
            elif isinstance(arg, dtype):
                module = getattr(arg.__class__, '__module__', None)
                _dtype = arg
            else: raise TypeError(f"Unrecognized dtype '{arg}' (of type {builtins.type(arg)})")

            class _this_dtype(builtins.type):

                def __getitem__(self, index):
                    class _sized_dtype(builtins.type):
                        def __instancecheck__(self, x):
                            if self.module is not None and self.module not in getattr(x.__class__, '__module__', ''): return False
                            if not hasattr(x, 'shape'): return False
                            if not (builtins.type(x.shape)(index) == x.shape or _touch(lambda: index == builtins.tuple(x.shape))): return False
                            if isinstance(self.dtype, builtins.type):
                                return isinstance(x[(0,) * len(x.shape)], self.dtype) or isinstance(x, self.dtype)
                            elif isinstance(self.dtype, dtype):
                                if not hasattr(x, 'dtype'): return False
                                return x.dtype == self.dtype
                            elif isinstance(self.dtype, builtins.str):
                                return self.dtype in builtins.str(getattr(x, 'dtype', ''))
                            else: raise TypeError(f"Invalid dtype of type {builtins.type(self.dtype)} read in dtype({self.dtype})[{builtins.str(index).strip('()')}]")
                        def __subclasscheck__(self, t): return self.module is None or self.module in getattr(t, 'module', getattr(t, '__module__', ''))
                    repr_dtype = repr(self.dtype).split('.')[-1]
                    return _sized_dtype(f"dtype[{repr_dtype}:{repr(index).strip('()')}]", (builtins.object,), builtins.dict(module = self.module, dtype = self.dtype))
                
                def __instancecheck__(self, x): return self[builtins.tuple()].__instancecheck__(x)
                def __subclasscheck__(self, t): return self.module is None or self.module in getattr(t, 'module', getattr(t, '__module__', ''))

            repr_dtype = repr(_dtype).split('.')[-1]
            return _this_dtype(f"dtype[{repr_dtype}]", (builtins.object,), builtins.dict(module = None if module is None else module.split('.')[0], dtype = _dtype))
        return super().__new__(cls, *args)
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'dtype' is not iterable")
        return ArrayType(iterable, (dtype, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t): return isinstance(t, builtins.type) and 'dtype' in t.__name__

class type(metaclass=ABCMeta):

    @builtins.staticmethod
    def __new__(cls, *args):
        if len(args) == 1: return args[0].__class__
        return super().__new__(cls, *args)
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'type' is not iterable")
        return ArrayType(iterable, (builtins.type, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.type in t.__mro__

class_type = type

class super(builtins.super, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'super' is not iterable")
        return ArrayType(iterable, (super, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return super in t.__mro__

class null(metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'null' is not iterable")
        return ArrayType(iterable, (builtins.type(None), index))

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.type(None) in t.__mro__

### >>> functions and methods <<< ###

callable = instance_satisfies(builtins.callable)
# class callable(metaclass=ABCMeta):

#     @builtins.staticmethod
#     def __new__(cls, *args):
#         if len(args) == 1: return builtins.callable(args[0])
#         return super().__new__(cls, *args)
    
#     @builtins.classmethod
#     def __class_getitem__(cls, index):
#         if not isinstance(index, builtins.int):
#             raise TypeError("type 'callable' is not iterable")
#         return ArrayType(iterable, (callable, index))
    
#     @builtins.classmethod
#     def __subclasshook__(cls, t):
#         return isinstance(t, builtins.type) and hasattr(t, '__call__')

class functional(metaclass=ABCMeta):

    @builtins.staticmethod
    def __new__(cls, *args):
        if len(args) == 1: return isinstance(args[0], functional)
        return super().__new__(cls, *args)
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'functional' is not iterable")
        return ArrayType(iterable, (functional, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return isinstance(t, builtins.type) and not issubclass(t, builtins.type) and hasattr(t, '__call__')

class lambda_func(metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'function' is not iterable")
        return ArrayType(iterable, ((lambda:...).__class__, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return (lambda:...) in t.__mro__
    
    def __call__(self, *args, **kwargs): ...

class method(metaclass=ABCMeta):

    @builtins.staticmethod
    def __new__(cls, *args):
        if len(args) == 2: return __info__.__enter__.__class__(*args)
        return super().__new__(cls, *args)
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'method' is not iterable")
        return ArrayType(iterable, (__info__.__enter__.__class__, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return __info__.__enter__.__class__ in t.__mro__

class function(metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'function' is not iterable")
        return ArrayType(iterable, (avoid.__str__.__class__, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return avoid.__str__.__class__ in t.__mro__
    
    def __call__(self, *args, **kwargs): ...

class builtin_function_or_method(metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'builtin_function_or_method' is not iterable")
        return ArrayType(iterable, (builtins.int.__new__.__class__, index))

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.int.__new__.__class__ in t.__mro__

class method_descriptor(metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'method_descriptor' is not iterable")
        return ArrayType(iterable, (builtins.int.to_bytes.__class__, index))

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.int.to_bytes.__class__ in t.__mro__

class method_wrapper(metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'method_wrapper' is not iterable")
        return ArrayType(iterable, (''.__str__.__class__, index))

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return ''.__str__.__class__ in t.__mro__

class classmethod(builtins.classmethod, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'classmethod' is not iterable")
        return ArrayType(iterable, (builtins.classmethod, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.classmethod in t.__mro__

class staticmethod(builtins.staticmethod, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("type 'staticmethod' is not iterable")
        return ArrayType(iterable, (builtins.staticmethod, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.staticmethod in t.__mro__

### >>> fixed element sequences <<< ###

class str(builtins.str, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("cannot define the type of elements in 'str'")
        return ArrayType(iterable, (builtins.str, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.str in t.__mro__

class bytes(builtins.bytes, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("cannot define the type of elements in 'bytes'")
        return ArrayType(iterable, (builtins.int, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.bytearray in t.__mro__

class bytearray(builtins.bytearray, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        if not isinstance(index, builtins.int):
            raise TypeError("cannot define the type of elements in 'bytearray'")
        return ArrayType(iterable, (builtins.int, index))
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.bytearray in t.__mro__

### >>> array or sequences <<< ###

class array(metaclass=ABCMeta):

    @builtins.classmethod
    def __class_getitem__(cls, index):
        """
        'array' is defined by subscripts ['type', element type such as 'dtype' and 'shape'].
            e.g. array[np.ndarray, np.int16, 3, 4] or array[torch.Tensor, 5, 8, 5]. 
        """
        if not isinstance(index, builtins.tuple): index = (index,)
        base_type = builtins.object
        base_dtype = scalar
        base_size = []
        stage = 0
        for arg in index:
            if stage == 0 and isinstance(arg, builtins.type) and not isinstance(arg, dtype):
                base_type = arg
                stage = 1
                continue
            elif stage == 0 and isinstance(arg, builtins.slice):
                if arg.step is not None: raise TypeError("Cannot identify 'array' type with two colons in subscript")
                base_type = arg.start
                base_dtype = dtype(arg.stop)
                stage = 2
                continue
            elif stage <= 1 and (isinstance(arg, dtype) or _touch(lambda: dtype(arg))):
                base_dtype = dtype(arg)
                stage = 2
                continue
            else: base_size.append(arg)
        if len(base_size) == 0: base_size = [-1]
        return ArrayType(base_type, (base_dtype, *base_size))

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return isinstance(t, builtins.type) and hasattr(t, 'shape') and hasattr(t, 'dtype')

class iterable(metaclass=ABCMeta):

    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(iterable, index)

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return isinstance(t, builtins.type) and (hasattr(t, '__iter__') or hasattr(t, '__len__') or hasattr(t, '__getitem__'))

class sequence(metaclass=ABCMeta):

    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(sequence, index)

    @builtins.classmethod
    def __subclasshook__(cls, t):
        return isinstance(t, builtins.type) and (hasattr(t, '__iter__') and hasattr(t, '__len__'))

class list(builtins.list, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __not__(cls): return not_list
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.list, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.list in t.__mro__

class tuple(builtins.tuple, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.tuple, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.tuple in t.__mro__

class dict(builtins.dict, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.dict, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.dict in t.__mro__

class set(builtins.set, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.set, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.set in t.__mro__

class reversed(builtins.reversed, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.reversed, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.reversed in t.__mro__

class frozenset(builtins.frozenset, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.frozenset, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.frozenset in t.__mro__

### >>> generators <<< ###

class zip(builtins.zip, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.zip, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.zip in t.__mro__

class enumerate(builtins.enumerate, metaclass=ABCMeta):
    
    @builtins.classmethod
    def __class_getitem__(cls, index):
        return ArrayType(builtins.enumerate, index)
    
    @builtins.classmethod
    def __subclasshook__(cls, t):
        return builtins.enumerate in t.__mro__

if __name__ == "__main__":
    ...