
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
    as_type
    get_type_name
    ArrayType

    type class_ class_type
    dtype
    union
    intersection intersect insct
    avoid
    tag
    note

    class_satisfies type_satisfies
    instance_satisfies
    
    object
    bool
    short
    int
    long
    half
    float
    double
    rational
    real
    complex
    number
    scalar
    property
    slice
    null
    
    callable
    functional
    lambda_ lambda_func
    method
    function
    builtin_function_or_method
    method_descriptor
    method_wrapper
    generator_function
    classmethod
    staticmethod
    
    str
    bytearray
    bytes
    memoryview
    map
    filter
    
    array
    iterable
    sequence
    list
    tuple
    dict
    set
    reversed
    frozenset

    range
    generator
    
    zip
    enumerate
""".split()

import re, random, copy, builtins
with __info__:
    from pycamia import alias, avouch, crashed

### >>> basic type functions <<< ###

def _touch(f, d=None):
    try: return f()
    except Exception: return d

def as_type(t: (type, tuple, str, None)):
    if isinstance(t, builtins.type): return t
    elif isinstance(t, builtins.tuple): return union(*t)
    elif isinstance(t, builtins.str): return tag(t)
    elif t is None: return builtins.object
    else: raise TypeError(f"Invalid 'type' {repr(t)}. ")

def get_type_name(x: (type, tuple, str, None)):
    if isinstance(x, builtins.type): return x.__name__
    elif isinstance(x, builtins.str): return repr(x)
    elif isinstance(x, builtins.tuple): return repr(builtins.tuple(get_type_name(i) for i in x))
    elif isinstance(x, builtins.slice): return f"{get_type_name(x.start)}:{get_type_name(x.stop)}"
    elif x is ...: return '...'
    elif x is None: return '-'
    else: raise TypeError(f"Invalid 'type' {repr(x)}. ")

### >>> basic type classes <<< ###

@alias("class_", "class_type")
class type(builtins.type):
    
    @builtins.staticmethod
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs: return args[0].__class__
        elif len(args) == 0 and 'name' not in kwargs or len(args) > 3:
            raise TypeError("type() takes 1 or 3 arguments")
        t_name = args[0] if len(args) >= 1 else kwargs['name']
        bases = args[1] if len(args) >= 2 else kwargs.pop('bases', (builtins.object,))
        dict = args[2] if len(args) >= 3 else (kwargs['dict'] if 'dict' in kwargs else kwargs)
        parent = None
        if not isinstance(bases, builtins.tuple):
            parent = bases
            bases = bases.__mro__
        elif bases[-1] != builtins.object: bases = bases + (builtins.object,)
        if bases[0].__class__ != builtins.type: bases = (builtins.object,)
        # dict may have 'is_iterable' for iterability test.
        self = super().__new__(cls, t_name, bases, dict)
        if 'is_iterable' not in dict: self.is_iterable = False
        self.properties = dict
        self.parent = dict.get('parent', parent)
        self.locked = False
        return self
    
    def __init__(self, *args, **kwargs): ...

    def __getitem__(self, index: (list, tuple[(type, slice, int, object)])) -> 'pyoverlaod.ArrayType':
        if self.locked: raise self.lock_error
        if '__getitem__' in self.properties: return self.properties['__getitem__'](self, index)
        if isinstance(index, builtins.list):
            index_names = ', '.join(get_type_name(t) for t in index)
            array_name = f"{get_type_name(self)}[{len(index)}]:[{index_names}]"
            avouch(self.is_iterable, f"Invalid subscripting {self.__name__} with list subscript for non-iterable object.")
            avouch(all(isinstance(x, builtins.type) for x in index if x != ...), f"types list for {array_name} contains non-type element. ")
            return ArrayType(array_name, self, element_types=index)
        if not isinstance(index, builtins.tuple): index = (index,)
        base_type = []
        base_size = None
        type_names = []
        for i in index:
            if isinstance(i, (builtins.type, builtins.tuple, builtins.str)):
                base_type.append(as_type(i))
                type_names.append(get_type_name(i))
            elif i is None:
                base_type.append(builtins.object)
                type_names.append('-')
            elif isinstance(i, builtins.slice):
                avouch(i.step is None, f"Invalid subscripting {self.__name__}[..., {get_type_name(i.start)}:{get_type_name(i.step)}:{get_type_name(i.stop)}, ...] with two colons in ArrayType subscript.")
                base_type.append(as_type(i.start))
                base_type.append(as_type(i.stop))
                type_names.append(get_type_name(i))
            else: break
        base_type = builtins.tuple(base_type)
        if len(base_type) == 0: base_type = None
        base_size = index[len(type_names):] if len(index) > len(type_names) else None
        size_name = '' if base_size is None else (f"[{base_size[0]}]" if len(base_size) == 1 else ('[-]' if len(base_size) == 0 else f"[{repr(base_size).strip('()')}]"))
        size_name = size_name.replace("Ellipsis", '...')
        if not self.is_iterable:
            avouch(len(type_names) == 0, 
                   f"cannot define the type of elements in '{self.__name__}', one can only use {self.__name__}[{{a size}}] to create iterable type with elements of type {self.__name__}. " if hasattr(getattr(self, 'parent', None), '__len__') else
                   f"type '{self.__name__}' is not iterable, one can only use {self.__name__}[{{a size}}] to create iterable type. ")
            return ArrayType(f"{self.__name__}{size_name}", iterable, base_type=self, base_size=base_size)
        if isinstance(self, ArrayType):
            origin_type = self.origin_type
            if base_type is not None: avouch(self.base_type is None, f"{self.__name__}[{repr(index)}] has conflict in element type: multiple types defined by two level subscripts.")
            elif self.base_type is not None:
                base_type = self.base_type
                type_names = [get_type_name(t) for t in base_type]
            if self.base_size is not None:
                if not self.can_add_size: raise TypeError(f"Compound ArrayType {self.__name__} can not accept more size information in subscript, please ensure that only size of 1 dimension in each pair of brackets")
                if base_size is not None:
                    if len(self.base_size) > 0 and len(base_size) > 1: raise TypeError(f"Please add size of one dimension a time, adding {len(base_size)} at a time is invalid: {self.__name__}[{repr(base_size).strip('()')}]")
                    if len(base_size) == 1 and base_size[0] == ...: base_size = (-1,)
                    if len(self.base_size) == 1 and self.base_size[0] == ...: base_size = (-1,) + base_size
                    else: base_size = self.base_size + base_size
                else: base_size = self.base_size
                size_name = f"[{base_size[0]}]" if len(base_size) == 1 else ('[-]' if len(base_size) == 0 else f"[{repr(base_size).strip('()')}]")
                size_name = size_name.replace("Ellipsis", '...')
        else: origin_type = self
        type_names = ': '.join(type_names)
        if not type_names: type_names = ''
        else: type_names = f'<{type_names}>'
        array_name = f"{get_type_name(origin_type)}{type_names}{size_name}"
        return ArrayType(array_name, origin_type, base_type=base_type, base_size=base_size, 
                         can_add_size=self.can_add_size if isinstance(self, ArrayType) else (base_size is None or len(base_size) <= 1))

    def __lshift__(self, ele_type): return self[ele_type].with_lock(True)
    def __rshift__(self, size):
        if self.locked:
            self.locked = False
            return self[*size]
        else: raise TypeError("ending tag '>>' before starting '<<'. ")

    def __invert__(self): return avoid(self)
    @alias('__rand__', '__iand__', '__mul__', '__rmul__', '__imul__')
    def __and__(self, other): return intersection(self, other)
    @alias('__ror__', '__ior__', '__add__', '__radd__', '__iadd__')
    def __or__(self, other): return union(self, other)
    @alias('__itruediv__', '__sub__', '__isub__')
    def __truediv__(self, other): return self & ~other
    @alias('__rsub__')
    def __rtruediv__(self, other): return other & ~self

    def __instancecheck__(self, x):
        if self.locked: raise self.lock_error
        if '__subclasshook__' in self.properties: return self.properties['__subclasshook__'](self, x.__class__)
        if '__instancecheck__' in self.properties: return self.properties['__instancecheck__'](self, x)
        if '__subclasscheck__' in self.properties: return self.properties['__subclasscheck__'](self, x.__class__)
        dtype_correct = False
        if not isinstance(x, builtins.type) and hasattr(x, 'shape') and hasattr(x, 'dtype') and len(x.shape) == 0:
            dtype_correct = issubclass(dtype(x.dtype), dtype(getattr(self, '__dtype__', self)))
        if self.parent is not None: return isinstance(x, self.parent) or x.__class__ == self or dtype_correct
        else: return dtype_correct
        return self in x.__class__.__mro__

    def __subclasscheck__(self, t):
        if self.locked: raise self.lock_error
        if '__subclasshook__' in self.properties: return self.properties['__subclasshook__'](self, t)
        if '__subclasscheck__' in self.properties: return self.properties['__subclasscheck__'](self, t)
        avouch(isinstance(t, builtins.type) or isinstance(t, dtype), f"Non type input for subclass check for {self.__name__}: {t}.")
        dtype_correct = False
        if isinstance(t, dtype): dtype_correct = issubclass(t, dtype(getattr(self, '__dtype__', self)))
        if self.parent is not None: return issubclass(t, self.parent) or t == self or dtype_correct
        else: return dtype_correct
        print(f"Waring: performing brutal check of subclass 'issubclass({t.__name__}, {self.__name__})' as type {self.__name__} does not have implementation '__subclasshook__' or '__subclasscheck__'. ")
        return self in t.__mro__

    def __call__(self, *args, **kwargs):
        if self.locked: raise self.lock_error
        if '__call__' in self.properties: return self.properties['__call__'](self, *args, **kwargs)
        if self.parent is not None: return self.parent(*args, **kwargs)
        if len(self.__mro__) > 2: self.__mro__[1](*args, **kwargs)
        raise NotImplementedError(f"Trying to call type '{self.__name__}', no implementation is found.")

def create_spec_dtype(self, arg):

    if hasattr(self, 'module'): raise TypeError(f"Specified '{self.__name__}' object is not callable")

    if isinstance(arg, builtins.str):
        module, *_, _dtype = arg.split('.')[0] if '.' in arg else [None, arg]
    elif isinstance(arg, builtins.type):
        if any(getattr(t, '__module__', '').startswith('pyoverload') for t in arg.__mro__):
            module = None
            _dtype = getattr(arg, '__dtype__', arg)
        elif any(getattr(t, '__module__', '').startswith('builtins') for t in arg.__mro__ if t != builtins.object):
            module = None
            _dtype = arg.__name__.split('.')[-1]
        else:
            module = getattr(arg, '__module__', None)
            _dtype = arg.__name__.split('.')[-1]
    elif hasattr(arg, 'dtype') and isinstance(arg.dtype, dtype):
        module = getattr(arg.__class__, '__module__', None)
        _dtype = getattr(arg, 'dtype')
    elif isinstance(arg, dtype):
        module = getattr(arg.__class__, '__module__', None)
        _dtype = arg
    else: raise TypeError(f"Unrecognized dtype '{arg}' (of type {builtins.type(arg)})")
    
    if isinstance(_dtype, type): _dtype = _dtype.__name__
    elif hasattr(_dtype, 'name'): _dtype = _dtype.name
    elif isinstance(_dtype, str): ...
    else: _dtype = str(_dtype)
    
    type_infos = re.findall(r"([<a-zA-Z]+)([x0-9]*)", _dtype.split('.')[-1])
    avouch(len(type_infos) == 1, TypeError(f"Unrecognized dtype '{_dtype}'"))
    name, bits = type_infos[0]
    bits = _touch(lambda: int(bits), bits)
    
    if name == 'long':
        avouch(bits is None, TypeError(f"Unrecognized dtype '{arg}': {name}{bits}"))
        name = 'int'
        bits = 64
    elif name == 'short':
        avouch(bits is None, TypeError(f"Unrecognized dtype '{arg}': {name}{bits}"))
        name = 'int'
        bits = 16
    elif name == 'double':
        avouch(bits is None, TypeError(f"Unrecognized dtype '{arg}': {name}{bits}"))
        name = 'float'
        bits = 64

    return type.__new__(dtype, f"dtype[{module + '.' if module is not None else ''}{name}{bits if bits is not None else ''}]", 
        module = module.split('.')[0] if module is not None else module, name = name, bits = bits
    )

def __dtype_subclasscheck__(self, t):
    if dtype not in t.__class__.__mro__: t = dtype(t)
    if self.module is not None and t.module != self.module: return False
    if self.bits is not None and t.bits != self.bits: return False
    return self.name in t.name

dtype = type("dtype", type,
    __call__ = create_spec_dtype,
    __instancecheck__ = lambda self, x: isinstance(x, scalar) and isinstance(x, self[...]),
    __subclasscheck__ = __dtype_subclasscheck__
)
dtype.properties['__instancecheck__'] = lambda self, x: x.__class__ == self or any('dtype' in mt.__name__ for mt in x.__class__.__mro__)
dtype.properties['__subclasscheck__'] = lambda self, t: isinstance(t, builtins.type) and any('dtype' in mt.__name__ for mt in t.__mro__)

class ArrayType(type):
    
    @builtins.staticmethod
    def __new__(cls, array_name, origin_type, **kwargs: 'dict(element_types=tuple[type], base_type=(null, tuple[type]), base_size=(null, tuple[(int, object)]), can_add_size=bool)'):
        avouch(
            'element_types' in kwargs or 'base_type' in kwargs and 'base_size' in kwargs,
            f"Initialization of 'ArrayType' takes either keyword argument 'element_types' or keyword arguments 'base_type' and 'base_size'."
        )
        if 'element_types' in kwargs:
            return super().__new__(cls, array_name, origin_type.__mro__, is_iterable=True, origin_type=origin_type, element_types=kwargs.get('element_types'), can_add_size = False)
        
        base_type = kwargs.get('base_type')
        base_size = kwargs.get('base_size')
        can_add_size = kwargs.get('can_add_size')
        if base_type is not None and not isinstance(base_type, builtins.tuple): base_type = (base_type,)
        avouch(base_type is None or len(base_type) <= 1 or
            len(base_type) == 2 and issubclass(origin_type, builtins.dict) or
            issubclass(origin_type, builtins.zip),
            f"Invalid ArrayType: {array_name}; at most 2 types for dict key-value pair or multiple types for 'zip' in subscript. One should use 'union' or tuple of types to include multiple probabilities, or use a list inside subscript to list the types for each item: e.g. list[[int, float, ...]]. \n" + 
            "P.S. Avoid using a single tuple of types such as 'list[(int, float)]', use 'union' instead or add another comma: 'list[(int, float),]'. "
        )
        if can_add_size is None: can_add_size = base_size is None or len(base_size) <= 1
        return super().__new__(cls, array_name, origin_type.__mro__, is_iterable=True, origin_type=origin_type, base_type=base_type, base_size=base_size, can_add_size=can_add_size)
    
    def __init__(self, array_name, origin_type, **kwargs): ...

    def __instancecheck__(self, x):
        if self.locked: raise self.lock_error
        if not isinstance(x, self.origin_type): return False
        
        if hasattr(self, 'element_types'):
            y = builtins.list(x)
            if ... in self.element_types:
                if len(self.element_types) <= len(y) + 1:
                    i_ellipsis = [i for i, t in enumerate(self.element_types) if t == ...]
                    if len(i_ellipsis) > 1: return False
                    return (
                        all(isinstance(u, t) for u, t in builtins.zip(y[:i_ellipsis[0]], self.element_types[:i_ellipsis[0]])) and 
                        all(isinstance(u, t) for u, t in builtins.zip(y[len(y) - len(self.element_types) + i_ellipsis[0] + 1:], self.element_types[i_ellipsis[0]+1:]))
                    )
                return False
            if len(y) != len(self.element_types):
                i_ellipsis = [i for i, t in enumerate(self.element_types) if getattr(t, 'base_size', None) == (...,)]
                if len(i_ellipsis) != 1: return False
                mid_t = self.element_types[i_ellipsis[0]]
                if mid_t.origin_type == iterable:
                    mid_t = builtins.tuple(mid_t.base_type)
                else: mid_t = mid_t.origin_type
                return (
                    all(isinstance(u, t) for u, t in builtins.zip(y[:i_ellipsis[0]], self.element_types[:i_ellipsis[0]])) and 
                    all(isinstance(u, t) for u, t in builtins.zip(y[len(y) - len(self.element_types) + i_ellipsis[0] + 1:], self.element_types[i_ellipsis[0]+1:])) and 
                    all(isinstance(u, mid_t) for u in y[i_ellipsis[0]:len(y) - len(self.element_types) + i_ellipsis[0] + 1])
                )
            return all(isinstance(u, t) for u, t in builtins.zip(y, self.element_types))
        n_max_ele = 50
        if issubclass(self.origin_type, builtins.dict):
            if self.base_size is not None:
                if len(self.base_size) != 1: raise TypeError(f"in {self}: dict size should be a single integer of total number of elements")
                if self.base_size[0] >= 0 and len(x) != self.base_size[0]: return False
            if self.base_type is None: self.base_type = (builtins.object, builtins.object)
            elif len(self.base_type) == 1: self.base_type = (builtins.object, self.base_type[0])
            if len(x) > n_max_ele: n_keys = n_max_ele
            else: n_keys = len(x)
            return all(isinstance(k, self.base_type[0]) and isinstance(x[k], self.base_type[1]) for k, _ in builtins.zip(x, builtins.range(n_max_ele)))
        generator_types = {
            builtins.zip: lambda x, t: all(isinstance(u, v) for u, v in builtins.zip(x, t)),
            builtins.enumerate: lambda x, t: isinstance(x[1], t[0]),
            generator: lambda x, t: isinstance(x, t) or isinstance(x, tuple) and isinstance(t, tuple) and all(isinstance(u, v) for u, v in builtins.zip(x, t))
        }
        for generator_type, check_generated in generator_types.items():
            if issubclass(self.origin_type, generator_type):
                if self.base_size is not None:
                    if len(self.base_size) != 1: raise TypeError(f"in {self}: {generator_type.__name__} size should be a single integer of total number of elements")
                    if self.base_type is None:
                        try: y = copy.deepcopy(x)
                        except TypeError: raise TypeError(f"Cannot copy the generator, pleas cast it to a list before checking its size. ")
                        if len(builtins.list(y)) != self.base_size[0]: return False
                        return True
                elif self.base_type is None: return True
                try: y = copy.deepcopy(x)
                except TypeError: raise TypeError(f"Cannot copy the generator, pleas cast it to a list before checking its value type. ")
                l = 0
                while True if self.base_size is not None and self.base_size[0] >= 0 else l < n_max_ele:
                    try: sample = next(y)
                    except StopIteration: break
                    if not check_generated(sample, self.base_type): return False
                    l += 1
                if self.base_size is not None and self.base_size[0] >= 0 and self.base_size[0] != l: return False
                return True
        if hasattr(x, 'shape'):
            if self.base_size is not None:
                tests = []
                if ... in self.base_size:
                    i_ellipsis = [i for i, x in enumerate(self.base_size) if x == ...]
                    if len(i_ellipsis) > 1: raise TypeError(f"Only one ellipsis (...) is allowed in size: {repr(self.base_size).replace('Ellipsis', '...')}")
                    tests.append((x.shape[:i_ellipsis[0]], self.base_size[:i_ellipsis[0]]))
                    tests.append((x.shape[len(x.shape)-len(self.base_size)+i_ellipsis[0]+1:], self.base_size[i_ellipsis[0]+1:]))
                else: tests.append((x.shape, self.base_size))
                for a, b in tests:
                    if builtins.type(a)(b) == a: ...
                    elif _touch(lambda: b == builtins.tuple(a)): ...
                    elif len(a) == len(b) and _touch(lambda: all(x == y or y < 0 for x, y in builtins.zip(a, b))): ...
                    else: return False
            if self.base_type is None: return True
            _dtype = getattr(x, 'dtype', None)
            if _dtype is None: raise TypeError(f"Cannot find 'dtype' in examination isinstance({x}, {self}), please contact the developer for more information (Error Code: H123).")
            if len(self.base_type) > 1: raise TypeError(f"{x.__class__.__name__} object can only have one dtype, instead of {len(self.base_type)}: {self.base_type}")
            return issubclass(dtype(_dtype), self.base_type[0])
            # return self.base_type[0] == builtins.object or dtype == self.base_type[0] or self.base_type[0].__name__.split('.')[-1] in builtins.str(dtype)

        x = builtins.list(x)
        if self.base_size is not None:
            if len(self.base_size) != 1: raise TypeError(f"in {self}: array size should be a single integer of total number of elements")
            if self.base_size[0] >= 0 and len(x) != self.base_size[0]: return False
        if self.base_type is None: return True
        if len(self.base_type) > 1: raise TypeError(f"{x.__class__.__name__} object can only have one type, instead of {len(self.base_type)}: {self.base_type}")
        return all(isinstance(u, self.base_type[0]) for u in x)
    
    def __len__(self):
        if hasattr(self, 'element_types'): return len(self.element_types)
        if self.base_size is None: return -1
        p = 1
        for x in self.base_size:
            p *= x
            if x < 0: break
        else: return p
        return -1
    
    @property
    def lock_error(self):
        if self.base_type is None: return TypeError(f"Missing ending note '>>' after {get_type_name(self.origin_type)}<<.")
        base_type_str = ', '.join(get_type_name(t) for t in self.base_type)
        return TypeError(f"Missing ending note '>>' for element-wise type in {get_type_name(self.origin_type)}<<{base_type_str}.")
    def with_lock(self, locked): self.locked = locked; return self

def union(*types):
    if len(types) == 1 and isinstance(types[0], builtins.tuple): types = types[0]
    new_types = []
    for t in types:
        if getattr(t, 'op_type', None) == 'union': new_types.extend(t.types)
        elif isinstance(t, builtins.type): new_types.append(t)
        elif isinstance(t, builtins.tuple): new_types.extend(union(*t).types)
        elif isinstance(t, builtins.str): new_types.append(tag(t))
        elif t is None: new_types.append(builtins.object)
        else: as_type(t)
    return type('[' + ' | '.join(t.__name__ for t in new_types) + ']',
        op_type = 'union',
        types = new_types,
        __instancecheck__ = lambda _, x: isinstance(x, builtins.tuple(new_types)),
        __subclasscheck__ = lambda _, t: issubclass(t, builtins.tuple(new_types))
    )

@alias('intersect', 'insct')
def intersection(*types):
    if len(types) == 1 and isinstance(types[0], builtins.tuple): types = types[0]
    new_types = []
    for t in types:
        if getattr(t, 'op_type', None) == 'intersection': new_types.extend(t.types)
        elif isinstance(t, builtins.type): new_types.append(t)
        elif isinstance(t, builtins.tuple): new_types.append(union(*t)) # this is different to the line 'union' as it cannot be expanded here
        elif isinstance(t, builtins.str): new_types.append(tag(t))
        elif t is None: new_types.append(builtins.object)
        else: as_type(t)
    return type('[' + ' & '.join(t.__name__ for t in new_types) + ']',
        op_type = 'intersection',
        types = new_types,
        __instancecheck__ = lambda _, x: all(isinstance(x, mt) for mt in new_types),
        __subclasscheck__ = lambda _, t: all(issubclass(t, mt) for mt in new_types)
    )

def avoid(origin):
    if getattr(origin, 'op_type', None) == 'avoid': return origin.types
    elif getattr(origin, 'op_type', None) == 'union': return intersection(*[avoid(t) for t in origin.types])
    elif getattr(origin, 'op_type', None) == 'intersection': return union(*[avoid(t) for t in origin.types])
    return type(f"!{origin.__name__}",
        op_type = 'avoid', 
        types = origin,
        __instancecheck__ = lambda _, x: not isinstance(x, origin),
        __subclasscheck__ = lambda _, t: not issubclass(t, origin)
    )

def tag(string):
    return type(f"tag[{string}]",
        __subclasshook__ = lambda _, t: isinstance(t, builtins.type) and any(string in getattr(it, '__qualname__', builtins.str(it)).split('.') for it in t.__mro__)
    )

def note(string):
    return type(f"note[{string}]", __subclasshook__ = lambda _: True)

@alias("type_satisfies")
def class_satisfies(func):
    return type(f"check_class_by_[{func.__qualname__}]",
        __call__ = lambda _, x: func(x),
        __subclasshook__ = lambda _, x: func(x)
    )
    
def instance_satisfies(func):
    return type(f"check_instance_by_[{func.__qualname__}]",
        __call__ = lambda _, x: func(x),
        __instancecheck__ = lambda _, x: func(x)
    )

def __array_getitem__(self, index):
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
    base_size = builtins.tuple(base_size)
    if len(base_size) == 0: base_size = None
    elif len(base_size) == 1 and isinstance(base_size[0], builtins.tuple): base_size = base_size[0]
    size_str = '-' if base_size is None else repr(base_size).strip('(,)')
    return ArrayType(f"{get_type_name(base_type)}<{base_dtype.__name__}>[{size_str}]", base_type, base_type=base_dtype, base_size=base_size)

array = type("array",
    is_iterable = True,
    __getitem__ = __array_getitem__,
    __subclasshook__ = lambda _, t: isinstance(t, builtins.type) and hasattr(t, 'shape') and hasattr(t, 'dtype')
)

### >>> basic element objects <<< ###

object = type("object", builtins.object)
bool = type("bool", builtins.int, parent = builtins.bool)
int = type("int", builtins.int)
short = type("short", builtins.int, __dtype__ = 'int16')
long = type("long", builtins.int, __dtype__ = 'int64')
float = type("float", builtins.float)
half = type("half", builtins.int, __dtype__ = 'float16')
double = type("double", builtins.float, __dtype__ = 'float64')
rational = type("rational", float) # not carefully implemented yet
real = type("real", parent = union(int, float), op_type = 'union', types = [int, float])
complex = type("complex", builtins.complex)
scalar = type("scalar", parent = ArrayType('scalar', array, base_type=None, base_size=builtins.tuple()))
number = type("number", parent = union(scalar, real, complex, rational), op_type = 'union', types = [scalar, real, complex, rational])

property = type("property", builtins.property)
slice = type("slice", parent = builtins.slice)
null = type("null", parent = builtins.type(None))

### >>> functions and methods <<< ###

callable = instance_satisfies(builtins.callable)
functional = type("functional",
    __call__ = lambda _, x: not isinstance(x, builtins.type) and hasattr(x, '__call__'),
    __subclasshook__ = lambda _, t: isinstance(t, builtins.type) and not issubclass(t, builtins.type) and hasattr(t, '__call__')
)
lambda_ = lambda_func = type("lambda", parent = (lambda:...).__class__)
method = type("method", parent = __info__.__enter__.__class__)
function = type("function", parent = avoid.__str__.__class__)
builtin_function_or_method = type("builtin_function_or_method", parent = builtins.int.__new__.__class__)
method_descriptor = type("method_descriptor", parent = builtins.int.to_bytes.__class__)
method_wrapper = type("method_wrapper", parent = ''.__str__.__class__)
def __is_generator_func__(_, func):
    while hasattr(func, '__wrapped__'): func = func.__wrapped__
    return func.__code__.co_flags &0x20
generator_function = type("generator_function", functional, 
    __instancecheck__ = __is_generator_func__
)
classmethod = type("classmethod", builtins.classmethod)
staticmethod = type("staticmethod", builtins.staticmethod)

### >>> fixed element sequences <<< ###

str = type("str", builtins.str)
bytes = type("bytes", builtins.bytes)
bytearray = type("bytearray", builtins.bytearray)
memoryview = type("memoryview", parent = builtins.memoryview)
map = type("map", builtins.map)
filter = type("filter", builtins.filter)

### >>> array or sequences <<< ###

iterable = type("iterable", 
    is_iterable = True,
    __call__ = lambda _, x: hasattr(x, '__iter__') or hasattr(x, '__len__') or hasattr(x, '__getitem__'),
    __subclasshook__ = lambda _, t: isinstance(t, builtins.type) and (hasattr(t, '__iter__') or hasattr(t, '__len__') or hasattr(t, '__getitem__'))
)

sequence = type("sequence",
    is_iterable = True,
    __subclasshook__ = lambda _, t: isinstance(t, builtins.type) and (hasattr(t, '__iter__') and hasattr(t, '__len__'))
)

list = type("list", builtins.list, is_iterable = True)
tuple = type("tuple", builtins.tuple, is_iterable = True)
dict = type("dict", builtins.dict, is_iterable = True)
set = type("set", builtins.set, is_iterable = True)
reversed = type("reversed", builtins.reversed, is_iterable = True)
frozenset = type("frozenset", builtins.frozenset, is_iterable = True)

### >>> generators <<< ###

range = type("range", parent = builtins.range, is_iterable = True)
generator = type("generator", parent = (_ for _ in []).__class__, is_iterable = True)
zip = type("zip", builtins.zip, is_iterable = True)
enumerate = type("enumerate", builtins.enumerate, is_iterable = True)

if __name__ == "__main__":
    ...