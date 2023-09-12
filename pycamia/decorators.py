
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File of important decorators.",
    help = "Use `@decorator` before functions. "
)

__all__ = """
    alias
    restore_type_wrapper
""".split()

from functools import wraps
from .environment import get_environ_vars

class alias: 
    """
    Create an alias function of the decorated function. 

    Note: 
        One can also use it to return the value after assignment in python<3.9.

    Example:
    ----------
    >>> @other_wrappers # wrappers for function `func_b` only. 
    ... @alias("func_a", b=1) # wrappers between two aliases are for functions `func_a` and `func_b`.
    ... @alias("func_c", b=2)
    ... @some_wrappers # wrappers for functions `func_a`, `func_b` and `func_c`.
    ... def func_b(x, b):
    ...     print(x+b)
    ... 
    >>> func_a(1), func_b(2, 4), func_c(7)
    (2, 6, 9)
    >>> def compose(f, g):
    ...     return lambda x: f(g(x))
    >>> compose(alias("square")(lambda x: x**2), square)(3)
    81
    
    Extension:
    ----------
    >>> a = alias('b')(10)
    >>> a, b
    (10, 10)
    >>> class A:
    ...     @alias('a')
    ...     @property
    ...     def name(self): return 1
    ...     @alias('a')
    ...     @name.setter
    ...     def name(self, value): pass
    ...
    >>> A().a, A().name
    (1, 1)
    >>> A().a = 2
    """
    def __init__(self, *names, **kwargs): self.names = names; self.kwargs = kwargs
    def __call__(self, func):
        vars = get_environ_vars()
        if len(self.kwargs) > 0:
            @wraps(func)
            def wrapper(*args, **kwargs):
                kwargs.update(self.kwargs)
                return func(*args, **kwargs)
        else: wrapper = func
        for n in self.names:
            vars[n] = wrapper
        return func

def _restore_type_wrapper(func, special_attr):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if len(args) == 0: return ret
        if str(type(args[0])) in func.__qualname__ and len(args) > 1: totype = type(args[1])
        else: totype = type(args[0])
        constructor = totype
        if "numpy.ndarray" in str(totype):
            import numpy as np
            constructor = np.array
        elif "batorch" in str(totype):
            import batorch as bt
            constructor = lambda x: x.as_subclass(bt.Tensor) if isinstance(x, torch.Tensor) else bt.tensor(x)
        elif "torch.Tensor" in str(totype):
            import torch
            constructor = lambda x: x.as_subclass(torch.Tensor) if isinstance(x, torch.Tensor) else torch.tensor(x)
        if not isinstance(ret, tuple): ret = (ret,)
        output = tuple()
        for r in ret:
            try: new_r = constructor(r)
            except: new_r = r
            for a in special_attr:
                if a in dir(r): exec(f"new_r.{a} = r.{a}")
            output += (new_r,)
        if len(output) == 1: output = output[0]
        return output
    return wrapper

def restore_type_wrapper(*args):
    """
    Restore type for array/tensor objects, as the first input. 
    """
    if len(args) == 1 and callable(args[0]):
        return _restore_type_wrapper(args[0], [])
    else:
        def restore_type_decorator(func):
            return _restore_type_wrapper(func, args)
        return restore_type_decorator

