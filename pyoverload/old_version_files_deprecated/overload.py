
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "pyoverload",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "overload functions",
    requires = ""
)

__all__ = """
    overload
    OverloadError
""".split()

from functools import wraps
# from .utils import decorator, get_environ_vars
from .typehint import typehint
from .typehint import TypeHintError, HintTypeError, get_virtual_declaration
from .typings import *

class OverloadError(Exception): ...

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError(f"@decorator wrapping a non-wrapper: {wrapper_func}")
    def wrapper(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) or \
            len(args) == 2 and callable(args[1]):
            func = args[-1]
            raw_func = func.__func__ if isinstance(func, method) else func
            func_name = f"{raw_func.__name__}:{wrapper_func.__name__}"
            outer_func = wraps(raw_func)(wrapper_func(*args))
            outer_func.__name__ = func_name
            outer_func.__doc__ = raw_func.__doc__
            if isinstance(func, staticmethod): trans = staticmethod
            elif isinstance(func, classmethod): trans = classmethod
            else: trans = lambda x: x
            return trans(outer_func)
        return decorator(wrapper_func(*args, **kwargs))
    return wraps(wrapper_func)(wrapper)

def wrapped(f):
    while hasattr(f, '__wrapped__'):
        f = f.__wrapped__
    return f

class Overloader:

    def __init__(self):
        """@overload decorator for overloading a function with multiple usages.
        The usages are given by annotations after colons.
        
        @overload find the first suitable function to run according to the order of definition.

        Examples::
            >>> @overload
            ... def func(self, pos1, pos2: str="def_val1", / , arg3: (int, float)=3, arg4=[], *args: int, kwarg5: str='extra', **kwargs) -> str:
            ...     ... # functional content
            ...
            >>> @overload
            ... def func(self, pos1, pos2: int=1, / , arg3: (int, float)=3, arg4=[], *args: int, kwarg5: str='extra', **kwargs) -> str:
            ...     ... # functional content
            ...
            >>> fun(self, 10, 20) # call the second function
        """
        self.prev_var_name = None
        self.funcset_collected = {}

    @decorator
    def __register__(self, func):
        var_name = (func.__func__ if isinstance(func, method) else func).__qualname__
        if var_name.endswith('._'):
            if self.prev_var_name is None:
                raise NameError("Cannot use '_' as a function name of the first @overload.")
            var_name = self.prev_var_name
            var_name_candidates = [var_name]
        else:
            var_name_candidates = [var_name]
            parts__ = var_name.split('__')
            if len(parts__) >= 2 and parts__[-1]: var_name_candidates.append('__'.join(parts__[:-1]))
            if len(parts__) >= 3 and parts__[-2] and not parts__[-1]: var_name_candidates.extend(['__'.join(parts__[:-2] + ['']), '__'.join(parts__[:-2])])
            parts_ = var_name.split('_')
            if len(parts_) >= 3 and parts_[-2] and not parts_[-1]: var_name_candidates.extend(['_'.join(parts_[:-2] + ['']), '_'.join(parts_[:-2])])
        
        if not func.__name__.endswith(':typehint'): func = typehint(func, check_return=True)
        func_index = None
        for v in var_name_candidates:
            if v in self.funcset_collected:
                if v != var_name: func_index = len(self.funcset_collected[v])
                self.funcset_collected[v].append(func)
                var_name = v
                break
        else:
            self.funcset_collected[var_name] = [func]
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.__call__(*args, __cur__func__ = (var_name, func_index), **kwargs)
        return wrapper

    def __call__(self, *args, __cur__func__=None, **kwargs):
        if len(args) == 1 and not kwargs and not __cur__func__ and isinstance(args[0], callable):
            return self.__register__(args[0])
        if __cur__func__ is None: raise TypeError("Invalid @overload sub-function without keyword argument '__cur__func__'")
        if __cur__func__[1] is not None:
            # direct call
            func = self.funcset_collected[__cur__func__[0]][__cur__func__[1]]
            try: return func(*args, **kwargs)
            except (HintTypeError, TypeHintError) as e:
                args_str = ', '.join([repr(x) for x in args] + ['='.join((str(x[0]), repr(x[1]))) for x in kwargs.items()])
                raise OverloadError(f"Failed in direct call of {wrapped(func).__name__}() with arguments {args_str}. The usage is:\n" + get_virtual_declaration(wrapped(func)) + f": [{e.__class__.__name__}] {str(e)}")
        func_list = self.funcset_collected[__cur__func__[0]]
        
        failed_func_list = []
        for i, func in enumerate(func_list):
            # return func(*args, **kwargs)
            try: return func(*args, **kwargs)
            except (TypeError, HintTypeError, TypeHintError) as e:
                if e.__class__.__name__.startswith('Hint'): error_name = e.__class__.__name__[4:]
                else: error_name = e.__class__.__name__
                failed_func_list.append((func, f": [{error_name}] {str(e)}"))
        args_str = ', '.join([repr(x) for x in args] + ['='.join((str(x[0]), repr(x[1]))) for x in kwargs.items()])
        raise OverloadError(f"No {__cur__func__[0]}() matches arguments {args_str}. All available usages are:\n" + '\n'.join([get_virtual_declaration(wrapped(f)) + e for f, e in failed_func_list]))

overload = Overloader()

# @decorator
# def overload(func):
#     # Usage @overload
#     func_name = func.__name__.split('[')[0] # function name without decorator sign
#     base_func_name = [func_name[:len(func_name)-len(tag)] for tag in ('_0', "__default", '') if func_name.endswith(tag)][0] # raw function name without default tag
#     local_vars = get_environ_vars(pivot='overload', offset=2).locals
#     overload_index = f"__{base_func_name}_overload_index__"
#     overload_function = f"__{base_func_name}_overload_function__"
#     if overload_index in local_vars:
#         local_vars[overload_index] = local_vars[overload_index] + 1
#         new_name = f"__{base_func_name}_overload{local_vars[overload_index]}__"
#         local_vars[new_name] = local_vars[overload_function](func)
#     else:
#         local_vars[overload_index] = 0
#         local_vars[overload_function] = overload_wrapper(func)
#     exec(f"def {base_func_name}(*args, **kwargs): return {overload_function}(*args, **kwargs)", local_vars)
#     return local_vars[base_func_name]

# class overload_kernel:

#     def __init__(self, argfunc, first_as_default_backup=False):
#         self.main_func = argfunc
#         self.func_list = [argfunc]
#         if isinstance(argfunc, method): argfunc = argfunc.__func__
#         func_name = argfunc.__name__.split('[')[0]
#         self.base_func_name = [func_name[:len(func_name)-len(tag)] for tag in ('_0', "__default", '') if func_name.endswith(tag)][0]
#         if func_name.endswith('_0') or func_name.endswith("__default"): self.default = 0
#         else: self.default = None
#         self.first_as_default_backup = first_as_default_backup

#     def __call__(self, *args, **kwargs):
#         # Usage 1: call to register a new function
#         if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], functional):
#             argfunc = args[0]
#             func_name = argfunc.__name__.split('[')[0]
#             if func_name == '_' or func_name.startswith(self.base_func_name):
#                 if func_name.endswith('_0') or func_name.endswith("__default"):
#                     raise OverloadError("Only one default function is acceptable, please pay attention to function names ending with '_0' or '__default'. ")
#                     self.default = len(self.func_list)
#                 self.func_list.append(argfunc)
#         # Usage 2: call to find the proper overload and run
#         else:
#             declaration_list = []
#             if self.first_as_default_backup and self.default is None: self.default = 0
#             for i, func in list(enumerate(self.func_list)) + ([(-1, self.func_list[self.default])] if self.default is not None else []):
#                 if i == self.default: continue
#                 try:
#                     typehint(func, check_annot_only=True)(*args, **kwargs)
#                     return typehint(func, check_return=True)(*args, **kwargs)
#                 except (TypeError, TypeHintError) as e:
#                     error_str = f": [{e.__class__.__name__}] {str(e)}"
#                     if i == -1: declaration_list = [get_virtual_declaration(func) + error_str] + declaration_list
#                     else: declaration_list.append(get_virtual_declaration(func) + error_str)
#             args_str = ', '.join([repr(x) for x in args] + ['='.join((str(x[0]), repr(x[1]))) for x in kwargs.items()])
#             raise OverloadError(f"No {self.base_func_name}() matches arguments {args_str}. All available usages are:\n" + '\n'.join(declaration_list))

# @decorator
# def overload_wrapper(func):
#     # Usage: @overload_wrapper
#     if not isinstance(func, functional): raise TypeError("Wrong usage of @overload_wrapper: need to be used as a decorator. ")

#     overloaded_func = overload_kernel(func)
#     @wraps(func)
#     def final_wrapper(*args, **kwargs): return overloaded_func(*args, **kwargs)
#     return final_wrapper
