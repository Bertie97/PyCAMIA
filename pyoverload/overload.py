
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
    
    deprecated override
""".split()

deprecated = ...
def override(*_a, **_k): raise NotImplementedError("""
[WARNING] pyoverload has upgraded into pyoverload II, please read the new README.md for detailed information.
To simply get your previous codes running, do this: 
from pyoverload.old_version_files_deprecated.override import overload
from pyoverload.old_version_files_deprecated.typehint import *
If you are using the 'decorator fashion' (@overload for all imp's) of pyoverload I, use the following imports of new functions will work.
from pyoverload import overload
from pyoverload import typehint as params
from pyoverload.old_version_files_deprecated.typehint import * # Special types such as Int, etc. still need this line of import. 
Do change the capitalized types into lower_cased ones, and @params into @typehint later.
""")

from functools import wraps
from .typehint import typehint, decorator
from .typehint import TypeHintError, HintTypeError, get_virtual_declaration
from .typings import *

with __info__:
    from pycamia import Path

class OverloadError(Exception): ...

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
        wrapped_func = wrapped(func)
        func_loc = Path(wrapped_func.__code__.co_filename)
        var_name = '.'.join([func_loc.parent.name, func_loc.name, (wrapped_func.__func__ if isinstance(wrapped_func, method) else wrapped_func).__name__])
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
        
        if not ':typehint' in func.__name__: func = typehint(func, check_return=True)
        func_index = None
        for v in var_name_candidates:
            if v in self.funcset_collected:
                if v != var_name: func_index = len(self.funcset_collected[v])
                self.funcset_collected[v].append(func)
                var_name = v
                break
        else:
            if var_name.endswith('__default__'): var_name = var_name[:-len('__default__')]
            elif var_name.endswith('__0__'): var_name = var_name[:-len('__0__')]
            self.funcset_collected[var_name] = [func]
        self.prev_var_name = var_name
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.__call__(*args, __cur__func__ = (var_name, func_index), **kwargs)
        return wrapper

    def __call__(self, *args, __cur__func__=None, **kwargs):
        if len(args) == 1 and not kwargs and not __cur__func__ and isinstance(args[0], callable):
            outer_func = self.__register__(args[0])
            var_name = self.prev_var_name
            declarations = '\n    '.join([get_virtual_declaration(wrapped(f)) for f in self.funcset_collected[var_name]])
            func_doc = [f.__doc__ for f in self.funcset_collected[var_name] if f.__doc__ is not None]
            if func_doc: func_doc = func_doc[0]
            else: func_doc = ''
            outer_func.__doc__ = f"""
@overload registered function at: '{var_name}', with the following usages:
    {declarations}
{func_doc}
            """
            return outer_func
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
            except (HintTypeError, TypeHintError) as e:
                failed_func_list.append((func, f": [{e.__class__.__name__}] {str(e)}"))
        args_str = ', '.join([repr(x) for x in args] + ['='.join((str(x[0]), repr(x[1]))) for x in kwargs.items()])
        raise OverloadError(f"No {__cur__func__[0]}() matches arguments {args_str}. All available usages are:\n" + '\n'.join([get_virtual_declaration(wrapped(f)) + e for f, e in failed_func_list]))

overload = Overloader()
