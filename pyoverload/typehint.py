
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "pyoverload",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Efficient typehint of input argument type for functions. ",
    requires = "pycamia"
)

__all__ = """
    typehint
    TypeHintError
    HintTypeError
    
    get_func_info
    get_arg_values
    get_virtual_declaration
    
    deprecated params
""".split()

deprecated = ...
def params(*_a, **_k): raise NotImplementedError("""
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

from .typings import *
from functools import wraps

def decorator(wrapper_func):
    if not isinstance(wrapper_func, functional):
        raise TypeError(f"@decorator wrapping a non-wrapper: {wrapper_func.__qualname__}")
    def wrapper(*args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], functional) or # functions
            len(args) == 2 and isinstance(args[1], functional)): # methods
            func = args[-1]
            raw_func = func.__func__ if isinstance(func, method) else func
            raw_func = getattr(raw_func, '__wrapped__', raw_func)
            func_name = f"{raw_func.__name__}:{wrapper_func.__name__}"
            outer_func = wraps(raw_func)(wrapper_func(*args, **kwargs))
            outer_func.__name__ = func_name
            outer_func.__doc__ = raw_func.__doc__
            if isinstance(func, staticmethod): trans = staticmethod
            elif isinstance(func, classmethod): trans = classmethod
            else: trans = lambda x: x
            return trans(outer_func)
        return decorator(wrapper_func(*args, **kwargs))
        # raise TypeError(f"Invalid decorator '@{wrapper_func.__qualname__}' for {args}")
    return wraps(wrapper_func)(wrapper)

class TypeHintError(Exception): ...
class HintTypeError(Exception): ...

def get_annot(annotation: (type, tuple, str, None, list, dict)):
    # type, tuple, str, None or list for *args, dict for **kwargs
    if isinstance(annotation, list): return '[' + ', '.join(get_annot(a) for a in annotation) + ']'
    if isinstance(annotation, dict): return '{' + ', '.join(repr(str(k)) + ': ' + get_annot(a) for k, a in annotation.items()) + '}'
    if annotation is ...: return '...'
    if annotation is None: return 'None'
    if isinstance(annotation, str): return repr(annotation)
    if isinstance(annotation, tuple):
        if len(annotation) >= 2: return '(' + ', '.join(get_annot(a) for a in annotation) + ')'
        if len(annotation) == 1: return '(' + get_annot[annotation[0]] + ',)'
        if len(annotation) == 0: return 'tuple()'
    return getattr(annotation, '__name__', str(annotation))

def get_virtual_declaration(func: functional):
    is_method = False
    if isinstance(func, method):
        is_method = True
        while hasattr(func, '__func__'): func = func.__func__
    annotations = func.__annotations__
    f_code = func.__code__
    f_name = func.__name__
    f_defaults = [] if func.__defaults__ is None else func.__defaults__
    f_kwdefaults = {} if func.__kwdefaults__ is None else func.__kwdefaults__
    co_nonvarargcount = f_code.co_argcount + f_code.co_kwonlyargcount
    parts = []
    for i, v in enumerate(f_code.co_varnames):
        if i == f_code.co_posonlyargcount and f_code.co_posonlyargcount > 0: parts.append('/')
        if i == f_code.co_argcount:
            if f_code.co_flags & 0x04:
                argv = f_code.co_varnames[co_nonvarargcount]
                if argv in annotations:
                    term = argv + ': ' + get_annot(annotations[argv])
                else: term = argv
                parts.append('*' + term)
            else: parts.append('*')
        if i >= co_nonvarargcount:
            if f_code.co_flags & 0x08:
                has_args = (f_code.co_flags & 0x04) >> 2
                kwargv = f_code.co_varnames[co_nonvarargcount + has_args]
                if kwargv in annotations:
                    term = kwargv + ': ' + get_annot(annotations[kwargv])
                else: term = kwargv
                parts.append('**' + term)
            break
        if i == 0 and is_method: continue
        if v in annotations:
            term = v + ': ' + get_annot(annotations[v])
        else: term = v
        j = i + len(f_defaults) - f_code.co_argcount
        if 0 <= j < len(f_defaults):
            term += '= ' + repr(f_defaults[j])
        if i >= f_code.co_argcount and v in f_kwdefaults:
            term += '= ' + repr(f_kwdefaults[v])
        parts.append(term)
    dec = f"{f_name}({', '.join(parts)})"
    if is_method: dec = '_.' + dec
    if 'return' in annotations:
        dec += ' -> ' + get_annot(annotations['return'])
    return dec

def get_arg_values(func_info: tuple, *args, **kwargs):
    (   f_name, has_args, has_kwargs, f_varnames,
        co_posonlyargcount, co_argcount, co_nonvarargcount, co_varcount,
        n_defaults, f_defaults, f_kwdefaults
    ) = func_info
    
    n_args = len(args)
    
    if has_kwargs:
        s = co_posonlyargcount
        f_varnames = f_varnames[s:]
        values = list(args[:co_posonlyargcount])
    else:
        s = 0
        values = []
    for i, v in zip(range(s, co_nonvarargcount), f_varnames):
        try:
            value = kwargs.pop(v, None)
            if i < co_posonlyargcount and value is not None:
                raise HintTypeError(f"{f_name}() got some positional-only arguments passed as keyword arguments: '{v}'")
            if i < n_args and i < co_argcount and value is not None:
                raise HintTypeError(f"{f_name}() got multiple values for argument '{v}'")
            if value is None:
                value = f_kwdefaults[v] if co_argcount <= i else (args[i] if i < n_args else f_defaults[i - co_argcount + n_defaults])
        except IndexError:
            missing_args = f_varnames[n_args:co_argcount - n_defaults]
            n_missing = len(missing_args)
            missing_args_str = ', '.join([repr(x) for x in missing_args[:-1]])
            missing_args_str += (' and ' if missing_args_str else '') + repr(missing_args[-1])
            raise HintTypeError(f"{f_name}() missing {n_missing} required position argument{'s' if n_missing > 1 else ''}: {missing_args_str}")
        except KeyError:
            missing_args = [v for v in f_varnames[co_argcount:co_nonvarargcount] if v not in f_kwdefaults]
            n_missing = len(missing_args)
            missing_args_str = ', '.join([repr(x) for x in missing_args[:-1]])
            missing_args_str += (' and ' if missing_args_str else '') + repr(missing_args[-1])
            raise HintTypeError(f"{f_name}() missing {n_missing} required keyword-only argument{'s' if n_missing > 1 else ''}: {missing_args_str}")
        values.append(value)
    if has_args:
        values.append(args[co_argcount:])
    elif n_args > co_argcount:
        raise HintTypeError(f"{f_name}() takes {co_argcount} positional argument but {n_args} were given")
    if has_kwargs: values.append(kwargs)
    elif len(kwargs) > 0:
        n_unexpkw = len(kwargs)
        unexpected_kw = list(kwargs.keys())
        if n_unexpkw == 1: unexpected_kw = repr(unexpected_kw[0])
        else: unexpected_kw = ' and '.join([', '.join([repr(x) for x in unexpected_kw[:-1]]), repr(unexpected_kw[-1])])
        raise HintTypeError(f"{f_name}() got{' an ' if n_unexpkw == 1 else ' '}unexpected keyword{'s' if n_unexpkw > 1 else ''} argument {unexpected_kw}")
    return values

def check_annotations(func_info: tuple, values: list, annotations: list):
    (   f_name, has_args, has_kwargs, f_varnames,
        co_posonlyargcount, co_argcount, co_nonvarargcount, co_varcount,
        n_defaults, f_defaults, f_kwdefaults
    ) = func_info

    for i, (v, a) in enumerate(zip(values, annotations)):
        if a is None: continue
        if i < co_nonvarargcount:
            if isinstance(a, (type, tuple, str)):
                if not isinstance(v, a):
                    raise TypeHintError(f"{f_name}() needs argument '{f_varnames[i]}' of type {get_annot(a)}, but got {repr(v)} of type {get_annot(v.__class__)}")
            else: raise HintTypeError(f"Invalid annotation for '{f_varnames[i]}': {get_annot(a)}")
        elif i == co_nonvarargcount and has_args:
            if isinstance(a, list):
                if ... in a:
                    if not len(a) <= len(v) + 1: raise TypeHintError(f"Variable-arguments (*{f_varnames[i]}) argument fewer thatn listed typehint annotation. Annotation of length {len(a)} but input of length {len(v)}")
                    i_ellipsis = [i for i, t in enumerate(a) if t == ...]
                    if len(i_ellipsis) > 1: raise TypeHintError(f"List typehint annotation for variable-arguments (*{f_varnames[i]}) argument can only have a single ellipsis '...' for omitted types with arbitrary length. Use None for others")
                    epsi = i_ellipsis[0]
                    if not all(isinstance(x, y) for x, y in zip(v[:epsi], a[:epsi])) or not all(isinstance(x, y) for x, y in zip(v[len(v) - len(a) + epsi + 1:], a[epsi+1:])):
                        raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[i]}' of types {get_annot(a)}, but got {get_annot([x.__class__ for x in v])}: {v}")
                elif len(a) != len(v):
                    i_ellipsis = [i for i, t in enumerate(a) if getattr(t, 'base_size', None) == (...,)]
                    if len(i_ellipsis) != 1: raise TypeHintError(f"List typehint annotation for variable-arguments (*{f_varnames[i]}) argument needs to be the same size of the input. Annotation of length {len(a)} but input of length {len(v)}, use ... for omitted types")
                    mid_t = a[i_ellipsis[0]]
                    if mid_t.origin_type == iterable:
                        mid_t = mid_t.base_type
                    else: mid_t = mid_t.origin_type
                    if (
                        not all(isinstance(x, y) for x, y in zip(v[:epsi], a[:epsi])) or 
                        not all(isinstance(x, y) for x, y in zip(v[len(v) - len(a) + epsi + 1:], a[epsi+1:])) or
                        not all(isinstance(x, mid_t) for x in v[epsi:len(v) - len(a) + epsi + 1])
                    ):
                        raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[i]}' of types {get_annot(a)}, but got {get_annot([x.__class__ for x in v])}: {v}")
                if not all(isinstance(x, y) for x, y in zip(v, a)):
                    raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[i]}' of types {get_annot(a)}, but got {get_annot([x.__class__ for x in v])}: {v}")
            else:
                if not all(isinstance(x, a) for x in v):
                    raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[i]}' with all elements of type {get_annot(a)}, but got {get_annot([x.__class__ for x in v])}: {v}")
        elif has_kwargs:
            if isinstance(a, dict):
                if not all(isinstance(v[k], a[k]) for k in v if k in a):
                    k = [k for k in v if k in a and not isinstance(v[k], a[k])][0]
                    raise TypeHintError(f"{f_name}() needs keyword argument '{k}' (in '**{f_varnames[i]}') of type {get_annot(a[k])}, but got {repr(v[k])} of type {get_annot(v[k].__class__)}")
            else:
                if not all(isinstance(x, a) for x in v.values()):
                    k = [k for k, x in v.items() if not isinstance(x, a)][0]
                    raise TypeHintError(f"{f_name}() needs keyword argument '{k}' (in '**{f_varnames[i]}') of type {get_annot(a)}, but got {repr(v[k])} of type {get_annot(v[k].__class__)}")

def match_and_check(func_info: tuple, annotations: list, *args, **kwargs):
    (   f_name, has_args, has_kwargs, f_varnames,
        co_posonlyargcount, co_argcount, co_nonvarargcount, co_varcount,
        n_defaults, f_defaults, f_kwdefaults
    ) = func_info
    
    n_args = len(args)
    
    for i, v, a in zip(range(co_nonvarargcount), f_varnames, annotations):
        
        if a is None: continue
        if not isinstance(a, (type, tuple, str)): raise HintTypeError(f"Invalid annotation for '{v}': {get_annot(a)}")
        try:
            if has_kwargs and i < co_posonlyargcount and i < n_args:
                if not isinstance(args[i], a):
                    raise TypeHintError(f"{f_name}() needs argument '{v}' of type {get_annot(a)}, but got {repr(args[i])} of type {get_annot(args[i].__class__)}")
                continue
            value = kwargs.pop(v, None)
            if i < co_posonlyargcount and value is not None:
                raise HintTypeError(f"{f_name}() got some positional-only arguments passed as keyword arguments: '{v}'")
            if i < n_args and i < co_argcount and value is not None:
                raise HintTypeError(f"{f_name}() got multiple values for argument '{v}'")
            if value is None:
                value = f_kwdefaults[v] if co_argcount <= i else (args[i] if i < n_args else f_defaults[i - co_argcount + n_defaults])
        except IndexError:
            missing_args = f_varnames[n_args:co_argcount - n_defaults]
            n_missing = len(missing_args)
            missing_args_str = ', '.join([repr(x) for x in missing_args[:-1]])
            missing_args_str += (' and ' if missing_args_str else '') + repr(missing_args[-1])
            raise HintTypeError(f"{f_name}() missing {n_missing} required position argument{'s' if n_missing > 1 else ''}: {missing_args_str}")
        except KeyError:
            missing_args = [v for v in f_varnames[co_argcount:co_nonvarargcount] if v not in f_kwdefaults]
            n_missing = len(missing_args)
            missing_args_str = ', '.join([repr(x) for x in missing_args[:-1]])
            missing_args_str += (' and ' if missing_args_str else '') + repr(missing_args[-1])
            raise HintTypeError(f"{f_name}() missing {n_missing} required keyword-only argument{'s' if n_missing > 1 else ''}: {missing_args_str}")
        except Exception as error:
            f_varnames = list(f_varnames)
            var_names = f_varnames[:co_posonlyargcount] + ['/'] + f_varnames[co_posonlyargcount:co_argcount]
            var_names += ['*' + ('' if not has_args else f_varnames[co_nonvarargcount])]
            var_names += f_varnames[co_argcount:co_nonvarargcount]
            if has_kwargs: var_names += ['**' + f_varnames[co_nonvarargcount + has_args]]
            i = f_varnames.index(v)
            var_names[i] = var_names[i] + ': ' + get_annot(a) + ' = ' + repr(value)
            args_str = ', '.join(var_names)
            raise error.__class__(f"In checking {f_name}({args_str}): " + error.__str__())
        if not isinstance(value, a):
            raise TypeHintError(f"{f_name}() needs argument '{v}' of type {get_annot(a)}, but got {repr(value)} of type {get_annot(value.__class__)}")
        continue

    if has_args:
        a = annotations[co_nonvarargcount]
        if a is not None:
            if isinstance(a, list):
                if ... in a:
                    if not len(a) <= n_args - co_argcount + 1: raise TypeHintError(f"Variable-arguments (*{f_varnames[co_nonvarargcount]}) argument fewer than annotated types. Annotation of length {len(a)} but unassigned input of length {n_args - co_argcount}")
                    i_ellipsis = [i for i, t in enumerate(a) if t == ...]
                    if len(i_ellipsis) > 1: raise TypeHintError(f"List typehint annotation for variable-arguments (*{f_varnames[co_nonvarargcount]}) argument can only have a single ellipsis '...' for omitted types with arbitrary length. Use None for others")
                    epsi = i_ellipsis[0]
                    if not all(isinstance(x, y) for x, y in zip(args[co_argcount:][:epsi], a[:epsi])) or not all(isinstance(x, y) for x, y in zip(args[co_argcount:][n_args - co_argcount - len(a) + epsi + 1:], a[epsi+1:])):
                        raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[co_nonvarargcount]}' of types {get_annot(a)}, but got {get_annot([x.__class__ for x in args[co_argcount:]])}: {args[co_argcount:]}")
                elif len(a) != n_args - co_argcount:
                    i_ellipsis = [i for i, t in enumerate(a) if getattr(t, 'base_size', None) == (...,)]
                    if len(i_ellipsis) != 1: raise TypeHintError(f"Variable-arguments (*{f_varnames[co_nonvarargcount]}) argument has a different length from the annotated types. Annotation of length {len(a)} but unassigned input of length {n_args - co_argcount}, use ... for omitted types")
                    epsi = i_ellipsis[0]
                    mid_t = a[epsi]
                    if mid_t.origin_type == iterable:
                        mid_t = mid_t.base_type
                    else: mid_t = mid_t.origin_type
                    if (
                        not all(isinstance(x, y) for x, y in zip(args[co_argcount:][:epsi], a[:epsi])) or 
                        not all(isinstance(x, y) for x, y in zip(args[co_argcount:][n_args - co_argcount - len(a) + epsi + 1:], a[epsi+1:])) or
                        not all(isinstance(x, mid_t) for x in args[co_argcount:][epsi:n_args - co_argcount - len(a) + epsi + 1])
                    ):
                        raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[co_nonvarargcount]}' of types {get_annot(a)}, but got {get_annot([x.__class__ for x in args[co_argcount:]])}: {args[co_argcount:]}")
            elif not isinstance(a, (type, tuple, str)): raise HintTypeError(f"Invalid annotation for '{f_varnames[co_nonvarargcount]}': {get_annot(a)}")
            elif not all(isinstance(x, a) for x in args[co_argcount:]):
                raise TypeHintError(f"{f_name}() needs variable-arguments '*{f_varnames[co_nonvarargcount]}' with all elements of type {get_annot(a)}, but got {get_annot([x.__class__ for x in args[co_argcount:]])}: {args[co_argcount:]}")
    elif n_args > co_argcount:
        raise HintTypeError(f"{f_name}() takes {co_argcount} positional argument but {n_args} were given")
    if has_kwargs:
        a = annotations[co_nonvarargcount + has_args]
        if a is not None:
            if isinstance(a, dict):
                if not all(isinstance(kwargs[k], a[k]) for k in kwargs if k in a):
                    v = f_varnames[co_nonvarargcount + has_args]
                    k = [k for k in kwargs if k in a and not isinstance(kwargs[k], a[k])][0]
                    raise TypeHintError(f"{f_name}() needs keyword argument '{k}' (in '**{v}') of type {get_annot(a[k])}, but got {repr(kwargs[k])} of type {get_annot(kwargs[k].__class__)}")
                return
            if not isinstance(a, (type, tuple, str)): raise HintTypeError(f"Invalid annotation for '{f_varnames[co_nonvarargcount + has_args]}': {get_annot(a)}")
            if not all(isinstance(x, a) for x in kwargs.values()):
                v = f_varnames[co_nonvarargcount + has_args]
                k = [k for k, x in kwargs.items() if not isinstance(x, a)][0]
                raise TypeHintError(f"{f_name}() needs keyword argument '{k}' (in '**{v}') of type {get_annot(a)}, but got {repr(kwargs[k])} of type {get_annot(kwargs[k].__class__)}")
    else:
        for v in f_varnames[co_posonlyargcount:]: kwargs.pop(v, None)
        if len(kwargs) > 0:
            n_unexpkw = len(kwargs)
            unexpected_kw = list(kwargs.keys())
            if n_unexpkw == 1: unexpected_kw = repr(unexpected_kw[0])
            else: unexpected_kw = ' and '.join([', '.join([repr(x) for x in unexpected_kw[:-1]]), repr(unexpected_kw[-1])])
            raise HintTypeError(f"{f_name}() got{' an ' if n_unexpkw == 1 else ' '}unexpected keyword{'s' if n_unexpkw > 1 else ''} argument {unexpected_kw}")

def v_args(func):
    return func.__code__.co_varnames[func.__code__.co_argcount + func.__code__.co_kwonlyargcount] if func.__code__.co_flags & 0x04 else None

def v_kwargs(func):
    return func.__code__.co_varnames[func.__code__.co_argcount + func.__code__.co_kwonlyargcount + ((func.__code__.co_flags & 0x04) >> 2)] if func.__code__.co_flags & 0x08 else None

def get_func_info(func):
    if isinstance(func, method): func = func.__func__
    wrapped_func = func
    while hasattr(wrapped_func, '__wrapped__'): wrapped_func = wrapped_func.__wrapped__
    f_name = wrapped_func.__name__
    f_code = wrapped_func.__code__
    has_args = (f_code.co_flags & 0x04) >> 2
    has_kwargs = (f_code.co_flags & 0x08) >> 3
    co_posonlyargcount = f_code.co_posonlyargcount
    co_argcount = f_code.co_argcount
    co_nonvarargcount = f_code.co_argcount + f_code.co_kwonlyargcount # co_nonvarargcount
    co_varcount = f_code.co_argcount + f_code.co_kwonlyargcount + has_args + has_kwargs # co_varcount
    f_varnames = f_code.co_varnames
    f_defaults = tuple() if wrapped_func.__defaults__ is None else wrapped_func.__defaults__
    n_defaults = len(f_defaults)
    f_kwdefaults = {} if wrapped_func.__kwdefaults__ is None else wrapped_func.__kwdefaults__
    return (
        f_name, has_args, has_kwargs, f_varnames,
        co_posonlyargcount, co_argcount, co_nonvarargcount, co_varcount,
        n_defaults, f_defaults, f_kwdefaults
    )

@decorator
def typehint(*types, **kwtypes):
    # Usage 1: @typehint(*, check_return=False, check_annot_only=False)
    if len(types) == 0 and (
        len(kwtypes) == 1 and ('check_return' in kwtypes or 'check_annot_only' in kwtypes) or
        len(kwtypes) == 2 and ('check_return' in kwtypes and 'check_annot_only' in kwtypes)
    ):
        return lambda f: typehint(f, **kwtypes)
    # Usage 2: func = typehint(func, /, *, annotations=None, check_return=False, check_annot_only=False) or @typehint
    elif len(types) == 1 and isinstance(types[0], functional) and (
        len(kwtypes) == 0 or
        len(kwtypes) == 1 and ('annotations' in kwtypes or 'check_return' in kwtypes or 'check_annot_only' in kwtypes) or
        len(kwtypes) == 2 and ('annotations' in kwtypes and 'check_return' in kwtypes or
                               'annotations' in kwtypes and 'check_annot_only' in kwtypes or
                               'check_return' in kwtypes and 'check_annot_only' in kwtypes) or
        len(kwtypes) == 3 and ('annotations' in kwtypes and 'check_return' in kwtypes and 'check_annot_only' in kwtypes)
    ):
        func = types[0]
        annotations = kwtypes.get('annotations', None)
        check_return = kwtypes.get('check_return', False)
        check_annot_only = kwtypes.get('check_annot_only', False)
        if isinstance(func, method): args = (func.__self__,) + args; func = func.__func__
        wrapped_func = func
        while hasattr(wrapped_func, '__wrapped__'): wrapped_func = wrapped_func.__wrapped__
        func_info = get_func_info(wrapped_func)
        f_name, _, _, f_varnames, *_ = func_info
        ret_annot = wrapped_func.__annotations__.get('return')
        if annotations is None: annotations = [wrapped_func.__annotations__.get(v, None) for v in f_varnames]
        def nest(a):
            if isinstance(a, str): return tag(a)
            if isinstance(a, tuple): return union(*a)
            if isinstance(a, list): return [nest(t) for t in a]
            if isinstance(a, dict): return {k: nest(t) for k, t in a.items()}
            if not isinstance(a, type) and isinstance(a, callable) or isinstance(a, tag('Type')): return instance_satisfies(a)
            return a
        annotations = [{k:nest(v) for k, v in a.items()} if isinstance(a, dict) else nest(a) for a in annotations]
        def wrapper(*args, **kwargs):
            # values = get_arg_values(func_info, *args, **kwargs)
            # check_annotations(func_info, values, annotations)
            match_and_check(func_info, annotations, *args, **kwargs)
            if check_annot_only: return lambda: None
            ret_value = func(*args, **kwargs)
            if check_return and ret_annot is not None:
                if not isinstance(ret_value, ret_annot):
                    raise TypeHintError(f"{f_name}() should have output of type {get_annot(a)}, but got {repr(ret_value)} of type {get_annot(ret_value.__class__)}")
            return ret_value
        return wrapper
    # Usage 3: @typehint(int, float, var1=str, var2=dict)
    else:
        @decorator
        def induced_decorator(func):
            __return__type__ = kwtypes.pop('__return__', None)
            annotations = get_arg_values(get_func_info(func), *types, **kwtypes)
            if __return__type__ is not None: annotations['return'] = __return__type__
            return typehint(func, annotations=annotations)
        return induced_decorator

if __name__ == "__main__":
    ...
