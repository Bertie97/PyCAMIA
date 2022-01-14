
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to handle the exceptions.",
    help = "Handle the exceptions. `touch` swallow the error, `crashed` to see if there is an erro, `avouch` to assert with text, `Error` to create new error types. "
)

__all__ = """
    touch
    crashed
    avouch
    Error
""".split()

import re, sys
from typing import Callable, Union

from .environment import get_args_expression, get_environ_locals, get_environ_globals
from .functions import const_function

def touch(v: Union[Callable, str], default=None):
    """
    Touch a function or an expression `v`, see if it causes exception. 
    If not, output the result, otherwise, output `default`. 
    
    Note:
        Use `default = pycamia.functions.identity_function` (or write one yourself)
        to return the exception object.
    
    Example:
    ----------
    >>> a = 0
    >>> touch(lambda: 1/a, default = 'fail')
    fail
    """
    if not callable(default):
        default = const_function(default)
    if isinstance(v, str):
        local_vars = get_environ_locals()
        local_vars.update(locals())
        locals().update(local_vars)
        try: return eval(v)
        except Exception as e: return default(e)
    else:
        try: return v()
        except Exception as e: return default(e)

def avouch(v: bool, txt=""):
    """
    Assert with text. 
    
    Inputs:
        v[bool]: the expression to be validated.
        txt[str]: the assertion message when the test fails.
    """
    if not v:
        if not txt:
            expr = get_args_expression().split(',')[0].strip()
            if (expr is not None):
                txt = f"Failure in assertion '{expr}'"
        raise AssertionError(txt)

def crashed(func):
    """
    Validate whether a function `func` would crash. 
    """
    try:
        func()
    except:
        return True
    return False

def Error(name: str):
    """
    Create a temporary error by text. 

    Inputs:
        name[str]: the name of the error; It is used to identify the error type. 

    Example:
    ----------
    >>> try:
    >>>     raise Error("TEST")()
    >>> except Error("TEST"):
    >>>     print('caught')
    ... 
    caught
    """
    v = get_environ_globals()
    error_name = f"{name}Error"
    if error_name in v: return v[error_name]
    exec(f"class {error_name}(Exception): pass")
    v[error_name] = eval(error_name)
    return eval(error_name)
