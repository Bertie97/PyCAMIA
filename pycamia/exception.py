
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to handle the exceptions.",
    help = "`touch` to try, check to validate `get_environ_vars` to obtain the `locals()` outside the function. "
)

__all__ = """
    touch
    crashed
    assertion
    Error
""".split()

from time import sleep
from typing import Callable, Union

from .environment import get_environ_vars

def touch(v: Union[Callable, str], default=None):
    """
    Touch a function or an expression `v`, see if it causes exception. 
    If not, output the result, otherwise, output `default`. 
    
    Example:
    ----------
    >>> a = 0
    >>> touch(lambda: 1/a, default = 'fail')
    fail
    """
    if isinstance(v, str):
        local_vars = get_environ_vars()
        local_vars.update(locals())
        locals().update(local_vars.simplify())
        try: return eval(v)
        except: return default
    else:
        try: return v()
        except: return default

def crashed(func):
    """
    Validate whether a function `func` would crash. 
    """
    try:
        func()
    except:
        return True
    return False

def assertion(v: bool, txt=""):
    """
    Assert with text. 
    
    Inputs:
        v[bool]: the expression to be validated.
        txt[str]: the assertion message when the test fails.
    """
    if not v:
        raise AssertionError(txt)

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
    v = get_environ_vars()
    print(v)
    error_name = f"{name}Error"
    if error_name in v: return v[error_name]
    exec(f"class {error_name}(Exception): pass")
    v[error_name] = eval(error_name)
    return eval(error_name)
