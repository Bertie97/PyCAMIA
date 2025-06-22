
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
    Warn
""".split()

from .environment import get_args_expression, get_environ_vars, update_locals_by_environ
from .functions import const_function
from .strop import is_snakename, tokenize

def touch(target, default=None, args=[], kwargs={}, error_type: (tuple, Exception)=Exception, print_error_message=False): 
    """
    Touch a function or an expression `v`, see if it causes exception in `error_type`. 
    If not, output the result, otherwise, output the `default` value (or return from the `default` function). 
    
    Args:
        target (callable or str): The expression (or function) to touch/run. 
        *args, **kwargs (variable arguments): The inputs of function target. 
        default (callable or value object): As the function, it takes inputs (error, *args, **kwargs) to cope with exceptions. 
            Use `default = lambda *a, **k: a[0]` to return the exception object.
        error_type (tuple or Exception): The exceptions to be caught. All exceptions by default. 
    
    Examples::
        >>> a = 0
        >>> touch(lambda: 1/a, 'failed')
        failed
        >>> touch(lambda x: 1/x, args=(a,), default = lambda e, x: f'failed for {x}')
        failed for 0
    """
    if not callable(default): default = const_function(default)
    if print_error_message: dfunc = default; default = lambda *a, **k: (print(e), dfunc(e, *a, **k))[1]
    
    if isinstance(target, str):
        update_locals_by_environ()
        try:
            res = eval(target)
            if not callable(res): return res
        except error_type as e: return default(e, *args, **kwargs)
    elif callable(target):
        try:
            return target(*args, **kwargs)
        except error_type as e: return default(e, *args, **kwargs)
    else: raise TypeError(f"The input of touch should be either a string or a callable, not {target}. ")

def avouch(v: (bool, callable), excp: (str, Exception)=""):
    """
    Assert with text. 
    
    Args:
        v (bool): the expression to be validated.
        excp (str, Exception, optional): the assertion message (AssertionError), user-designed Exception when the test fails. Defaults to the asserted expression of `v`.
        
    Examples::
        >>> a = 0
        >>> avouch(a == 1, "Variable 'a' should be 0. ")
        Traceback (most recent call last):
        ...
        AssertionError: Variable 'a' should be 0.
        >>> avouch(a == 1)
        Traceback (most recent call last):
        ...
        AssertionError: Failure in assertion 'a == 1'
        >>> # The above line of code may lead to <unreachable arg expression> in native Python IDLE.
    """
    if not v:
        if not excp:
            expr = tokenize(get_args_expression('avouch'), sep=',')[0].strip()
            if (expr is not None):
                excp = f"Failure in assertion '{expr}'"
        if isinstance(excp, str):
            excp = ' '.join(excp.split())
            raise AssertionError(excp)
        else: raise excp # Inside 'avouch': a forward of previous Exception. 

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

    Args:
        name (str): the name of the error; It is used to identify the error type. 

    Examples::
        >>> try:
        >>>     raise Error("TEST")()
        >>> except Error("TEST"):
        >>>     print('caught')
        ... 
        caught
        >>> raise Error('Type')("description")
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        pycamia.exception.TESTError: description
    """
    if not is_snakename(name):
        raise TypeError(f"Invalid name '{name}' for an error: it should be alphabets/digits/underlines without spaces (as long as all other symbols). ")
    v = get_environ_vars().globals
    error_name = f"{name}Error"
    if error_name in v: return v[error_name]
    exec(f"class {error_name}(Exception): pass")
    v[error_name] = eval(error_name)
    return eval(error_name)

class Warn:
    
    debugging = False
    suppressing = False
    
    def __init__(self, message, by_func=None):
        if by_func is None: by_func = print
        if isinstance(self.debugging, str):
            if self.debugging.lower().strip() == message.lower().strip():
                raise Error("Warning")(message)
        elif self.debugging: raise Error("Warning")(message)
        if isinstance(self.suppressing, str):
            if self.suppressing.lower().strip() == message.lower().strip():
                return
        elif self.suppressing: return
        by_func("Warning: " + message.lstrip())
    
    @classmethod
    def debug(cls, message=None):
        return Warn.Debugger(cls, message)
    
    @classmethod
    def suppress(cls, message=None):
        return Warn.Suppresser(cls, message)
    
    class Debugger:
        def __init__(self, cls, message): self.cls = cls; self.message = message
        def __enter__(self): self.cls.debugging = True if self.message is None else self.message
        def __exit__(self, exc_type, exc_value, traceback): self.cls.debugging = False
    
    class Suppresser:
        def __init__(self, cls, message): self.cls = cls; self.message = message
        def __enter__(self): self.cls.suppressing = True if self.message is None else self.message
        def __exit__(self, exc_type, exc_value, traceback): self.cls.suppressing = False
