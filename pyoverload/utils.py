
from pycamia import info_manager

__info__ = info_manager(
    project = "PyZMyc",
    package = "pyoverload",
    fileinfo = "Useful tools for decorators."
)

__all__ = """
    raw_function
    return_type_wrapper
    decorator
""".split()

import sys
from functools import wraps

def raw_function(func):
    if hasattr(func, "__func__"):
        return func.__func__
    return func

def _get_wrapped(f):
    while hasattr(f, '__wrapped__'): f = f.__wrapped__
    return f

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError("@decorator wrapping a non-wrapper")
    def wrapper(*args, **kwargs):
        if not kwargs and len(args) == 1:
            func = args[0]
            raw_func = raw_function(func)
            if callable(raw_func):
                func_name = f"{raw_func.__name__}[{wrapper_func.__qualname__.split('.')[0]}]"
                wrapped_func = wraps(raw_func)(wrapper_func(raw_func))
                wrapped_func.__name__ = func_name
                wrapped_func.__doc__ = raw_func.__doc__
                # return wrapped_func
                if 'staticmethod' in str(type(func)): trans = staticmethod
                elif 'classmethod' in str(type(func)): trans = classmethod
                else: trans = lambda x: x
                return trans(wrapped_func)
        return decorator(wrapper_func(*args, **kwargs))
    return wraps(wrapper_func)(wrapper)

def _mid(x): return x[1] if len(x) > 1 else x[0]
def _rawname(s): return _mid(str(s).split("'"))

stack_error = lambda x: TypeError(f"Unexpected function stack for {x}, please contact the developer for further information. ")

def _get_frames():
    frames = []
    frame = sys._getframe()
    fname = frame.f_back.f_code.co_name
    while frame is not None:
        frame_file = _rawname(frame)
        if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>':
            frame = frame.f_back
            continue
        frames.append(frame)
        if len(frames) >= 4: return frames[2:]
        frame = frame.f_back
    raise stack_error(fname)

def get_environ_locals():
    _, client_frame = _get_frames()
    return client_frame.f_locals

def get_environ_globals():
    _, client_frame = _get_frames()
    return client_frame.f_globals
