
from pycamia import info_manager

__info__ = info_manager(
    project = "PyZMyc",
    package = "pyoverload",
    fileinfo = "Useful tools for decorators."
)

__all__ = """
    decorator
    get_environ_vars
""".split()

import os, sys
from functools import wraps

try:
    import ctypes
except ModuleNotFoundError:
    ctypes = None
else:
    if hasattr(ctypes, "pythonapi") and \
       hasattr(ctypes.pythonapi, "PyFrame_LocalsToFast"): pass
    else: ctypes = None

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError(f"@decorator wrapping a non-wrapper: {wrapper_func}")
    def wrapper(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) or \
            len(args) == 2 and callable(args[1]):
            func = args[0]
            raw_func = getattr(func, '__func__', func)
            func_name = f"{raw_func.__name__}[{wrapper_func.__qualname__.split('.')[0]}]"
            wrapped_func = wraps(raw_func)(wrapper_func(raw_func))
            wrapped_func.__name__ = func_name
            wrapped_func.__doc__ = raw_func.__doc__
            if 'staticmethod' in str(type(func)): trans = staticmethod
            elif 'classmethod' in str(type(func)): trans = classmethod
            else: trans = lambda x: x
            return trans(wrapped_func)
        return decorator(wrapper_func(*args, **kwargs))
    return wraps(wrapper_func)(wrapper)

stack_error = lambda x, ext: TypeError(f"Unexpected function stack for {x}, please contact the developer for further information (Error Code: E001*). {ext}")

def _get_frames(i = [2, 3], key=''):
    """
    Get frames in stack. 
    By default: it gets frame of the function calling get_environ (function frame) and the frame calling this function (client frame). 
    Returns: function frame, client frame
    """
    frames = []
    frame = sys._getframe()
    fname = frame.f_back
    if isinstance(i, int): i = [i]
    if i is not None:
        if len(i) == 0: raise IndexError("Invalid index for _get_frames")
        max_i = max(i)
    while frame is not None:
        frame_file = frame.f_code.co_filename
        if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>':
            frame = frame.f_back
            continue
        if i is None:
            if frame.f_code.co_name == key: frames.append(frame)
        else:
            frames.append(frame)
            if len(frames) >= max_i + 1:
                domain = [frames[j] for j in i]
                if key != '': domain = [f for f in domain if f.f_code.co_name == key]
                return domain if len(domain) > 1 else domain[0]
        frame = frame.f_back
    if i is not None or len(frames) == 0:
        try: f_all = _get_frames(-1)
        except: raise stack_error(fname, f"\n_get_frames({i}) got stack: \n" + '\n'.join(map(str, frames)))
        raise stack_error(fname, "\nComplete stack: \n" + '\n'.join(map(str, f_all)) + f"\n_get_frames({i}) got stack: \n" + '\n'.join(map(str, frames)))
    return frames

class EnvironVars():
    
    def __init__(self, frame): self.frame = frame

    def get(self, name, default=None):
        res = self.frame.f_locals.get(name, self.frame.f_globals.get(name, default))
        if res is None: raise AttributeError(f"No variable {name} found in the environment. ")
        return res

    def set(self, name, value, in_dict=None):
        if not in_dict: in_dict = 'local'
        if in_dict.lower().startswith('loc'): self.frame.f_locals[name] = value
        else: self.frame.f_globals[name] = value
        if ctypes is not None:
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.frame), ctypes.c_int(0))
    
    def update(self, dic, in_dict=None):
        for k, v in dic.items():
            if not in_dict: in_dict = 'local'
            if in_dict.lower().startswith('loc'): self.frame.f_locals[k] = v
            else: self.frame.f_globals[k] = v
        if ctypes is not None:
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.frame), ctypes.c_int(0))
        
    def __contains__(self, key): return key in self.frame.f_locals or key in self.frame.f_globals
    def __getitem__(self, key): return self.get(key)
    def __setitem__(self, key, value): return self.set(key, value)
    def __getattr__(self, key):
        if key in self.__dict__: return super().__getattr__(key)
        return self.get(key)
    def __setattr__(self, key, value):
        if key == 'frame': return super().__setattr__(key, value)
        return self.set(key, value)
    
    @property
    def locals(self): return self.frame.f_locals
    
    @property
    def globals(self): return self.frame.f_globals
    
    @property
    def all(self):
        all = self.frame.f_globals.copy()
        all.update(self.frame.f_locals)
        return all
    
def get_environ_vars(offset=0, pivot=''):
    client_frame = _get_frames(3) # offset of frame
    if pivot: client_frame = _get_frames(None, key=pivot)
    if isinstance(client_frame, list): client_frame = client_frame[-1]
    for _ in range(offset):
        client_frame = client_frame.f_back
    return EnvironVars(client_frame)

def update_locals_by_environ():
    module_frame, client_frame = _get_frames()
    vars_set = client_frame.f_locals.copy()
    vars_set.update(module_frame.f_locals)
    module_frame.f_locals.update(vars_set)
