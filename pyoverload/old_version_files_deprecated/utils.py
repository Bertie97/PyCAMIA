
from pycamia import info_manager

__info__ = info_manager(
    project = "PyZMyc",
    package = "pyoverload",
    fileinfo = "Useful tools for decorators."
)

__all__ = """
    raw_function
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

def raw_function(func):
    if hasattr(func, "__func__"):
        return func.__func__
    return func

def _get_wrapped(f):
    while hasattr(f, '__wrapped__'): f = f.__wrapped__
    return f

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError(f"@decorator wrapping a non-wrapper: {wrapper_func}")
    def wrapper(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            func = args[0]
            raw_func = raw_function(func)
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

stack_error = lambda x, ext: TypeError(f"Unexpected function stack for {x}, please contact the developer for further information (Error Code: E001*). {ext}")

# def _get_frames():
#     frames = []
#     frame = sys._getframe()
#     fname = frame.f_back.f_code.co_name
#     while frame is not None:
#         frame_file = _rawname(frame)
#         if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>':
#             frame = frame.f_back
#             continue
#         frames.append(frame)
#         if len(frames) >= 4: return frames[2:]
#         frame = frame.f_back
#     raise stack_error(fname)

# def get_environ_locals():
#     _, client_frame = _get_frames()
#     return client_frame.f_locals

# def get_environ_globals():
#     _, client_frame = _get_frames()
#     return client_frame.f_globals

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

class StrIO:
    def __init__(self, file_name = os.path.abspath('.null')):
        self._str_ = None
        self._file_ = open(file_name, 'w+')
        self.file_name = file_name
        self.fileno = self._file_.fileno
    def write(self, s): self._file_.write(s)
    def __str__(self):
        if self._str_ is not None: return self._str_
        self._file_.seek(0)
        self._str_ = self._file_.read()
        self.close()
        return self._str_
    def split(self, c=None): return str(self).split(c)
    def string(self): return str(self)
    def close(self):
        self._file_.close()
        if self.file_name == os.path.abspath('.null'):
            os.remove(self.file_name)
            self._file_ = None
    
def get_args_expression():
    module_frame, client_frame = _get_frames()
    func_name = module_frame.f_code.co_name
    if os.path.exists(client_frame.f_code.co_filename):
        with open(client_frame.f_code.co_filename) as fp:
            for _ in range(client_frame.f_lineno-1): fp.readline()
            l = fp.readline()
            if func_name not in l:
                raise TypeError(f"Cannot find function name `{func_name}` in {client_frame.f_code.co_filename} line {client_frame.f_lineno}:\n\t{l}. \n" +
                    "Problem occurs in code stack, please contact the developer for further information (Error Code: E002*). ")
            exp = l.split(func_name)[-1].split(';')[0].strip()
            if exp.startswith('('): exp = exp[1:]
            if exp.endswith(')'): exp = exp[:-1]
            return exp
    else: return "<unreachable arg expression>"

def get_declaration(func):
    func_code = func.__code__
    func_name = func.__name__
    if os.path.exists(func_code.co_filename):
        with open(func_code.co_filename) as fp:
            for _ in range(func_code.co_firstlineno - 1): fp.readline()
            i = func_code.co_firstlineno
            while True:
                l = fp.readline()
                if func_name not in l:
                    if not l.strip() or l.strip().startswith('@') or l.strip().startswith('#'): i += 1; continue
                    raise TypeError(f"Cannot find function name `{func_name}` in {func_code.co_filename} line {i}:\n\t{l}. \n" +
                        "Problem occurs in code stack, please contact the developer for further information (Error Code: E003*). ")
                dec_line = l
                break
    else:
        ss = StrIO()
        oldout = sys.stdout
        sys.stdout = ss
        help(func)
        sys.stdout = oldout
        dec_line = [l for l in ss.split('\n') if len(l) > 0 and 'Help' not in l][0]
    if not dec_line: return "<unreachable declaration>"
    dec = dec_line.strip().split(' ', 1)[-1].rstrip(':')
    return dec
