
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to manage the environment.",
    help = "Use `get_**` to obtain the the variables etc. outside the function. "
)

__all__ = """
    get_environ_vars
    get_environ_globals
    get_environ_locals
    update_locals_by_environ
    get_args_expression
    get_declaration
    EnvironVars
""".split()

import os, sys
from .strop import tokenize

try:
    import ctypes
except ModuleNotFoundError:
    ctypes = None
else:
    if hasattr(ctypes, "pythonapi") and \
       hasattr(ctypes.pythonapi, "PyFrame_LocalsToFast"): pass
    else: ctypes = None

stack_error = lambda x, ext: TypeError(f"Unexpected function stack for {x}, please contact the developer for further information (Error Code: E001). {ext}")

def _get_frames(i = [2, 3], key=''):
    """
    Get frames in stack. 
    By default: it gets frame of the function calling get_environ (function frame) and the frame calling this function (client frame). 
    i = -1 for all frames
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
        elif i == [-1]:
            if frame is None: return frames
            frames.append(frame) 
        else:
            frames.append(frame)
            if len(frames) >= max_i + 1:
                domain = [frames[j] for j in i]
                if key != '': domain = [f for f in domain if f.f_code.co_name == key]
                return domain if len(domain) > 1 else domain[0]
        frame = frame.f_back
    if i is not None and i != [-1] or len(frames) == 0:
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
    """If there is a function 'f' called in script 's', one can use 'get_environ_vars' in 'f' to obtain the variables in 's'.
    offset (int, optional): If 's' calls 'u' and 'u' calls 'f', one need to set 'offset' to 1 for the additional scope 'u'. Defaults to 0.
    pivot (str, optional): If one has the name of the function of which the variables are needed, place it here. The most previous call would be obtained. Defaults to ''.
    """
    client_frame = _get_frames(3) # offset of frame
    if pivot: client_frame = _get_frames(None, key=pivot)
    if isinstance(client_frame, list): client_frame = client_frame[-1]
    for _ in range(offset):
        client_frame = client_frame.f_back
    return EnvironVars(client_frame)

def get_environ_globals(offset=0, pivot=''):
    """If there is a function 'f' called in script 's', one can use 'get_environ_globals' in 'f' to obtain the global variables in 's'.
    offset (int, optional): If 's' calls 'u' and 'u' calls 'f', one need to set 'offset' to 1 for the additional scope 'u'. Defaults to 0.
    pivot (str, optional): If one has the name of the function of which the global variables are needed, place it here. The most previous call would be obtained. Defaults to ''.
    """
    return get_environ_vars(offset, pivot).globals()

def get_environ_locals(offset=0, pivot=''):
    """If there is a function 'f' called in script 's', one can use 'get_environ_locals' in 'f' to obtain the local variables in 's'.
    Note that changing the values in the local object will not be assigned to the original scope, please use methods 'set' or 'update' of variables object obtained by 'get_environ_vars' to perform assignment.
    offset (int, optional): If 's' calls 'u' and 'u' calls 'f', one need to set 'offset' to 1 for the additional scope 'u'. Defaults to 0.
    pivot (str, optional): If one has the name of the function of which the local variables are needed, place it here. The most previous call would be obtained. Defaults to ''.
    """
    return get_environ_vars(offset, pivot).locals()

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
    
def get_args_expression(func_name = None):
    module_frame, client_frame = _get_frames()
    if func_name is None: func_name = module_frame.f_code.co_name
    if os.path.exists(client_frame.f_code.co_filename):
        with open(client_frame.f_code.co_filename) as fp:
            for _ in range(client_frame.f_lineno-1): fp.readline()
            l = fp.readline().rstrip('\n')
            if func_name not in l:
                raise TypeError(f"Cannot find function name `{func_name}` in {client_frame.f_code.co_filename} line {client_frame.f_lineno}:\n\t{l}" +
                    "Problem occurs in code stack, please contact the developer for further information (Error Code: E002). ")
            error = TypeError(f"Cannot get args expression. Please avoid using two function '{func_name}'s in a single code line. ")
            lines_with_func = [x for x in tokenize(l, sep=[';']) if func_name in x]
            if len(lines_with_func) > 1: raise error
            line = lines_with_func[0]
            parts = line.split(func_name)
            if len(parts) > 2: raise error
            exp = tokenize(parts[-1], sep=[',', ')'])[0]
            exp = tokenize(exp, sep='.')[0]
            while True:
                if exp.startswith('(') and exp.endswith(')'): exp = exp[1:-1]
                else: break
            return exp
    else: return "<unreachable arg expression>"

def get_declaration(func, func_name = None):
    func_code = func.__code__
    if func_name is None: func_name = func.__name__
    if os.path.exists(func_code.co_filename):
        with open(func_code.co_filename) as fp:
            for _ in range(func_code.co_firstlineno - 1): fp.readline()
            i = func_code.co_firstlineno
            while True:
                l = fp.readline()
                if func_name not in l:
                    if not l.strip() or l.strip().startswith('@') or l.strip().startswith('#'): i += 1; continue
                    raise TypeError(f"Cannot find function name `{func_name}` in {func_code.co_filename} line {i}:\n\t{l}" +
                        "Problem occurs in code stack, please contact the developer for further information (Error Code: E003). ")
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
