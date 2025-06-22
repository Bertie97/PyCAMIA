
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
    get_reference_line
    get_declaration
    EnvironVars
""".split()

import os, re, sys, builtins
from .strop import tokenize
from .pythonop import python_line

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
        # if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>':
        #     frame = frame.f_back
        #     continue
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

def _update_locals(frame):
    if ctypes is not None:
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))

class VarDict(dict):

    def __new__(cls, frame, method = 'global'):
        self = super().__new__(cls, getattr(frame, f"f_{method}s"))
        self.frame = frame
        self.method = method
        return self
    
    def __init__(self, frame, method):
        super().__init__(getattr(frame, f"f_{method}s"))
        
    @property
    def data(self): return getattr(self.frame, f"f_{self.method}s")

    def set(self, name, value):
        self[name] = value

    def __getitem__(self, name):
        if name not in self.data:
            if hasattr(builtins, name): return getattr(builtins, name)
            raise NameError(f"No variable '{name}' found in the {self.method} environment. ")
        return self.data[name]

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        if self.method == 'local':
            self.frame.f_locals[name] = value
            _update_locals(self.frame)
        else: self.frame.f_globals[name] = value

    def __getattr__(self, key):
        if key in ['frame', 'method', 'set'] + dir(dict): return super().__getattribute__(key)
        return self[key]

    def __setattr__(self, key, value):
        if key in ['frame', 'method', 'set'] + dir(dict): return super().__setattr__(key, value)
        self[key] = value

class EnvironVars:
    
    def __init__(self, frame): self.frame = frame

    def get(self, name, default=None):
        return self.locals.get(name, self.globals.get(name, default))

    def set(self, name, value, in_dict=None):
        if not in_dict: in_dict = 'local'
        if len(self.locals) == len(self.globals): in_dict = 'global'
        if in_dict.lower().startswith('loc'): self.locals.set(name, value)
        else: self.globals.set(name, value)
    
    def update(self, dic, in_dict=None):
        for k, v in dic.items(): self.set(k, v, in_dict=in_dict)
        
    def __contains__(self, key): return key in self.locals or key in self.globals
    def __getitem__(self, key):
        if key in self.locals: return self.locals[key]
        return self.globals[key]
    def __setitem__(self, key, value): self.set(key, value)
    def __getattr__(self, key):
        if key in ['frame', 'locals', 'globals']: return super().__getattr__(key)
        return self[key]
    def __setattr__(self, key, value):
        if key in ['frame', 'locals', 'globals']: return super().__setattr__(key, value)
        return self.set(key, value)
    
    @property
    def locals(self): return VarDict(self.frame, method='local')
    
    @property
    def globals(self): return VarDict(self.frame, method='global')
    
    @property
    def all(self):
        all_dict = self.frame.f_globals.copy()
        all_dict.update(self.frame.f_locals)
        return VarDict(self.frame, all_dict)
    
    def __str__(self):
        return "[ locals]: " + str(self.locals) + '\n[globals]: ' + str(self.globals)

def get_environ_vars(offset=0, pivot=''):
    """If there is a function 'f' called in script 's', one can use 'get_environ_vars' in 'f' to obtain the variables in 's'.
    Args:
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
    Args:
        offset (int, optional): If 's' calls 'u' and 'u' calls 'f', one need to set 'offset' to 1 for the additional scope 'u'. Defaults to 0.
        pivot (str, optional): If one has the name of the function of which the global variables are needed, place it here. The most previous call would be obtained. Defaults to ''.
    """
    return get_environ_vars(offset, pivot).globals

def get_environ_locals(offset=0, pivot=''):
    """If there is a function 'f' called in script 's', one can use 'get_environ_locals' in 'f' to obtain the local variables in 's'.

    Note: Changing the values in the local object will not be assigned to the original scope, please use methods 'set' or 'update' of variables object obtained by 'get_environ_vars' to perform assignment.

    Args:
        offset (int, optional): If 's' calls 'u' and 'u' calls 'f', one need to set 'offset' to 1 for the additional scope 'u'. Defaults to 0.
        pivot (str, optional): If one has the name of the function of which the local variables are needed, place it here. The most previous call would be obtained. Defaults to ''.
    """
    return get_environ_vars(offset, pivot).locals

def update_locals_by_environ():
    module_frame, client_frame = _get_frames()
    vars_set = client_frame.f_locals.copy()
    vars_set.update(module_frame.f_locals)
    module_frame.f_locals.update(vars_set)
    _update_locals(module_frame)

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
    
def get_args_expression(func_name = None, search_more=False, with_line_info=False):
    _, _, module_frame, *client_frames = _get_frames(i=-1)
    if func_name is None: func_name = module_frame.f_code.co_name
    if func_name.startswith('__') and func_name.endswith('__'): func_name = ''
    if func_name == '<lambda>':
        if os.path.exists(module_frame.f_code.co_filename):
            with open(module_frame.f_code.co_filename) as fp:
                for _ in range(module_frame.f_lineno-1): fp.readline()
                l = python_line(fp)
                func_name = l.split("lambda", 1)[0].split(';')[-1].rstrip(" =").strip()
    for client_frame in client_frames:
        if os.path.exists(client_frame.f_code.co_filename):
            with open(client_frame.f_code.co_filename) as fp:
                for _ in range(client_frame.f_lineno-1): fp.readline()
                l = python_line(fp)
                if func_name not in l:
                    if search_more: continue
                    raise TypeError(f"Cannot find function name `{func_name}` in {client_frame.f_code.co_filename} line {client_frame.f_lineno}:\n\t> {l.lstrip()}\n" +
                        "Problem occurs in code stack, please contact the developer for further information (Error Code: E002). ")
                error = TypeError(f"Cannot get args expression. Please avoid using two/none function '{func_name}'s in a single code line: {l}. ")
                lines_with_func = [x for x in tokenize(l, sep=[';']) if func_name in x]
                if len(lines_with_func) > 1: raise error
                line = lines_with_func[0]
                parts = re.split(rf'[^\w]{func_name}([^\w])', ' ' + line)
                if len(parts) != 3: raise error
                exp = tokenize(''.join(parts[1:]), sep=['.', ')'])[0]
                while True:
                    if exp.startswith('(') and exp.endswith(')'): exp = exp[1:-1]
                    else: break
                if with_line_info: return client_frame.f_code.co_filename, client_frame.f_lineno, exp
                else: return exp
        elif with_line_info: return None, None, "<unreachable arg expression>"
        else: return "<unreachable arg expression>"
    raise TypeError(f"Cannot find function name `{func_name}` in {client_frame.f_code.co_filename} line {client_frame.f_lineno}:\n\t{l}" +
        "Problem occurs in code stack, please contact the developer for further information (Error Code: E002). ")
    
def get_reference_line(func_name = None, search_more=False, with_line_info=False):
    _, _, module_frame, *client_frames = _get_frames(i=-1)
    if func_name is None: func_name = module_frame.f_code.co_name
    if func_name.startswith('__') and func_name.endswith('__'): func_name = ''
    for client_frame in client_frames:
        if os.path.exists(client_frame.f_code.co_filename):
            with open(client_frame.f_code.co_filename) as fp:
                for _ in range(client_frame.f_lineno-1): fp.readline()
                l = python_line(fp)
                if func_name not in l:
                    if search_more: continue
                    raise TypeError(f"Cannot find function name `{func_name}` in {client_frame.f_code.co_filename} line {client_frame.f_lineno}:\n\t{l}" +
                        "Problem occurs in code stack, please contact the developer for further information (Error Code: E002). ")
                if with_line_info: return client_frame.f_code.co_filename, client_frame.f_lineno, l
                else: return l
        elif with_line_info: return None, None, "<unreachable reference>"
        else: return "<unreachable reference>"
    raise TypeError(f"Cannot find function name `{func_name}` in {client_frame.f_code.co_filename} line {client_frame.f_lineno}:\n\t{l}" +
        "Problem occurs in code stack, please contact the developer for further information (Error Code: E002). ")

def get_declaration(func, func_name = None):
    func_code = func.__code__
    if func_name is None: func_name = func.__name__
    if os.path.exists(func_code.co_filename):
        with open(func_code.co_filename) as fp:
            for _ in range(func_code.co_firstlineno - 1): fp.readline()
            i = func_code.co_firstlineno
            while True:
                l = python_line(fp)
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
        dec_line = [l for l in python_lines(ss) if len(l) > 0 and 'Help' not in l][0]
    if not dec_line: return "<unreachable declaration>"
    dec = dec_line.strip().split(' ', 1)[-1].rstrip(':')
    return dec
