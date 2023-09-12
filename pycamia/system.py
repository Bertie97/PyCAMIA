
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "System operations including class dealing with path or shell menagements.",
    requires = "tqdm"
)

__all__ = """
    path Path
    path_list PathList
    set_black_list
    get_black_list
    curdir
    pardir
    homedir
    rootdir
    pwd ls
    cp copy
    mv move
    rename
    is_valid_command
    Shell
    execblock
    get_volumes
    ByteSize
""".split()

import os, re, math, platform, psutil
from subprocess import PIPE, Popen
from .strop import no_indent
from .exception import avouch, touch
from .listop import arg_tuple, to_tuple
from .decorators import alias
from .functions import identity_function
from .environment import get_environ_vars, update_locals_by_environ

with __info__:
    from tqdm import tqdm

black_list = [".DS_Store", ".git"]

def get_black_list():
    global black_list
    return black_list

def set_black_list(blacklist):
    global black_list
    black_list = blacklist

@alias("path_list")
class PathList(list):
    def __new__(cls, *args, ref=None):
        args = arg_tuple(args)
        if ref is None: self = super().__new__(cls, args)
        else: self = super().__new__(cls, (Path(a, ref=ref) for a in args))
        self._ref_dir = ref
        return self
    
    def __getitem__(self, key):
        if touch(lambda: len(key), -1) == len(self): return PathList([x for x, b in zip(self, key) if b])
        res = super().__getitem__(key)
        if isinstance(res, (tuple, list)): return PathList(res)
        return res
        
    def sort(self, *, key=identity_function):
        super().sort(key=key)
        
    def append(self, p):
        if self._ref_dir == p._ref: super().append(p)
        super().append(Path(p, ref=self._ref_dir))

    @alias("__or__", "filter")
    def select_file_type(self, k): return self[[x | k for x in self]]
    @alias("__sub__")
    def relative_to(self, y): return PathList([x - y for x in self])
    def __mod__(self, k): return PathList([x % k for x in self])
    def __truediv__(self, k): return PathList([x / k for x in self])
    def __rtruediv__(self, k): return PathList([k / x for x in self])

@alias("path")
class  Path(str):
    """
    Path("abc") -> path_object

    An object managing file path.

    Example:
    ----------
    >>> Path()
    {working directory}
    >>> Path("abc", "sub_folder")
    abc/sub_folder
    >>> Path("abc")/"sub_folder"
    abc/sub_folder
    >>> Path("abc", ref="bcd") # abc and bcd are brother directories
    ../abc
    """
    
    sep = os.path.sep #[In Mac]: '/'
    extsep = os.path.extsep #[In Mac]: '.'
    pathsep = os.path.pathsep #[In Mac]: ':'
    _curdir = os.path.curdir #[In Mac]: '.'
    _pardir = os.path.pardir #[In Mac]: '..'
    _homedir = os.path.expanduser("~") # [In Mac]: '/Users/admin'; [In Win]: Unknown
    _rootdir = os.path.abspath(os.path.curdir).split(sep)[0] + sep
        # [In Mac or Linux]: '/'; [In Win]: 'C:\\' for disk C
    namesep = '_'
    File = b'\x04'
    Folder = Dir = b'\x07'
    invalid_char_regx = r"[:\?$]"

    @alias('list_all', 'listall', 'ls_a', all_items=True, depth=1)
    @alias('list_dir', 'listdir', 'list', 'ls', depth=1)
    @alias('list_subdirs', 'listsubdirs', depth=-1)
    @alias('list_files', 'listfiles', depth=0)
    @alias('walk')
    def _create_dir_list(self, all_items=False, depth=None):
        res = PathList(self._recursively_listdir(all_items, depth))
        res.sort()
        return res

    @alias('iter_all', 'iterall', all_items=True, depth=1)
    @alias('iter_dir', 'iterdir', 'iter', depth=1)
    @alias('iter_subdirs', 'itersubdirs', depth=-1)
    @alias('iter_files', 'iterfiles', depth=0)
    @alias('iter_walk')
    def _recursively_listdir(self, all_items=False, depth=None):
        """
        parameters:
            all_items: whether to search hidden files / directories or not
            depth: [-1] means folders with no subfolders
                    [0] means all files in the directory (recursively)
                    [d] means paths with relative depth d (d > 0)
                 [None] means all relative paths in the folder in a recursive scan
            listing with depth = 1 is equivalent to os.listdir. 
        ----------
        Function code provided by @Yiteng Zhang
        """
        recursively_searched = False
        for f in os.listdir(self._abs):
            if f in get_black_list():
                continue
            p = Path(f, ref=self)
            if not all_items and p.is_hidden():
                continue
            if depth is None:
                yield p
                if p.is_dir():
                    for cp in p._recursively_listdir(all_items=all_items, depth=depth):
                        yield p/cp
            else:
                assert isinstance(depth, int)
                if p.is_file() and depth >= 0:
                    yield p
                elif p.is_dir():
                    if depth != 1:
                        for cp in p._recursively_listdir(all_items=all_items, depth=depth-1 if depth > 0 else depth):
                            yield p/cp
                        recursively_searched = True
                    else:
                        yield p
        if depth == -1 and not recursively_searched:
            yield self

    def __new__(cls, *init_texts, ref=None):
        """
        path object containing path in `init_texts` and a reference folder `ref`. 

        Examples:
        -----------
        >>> Path("a", "b")
        a/b
        >>> Path("a/b/c/d", ref="a")
        b/c/d
        """
        if ref is not None:
            if not isinstance(ref, str): raise TypeError(f"Path reference should be a string, not {ref} of type {type(ref)}. ")
            if isinstance(ref, Path): ref = ref._abs
            else:
                if ref != Path._rootdir: ref = os.path.normpath(str(ref))
                if not os.path.isabs(ref): ref = os.path.abspath(ref)
        if len(init_texts) == 1 and isinstance(init_texts[0], (list, tuple)):
            init_texts = init_texts[0]
        if len(init_texts) == 1 and isinstance(init_texts[0], Path):
            path_object = init_texts[0]
            if ref is None: return path_object
            else:
                abs_path = path_object._abs
                self = super().__new__(cls, abs_path if ref == Path._rootdir else os.path.relpath(abs_path, ref))
                self._ref_dir = ref
            return self
        init_texts = [str(x) for x in init_texts]
        # avouch(all([isinstance(x, str) for x in init_texts]), f"Cannot create path from {init_texts} as there are non-string elements. ")
        if len(init_texts) <= 0 or init_texts[0] == '' or (len(init_texts) == 1 and init_texts[0] == os.path.curdir): string = os.path.curdir
        else:
            init_texts = [str(x).strip() for x in init_texts]
            for x in init_texts:
                if len(re.findall(Path.invalid_char_regx, x)) > 0:
                    print(f"Warning: Invalid characters in path '{x}'.")
            if os.name in ('posix',) and init_texts[0].startswith('~'):
                init_texts = (Path._homedir + init_texts[0][1:],) + init_texts[1:]
            string = os.path.normpath(os.path.join(*init_texts))
        if ref == Path._rootdir:
            self = super().__new__(cls, os.path.abspath(string) if Path._curdir in string else string)
            self._ref_dir = Path._rootdir
            return self
        if os.path.isabs(string):
            if ref is None or ref == Path._rootdir:
                self = super().__new__(cls, string)
                self._ref_dir = Path._rootdir
            else:
                self = super().__new__(cls, os.path.relpath(string, ref))
                self._ref_dir = ref
            return self
        if ref is None: ref = os.path.abspath(os.path.curdir)
        self = super().__new__(cls, string)
        self._ref_dir = ref
        return self

    @property
    def _ref(self): return self._ref_dir
    @property
    def _abs(self): return os.path.normpath(os.path.join(self._ref_dir, str(self)))
    @property
    def _rel(self): return os.path.relpath(self, Path._rootdir) if self._ref_dir == Path._rootdir else str(self)

    @alias('ref')
    @property
    def reference_dir(self): return Path(self._ref, ref=Path._rootdir)
    @alias("__invert__", "__abs__", "abs")
    @property
    def absolute_path(self): return Path(self._abs, ref=Path._rootdir)
    @alias('rel')
    @property
    def relative_path(self): return Path(self._rel, ref=self._ref)

    def __mod__(x, y): return Path(super().__mod__(to_tuple(y)), ref=x._ref)
    
    @alias("__sub__")
    def relative_to(x, y):
        if isinstance(y, Path): y = y._abs
        return Path(x._abs, ref=y)
    @alias("__add__")
    def name_add(x, y):
        y = str(y)
        if x.is_filepath():
            return x.parent / (x.name + y + Path.extsep + x.ext)
        else: return Path(super(Path, x).__add__(y), ref=x._ref)
    @alias("__xor__")
    def name_subscript(x, y):
        y = str(y).lstrip(Path.namesep)
        if x.is_filepath():
            return x.parent / (x.name.rstrip(Path.namesep) + Path.namesep + y + Path.extsep + x.ext)
        else: return Path(super(Path, x).__add__(Path.namesep + y), ref=x._ref)
    @alias("__pow__")
    def common_parent(x, y):
        return Path(os.path.commonpath(x._abs, y._abs), ref=x._ref)

    @alias("__floordiv__")
    def add_ext(x, y):
        if not y: return x
        return Path(Path.extsep.join((x.rstrip(Path.extsep), y.lstrip(Path.extsep))), ref=x._ref)
    @alias("__truediv__", "cd", "chdir", "concat", "append")
    def add_dir(x, y): return Path(x, y, ref=x._ref)
    @alias("__rtruediv__", "prefix", "with_pardir")
    def pre_dir(x, y):
        if isinstance(y, Path):
            ref = y._ref
        else:
            if x._ref.endswith(y): ref = x._ref[:-len(y)]
            else: ref = x._ref
        return Path(y, x, ref=ref)
    @alias("__or__")
    def is_file_type(x, y):
        for y in to_tuple(y):
            if (y == "FILE" or y == Path.File) and x.is_file(): return True
            if (y == "FOLDER" or y == Path.Folder) and x.is_dir(): return True
            if x.ext.lower() == y.lower(): return True
        return False
    def __eq__(x, y): return x._abs == y._abs if isinstance(y, Path) else super().__eq__(y)
    def __len__(self): return len(str(self))
    def __hash__(self): return super().__hash__()

    def __iter__(self):
        for p in self.list(): yield p

    def files(self):
        for p in self.list() | 'FILE': yield p

    def folders(self):
        for p in self.list() | 'FOLDER': yield p

    def __contains__(self, x):
        for p in self:
            if p == x: return True
        return False

    def str_contains(self, x): return x in str(self)
    
    @classmethod
    def join(cls, *args):
        args = arg_tuple(args)
        return cls(*args)

    @property
    def ext(self):
        if hasattr(self, '_ext') and self._ext is not None:
            return self._ext
        if self.is_dir():
            self._ext = ""
            return ""
        file_name = self.fullname
        parts = file_name.split(Path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        self._ext = Path.extsep.join(parts[brk:])
        return self._ext

    @property
    def name(self):
        if hasattr(self, '_name') and self._name is not None:
            return self._name
        file_name = self.fullname
        if self.is_dir():
            self._name = file_name
            return file_name
        parts = file_name.split(Path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        self._name = Path.extsep.join(parts[:brk])
        return self._name

    def with_name(self, name):
        return self.parent / name // self.ext

    def with_ext(self, ext: str):
        return self.parent / self.name // ext

    @alias("fullname", "basename")
    @property
    def filename(self):
        if not hasattr(self, '_filename'):
            self._filename = self.split()[-1]
        return self._filename
    
    @property
    def dirname(self):
        if not hasattr(self, '_dirname'):
            path_tokens = self.split()
            if len(path_tokens) == 0 or path_tokens[-1] in (Path._curdir, Path._pardir, Path._homedir, Path.rootdir):
                path_tokens = self.absolute_path.split()
            self._dirname = Path.sep.join(path_tokens[:-1])
        return self._dirname
    
    @property
    def parent(self):
        return Path(self.dirname, ref=self._ref)
    
    @property
    def children(self):
        return self.ls()

    def is_hidden(self):
        if os.name== 'nt':
            try: import win32api, win32con
            except ModuleNotFoundError: raise ModuleNotFoundError("Packages win32api, win32con needed for hidden file recognition in windows system. Please manually install them. ")
            attribute = win32api.GetFileAttributes(self._abs)
            return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
        elif os.name == 'posix':
            return self.name.startswith('.') #linux or macosx
        else: raise TypeError(f"Unrecognized operating system '{os.name}' for hidden state.")

    def split(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return super().split(Path.sep)
        return super().split(*args, **kwargs)

    def is_abs(self): return self._ref == Path._rootdir
    def is_rel(self): return self._ref != Path._rootdir
    def exists(self): return os.path.exists(self._abs)
    @alias("isfile", "is_folder")
    def is_file(self): return os.path.isfile(self._abs)
    @alias("isdir", "is_folder")
    def is_dir(self): return os.path.isdir(self._abs)
    @alias("is_file_path")
    def is_filepath(self): return True if os.path.isfile(self._abs) else all([0 < len(e) <= 4 for e in self.ext.split(Path.extsep)])
    @alias("is_dir_path", "is_folderpath", "is_folder_path")
    def is_dirpath(self): return True if os.path.isdir(self._abs) else not all([0 < len(e) <= 4 for e in self.ext.split(Path.extsep)])
    
    @alias("delete", "del", "rm")
    def remove(self, verbose=True):
        if self.is_dir():
            if verbose and self.ls():
                print(f"You want to delete directory: {self}")
                if 'y' not in input("Do you want to continue? [Y/n]: ").lower(): return
            for f in self: f.remove(verbose=verbose)
            os.rmdir(self._abs)
        else: os.remove(self._abs)
        
    def rename(self, new_name):
        old_wd = os.path.abspath(os.curdir)
        os.chdir(self.parent._abs)
        os.rename(self.filename, (new_name - self.parent) if isinstance(new_name, Path) else new_name)
        os.chdir(old_wd)

    @alias('cmd', "system")
    def command(self, command):
        try:
            cmd = command.format(self, file=self, path=self)
            old_wd = os.path.abspath(os.curdir)
            os.chdir(self.ref)
            os.system(cmd)
            os.chdir(old_wd)
        except Exception as e:
            print(f"Command error in {cmd}:", e)
            
    def move_to(self, path):
        move(self, path)

    def copy_to(self, path):
        copy(self, path)

    def open(self, mode='r+'):
        if 'w' not in mode and not self.exists(): raise FileNotFoundError(f"Cannot find file {self._abs}. ")
        elif not self.parent.exists(): raise FileNotFoundError(f"Cannot find file folder {self.parent._abs}. ")
        avouch(self.is_file() or self.is_filepath() and 'w' in mode, "Only files can be opened as python stream. ")
        return open(self._abs, mode)

    @alias('browse')
    def open_in_browser(self):
        if os.name == 'posix':
            self.command("open {path}")
        else:
            raise TypeError(f"Unrecognized operating system '{os.name}' for default open.")

    def mkdir(self, new_folder = None):
        """
        Make directory along the path. Create `new_folder` if it is provided. 

        i.e., Path("/Users/username/code/dataset").mkdir()
        will recursive check if "/Users", "/Users/username", "/Users/username/code", "/Users/username/code", "/Users/username/code/dataset"
        is exists or not and make the corresponding directory.
        """
        if not self.is_dir():
            p = self.ref
            if not p.is_dir(): p.mkdir()
            for seg in self.split():
                p = p / seg
                # if not p.exists() and p.is_filepath():break
                if not p.is_dir(): os.mkdir(p)
        if not new_folder: return self
        os.mkdir(self / new_folder)
        return self / new_folder

    def size(self):
        return ByteSize(self._scan_size())
    
    def _scan_size(self):
        fsize = os.stat(self._abs).st_size
        if not self.is_dir(): return fsize
        for p in self.list_all():
            fsize += p._scan_size()
        return fsize
    
    @classmethod
    @property
    def curdir(self): return Path(Path._curdir)
    @classmethod
    @property
    def pardir(self): return Path(Path._pardir)
    @classmethod
    @property
    def homedir(self): return Path(Path._homedir)
    @classmethod
    @property
    def rootdir(self): return Path(Path._rootdir)

curdir = Path(Path._curdir)
pardir = Path(Path._pardir)
homedir = Path(Path._homedir)
rootdir = Path(Path._rootdir)

def pwd(): return curdir.abs
def ls(): return curdir.ls()

@alias('cp')
def copy(src, dst, copy_unit = 1 << (10 * 2), pbar = None, force=None):
    """
    Copy file from 'src' to 'dst'.

    Arguments:
    ----------
    src (str or Path): The source file path.
    dst (str or Path): The target file path.
    copy_unit (_type_, optional): _description_. Defaults to 1<<(10 * 2).
    pbar (_type_, optional): _description_. Defaults to None.
    force (_type_, optional): _description_. Defaults to None.

    Raises:
        FileNotFoundError: _description_
    """
    if pbar is None:
        avouch(isinstance(src, str))
        avouch(isinstance(dst, str))
        src = Path(src)
        dst = Path(dst)
        if not src.exists(): raise FileNotFoundError(f"Cannot find the copy source '{src}'.")
        if dst.is_dir(): dst = dst / src.filename
        if dst.exists():
            
            i_copy = 1
            while True:
                i_copy += 1
                new_name = dst.name + f" {i_copy}"
                if dst.with_name(new_name).exists(): continue
                break
            dst = dst.with_name(new_name)
        size_all = src.size()
        n_share = math.ceil(size_all / copy_unit)
        pbar = tqdm(total=size_all, desc="Copy", unit="item", unit_scale=True, 
                    ncols=80, bar_format="{l_bar}{bar}|")
    if src.is_dir():
        if not dst.exists(): dst.mkdir()
        for f in src: copy(src/f, dst/f, pbar=pbar)
    else:
        file_size = src.size()
        n_file_share = math.ceil(file_size / copy_unit)
        n_remain = file_size - n_file_share * copy_unit
        with open(dst, 'w'): ...
        with open(src, 'rb') as fp_src:
            for i in range(n_file_share):
                if i == n_file_share - 1: n_copy = n_remain
                n_copy = copy_unit
                with open(dst, 'ab') as fp_dst:
                    fp_dst.write(fp_src.read(n_copy))
                pbar.update(n_copy)
    # src = Path(src, ref=src.parent)
    # dst = dst - src.parent
    # opt = '-r' if src.is_dir() else ''
    # src.command(f"cp {opt} {{path}} {dst}")

@alias('mv')
def move(src, dst):
    avouch(isinstance(src, str))
    avouch(isinstance(dst, str))
    src = Path(src)
    dst = Path(dst)
    src = Path(src, ref=src.parent)
    dst = dst - src.parent
    src.command(f"mv {{path}} {dst}")
    
def rename(src, dst):
    avouch(isinstance(src, str))
    avouch(isinstance(dst, str))
    src = Path(src)
    dst = Path(dst)
    src = Path(src, ref=src.parent)
    if dst.parent == curdir:
        dst = dst.filename
    else: dst = dst - src.parent
    src.command(f"mv {{path}} {dst}")
    
def is_valid_command(cmd, error_name='command not found'):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    try: stdout, stderr = p.communicate(timeout=0.05)
    except: stderr = b''
    return bytes(error_name, 'utf8') not in stderr
    
def get_volumes(**kwargs):
    drives = []
    for p in psutil.disk_partitions():
        for k, v in kwargs.items():
            this_v = getattr(p, k, None)
            if isinstance(this_v, str):
                this_v = this_v.split(',')
            else: this_v = [this_v]
            if callable(v):
                if not v(this_v): break
            elif isinstance(v, str) and v.startswith('!'):
                v = v[1:]
                if len(this_v) == 1 and v == str(this_v[0]): break
                elif v in this_v: break
            elif v not in this_v: break
        else:
            drives.append(p)
    return drives
    
class Shell:

    def __init__(self, *directories, verbose=False):
        self.print = print if verbose else lambda x: None
        self.tools = []
        directories = arg_tuple(directories)
        for d in directories:
            for f in Path(d).files():
                if f | 'py': setattr(self, f.name, f"python3 {f.abs}")
                elif f | ('sh', 'exe', ''): setattr(self, f.name, f.abs)
                else: continue
                self.tools.append(f.name)

    def __getattr__(self, key):
        update_locals_by_environ()
        def run(*args):
            self(getattr(self, key, key) + ' ' + ' '.join(args))
        return run

    def __call__(self, string):
        update_locals_by_environ()
        vars = re.findall(r'{(\w+)}', string)
        command = string.format(**{v: eval(v).replace(' ', '\ ') if isinstance(eval(v), str) else eval(v) for v in vars})
        key, *args = re.sub(r'[^\\] ', lambda x: x.group().replace(' ', '\n'), command).split('\n')
        do = getattr(self, key, key)
        command = do + command[len(key):]
        self.print("Running:", command)
        opt = None
        for arg in enumerate(args):
            arg = arg.strip()
            if arg.startswith('-'): opt = arg; continue
            if os.path.sep in arg:
                arg = arg.replace('\ ', ' ')
                arg = arg.strip("""'"'""")
                p = Path(arg)
                file_out = False
                if opt is not None and (opt == '-o' or opt.startswith('--out')):
                    file_out = True
                    p = p.parent
                if not p.exists():
                    self.print(f"Warning: Path doesn't exists: {p}. Trying to run the command. ")
                    if file_out: self.print(f"Creating directory {p.mkdir()}... ")
        if self.verbose: os.system(command)
        else: return os.popen(command).read()
    
def execblock(code):
    """
    Execute `code` with indents eliminated. 
    
    Note: Assigning local variables in functions would fail just as built-in 
        method `exec`. Use `locals()[var_name]` instead to fetch the result. 

    Examples:
    ----------
    >>> class A:
    ...     def run(self, x): return x ** 2
    ...     exec('''
    ...     def additional_method(self, x):
    ...         return self.run(x)
    ...          ''')
    ...
    Traceback (most recent call last):
    ...... [omitted]
    IndentationError: unexpected indent
    >>> class A:
    ...     def run(self, x): return x ** 2
    ...     execblock('''
    ...     def additional_method(self, x):
    ...         return self.run(x) + self.run(x+1)
    ...          ''')
    ...
    >>> A().additional_method(3)
    25
    """
    code = no_indent(code)
    vars = get_environ_vars()
    loc_vars = {}
    try: exec(code, vars.globals, loc_vars)
    except Exception as e:
        raise NameError(f"Error ({e}) in block execution: \n{code}")
    vars.update(loc_vars)

ByteSize_class = {}
def ByteSize(x):
    T = type(x)
    if T in ByteSize_class: return ByteSize_class[T](x)
    class ByteSizeClass(T):
        units = " K M G T P E Z Y".split(' ')
        def __str__(self):
            i = int(math.log2(self) // 10)
            u = ByteSizeClass.units[i] + 'B'
            if len(u) == 1: u += ' '
            x = float(self / (1 << (10 * i)))
            return f"{x: =9.04f} {u}"
    ByteSize_class[T] = ByteSizeClass
    return ByteSizeClass(x)
