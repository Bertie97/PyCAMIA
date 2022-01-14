
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to manage package info.",
    help = "Use `info_manager(**kwargs)` to create an info object and use `with info` to check for imports. "
)

__all__ = """
    info_manager
""".split()

import os, re

from .environment import get_environ_globals
from .strop import tokenize

class info_manager:
    """
    info_manager() -> info_object

    An object storing the info of the current file.

    Note:
        It is currently provided for private use in project PyCAMIA only. 
        But it can also be used outside.

    Example:
    ----------
    Code:
        __info__ = info_manager(project="PyCAMIA", package="", requires="xxx").check()
        print("Project is", __info__.project)
        with __info__:
            import xxx
            from xxx import yyy
    Output:
        Project is PyCAMIA
    Error if xxx doesnot exist: ModuleNotFoundError
    Error if yyy doesnot exist in xxx: ImportError
    """
    
    @staticmethod
    def parse(code):
        info = eval(code)
        raw_args = tokenize(code, sep=['(',')'])[1]
        info.tab = raw_args[:re.search(r'\w', raw_args).span()[0]].strip('\n')
        info.order = tokenize(raw_args, sep=[',', '='], strip=None)[::2]
        return info

    def __init__(self, project = "", package = "", requires = "", **properties):
        if isinstance(requires, str): requires = requires.split()
        properties.update(dict(project=project, package=package, requires=requires))
        self.properties = properties
        self.__dict__.update(properties)
        file_path = get_environ_globals()['__file__']
        file = os.path.extsep.join(os.path.basename(file_path).split(os.path.extsep)[:-1])
        self.name = '.'.join([x for x in [project, package, file] if x and x != "__init__"])
        self.tab = ' ' * 4
        major_keys = "project package requires".split()
        self.order = major_keys + list(set(properties.keys()) - set(major_keys))
        
    def check_requires(self):
        not_found_packages = []
        for r in self.requires:
            tokens = tokenize(r, ['<', '>', '='], strip=None, keep_empty=False)
            if len(tokens) == 2: rname, rversion = tokens
            else: rname, rversion = tokens[0], None
            try:
                package = __import__(rname)
                if rversion is not None:
                    op = r.replace(rname, '').replace(rversion, '').strip()
                    if not eval(repr(tuple(float(x) for x in package.__version__.split('.'))) + op + repr(tuple(float(x) for x in rversion.split('.')))):
                        not_found_packages.append(r)
            except ModuleNotFoundError: not_found_packages.append(rname)
        if len(not_found_packages) > 0:
            raise ModuleNotFoundError(f"'{self.name}' cannot be used without dependencies {repr(not_found_packages)}.")
            
    def check(self):
        self.check_requires()
        return self
    
    def version_update(self):
        if hasattr(self, 'version'):
            version = re.sub(r"((\d+\.){2})(\d+)", lambda x: x.group(1)+str(eval(x.group(3))+1), self.version)
        else: version = '1.0.0'
        self.version = version
    
    def __enter__(self): return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == ModuleNotFoundError:
            module_name = str(exc_value).split("'")[1]
            raise ModuleNotFoundError(f"'{self.name}' cannot be used without dependency '{module_name}'. ")
        elif exc_type == ImportError:
            func_name, _, module_name = str(exc_value).split("'")[1:4]
            raise ImportError(f"'{self.name}' requires '{func_name}' in module '{module_name}'. Please double check the version of packages. ")
        
    def __str__(self):
        args = ',\n'.join([self.tab + k + ' = ' + repr(getattr(self, k)) for k in self.order])
        return "info_manager(\n" + args + "\n)"
    
    def __getitem__(self, name):
        return getattr(self, name)
    
    def __setattr__(self, name, value):
        if hasattr(self, 'order') and name not in self.order: self.order.append(name)
        super().__setattr__(name, value)
    
    def get(self, name, value):
        if hasattr(self, name): return self[name]
        return value
