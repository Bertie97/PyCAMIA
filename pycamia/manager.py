
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
    Hyper
    hypers
    Version
""".split()

import os, re, sys, time, builtins
from optparse import OptionParser

from .environment import get_environ_vars
from .strop import tokenize

update_format = "%Y-%m-%d %H:%M:%S"

class Version(tuple):
    def __new__(cls, string):
        matches = re.findall(r"[0-9\.]+", string)
        if len(matches) == 0: raise TypeError(f"{string} is not a version")
        return super().__new__(cls, [eval(x) for x in matches[0].split('.') if x])
    for op in "gt lt ge le eq ne".split():
        exec(f"def __{op}__(self, x): return super(Version, self).__{op}__(Version(x) if isinstance(x, str) else x)")

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
        file_path = get_environ_vars()['__file__']
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
            except ModuleNotFoundError: not_found_packages.append(rname)
            if rversion is not None:
                op = r.replace(rname, '').replace(rversion, '').strip()
                if not eval("Version(package.__version__)" + op + "'" + rversion + "'"):
                    not_found_packages.append(r)
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
        self.update = time.strftime(update_format, time.localtime(time.time()))
    
    def __enter__(self): return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == ModuleNotFoundError:
            module_name = str(exc_value).split("'")[-2]
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

class Hyper:
    def __init__(self):
        self.optParser = OptionParser()

    def add_hyper(self, *names, default=None, type=None, help_str='', **kwargs):
        if 'name' in kwargs: names += (kwargs['name'],)
        if len(names) == 0: raise TypeError("add_hyper requires a name. ")
        if type is None: type = builtins.type(default)
        if help_str == '': help_str = f"variable \'{kwargs.get('name', names[0])}\'"

        new_names = []
        for name in names:
            name = name.lstrip('-')
            abbr_match = re.match(r'\[(.)\]', name)
            if abbr_match is not None:
                abbr = abbr_match.group(1)
                name = name[:abbr_match.span()[0]] + abbr + name[abbr_match.span()[1]+1:]
                new_names.extend([f'-{abbr}', f'--{name}'])
            else:
                new_names.append(('-' if len(name) > 1 else '') + f'-{name}')

        if type == bool:
            self.optParser.add_option(
                *new_names, action = 'store_true' if default else 'store_false', 
                dest = name, default = default, help = help_str
            )
        else:
            self.optParser.add_option(
                *new_names, action = 'store', type = type, 
                dest = name, default = default, help = help_str
            )
            
    def parse_args(self, *args):
        res = self.optParser.parse_args(*args)
        self.__dict__.update(vars(res[0]))
        return res
    
    def parse_name(self, *args):
        self.parse_args(*args)
        return str(self)

    def __str__(self):
        defaults = vars(self.optParser.get_default_values())
        values = vars(self)
        values.pop('optParser')
        updated = {k: v for k, v in values.items() if k in defaults and v != defaults[k]}
        if len(updated) == 0: return "baseline"
        token = lambda k, v: str(k) + '_' + re.sub(r'\.0+$', '', str(v)[-10:])
        return '-'.join([token(k, v) for k, v in updated])

def hypers(**kwargs):
    hyper = Hyper()
    for k, v in kwargs.items():
        hyper.add_hyper(k, default=v, help_str=kwargs.get(k+'_help', ''))
    variables, args = hyper.parse_args(sys.argv)
    vs = get_environ_vars()
    vs.update(vars(variables))
    return hyper
