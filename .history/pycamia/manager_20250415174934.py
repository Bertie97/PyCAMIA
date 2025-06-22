
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
    Args
    args
""".split()

import os, re, sys, math, time, builtins, types, json
from optparse import OptionParser
from argparse import ArgumentParser

from .environment import get_environ_vars, get_reference_line
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

    Examples::
        Code.py:
            __info__ = info_manager(
                project="PyCAMIA", 
                package="", 
                requires="xxx"
            ).check()
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
        env = get_environ_vars()
        file_path = env['__file__']
        file = os.path.extsep.join(os.path.basename(file_path).split(os.path.extsep)[:-1])
        self.name = '.'.join([x for x in [project, package, file] if x and x != "__init__"])
        self.tab = ' ' * 4
        major_keys = "project package requires".split()
        self.order = major_keys + list(set(properties.keys()) - set(major_keys))
        env['__args__'] = Args(sys.argv)
        
    def check_requires(self):
        not_found_packages = []
        for r in self.requires:
            tokens = tokenize(r, ['<', '>', '='], strip=None, keep_empty=False)
            if len(tokens) == 2: rname, rversion = tokens
            else: rname, rversion = tokens[0], None
            try:
                package = __import__(rname)
            except ModuleNotFoundError: not_found_packages.append(rname); continue
            if rversion is not None:
                op = r.replace(rname, '').replace(rversion, '').strip()
                if not eval("Version(package.__version__)" + op + "'" + rversion + "'"):
                    not_found_packages.append(r)
        if len(not_found_packages) > 0:
            raise ModuleNotFoundError(f"'{self.name}' cannot be used without dependencies {repr(not_found_packages)}.")
            
    def check(self):
        self.check_requires()
        return self
    
    def auto_install_packages(self):
        queue = self.requires.copy()
        def install(r, update=False):
            os.system(f"pip install{' --update' if update else ' '}{r}")
        while True:
            try: r = queue.pop(0)
            except IndexError: break
            tokens = tokenize(r, ['<', '>', '='], strip=None, keep_empty=False)
            if len(tokens) == 2: rname, rversion = tokens
            else: rname, rversion = tokens[0], None
            
            try: package = __import__(rname)
            except ModuleNotFoundError as error:
                error_msg = error.__str__()
                if "without dependencies" in error_msg:
                    module_names = eval(error_msg.split('[')[-1].split(']')[0])
                else: module_names = [error_msg.split("'")[1]]
                for module_name in module_names:
                    install(module_name)
                queue.append(r); continue
            if rversion is not None:
                op = r.replace(rname, '').replace(rversion, '').strip()
                if not eval("Version(package.__version__)" + op + "'" + rversion + "'"):
                    install(r, update=True); queue.append(r)
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

class const:

    value = None
    
    @classmethod
    def __class_getitem__(cls, value):
        cls.value = value
        return cls

class Args:
    
    """
    Argument manager for a python script. An enclosure of argparse.ArgumentParser. 

    Example::
        >>> args = Args("a.py -fv --n_epoch 4 --n_batch 3".split()[1:])
        >>> args.n_epoch = 5
        >>> args.n_batch[int: "The batch size for the network training. "] = 5
        >>> args.gpu[list[2]: "The 2 GPU ids. "] = [2, 3]
        >>> with args('-f'): args.flag[bool: "A flag argument which is False if --flag is not given. "] = False
        >>> with args('-v', '--use_valid'): args.n_iter_each_valid[Args.const[10]: "Whether to use validation or not. "] = int(1e6)
        >>> args
        Args(n_epoch=4, n_batch=3, gpu=[2, 3], flag=True, n_iter_each_valid=10)
        >>> args.n_batch
        3
        >>> print(args.help())
        usage: <code> [-h] [--n_epoch N_EPOCH] [--n_batch N_BATCH] [--gpu GPU GPU] [-f [FLAG]] [-v [N_ITER_EACH_VALID]]

        options:
        -h, --help            show this help message and exit
        --n_epoch N_EPOCH     The maximal number of epochs for training.
        --n_batch N_BATCH     The batch size for the network training.
        --gpu GPU GPU         The 2 GPU ids.
        -f [FLAG]             A flag argument which is False if --flag is not given.
        -v [N_ITER_EACH_VALID], --use_valid [N_ITER_EACH_VALID]
                                Whether to use validation or not.
    """
    
    default_help_strs = dict(
        n_batch = "Batch size for each iteration. ", 
        batch_size = "Batch size for each iteration. ", 
        n_epoch = "The maximal number of epochs for training. ", 
        epochs = "The maximal number of epochs for training. ", 
        max_epoch = "The maximal number of epochs for training. ", 
        init_epoch = "The index of epoch for the first iteration, used in restarts of training. ",
        i_gpu = "The id of GPU(s)", 
        gpu = "The id of GPU(s)", 
    )
    
    const = const
    
    @staticmethod
    def tags(self, name):
        abbr_match = re.match(r'\[(.)\]', name)
        if abbr_match is not None:
            abbr = abbr_match.group(1)
            name = name[:abbr_match.span()[0]] + abbr + name[abbr_match.span()[1]+1:]
            names=[f'-{abbr}', f'--{name.lstrip("-")}']
        else:
            names=[('-' if len(name) > 1 else '') + f'-{name}']
        return names
    
    class Arg:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            
        def __setitem__(self, subscript, value):
            if isinstance(subscript, str):
                _type = value.__class__
                _help = subscript
            elif isinstance(subscript, (type, types.GenericAlias)):
                _type = subscript
                _help = Args.default_help_strs.get(self.name, f"Argument '{self.name}' for {self.parent.prog}. ")
            elif isinstance(subscript, slice):
                if subscript.step is not None: raise TypeError("Only one colon is allowed in argument expression. ")
                _type = subscript.start
                _help = subscript.stop
            else: raise TypeError(f"Only types and help string can be specified in subscripts: such as args.{self.name}['help'], args.{self.name}[list[{{len}}]] or args.{self.name}[Args.const(2):'help']")
            if _type == Args.const: nargs = '?'; const = _type.value; _type = type(_type.value)
            elif _type == bool:
                nargs = '?'
                if isinstance(value, bool): const = not value
                else: const = True
            else:
                if not isinstance(_type, (type, types.GenericAlias)): raise TypeError("Only types or Args.const is allowed before colon. ")
                if issubclass(getattr(_type, '__origin__', _type), (tuple, list)) and isinstance(_type, types.GenericAlias):
                    nargs = _type.__args__
                    if len(nargs) > 2: raise TypeError("Only type and length can be set for list/tuple arguments. ")
                    elif len(nargs) == 2:
                        if not isinstance(nargs[0], type): raise TypeError(f"The first item of list/tuple[*, 1] representation should be a type, instead of {type(args[0])}. ")
                        _type = _type.__origin__[nargs[0]]
                        nargs = nargs[1]
                    elif isinstance(nargs[0], type):
                        _type = _type.__origin__[nargs[0]]
                        nargs = None
                    if nargs is not None:
                        if not isinstance(nargs, (int, slice)): raise TypeError(f"The length identifier of list/tuple[int, *] representation should be an integer or slice, instead of {type(args[0])}. ")
                        if isinstance(nargs, slice): nargs = '+'
                        nargs = {'>0': '+', ...: '*', -1: '*'}.get(nargs, nargs)
                elif str(_type.__class__) == "<class 'pyoverload.typings.ArrayType'>":
                    nargs = len(_type)
                    if nargs < 0: nargs = '*'
                    _type = _type.origin_type[_type.base_type[0]]
                else:
                    nargs = (len(value) if len(value) > 0 else '*') if isinstance(value, (tuple, list)) else None
                const = None
            tags = self.parent.additional_tags
            if not tags: tags = ['--' + self.name]
            dest = self.name
            if callable(value): func_str = get_reference_line().split('=', 1)[1].strip()
            elif isinstance(value, str) and value.startswith("[func:]"):
                func_str = value[len("[func:]"):]
                value = eval(func_str)
            else: func_str = ''
            self.parent.add_argument(*tags, nargs=nargs, const=const, dest=dest, type=_type, default=value, help=_help, func_str=func_str)
            if not self.parent.parse_attempt(key=self.name):
                return super(Args, self.parent).__setattr__(self.name, value)
            
    class Scope:
        
        def __init__(self, parent, *tags):
            if len(tags) == 1 and isinstance(tags[0], (tuple, list)): tags = tags[0]
            self.tags = list(tags)
            self.parent = parent
            
        def __enter__(self): self.parent.additional_tags = self.tags
        def __exit__(self, type, value, trace): self.parent.additional_tags = []
        
    class Argument:
        
        def __init__(self, *tags, nargs=1, const=None, dest='', type=None, default=None, help=''):
            self.name = dest
            fullnames = []
            abbrs = []
            for tag in tags:
                if tag.startswith('--'): fullnames.append(tag.lstrip('-'))
                elif tag.startswith('-'): abbrs.append(tag.lstrip('-'))
                else: raise TypeError(f"Invalid argument tag '{tag}' in '{tags}' for variable {dest}. ")
            if len(set(abbrs)) < len(abbrs): raise TypeError(f"Duplicated abbreviation in {abbrs}. ")
            if len(fullnames) > 0:
                found_abr = {}
                for abr in abbrs:
                    for name in fullnames:
                        if name in found_abr: continue
                        if abr.lower() in name.lower(): found_abr[name] = [abr]; break
                    else:
                        for name in fullnames:
                            if name not in found_abr: continue
                            if abr.lower() in name.lower(): found_abr[name].append(abr); break
                        else: raise TypeError(f"Inappropriate abbreviation '{abr}' as it is not in '{'/'.join(fullnames)}'. ")
                def mark_name(name, chars):
                    for ch in chars:
                        i = name.lower().index(ch.lower())
                        name = name[:i] + '[' + ch + ']' + name[i+1:]
                    return name
                self.tag_label = '--' + '/'.join(name if name not in found_abr else mark_name(name, found_abr[name]) for name in fullnames)
            else: self.tag_label = '-' + ''.join(abbrs)
            self.tags = tags
            self.nargs = 1 if nargs is None else nargs
            self.const = const
            self.dest = dest
            self.type = builtins.type(default) if type is None else type
            self.default = default
            self.help = help
            self.env = None
        
        @property
        def optional(self): return self.default is not None
        
        def __repr__(self): return f"<Argument {self.name}[{self.type.__name__}]: {'/'.join(self.tags)}>"

    properties = ['argv', 'prog', 'description', 'arguments', 'arg_names', 'func_strs', 'additional_tags', 'end_parse', 'calling_help', 'env']
    
    def __init__(self, argv=None, prog='', description=''):
        if not prog:
            if argv is None: prog, *argv = sys.argv
            else: prog = '<code>'
        elif argv is None: _, *argv = sys.argv
        self.argv = argv
        self.prog = prog
        self.description = description
        
        self.arguments = [Args.Argument('-h', '--help', nargs='?', dest='help', help="Show this help message and exit. ")]
        self.arg_names = []
        self.func_strs = {}
        self.additional_tags = []
        self.end_parse = False
        self.calling_help = False
    
    def __getattribute__(self, name):
        try: return super().__getattribute__(name)
        except AttributeError:
            if self.end_parse: raise AttributeError(f"Argument '{name}' is not defined. ")
            return Args.Arg(self, name)
    
    def __setattr__(self, name, value):
        if name in Args.properties: return super(Args, self).__setattr__(name, value)
        if self.end_parse and name in self.arg_names: return super(Args, self).__setattr__(name, value)
        if name in self.arg_names:
            i = self.arg_names.index(name)
            self.arg_names.pop(i)
            self.arguments.pop(i)
        help_str = Args.default_help_strs.get(name, f"Argument '{name}' for {self.prog}. ")
        tags = self.additional_tags
        if not tags: tags = ['--' + name]
        dest = name
        if callable(value): func_str = get_reference_line().split('=', 1)[1].strip()
        elif isinstance(value, str) and value.startswith("[func:]"):
            func_str = value[len("[func:]"):]
            value = eval(f"lambda args: ({func_str})", self.env.globals, self.env.locals)(self)
        else: func_str = ''
        self.add_argument(*tags, dest=dest, type=type(value), default=value, help=help_str, func_str=func_str)
        if not self.parse_attempt(key=name):
            return super(Args, self).__setattr__(name, value)
        
    def __call__(self, *tags):
        return Args.Scope(self, *tags)
    
    def __enter__(self):
        self.end_parse = False
        return self
    
    def __exit__(self, error_type, error_str, traceback):
        if error_type is not None:
            raise error_type(error_str)
        self.fix_arguments()
        
    @property
    def usage(self):
        flags = []
        fixeds = []
        optionals = []
        for arg in self.arguments:
            if not arg.tag_label.startswith('--') and arg.nargs in ('?', 0):
                flags.append(arg.tag_label.lstrip('-')); continue
            optional = arg.optional
            if arg.nargs == '?': optional = True; nargs = 0
            elif arg.nargs in ('+', '*'): nargs = 1
            else: nargs = arg.nargs
            arg_usage = ' '.join([arg.tag_label] + [arg.name.upper()] * nargs)
            if arg.nargs in ('+', '*'): arg_usage += arg.nargs
            if optional: optionals.append(f'[{arg_usage}]')
            else: fixeds.append(arg_usage)
        fregs = [self.prog]
        fregs.extend(fixeds)
        if flags: fregs.append('-' + ''.join(flags))
        fregs.extend(optionals)
        return ' '.join(fregs)
    
    @property
    def help_str(self, indent=2, usage_width=20, help_width=90):
        el = '\n'; usage_indent = ' ' * indent; help_indent = ' ' * (indent * 2 + usage_width)
        string = "usage: " + self.usage + el*2
        string += "options: " + el
        for arg in self.arguments:
            optional = "; optional" if arg.optional else ''
            if arg.nargs == '?': optional = "; flag"; nargs = 0
            elif arg.nargs in ('+', '*'): nargs = 1
            else: nargs = arg.nargs
            arg_usages = ', '.join(' '.join([tag] + [arg.name.upper()] * nargs) + (arg.nargs if arg.nargs in ('+', '*') else '') for tag in arg.tags)
            if arg.name != 'help': help_content = f"[{arg.type.__name__}{optional}] " + arg.help.strip() + f" Defaults to {arg.default!r}. "
            else: help_content = arg.help
            help_lines = []
            ipointer = 0
            while True:
                try: last_space = help_content.rindex(' ', ipointer, ipointer + help_width)
                except ValueError:
                    line = help_content[ipointer:]
                    if line.strip(): help_lines.append(line)
                    break
                line = help_content[ipointer:last_space]
                if line.strip(): help_lines.append(line)
                ipointer = last_space + 1
            help_string = (el+help_indent).join(help_lines)
            help_string += el
            if len(arg_usages) > usage_width:
                string += usage_indent + arg_usages + el
                string += help_indent + help_string
            else: string += usage_indent + arg_usages + ' ' * (usage_width - len(arg_usages) + indent) + help_string
        return string
    
    def register(self, *tags, nargs=0, const=None, dest='', type=None, default=None, help=''):
        if not dest: raise TypeError(f"Undecided variable name for '{tags}'")
        self.arguments.append(Args.Argument(*tags, nargs=nargs, const=const, dest=dest, type=type, default=default, help=help))
        self.arg_names.append(dest)
    
    def add_argument(self, *tags, nargs=0, const=None, dest='', type=None, default=None, help='', func_str=''):
        if not dest: raise TypeError(f"Undecided variable name for '{tags}'")
        self.arguments.append(Args.Argument(*tags, nargs=nargs, const=const, dest=dest, type=type, default=default, help=help))
        self.arg_names.append(dest)
        if func_str: self.func_strs[dest] = "[func:]" + func_str
    
    def parse_attempt(self, input_argv=None, key=None):
        if self.calling_help: return False
        if input_argv is None: input_argv = self.argv
        # flags = []
        # for arg in self.arguments:
        #     if not arg.tag_label.startswith('--') and arg.nargs in ('?', 0):
        #         flags.append(arg.tag_label.lstrip('-'))
        key_found = False
        section = []
        cur_tag = None
        for va in input_argv + ['-']:
            if not va.startswith('-'):
                # All tokens between two adjacent tags are sent two the previous tag.
                if cur_tag is not None: section.append(va)
                continue
            if cur_tag is not None:
                # Find the argument. 
                if not cur_tag.startswith('-'): raise RuntimeError(f"Invalid form of tag: {cur_tag!r}")
                if not cur_tag.startswith('--'): tags = ['-' + t for t in cur_tag[1:]]
                else: tags = [cur_tag]
                for tag in tags:
                    for arg in self.arguments:
                        if tag in arg.tags: break
                    else:
                        # Failed to find argument. 
                        if key is None: return False
                        continue
                    # Argument found, proceed. 
                    if key is not None and arg.name == key: key_found = True
                    if arg.name == 'help': self.calling_help = True; return False
                    if isinstance(arg.nargs, int) and arg.nargs != 1 or arg.nargs in ('+', '*'):
                        # Deal with the list. 
                        try:
                            if issubclass(getattr(arg.type, '__origin__', arg.type), (tuple, list)) and isinstance(arg.type, types.GenericAlias):
                                subtype = arg.type.__args__[0]
                                value = arg.type.__origin__([subtype(x) for x in section])
                            else: value = arg.type(section)
                        except ValueError: raise TypeError(f"Error in casting {' '.join(section)} to type {arg.type} for argument {arg.name}: {tag} {' '.join(section)}. ")
                        if isinstance(arg.nargs, int) and arg.nargs > 0 and len(value) != arg.nargs:
                            raise TypeError(f"Error in assigning {value} (of length {len(value)}) to argument {arg.name}: {tag} {' '.join(section)}. ")
                    else:
                        value_str = ' '.join(section)
                        if not value_str and arg.nargs == '?':
                            value = arg.const
                        elif value_str.startswith("[func:]"):
                            func_str = value_str[len("[func:]"):]
                            value = eval(func_str)
                            if issubclass(arg.type, builtins.type) or not hasattr(arg.type, '__call__'):
                                raise TypeError(f"Error in casting function string {value_str!r} to non-function type {arg.type} for argument {arg.name}: {tag} {value_str}. ")
                        else:
                            try: value = arg.type(value_str)
                            except ValueError: raise TypeError(f"Error in casting {value_str!r} to type {arg.type} for argument {arg.name}: {tag} {value_str}. ")
                    super(Args, self).__setattr__(arg.name, value)
            cur_tag = va
            section = []
        if key is not None: return key_found
        return True
    
    def complete(self):
        self.fix_arguments()
    
    def fix_arguments(self):
        self.end_parse = True
        if self.calling_help:
            print(self.help_str)
            exit(0)
    
    def help(self):
        return self.help_str
    
    def to_dict(self):
        return {a: self.func_strs.get(a, getattr(self, a)) for a in self.arg_names}
    
    @classmethod
    def from_dict(cls, obj):
        with cls() as args:
            args.env = get_environ_vars()
            for k, v in obj.items(): setattr(args, k, v)
        return args
    
    def save(self, file_path):
        with open(file_path, "w") as arg_file:
            json.dump(self.to_dict(), arg_file, indent=4)
    
    @classmethod
    def load(self, file_path):
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found. ")
        with open(file_path, "r") as arg_file:
            args = Args.from_dict(json.load(arg_file))
        return args
    
    def __str__(self):
        return f"Args({', '.join(f'{a}={getattr(self, a)!r}' for a in self.arg_names)})"
    
    __repr__ = __str__
    
args = Args()
