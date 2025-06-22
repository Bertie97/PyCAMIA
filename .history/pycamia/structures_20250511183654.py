
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Useful python structures",
    requires = ""
)

__all__ = """
    struct
    odict
    protocol
""".split()

from typing import Mapping, Iterable

class struct(dict):
    def __setattr__(self, name, value):
        self[name] = value
    def __getattribute__(self, name):
        try: return super().__getattribute__(name)
        except AttributeError: return self[name]
    def __str__(self):
        return "struct {\n" + ',\n'.join(f"    {n} = {v}" for n, v in self.items()) + "\n}"
    
class odict(dict):
    def __new__(cls, *args, **kwargs):
        if len(args) == 1: arg = args[0]
        self = super().__new__(cls, *args, **kwargs)
        if len(args) == 0: self._keys = []
        elif isinstance(arg, Mapping): self._keys = [x[0] for x in arg.items()]
        elif isinstance(arg, Iterable): self._keys = [x[0] for x in arg]
        elif len(kwargs) > 0: self._keys = list(kwargs.keys())
        else: self._keys = []
        return self

    @classmethod
    def __class_getitem__(cls, args: tuple[slice]):
        if not isinstance(args, tuple): args = (args,)
        self = cls()
        for arg in args:
            if not isinstance(arg, slice): raise TypeError("Creating ordered dict by odict[x:y] format accepts only 'slice' objects in subscript. ")
            if arg.step is not None: raise TypeError("No more than one colon is accepted for odict[x:y] format. ")
            self._keys.append(arg.start)
            self[arg.start] = arg.stop
        return self
    
    def __iter__(self):
        for k in self._keys: yield k
    def __setitem__(self, key, value):
        if key not in self._keys:
            self._keys.append(key)
        return super().__setitem__(key, value)
    def update(self, new_dict):
        self._keys.extend([k for k in new_dict.keys() if k not in self._keys])
        return super().update(new_dict)
    def items(self):
        for k in self._keys:
            yield k, self[k]
    def pop(self, key):
        self._keys.remove(key)
        return super().pop(key)
    def index(self, key):
        return self._keys.index(key)
    def key(self, index):
        return self._keys[index]
    def __str__(self):
        return '[' + ', '.join(f"{k!r}: {v!r}" for k, v in self.items()) + ']'

class protocol:
    def __new__(cls, *requires, optional=[]):
        requires = list(requires)
        if isinstance(optional, tuple): optional = list(optional)
        if not isinstance(optional, list): optional = [optional]
        for name, value in cls.__dict__.items():
            if name.startswith('__') and name.endswith('__'): continue
            if not callable(value): continue
            if name in requires: requires.remove(name)
            elif name not in optional: raise NameError(f"Protocol {cls.__name__} should not have {name!r} method. ")
            if callable(value):
                setattr(cls, name, staticmethod(value))
        if len(requires) > 0:
            raise NameError(f"Protocol {cls.__name__} requires {', '.join(requires)} method(s). ")
        return super().__new__(cls)
