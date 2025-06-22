
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File to read image data.",
    help = "Use `@register.subject` to create subjects and `@register.data` to create different data types.",
    requires = ''
)

import gc
import random
from .trans import Transformation
from .stdio import IMG as _IMG
from .stdio import affine2orient
IMG = lambda *a, **k: _IMG(*a, **k, warn=False)
from .zxhtools.TRS import TRS
try: from PIL import Image; available_Image = True
except ImportError: available_Image = False

with __info__:
    import batorch as bt
    from pycamia import alias, Path, SPrint, ByteSize
    from pycamia import avouch, touch, to_tuple, arg_tuple, execblock, unique, protocol

__all__ = """
    Key
    DataObject
    BatchObject
    Dataset
    LossCollector
""".split()
    # Subject
    # ImageObject
    # Dataset
    # MedicalDataset
    
def deapprox(x):
    if x.__class__.__name__ == 'approx_T': return x.__class__.__mro__[1](x)
    return x

def approx(x):
    """
    A value that prints with prefix '≈' indicating itself as an approximation
    """
    if x == NotImplemented: return x
    class approx_T(x.__class__):
        def __init_subclass__(cls, **kwargs): raise TypeError("Class 'approx_T' should not be inheritted. ")
        for __op__ in [f"__{op}__" for op in """
                add radd iadd sub rsub isub mul rmul imul
                truediv rtruediv itruediv floordiv rfloordiv ifloordiv
                mod rmod imod pow rpow ipow lshift rshift
            """.split() if op]:
            exec(f"def {__op__}(self, other): return approx(getattr(deapprox(self), '{__op__}')(deapprox(other)))")
        def __str__(self): return '≈' + super().__str__()
    return approx_T(x)

class SortedDict(dict):
    
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], dict):
            return self.__init__dict(*args, **kwargs)
        if len(args) > 1 and not isinstance(args[0], str):
            return self.__init__keyvalue(*args, **kwargs)
        return self.__init__default(*args, **kwargs)

    def __init__keyvalue(self, keys:list=[], values:list=[]):
        super().__init__(zip(keys, values))
        self.key_order = keys

    def __init__dict(self, dic:dict, order=None):
        super().__init__(dic)
        if order is None: self.key_order = list(dic.keys())
        else: self.key_order = order

    def __init__default(self, *args, **kwargs):
        super().__init__(kwargs)
        if len(args) == 0: self.key_order = list(kwargs.keys())
        else: self.key_order = list(arg_tuple(args))
    
    # def __init__(self, *args, **kwargs): ...
    
    # def __new__(cls, *args, **kwargs):
    #     if len(args) > 0 and isinstance(args[0], dict):
    #         return SortedDict.__new__dict(*args, **kwargs)
    #     if len(args) > 1 and not isinstance(args[0], str):
    #         return SortedDict.__new__keyvalue(*args, **kwargs)
    #     return SortedDict.__new__default(*args, **kwargs)

    # @classmethod
    # def __new__keyvalue(cls, keys:list=[], values:list=[]):
    #     self = super().__new__(cls, zip(keys, values))
    #     self.key_order = keys
    #     return self

    # @classmethod
    # def __new__dict(cls, dic:dict, order=None):
    #     self = super().__new__(cls, dic)
    #     if order is None: self.key_order = list(dic.keys())
    #     else: self.key_order = order
    #     return self

    # @classmethod
    # def __new__default(cls, *args, **kwargs):
    #     self = super().__new__(cls, kwargs)
    #     if len(args) == 0: self.key_order = list(kwargs.keys())
    #     else: self.key_order = list(arg_tuple(args))
    #     return self
    
    # def __init__(self, *args, **kwargs): ...

    def sort(self): self.key_order.sort()

    def shuffle(self): random.shuffle(self.key_order)
        
    def first(self): return self.key_order[0], self[self.key_order[0]]

    def keys(self):
        for k in self.key_order: yield k

    def values(self):
        for k in self.key_order: yield self[k]

    def items(self):
        for k in self.key_order: yield (k, self[k])

    def setdefault(self, k, v):
        if k not in self: self.key_order.append(k)
        super().setdefault(k, v)

    def pop(self, k, value):
        if k in self.key_order: self.key_order.remove(k)
        return super().pop(k, value)

    def __setitem__(self, k, v):
        if k not in self: self.key_order.append(k)
        super().__setitem__(k, v)

    def update(self, dic):
        for k, v in dic.items(): self[k] = v
        
    def copy(self):
        return SortedDict(super().copy(), order = self.key_order)
    
    def __iter__(self): return self.keys()
    
    @alias('__repr__')
    def __str__(self): return '{' + ', '.join([f"{repr(k)}: {repr(v)}" for k, v in self.items()]) + '}'
    
    def __len__(self): return len(self.key_order)
    
    def get_key(self, i): return self.key_order[i]

    def get_value(self, i): return self[self.key_order[i]]
    
    def get_item(self, i): k = self.key_order[i]; return (k, self[k])
    
    def __getitem__(self, k):
        if isinstance(k, slice):
            key_order = self.key_order[k]
            return SortedDict({i: self[i] for i in key_order}, key_order)
        return_value = super().get(k, None)
        if return_value is None and isinstance(k, int): return_value = self[self.get_key(k)]
        return return_value
    
    def filter(self, new_key_list):
        return SortedDict(
            [k for k in new_key_list if k in self.key_order], 
            [self[k] for k in new_key_list if k in self.key_order]
        )

def is_placeholder(token): return token is ... or isinstance(token, str) and token.startswith('[') and token.endswith(']')
def placeholder_values(key, pattern):
    if not isinstance(key, Key) or not isinstance(pattern, Key): return
    values = Key()
    for n, p in pattern.items():
        if not isinstance(p, (list, tuple)): p = [p]
        accept = False
        got_placeholder = None
        for choice in p:
            if is_placeholder(choice): got_placeholder = choice; continue
            key_value = key.get(n, None)
            if key_value == choice: accept = True; break
        if not accept:
            if got_placeholder is None: return
            values[n if got_placeholder is ... else (n+got_placeholder)] = key[n]
    return values
def with_placeholder_values(pattern, values):
    if not isinstance(pattern, Key): return
    new_pattern = Key()
    for n, p in pattern.items():
        if not isinstance(p, (list, tuple)): p = [p]
        q = []
        for choice in p:
            if choice is ...: q.append(values[n])
            elif is_placeholder(choice):
                if n + choice not in values: return
                q.append(values[n + choice])
            else: q.append(choice)
        if len(q) == 1: new_pattern[n] = q[0]
        else: new_pattern[n] = q
    return new_pattern
def is_subkey(key_sub, key_cons):
    return placeholder_values(key_sub, key_cons) is not None
    
class Key:
    
    main_key_name = None
    
    @classmethod
    def __class_getitem__(cls, main_key_name):
        cls.main_key_name = main_key_name
        return cls
    
    def __init__(self, **kwargs):
        self._property_only = kwargs.pop('property_only', None)
        self._keys = []
        self._is_hidden = []
        self._hidden_values = {}
        self._visible_values = {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                self._keys.append(k[1:])
                self._is_hidden.append(True)
                self._hidden_values[k[1:]] = v
            else:
                self._keys.append(k)
                self._is_hidden.append(False)
                self._visible_values[k] = v
                
    def update(self, **kwargs):
        for k, v in kwargs.items():
            is_hidden = False
            if k.startswith('_'): k = k[1:]; is_hidden = True
            elif k in self._keys: is_hidden = self._is_hidden[self._keys.index(k)]
            if k not in self._keys:
                self._keys.append(k)
                self._is_hidden.append(is_hidden)
            if is_hidden:
                self._hidden_values[k] = v
            else: self._visible_values[k] = v
        return self
            
    def pop(self, k):
        i = self._keys.index(k)
        self._keys.pop(i)
        is_hidden = self._is_hidden.pop(i)
        if is_hidden: return self._hidden_values.pop(k)
        return self._visible_values.pop(k)
            
    def tuple(self):
        return tuple((k, self[k]) for k in self._keys) + (self._property_only,)
    
    def __getitem__(self, k):
        if isinstance(k, int): k = self._keys[k]
        if k.startswith('_'): return self._hidden_values[k[1:]]
        if k in self._keys:
            if self._is_hidden[self._keys.index(k)]:
                return self._hidden_values[k]
            else: return self._visible_values[k]
        raise IndexError(f"No '{k}' in Key object. ")
    
    def __setitem__(self, k, v):
        if isinstance(k, int): k = self._keys[k]
        if k.startswith('_'):
            self._hidden_values[k[1:]] = v
            if k[1:] not in self._hidden_values:
                self._keys.append(k[1:])
                self._is_hidden.append(True)
            return
        if k in self._keys:
            if self._is_hidden[self._keys.index(k)]:
                self._hidden_values[k] = v
            else: self._visible_values[k] = v
        else:
            self._keys.append(k)
            self._is_hidden.append(False)
            self._visible_values[k] = v
            
    def __getattr__(self, k):
        return self[k]
            
    def __len__(self): return len(self._keys)
    
    def __contains__(self, k): return k.lstrip('_') in self._keys
    
    def fuzzy_get(self, *keys):
        for k in keys:
            if k in self: return self[k]
    
    def keys(self): return ['_' if h else '' + k for h, k in zip(self._is_hidden, self._keys)]
    def visible_keys(self): return [k for h, k in zip(self._is_hidden, self._keys) if not h]
    def hidden_keys(self): return [k for h, k in zip(self._is_hidden, self._keys) if h]
    def values(self): return [self[k] for k in self._keys]
    def items(self): return zip(self.keys(), self.values())
    def get(self, k, v): return self[k] if k in self else v
    def dict(self): return dict(self.items())
    def copy(self): return Key(**self.dict())
    for __op__ in [f'__{op}__' for op in "lt le gt ge eq".split()]:
        execblock(f"""
        def {__op__}(x, y):
            return x.tuple().{__op__}(getattr(y, 'tuple', lambda: touch(lambda: tuple(y), y))())
        """)

    def __hash__(self): return hash(self.tuple())
    
    @alias('__repr__')
    def __str__(self):
        main_key_list = [f"[{self.main_key_name}]: {repr(self.main_key)}"] if self.main_key_name in self._keys else []
        return '{' + ', '.join(main_key_list + [('({})' if h else '{}').format(f"{k}: {repr(self[k])}") for k, h in zip(self._keys, self._is_hidden) if k != self.main_key_name]) + '}' + (f".{self._property_only}" if self._property_only is not None else '')
    
    def __call__(self, **kwargs):
        self = self.copy()
        for k, v in kwargs.items():
            avouch(k in self._keys, f"cannot change '{k}' in Key as no such property is defined. ")
            self[k] = v
        return self
    
    def with_item(self, **kwargs):
        self = self.copy()
        for k, v in kwargs.items(): self[k] = v
        return self
    
    @property
    def main_key(self):
        return self.get(self.main_key_name, None)
    
    @property
    def object(self):
        return Key(property_only=None, **self.dict())
    
    @property
    def data(self):
        return Key(property_only='data', **self.dict())
    
    @property
    def orientation(self):
        return Key(property_only='orientation', **self.dict())
    
    @property
    def affine(self):
        return Key(property_only='affine', **self.dict())
    
    def to_property(self, target):
        if isinstance(target, str):
            return Key(property_only=target, **self.dict())
        elif isinstance(target, Key):
            return Key(property_only=target._property_only, **self.dict())
        else: raise TypeError(f"target should be str or Key, but got {type(target)}. ")
    
    @property
    def is_object(self): return self._property_only == None
    
    @property
    def is_data(self): return self._property_only == 'data'
    
    @property
    def is_orientation(self): return self._property_only == 'orientation'
    
    @property
    def is_affine(self): return self._property_only == 'affine'

# class Info:
#     def __init__(self, **kwargs):
#         self._keys = []
#         self._values = []
#         self._hidden_keys = []
#         self._hidden_values = []
#         for k, v in kwargs.items():
#             if k.startswith('_'): 
#                 self._hidden_keys.append(k[1:])
#                 self._hidden_values.append(v)
#                 continue
#             self._keys.append(k)
#             self._values.append(v)

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if k in self._keys:
#                 self._values[self._keys.index(k)] = v
#                 continue
#             if k.startswith('_') and k[1:] in self._hidden_keys:
#                 self._hidden_values[self._hidden_keys.index(k[1:])] = v
#                 continue
#             if k.startswith('_'): 
#                 self._hidden_keys.append(k[1:])
#                 self._hidden_values.append(v)
#                 continue
#             self._keys.append(k)
#             self._values.append(v)

#     def tuple(self):
#         return tuple(self._values)
    
#     def visible(self):
#         return self._keys

#     def keys(self):
#         return self._keys + ['_'+k for k in self._hidden_keys]

#     def values(self):
#         return self._values + self._hidden_values
    
#     def items(self): return zip(self.keys(), self.values())
    
#     def get(self, k, v): return self[k] if k in self else v
    
#     def dict(self): return dict(self.items())
        
#     def copy(self): return Info(**self.dict())

#     def __getitem__(self, k):
#         if isinstance(k, int): return self.tuple()[k]
#         elif k in self._keys: return self._values[self._keys.index(k)]
#         elif k in self._hidden_keys: return self._hidden_values[self._hidden_keys.index(k)]
#         elif k.startswith('_') and k[1:] in self._hidden_keys: return self._hidden_values[self._hidden_keys.index(k[1:])]
#         else: raise IndexError(f"No '{k}' in Info object. ")

#     def __setitem__(self, k, v):
#         if k in self._keys: self._values[self._keys.index(k)] = v
#         elif k in self._hidden_keys: self._hidden_values[self._hidden_keys.index(k)] = v
#         elif k.startswith('_') and k[1:] in self._hidden_keys: self._hidden_values[self._hidden_keys.index(k[1:])] = v
#         elif k.startswith('_'): self._hidden_keys.append(k[1:]); self._hidden_values.append(v)
#         else: self._keys.append(k); self._values.append(v)

#     def __len__(self): return len(self._keys) + len(self._hidden_keys)

#     def __contains__(self, k): return k in self._keys or k in self._hidden_keys or k.startswith('_') and k[1:] in self._hidden_keys

#     for __op__ in [f'__{op}__' for op in "lt le gt ge eq".split()]:
#         execblock(f"""
#         def {__op__}(x, y):
#             return x.tuple().{__op__}(getattr(y, 'tuple', lambda: tuple(y))())
#         """)

#     def __hash__(self): return hash(self.tuple())
    
#     @alias('__repr__')
#     def __str__(self):
#         return '{' + ', '.join([f"{k}: {repr(v)}" for k, v in zip(self._keys, self._values)]) + \
#             ('; ' + ', '.join([f"{k}: {repr(v)}" for k, v in zip(self._hidden_keys, self._hidden_values)]) if len(self._hidden_keys) > 0 else '') + '}'

# class Subject:
#     def __init__(self, x=None, **kwargs):
#         if x is not None and isinstance(x, (dict, Info)):
#             kwargs = x
#         self.subject = Info(**kwargs)
#         self.image = Info(_n_subimage = 1)
#         self.subimage = Info(subimage_id = 'whole')

#     def Image(self, **kwargs):
#         if 'n_subimage' in kwargs: kwargs['_n_subimage'] = kwargs.pop('n_subimage')
#         if 'path' in kwargs: kwargs['_path'] = kwargs.pop('path')
#         self.image.update(**kwargs)
#         return self

#     def SubImage(self, *args, **kwargs):
#         if len(args) == 1:
#             self.Image(n_subimage = args[0])
#             return self
#         elif len(args) == 0:
#             self.subimage.update(**kwargs)
#             return self
#         else: raise TypeError()

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if k in self.subject: self.subject[k] = v
#             elif k in self.image: self.image[k] = v
#             elif k in self.subimage: self.subimage[k] = v
#         return self

#     def get_subimage(self, i):
#         return Info(subimage_id = f"%0{len(str(self.image['n_subimage']))}d"%i)

#     def to_subimage(self, i):
#         ret = self.copy()
#         if i < 0: ret.update(subimage_id = 'whole')
#         else: ret.update(subimage_id = f"%0{len(str(self.image['n_subimage']))}d"%i)
#         return ret
    
#     def to_wholeimage(self):
#         return self.to_subimage(-1)

#     def tuple(self):
#         return self.subject.tuple() + self.image.tuple() + self.subimage.tuple()

#     def keys(self):
#         return self.subject.keys() + self.image.keys() + self.subimage.keys()
        
#     def copy(self):
#         subject = Subject(**self.subject.dict())
#         subject.image = self.image.copy()
#         subject.subimage = self.subimage.copy()
#         return subject

#     def __contains__(self, k):
#         if isinstance(k, int): return k < len(self)
#         elif k in self.subject: return True
#         elif k in self.image: return True
#         elif k in self.subimage: return True
#         else: return False

#     def __getitem__(self, k):
#         if isinstance(k, int): return self.tuple()[k]
#         elif k in self.subject: return self.subject[k]
#         elif k in self.image: return self.image[k]
#         elif k in self.subimage: return self.subimage[k]
#         else: raise IndexError(f"No '{k}' in micomputing.Subject-Image-SubImage-Info object. ")

#     def __setitem__(self, k, v):
#         if k in self.subject: self.subject[k] = v
#         elif k in self.image: self.image[k] = v
#         elif k in self.subimage: self.subimage[k] = v
#         else: raise IndexError(f"No '{k}' in micomputing.Subject-Image-SubImage-Info object. ")

#     def __len__(self): return len(self.subject) + len(self.image) + len(self.subimage)

#     for __op__ in [f'__{op}__' for op in "lt le gt ge eq".split()]:
#         execblock(f"""
#         def {__op__}(x, y):
#             return x.tuple().{__op__}(getattr(y, 'tuple', lambda: tuple(y))())
#         """)

#     def __hash__(self): return hash(self.tuple())
    
#     @alias('__repr__')
#     def __str__(self): return '{' + ' | '.join([str(self.subject).strip('{}'), str(self.image).strip('{}'), str(self.subimage).strip('{}')]) + '}'
    
#     def __call__(self, **kwargs):
#         subject = self.copy()
#         for k, v in kwargs.items(): subject[k] = v
#         return subject
    
class DataObject:
    
    preset_loaders = [IMG, TRS.load, Transformation.load, lambda x: x.open().read(), lambda x: open(x).read()] + [Image.open] if available_Image else []
    
    def __init__(self, obj, loader=None, *, rois=None, prop=None):
        assert rois is None or isinstance(rois, list)
        if isinstance(obj, str):
            self.path = Path(obj)
            self.subimage_rois = rois
            self.loader = loader
            self.property = prop
            self._data = None
            self.loaded_info = {}
            self.final_loader = None
        elif isinstance(obj, DataObject):
            self.path = obj.path
            self.subimage_rois = rois if rois is not None else obj.subimage_rois
            self.loader = obj.loader
            self.property = prop if prop is not None else obj.property
            self._data = obj._data
            self.loaded_info = obj.loaded_info
            self.final_loader = obj.final_loader
        else: raise TypeError(f"Invalid input type {type(obj)} for DataObject.")
        
    @property
    def is_loaded(self): return self._data is not None

    def load(self, loader=None):
        if self.is_loaded: return self._data
        path = self.path.abs
        if loader is None and self.loader is not None: loader = self.loader
        if loader is None:
            for loader in DataObject.preset_loaders:
                try: self._data = loader(path); break
                except Exception as e:
                    if isinstance(e, (TypeError, ValueError, AssertionError)): pass
                    elif isinstance(e, FileNotFoundError): raise e
                    elif 'DecodeError' in str(type(e)): pass
                    else: raise e
            self.final_loader = ["IMG", "TRS", "Trans", "Path", "txt", "Image"][DataObject.preset_loaders.index(loader)]
            if self._data is None: raise TypeError(f"Cannot open file {path} yet (only *.nii, *.nii.gz, *.dcm, *.ima, *.nrrd, *.jpg, *.png, *.txt, *.trs, *.AFF, *.FFD, *.DDF are allowed), please contact the developpers (Error Code: D651). ")
        else:
            self._data = loader(path)
            self.final_loader = loader.__name__ if getattr(loader, '__name__', '') and loader.__name__ != "<lambda>" else "DIY"
        if loader == IMG:
            orient=self._data.orientation
            self.loaded_info = dict(
                shape=self._data.shape,
                affine=bt.tensor(self._data.affine), 
                header=self._data.header, 
                orient=orient, 
                final_loader=self.final_loader
            )
            self._data = self._data.to_tensor()
            self._data.orient = orient
        return self._data
    
    @property
    def n_subimage(self): return len(self.subimage_rois) if self.subimage_rois is not None else 1
    
    @property
    def shape(self): return DataObject(self, prop='shape')
    
    @property
    def affine(self): return DataObject(self, prop='affine')
    
    @property
    def header(self): return DataObject(self, prop='header')
    
    @property
    def orient(self): return DataObject(self, prop='orient')
    
    def __len__(self): return len(self.subimage_rois) if self.subimage_rois else 1
    
    def __getitem__(self, i_subimage):
        if self.subimage_rois is None or not self.subimage_rois:
            raise IndexError(f"No subimage is defined for {self.path.filename}, please use *.slices(axis) or *.patches(rois) to create subimages. ")
        if isinstance(i_subimage, int): i_subimage = slice(i_subimage, i_subimage+1)
        rois = self.subimage_rois[i_subimage]
        return DataObject(self, rois=rois)

    def release(self):
        del self._data
        gc.collect()
        self._data = None

    @alias('__repr__')
    def __str__(self):
        if self.is_loaded: load_state = f"loaded by {self.final_loader}"
        else: load_state = "not loaded"
        sub_str = ''
        if self.subimage_rois: sub_str = f" ({self.n_subimage} subimages)"
        return f"<DataObject at {self.path.filename}{sub_str} [{load_state}]>"
    
    @property
    def data(self):
        if self._data is None: self.load()
        if self.property: return self.loaded_info.get(self.property, None)
        if self.subimage_rois:
            shape = tuple(s.stop - s.start for s in self.subimage_rois[0])
            return bt.stack([bt.crop_as(self._data[roi], shape).squeeze() for roi in self.subimage_rois], {})
        return self._data
    
    def slice(self, i_slice:int):
        return self.slices('IS', i_slices=[i_slice])
    
    def patch(self, roi:tuple[slice]):
        return self.patches(rois=[roi])
    
    def slices(self, axis:(int,str), i_slices=None):
        if i_slices is None or isinstance(axis, str) and 'orient' not in self.loaded_info:
            if not self.is_loaded: self.load(loader=IMG)
        if isinstance(axis, str):
            orient = self.loaded_info['orient']
            axis = [orient.index(a) for a in axis if a in orient]
            if not axis: raise TypeError(f"Cannot finde axis {axis!r} in {self.final_loader} {self.path}(orient: {orient}). ")
            axis = axis[0]
        n_dim = len(self._data.shape)
        n_slice = self._data.size(axis)
        if i_slices is None: i_slices = list(range(n_slice))
        rois = [(slice(None),)*axis + (slice(i,i+1),) + (slice(None),)*(n_dim-axis-1) for i in i_slices]
        return DataObject(self, rois=rois)
    
    def patches(self, rois):
        return DataObject(self, rois=rois)
    
    def random_patches(self, shape, n_patches, *, seed=None):
        if 'shape' not in self.loaded_info:
            if not self.is_loaded: self.load(loader=IMG)
        data_shape = self.loaded_info['shape']
        if seed is not None: bt.manual_seed(seed)
        rois = []
        for _ in range(n_patches):
            starts = tuple(random.randint(0, n-s) for n, s in zip(data_shape, shape))
            rois.append(tuple(slice(st, st+s) for st, s in zip(starts, shape)))
        return DataObject(self, rois=rois)
    
    def uniform_patches(self, shape, stride=None):
        if 'shape' not in self.loaded_info:
            if not self.is_loaded: self.load(loader=IMG)
        data_shape = self.loaded_info['shape']
        if stride is None: stride = shape
        n_patches = tuple(max(int((n-s)/st), 1) for n, s, st in zip(data_shape, shape, stride))
        starts = tuple(int((n - st * (np-1) - s) // 2) for n, s, st, np in zip(data_shape, shape, stride, n_patches))
        
        rois = []
        for i_patches in bt.image_grid(n_patches).flatten(...).transpose(0, 1):
            patch_starts = tuple(s + i * st for s, i, st in zip(starts, i_patches, stride))
            rois.append(tuple(slice(int(st), int(st+s)) for st, s in zip(patch_starts, shape)))
        return DataObject(self, rois=rois)

# class Slicer:
    
#     def __init__(self, path, loader=None):
#         self.data_obj = DataObject(path, loader=loader)
#         self.valid_slices = None
#         self.slice_pointer = 0
        
#     def items(self, key=Key(slice_id=None)):
#         i_slice = 0
#         while True:
#             yield key(slice_id=i_slice), self.data_obj[i_slice]
#             i_slice += 1
#             if self.is_loaded and i_slice >= self.n_slice: break
#         self.release()
        
#     @property
#     def is_loaded(self): return self.data_obj.is_loaded
#     def load(self, loader=None):
#         self.slice_pointer = 0
#         return self.data_obj.load(loader=loader)
#     def release(self): return self.data_obj.release()
    
#     def __getitem__(self, index):
#         if not self.data_obj.is_loaded: self.slice_pointer = 0
#         if self.valid_slices is not None:
#             if callable(self.valid_slices):
#                 self.valid_slices = self.valid_slices(self.data_obj.data)
#             avouch(isinstance(self.valid_slices, list))
#             index = self.valid_slices[index]
#         return self.data_obj[index]
    
#     def slice(self, i_slice):
#         return self.data_obj.slice(i_slice)
    
#     @property
#     def n_slice(self):
#         if self.valid_slices is None: return getattr(self.data_obj, 'n_slice', None)
#         return len(self.valid_slices) if isinstance(self.valid_slices, list) else None

#     @alias('__repr__')
#     def __str__(self):
#         if self.is_loaded: load_state = f"loaded by {self.final_loader}"
#         else: load_state = "not loaded"
#         return f"<SlicerObject at {self.path.filename} [{load_state}]>"
    
#     def __getattribute__(self, name):
#         if name in ("path", "loader", "final_loader", "i_slice", 
#                     "is_loaded", "header", "affine", "data"):
#             return getattr(self.data_obj, name)
#         return super().__getattribute__(name)
    
#     def set_valid_slices(self, obj):
#         """
#         Set the slices that are valid to pick from. 

#         Args:
#             obj (str or list or func): object determining the valid slice indices. 
#                 str: path to a file with content of format "s1, s2, ..., sn"
#                 list: a list of slice indices
#                 func: a function that returns a list of valid slice indices during data loading
#                 None: all slices are valid
#         """
#         if isinstance(obj, str):
#             obj = Path(obj)
#             avouch(obj.exists(), FileNotFoundError(f"Unknown file path '{obj}'. "))
#             with open(obj) as fp:
#                 content = fp.read().strip()
#                 if content: self.valid_slices = list(eval(content))
#                 else: content = []
#         elif isinstance(obj, list):
#             self.valid_slices = obj
#         elif callable(obj): self.valid_slices = obj
#         else: raise TypeError(f"Unknown valid slice type {obj}. ")
#         return self
    
class BatchObject:
    def __init__(self, obj=None, values=None, **kwargs):
        if obj is None: obj = kwargs
        elif kwargs: raise ValueError("Cannot pass both obj and kwargs for BatchObject. ")
        if values is None:
            if isinstance(obj, dict):
                self._names, self._values = zip(*obj.items())
            elif isinstance(obj, (tuple, list)):
                self._names = None
                self._values = list(obj)
            else: raise TypeError(f"BatchObject only accept tuple or dict for input, not {type(obj)}. ")
        else: self._names = obj; self._values = values
        assert len(self._values) > 0, "BatchObject cannot be empty. "
        self.flexible_batch = [False] * len(self._values)
        self.n_batch = None
        for i, value in enumerate(self._values):
            if isinstance(value, bt.Tensor):
                nb = value.n_batch if value.has_batch else 1
            elif isinstance(value, (Key, str)): nb = 1
            else: nb = getattr(value, '__len__', lambda:1)()
            if self.n_batch is not None:
                if nb != self.n_batch and nb != 1:
                    raise TypeError(f"All values in BatchObject should have the same length, but got {nb} in {value!r} mismatching {self.n_batch}. Please ensure to unsqueeze the batch dimension. ")
            elif nb != 1: self.n_batch = nb
            if nb== 1: self.flexible_batch[i] = True
        if self.n_batch is None: self.n_batch = 1
    def __iter__(self): return iter(self._values)
    def __len__(self): return len(self._values)
    def __getitem__(self, i):
        if isinstance(i, int): return self._values[i]
        elif isinstance(i, str): return self._values[self._names.index(i)]
        return BatchObject(self._names, [v if f else v[i] for v, f in zip(self._values, self.flexible_batch)])
    def items(self):
        for k, v in zip(self._names, self._values): yield k, v
    def keys(self): return self._names
    def values(self): return self._values
    @property
    def has_names(self): return self._names is not None
    def __str__(self):
        if self._names is None: str_ = "BatchObject(" + ', '.join([f"{v}" for v in self.values()]) + ")"
        else: str_ = "BatchObject{" + ', '.join([f"{k} = {v!r}" for k, v in self.items()]) + "}"
        return str_.replace("Ellipsis", '...')

class Dataset:
    """
    Create a dataset object. 
    
    Args:
        paths (str): The main folders of the dataset. Dataset will browse all the files within. 
        name (str): The name of the dataset, one can also identify it in the function name of the loading function. 
        main_key_name (str): The main key defined to distinguish different subjects. Defaults to 'subject_id'. 
        batch_pattern (str: unpaired / paired / random or list[Key] / dict[str:Key]): Whether the images inside are paired. Defaults to 'unpaired'.
            unpaired: Values with different Key should be taken as independent data. 
            paired: Only values with different 'main_key' should be regarded as independent data, the input for batch function would be a tuple of such data. 
            random: Values with different Key are regarded as independent data, but the input for batch function would randomly pick the data of the same non-main_key keys. 
            list[Key] or dict[str:Key]: A list or dict of Key objects to constraint each element of the batch input. 
                unpaired <==> Key()
                paired   <==> e.g. [Key(subject_id=..., modality='CT', is_label=False), Key(subject_id=..., modality='T1', is_label=False), Key(subject_id=..., modality='T1', is_label=True)] # ... here is Ellipsis indicating the same subject_id
                random   <==> e.g. [Key(modality='CT', is_label=False), Key(modality='T1', is_label=False), Key(modality='T1', is_label=True)]
        Note: the training/ validation/ test sub-datasets are split according to the 'main_key'.
        preprocess (functional): The preprocess function. 
            Inputs:
                data (dict): The dict of Key-DataObject pairs.
                key (Key): The key of the current data object.
                The default result is data[key], use key.object/data/affine/orientation to quickly convert them into other forms. 
            Outputs: The resulting tensor. 
        max_value (dict or float): The max value of each modality.
    
    Decorated Object:
        The decorated object should be a protocol type. One can create one using:
            (1) inheritence from pycamia.protocol;
            (2) a class with static methods; 
            (3) a regular class with methods with the first argument 'self'. 
        The protocol should have the following methods:
            get_key: a function that maps a Path object into the Key representing the object.
                Note I:  No identical keys are allowed. One needs to set another property even for elements from a single datum, e.g.,
                            Key(..., slice_id = xx) for '*.dcm' or '*.ima' slice files. 
                Note II: None should be returned if the path is to be omitted. 
            get_path: a function that maps a set of batch Keys into a BatchObject. 
                Args:
                    dataset (DataDict): The internal collection of data.
                    *keys (Key): The keys of the data in a single batch, the annotations represents the pattern filter. 
                        e.g., k_image: Key(subject_id=..., is_label=False) represents a batch element with 'is_label' set to False.
                Outputs: A BatchObject containing batch tensors.
    
    Examples::
        >>> from micomputing import data
        >>> @data.Dataset("folderpath1", "folderpath2", main_key_name='instance_id')
        ... class <datasetname>(protocol):
        ...     def get_key(path):
        ...         # Return None to wash out unnecessary data. 
        ...         return Key(instance_id = path.parent.name, is_label = path.name.endswith('label'))
        ...     def get_batch(collection, key_image: Key(instance_id=..., is_label=False), key_label: Key(instance_id=..., is_label=True)):
        ...         # Preprocess the batch data here. 
        ...         return DataObject(image=collection[key_image], label=collection[key_label])
        ... 
        >>> <datasetname>
        <datasetname> Dataset (121 images): 
        ==================================================
        instance_id = 152
        || is_label = False
        || is_label = True
        ...
        instance_id = 174
        || is_label = False
    """
    
    class DataDict(SortedDict):
        def __getitem__(self, k):
            if isinstance(k, slice): return super().__getitem__(k)
            if k.is_data: return super().__getitem__(k.object).data
            if k.is_affine: return super().__getitem__(k.object).affine
            if k.is_orientation: return super().__getitem__(k.object).orient
            return_value = super().get(k, None)
            if return_value is None and isinstance(k, int): return_value = self[self.get_key(k)]
            return return_value
    
    def __init__(self, *args, name = None, main_key_name = 'subject_id', batch_pattern = 'unpaired', preprocess = None, max_value={}):
        self.n_subimage_record = []
        self.use_subimages = None
        self.name = name
        self.main_key_name = main_key_name
        self.batch_pattern = batch_pattern
        self.preprocess = preprocess
        self.splitted = False
        self.batch_index = {}   # 3-level indexing: batch pattern, common pair index, and specified pattern. 
        # self.subject_index = {}    # {(main key)-[keys list] paired dict}
        # self.batch_index = {}      # {batch pattern (in Key): (main key)-(key candidates list) paired sorted dict}
        self._training_set = []    # list of main keys
        self._validation_set = []
        self._testing_set = []
        if len(args) == 1:
            x = args[0]
            if isinstance(x, dict): self.__init__dict(*args)
            elif isinstance(x, str): self.__init__str(*args)
            else: self.__init__sequence(*args)
        else: self.__init__str(*args)

    def __init__dict(self, x: DataDict):
        self.data = x
        self.batch_pointer = {'training': {}, 'validation': {}, 'testing': {}}
        self.directories = []
        for k in self.data.keys():
            if 'path' not in k: continue
            if k.path.ref not in self.directories: self.directories.append(k.path.ref)
        self._cache = []

    def __init__sequence(self, x: (list, tuple)):
        self.__init__(*x)

    def __init__str(self, *directories: str):
        self.data = Dataset.DataDict()
        self.batch_pointer = {'training': {}, 'validation': {}, 'testing': {}}
        self.directories = map(Path, directories)
        self._cache = []

    def __call__(self, prot):
        if self.name is None: self.name = prot.__name__
        if not isinstance(prot, type):
            self.path2key = prot
            self.key2batch = None
        else:
            if issubclass(prot, protocol):
                try: prot = prot("get_key", optional="get_batch", allow_properties=False)
                except NameError as e:
                    if "allow property member" not in str(e): raise e
                    else: raise NameError(f"Dataset protocol '{prot.__name__}' should not have property members, please use a class and 'self/cls.[property]' instead. ")
            else: prot = prot()
            avouch(hasattr(prot, 'get_key'), "Dataset protocol requires function 'get_key' to map path objects into dataset keys. ")
            self.path2key = prot.get_key
            self.key2batch = getattr(prot, 'get_batch', None)
        self.sort_files()
        return self

    def preprocessor(self, func):
        self.preprocess = func
        return self

    def __len__(self):
        self.check_data()
        return len(self.data)

    @alias("manual_seed")
    def seed(self, s): random.seed(s)

    def check_data(self): avouch(len(self.data) > 0, "Dataset not created yet. Use `@Dataset(directory_paths)` in front of a function mapping a path to key structure to create Dataset. ")

    def cache(self, k, v=None):
        if v is None: return self._cache[k]
        self._cache[k] = v

    def byte_size(self):
        total_size = 0
        for ele in self.data.values():
            if isinstance(ele, Dataset):
                total_size += ele.byte_size()
            elif isinstance(ele, DataObject):
                if isinstance(ele.data, bt.Tensor):
                    total_size += ele.data.byte_size()
            elif isinstance(ele, bt.Tensor):
                total_size += ele.byte_size()
        return ByteSize(total_size)
    
    def append_n_subimage_record(self, new_sample):
        n_sample = 20
        if len(self.n_subimage_record) < n_sample: self.n_subimage_record.append(new_sample); return
        i = [r < new_sample for r in self.n_subimage_record].index(False)
        self.n_subimage_record.insert(i, new_sample)
        if i < n_sample // 2: self.n_subimage_record.pop(0)
        else: self.n_subimage_record.pop(-1)
    
    @property
    def estimated_n_subimage(self):
        if len(self.n_subimage_record) > 0: return approx(round(sum(self.n_subimage_record) / len(self.n_subimage_record)))
        else: return approx(20)
    
    def default_preproc(self, key, obj):
        if key.is_affine: return bt.auto_device(obj)
        if key.fuzzy_get('is_label', 'islabel', 'label', 'is_seg', 'isseg', 'seg'):
            return bt.auto_device(obj.long())
        if self.max_value is None: return bt.auto_device(obj)
        mod = key.fuzzy_get('modality', 'mod', 'sequence', 'seq')
        return bt.auto_device((obj / (self.max_value.get(mod, self.max_value['default']) if mod in self.max_value or 'default' in self.max_value else self.max_value[mod])).clamp(0, 1))
    
    def sort_files(self):
        # Scan files
        for d in self.directories:
            for f in d.iter_files():
                key = self.path2key(f)
                if key is None: continue
                if not isinstance(key, Key): raise TypeError(f"The 'get_key' method in dataset protocol should return a Key object, but got {key}. ")
                if key.main_key_name is None: key.__class__.main_key_name = self.main_key_name
                self.data[key] = DataObject(f)
                # self.subject_index.setdefault(key.main_key, [])
                # self.subject_index[key.main_key].append(key)
        
        # Decide batch pattern
        if self.batch_pattern == 'unpaired':
            self.batch_pattern = BatchObject([Key()])
            if self.key2batch is not None:
                annots = self.key2batch.__annotations__
                self.batch_pattern = BatchObject([annots[v] for v in self.key2batch.__code__.co_varnames if v in annots])
        if isinstance(self.batch_pattern, str):
            all_items = []
            for k in self.data:
                if self.batch_pattern == 'paired': k = k(**{k.main_key_name: ...})
                else: k = k.pop(k.main_key_name)
                if k not in all_items: all_items.append(k)
            self.batch_pattern = BatchObject(all_items)
        elif isinstance(self.batch_pattern, dict):
            keys, values = [], []
            for k, v in self.batch_pattern.items(): keys.append(k); values.append(v)
            self.batch_pattern = BatchObject(keys, values)
        elif isinstance(self.batch_pattern, Key): self.batch_pattern = BatchObject([self.batch_pattern])
        avouch(isinstance(self.batch_pattern, BatchObject), f"Property batch_pattern={self.batch_pattern} is invalid for Dataset object. ")
        
        n_common_keys = {}
        valid = {}
        found_pattern = {}
        for key in self.data:
            # if valid.get(key, False): continue
            for bpk in self.batch_pattern.values():
                if is_placeholder(bpk): found_pattern[bpk] = True; continue
                assert isinstance(bpk, Key), f"The batch pattern should be a list of Key (or placeholders ... or [i]) objects, not {bpk}."
                pair_index = placeholder_values(key, bpk)   # The determined placeholder values. 
                if pair_index is None: continue
                pair_keys = {bpk: key}                      # The decided keys. 
                for trybpk in self.batch_pattern.values():  # Test if the placeholder values can find the data. 
                    if trybpk == bpk: continue
                    if is_placeholder(trybpk): continue
                    this_key = with_placeholder_values(trybpk, pair_index)
                    if this_key is None: continue
                    possible_keys = [this_key]
                    pair_keys[trybpk] = []
                    while len(possible_keys) > 0:
                        k = possible_keys.pop(0)
                        for n, v in k.items():
                            if isinstance(v, list):
                                possible_keys.extend([k.with_item(**{n:u}) for u in v])
                                break
                        else:
                            if k not in self.data: continue
                            pair_keys[trybpk].append(this_key)
                    if not pair_keys[trybpk]: break
                else: # all keys in pair_keys are available. 
                    for p in pair_keys.keys():
                        self.batch_index.setdefault(p, SortedDict())
                        decided_keys = {}
                        for k, v in pair_keys.items():
                            if isinstance(v, Key): v = v.to_property(k)
                            elif isinstance(v, list): v = [u.to_property(k) if isinstance(u, Key) else u for u in v]
                            decided_keys[k] = v
                        self.batch_index[p][pair_index] = decided_keys
                        found_pattern[p] = True
                    for ks in pair_keys.values():
                        if isinstance(ks, list):
                            for k in ks: valid[k] = True
                        else: valid[ks] = True
            else: ... # no trial succeeded for this key.
        for p in self.batch_pattern.values():
            if not found_pattern.get(p, False):
                raise TypeError(f"Invalid batch patern for dataset {self.name}: {p}")
        for key in self.data:
            if not valid.get(key, False): self.data.pop(key, None)
        
        # Create key2batch function if it does not exists
        if self.max_value is not None:
            if isinstance(self.max_value, (int, float)): self.max_value = {'default': self.max_value}
            if self.key2batch is not None: raise TypeError("Cannot set 'max_value' and 'get_batch' protocol for Dataset at the same time. ")
        if self.key2batch is None:
            if self.batch_pattern.has_names:
                def key2batch(collection, **keys):
                    return BatchObject({n: self.default_preproc(k, collection[k]) for n, k in keys.items()})
            else:
                def key2batch(collection, *keys):
                    return BatchObject([self.default_preproc(k, collection[k]) for k in keys])
            self.key2batch = key2batch
        
        # # Create subject index and batch index
        # n_data = len(self.main_keys)
        # avouch(n_data > 5, f"Only {n_data} subject{'s' if n_data > 1 else ''} found for dataset {self.name}, please check the `get_key` protocol. "
        #        "Make sure that it returns a Key with a main key by name in `main_key_name`. ")
        # remove_list = []
        # for bk in self.batch_pattern: # for each batch constraint
        #     # Create batch_index which has 2 levels of keys, i.e., batch pattern and sid to record candidates
        #     if bk == ...: continue
        #     self.batch_index.setdefault(bk, SortedDict())
        #     for sid in self.main_keys:
        #         if sid in remove_list: continue
        #         paired = bk.get(bk.main_key_name, None) == ...
        #         if paired: # if the element needs paired
        #             candidates = [k for k in self.subject_index[sid] if is_subkey(k, bk(**{bk.main_key_name: sid}))]
        #         else: # no paired element needed
        #             candidates = [k for k in self.subject_index[sid] if is_subkey(k, bk)]
        #         if len(candidates) > 0: self.batch_index[bk][sid] = candidates; continue # subimage found
        #         elif paired: remove_list.append(sid) # paired images missing member
        # if len(remove_list) == self.n_main_key or any(len(cand) == 0 for cand in self.batch_index.values()):
        #     error_bps = [bp for bp, cand in self.batch_index.items() if len(cand) == 0]
        #     raise TypeError(f"Cannot find required image output following batch pattern {error_bps} for {self.name} dataset. ")
        # for sid in remove_list:
        #     for k in self.subject_index[sid]: self.data.pop(k, None) # remove from dataset
        #     for bk in self.batch_pattern:
        #         if bk == ...: continue
        #         self.batch_index[bk].pop(sid, None) # remove from content index

        self.data.sort()
        # self.split_datasets(training=0.7, validation=0.2, shuffle=shuffle)
    
    @property
    def main_keys(self):
        return unique(k.main_key for k in self.data.keys())
    
    @alias("n_subject")
    @property
    def n_main_key(self): return len(self.main_keys)

    def to_str(self, list_all=True) -> str:
        self.check_data()
        self.data.sort()
        str_print = SPrint()
        s = lambda n: "s" if n > 1 else ''
        item = 'sub-image' if self.use_subimages else 'image'
        str_print(f"{'' if self.name is None else self.name} Dataset", f"({self.n_data} {item}{s(self.n_data)}, {self.n_subject} subject{s(self.n_subject)}): ")
        str_print('=' * 50)
        if self.splitted:
            str_print(
                f"{self.n_training} training {item}{s(self.n_training)} ({self.n_training_subject} subject{s(self.n_training_subject)})", 
                f"{self.n_validation} validation {item}{s(self.n_validation)} ({self.n_validation_subject} subject{s(self.n_validation_subject)})", 
                f"{self.n_testing} test {item}{s(self.n_testing)} ({self.n_testing_subject} subject{s(self.n_testing_subject)}).", sep=', ')
            str_print('=' * 50)
        
        n_main_keys = len(set([k.main_key for k in self.data.keys()]))
        should_omit = not list_all and self.n_main_key > 6 # whether to omit the middle data. 
        count = 0 # the counter of number of data, used for omission. 
        prefix = lambda i: ' |  ' * i
        prev_key = None
        for d, v in self.data.items():
            first_diff = None
            for i, k in enumerate(d.visible_keys()):
                if first_diff is not None and i > first_diff or prev_key is None or prev_key[k] != d[k]:
                    if first_diff is None: first_diff = i
            prev_key = d
            if should_omit and first_diff == 0:
                count += 1
                if count == 3: str_print('...')
                if count > self.n_main_key - 2: count = 0
            if should_omit and count > 2: continue
            for i, k in enumerate(d.visible_keys()):
                if i < first_diff: continue
                if i < len(d) - 1:
                    subset_label = ''
                    if self.splitted and k == d.main_key_name:
                        subset_label = f" [{self.in_subset(d.main_key)}]"
                    str_print(prefix(i), f"{k} = {d[k]}{subset_label}", sep=''); continue
                used = ''
                if not self.needed_in_batch(d): used = " [not used]"
                skey = f"{k} = {d[k]}{used}: "
                str_v = str(v).split('=' * 10)[-1].lstrip('=').strip()
                str_print(prefix(i), skey, str_v.replace('\n', '\n' + prefix(i) + ' ' * len(skey)), sep='')
        return str_print.text
    
    def __str__(self): return self.to_str(list_all=False)

    def setdefault(self, k, v):
        if k not in self.data: self.data[k] = v

    def get_key(self, *args, **kwargs):
        all_candidates = kwargs.pop('all_candidates', False)
        avouch(len(args) > 0 and len(kwargs) == 0 or len(args) == 0 and len(kwargs) > 0)
        if len(args) > 0:
            args = to_tuple(args)
            if len(args) == 1 and isinstance(args[0], (list, tuple)): args = args[0]
        candidates = []
        for k in self.data.keys():
            if len(args) > 0:
                if all([x in k.tuple() for x in args]):
                    candidates.append(k)
            else:
                if all([k.get(a, None) == kwargs[a] for a in kwargs]):
                    candidates.append(k)
        if len(candidates) == 1: return candidates[0]
        if all_candidates: return candidates
        raise TypeError(f"Cannot use incomplete of keys {args if len(args) > 0 else kwargs} to find non-unique or non-existed items. Number of matches: {len(candidates)} ")
    
    def get(self, *args, **kwargs):
        """
        get(key, default_value) OR get(constraint_key_name=constraint_value)
        """
        avouch(len(args) > 0 and len(kwargs) == 0 or len(args) == 0 and len(kwargs) > 0)
        if len(args) > 0:
            k = args[0]
            v = None
            if len(args) > 1: v = args[1]
            if k not in self.data: return v
            return self.data[k]
        try: return self.data[self.get_key(**kwargs)]
        except TypeError: raise KeyError(f"Invalid key {kwargs}.")

    def items(self): return self.data.items()

    def first(self): return self.data.first()

    def shuffle(self): return self.data.shuffle()

    def __setitem__(self, k, v):
        self.data[k] = v

    def __getitem__(self, k):
        if isinstance(k, Key) and k == Key(): return self
        if isinstance(k, list): return Dataset(
            Dataset.DataDict({l: self[l] for l in k}), 
            name=self.name+'[subset]', 
            main_key_name=self.main_key_name, 
            batch_pattern=self.batch_pattern
        )
        if k in self.data: return self.data[k]
        if isinstance(k, (dict, Key)): return self.data[self.get_key(**k)]
        try: return self.data[self.get_key(k)]
        except TypeError: raise KeyError(f"Invalid key {k}.")
    
    def __getattr__(self, k):
        try: return getattr(super(), k, None)
        except KeyError as e: raise AttributeError(str(e))

    def __iter__(self):
        return iter(self.data)
    
    def has_prop(self, prop):
        key = self.data.first()[0]
        return prop in key.keys()
    
    def split_datasets(self, *, byprop = None, training = None, validation = None, testing = None, shuffle = True):
        """
        Split dataset to training, validation and test sets by ratio. e.g. split_datasets(training=0.8, validation = 0.1)
        """
        self.check_data()
        main_keys = self.main_keys.copy()
        if shuffle: random.shuffle(main_keys)
        if byprop is not None:
            self._training_set = []
            self._validation_set = []
            self._testing_set = []
            for sid in main_keys:
                subject_set = self.subject_index[sid]
                assert len(subject_set) > 0, f"Empty subject {sid} during dataset splitting. "
                subset_name = subject_set[0][byprop]
                if subset_name.startswith("train"): self._training_set.append(sid)
                elif subset_name.startswith("val"): self._validation_set.append(sid)
                elif subset_name.startswith("test"): self._testing_set.append(sid)
                else: raise TypeError(f"Unrecognized subset property '{byprop} = {subset_name}'. ")
        else:
            if validation is None and (training is None or testing is None): validation = 0.
            if testing is None and training is None: testing = 0.
            if training is None: training = 1. - validation - testing
            elif validation is None: validation = 1. - testing - training
            elif testing is None: testing = 1. - training - validation
            avouch(abs(training + testing + validation - 1.) < 1e-3, "Invalid ratios for function 'split_datasets' (Sum is not 1). ")
            n_train = int(training * len(main_keys))
            n_valid = int(validation * len(main_keys))
            self._training_set = main_keys[:n_train]
            self._validation_set = main_keys[n_train:n_train + n_valid]
            self._testing_set = main_keys[n_train + n_valid:]
        self.splitted = True
        # def add_subimage(main_key_set):
        #     return sum((self.subject_index[mk] for mk in main_key_set), [])
        # self._training_set = add_subimage(main_keys[:n_train])
        # self._validation_set = add_subimage(main_keys[n_train:n_train + n_valid])
        # self._testing_set = add_subimage(main_keys[n_train + n_valid:])
        return self
    
    def needed_in_batch(self, key, batch_pattern=None):
        """
        Whether the image with `key` is needed in (any) batches defined by `batch_pattern`. 
        """
        if batch_pattern is None: batch_pattern = self.batch_pattern
        for const in batch_pattern.values():
            if is_subkey(key, const): return True
        return False
    
    def in_subset(self, main_key):
        if not self.splitted: return
        if main_key in self._training_set: return 'training'
        if main_key in self._validation_set: return 'validation'
        if main_key in self._testing_set: return 'testing'
    
    @property
    def n_training_subject(self): return len(self._training_set) if self.splitted else None
    @property
    def n_validation_subject(self): return len(self._validation_set) if self.splitted else None
    @property
    def n_testing_subject(self): return len(self._testing_set) if self.splitted else None
    
    # @property
    # def n_training_subject(self): return len(list(set(k.main_key for k in self._training_set))) if self._training_set else None
    # @property
    # def n_validation_subject(self): return len(list(set(k.main_key for k in self._validation_set))) if self._validation_set else None
    # @property
    # def n_testing_subject(self): return len(list(set(k.main_key for k in self._testing_set))) if self._testing_set else None
    
    @property
    def n_data(self):
        if self.data is None: return
        total_item = 0
        for key in self.data:
            if not self.needed_in_batch(key): continue
            if self.use_subimages:
                if not self.data[key].is_loaded: total_item = total_item + self.estimated_n_subimage
                else: total_item += self.data[key].n_subimage
            else: total_item += 1
        return total_item
    @property
    def n_training(self):
        if not self.splitted: return
        total_item = 0
        for key in self.data:
            if key.main_key not in self._training_set: continue
            if not self.needed_in_batch(key): continue
            if self.use_subimages: 
                if not self.data[key].is_loaded: total_item = total_item + self.estimated_n_subimage
                else: total_item += self.data[key].n_subimage
            else: total_item += 1
        return total_item
    @property
    def n_validation(self):
        if not self.splitted: return
        total_item = 0
        for key in self.data:
            if key.main_key not in self._validation_set: continue
            if not self.needed_in_batch(key): continue
            if self.use_subimages: 
                if not self.data[key].is_loaded: total_item = total_item + self.estimated_n_subimage
                else: total_item += self.data[key].n_subimage
            else: total_item += 1
        return total_item
    @property
    def n_testing(self):
        if not self.splitted: return
        total_item = 0
        for key in self.data:
            if key.main_key not in self._testing_set: continue
            if not self.needed_in_batch(key): continue
            if self.use_subimages: 
                if not self.data[key].is_loaded: total_item = total_item + self.estimated_n_subimage
                else: total_item += self.data[key].n_subimage
            else: total_item += 1
        return total_item

    @alias("train_batch")
    def training_batch(self, n, **kwargs): kwargs.update(dict(restart=True)); return self.batch('training', n, **kwargs)
    @alias("valid_batch")
    def validation_batch(self, n, **kwargs): kwargs.update(dict(restart=True)); return self.batch('validation', n, **kwargs)
    @alias("test_batch")
    def testing_batch(self, n, **kwargs): kwargs.update(dict(restart=True)); return self.batch('testing', n, **kwargs)

    @alias("train_batch")
    def training_batches(self, n, **kwargs):
        while True:
            cur_batch = self.batch('training', n, **kwargs)
            if cur_batch is None: break
            yield cur_batch
    @alias("valid_batch")
    def validation_batches(self, n, **kwargs):
        while True:
            cur_batch = self.batch('validation', n, **kwargs)
            if cur_batch is None: break
            yield cur_batch
    @alias("test_batch")
    def testing_batches(self, n, **kwargs):
        while True:
            cur_batch = self.batch('testing', n, **kwargs)
            if cur_batch is None: break
            yield cur_batch
    
    def batch(self, stage='training', n_batch=4, shuffle=True, drop_last=True, none_each_epoch=True, restart=False):
        """
        Create a batch data. 

        Args:
            stage (str, 'training'|'validation'|'testing'): The stage we are in, determining the subset we are using. Defaults to 'training'.
            n_batch (int): The size of batch. Defaults to 4.
            shuffle (bool): Whether to randomly pick data from the dataset or not. Defaults to True.
                [True]: The dataset will automatically shuffle for each epoch. 
                [False]: No randomization is used in selecting batches, they are selected in order. 
            drop_last (bool):
                Whether to drop the last data which is less than a batch. Defaults to True.
                [True]: Drop the remaining data.
                [False]: Use the remaining data to create a batch with a batch size smaller than normal ones. 
            none_each_epoch (bool): Whether to output a None value at the end of each epoch. Defaults to True.
            restart (bool): Manually use a restart of epoch for this batch. Defaults to False.
        """
        self.check_data()
        if not self.splitted: self.split_datasets(training=0.7, validation=0.2, shuffle=shuffle)
        stage = stage.lower()
        if stage == 'training': subset = self._training_set
        elif stage == 'validation': subset = self._validation_set
        elif stage == 'testing': subset = self._testing_set
        
        batch_list = []
        remain_batch_size = n_batch
        while True:
            if remain_batch_size == 0: break
            if self.use_subimages and self.batch_pointer[stage]['i_sub'] > 0:
                i_sub = self.batch_pointer[stage]['i_sub']
                batch_obj = self.prev_batch_obj
                if batch_obj.n_batch - i_sub <= remain_batch_size:
                    batch_list.append(batch_obj[i_sub:])
                    self.batch_pointer[stage]['i_sub'] = 0
                    remain_batch_size -= batch_obj.n_batch - i_sub
                    continue
                batch_list.append(batch_obj[i_sub:i_sub+remain_batch_size])
                self.batch_pointer[stage]['i_sub'] = i_sub+remain_batch_size
                remain_batch_size = 0
                break
                
            key_indices = []
            decided = {}
            decided_key = Key()
            main_pointer_bp = None
            reaches_end = False
            for bp in self.batch_pattern.values():
                if decided.get(bp, None) is not None: continue
                if not isinstance(bp, Key): key_indices.append(bp); continue
                candidates = self.batch_index[bp]
                cur_pointer = 0 if restart else self.batch_pointer[stage].get(bp, 0)
                if cur_pointer < len(candidates):
                    pair_index = candidates.get_key(cur_pointer)
                else: reaches_end = True; break
                self.batch_pointer[stage][bp] = cur_pointer + 1
                if main_pointer_bp is None: main_pointer_bp = bp
                decided_key.update(**pair_index)
                pair_keys = candidates[pair_index]
                for p, k in pair_keys.items():
                    decided[p] = random.choice(k) if isinstance(k, list) else k
            if reaches_end: break
            for i, bp in enumerate(self.batch_pattern.values()):
                if bp in key_indices:
                    if bp is ...: decided[bp] = Key(**{k:v for k, v in decided_key.items() if not k.endswith(']')})
                    elif is_placeholder(bp):
                        cands = [(k, v) for k, v in decided_key.items() if k.endswith(bp)]
                        if len(cands) > 1: raise TypeError(f"Multiple pattern for placeholder {bp} detected. ")
                        elif len(cands) < 1: raise NameError(f"Unrecognized placeholder {bp}. ")
                        decided[bp] = Key(**{cands[0][0].split('[', 1)[0]:v})
                    else: raise TypeError(f"Invalid key index pattern: {bp}.")
            if len(decided) < len(self.batch_pattern): raise TypeError("Not fully decided batch: please check the pattern definitions. ")
            chosen = []
            for bp in self.batch_pattern:
                cands = decided[bp]
                if not isinstance(bp, Key): chosen.append(cands); continue
                if isinstance(cands, list): cands = random.choice(cands)
                key_cands = self.get_key(**cands, all_candidates=True)
                if isinstance(key_cands, list): key_cands = random.choice(key_cands)
                key_cands = key_cands.to_property(cands)
                chosen.append(key_cands)
            batch_obj = self.key2batch(self.data, *chosen)
            if batch_obj.n_batch > 1:
                self.use_subimages = True
                n_subimage = batch_obj.n_batch
                self.append_n_subimage_record(n_subimage)
                if n_subimage <= remain_batch_size:
                    batch_list.append(batch_obj)
                    remain_batch_size -= n_subimage
                else:
                    batch_list.append(batch_obj[:remain_batch_size])
                    self.batch_pointer[stage]['i_sub'] = remain_batch_size
                    self.prev_batch_obj = batch_obj
                    remain_batch_size = 0
            else:
                batch_list.append(batch_obj)
                remain_batch_size -= 1
        
        if remain_batch_size > 0 and drop_last:
            self.batch_pointer = {'training': {}, 'validation': {}, 'testing': {}}
            if shuffle:
                for bp in self.batch_pattern:
                    if not isinstance(bp, Key): continue
                    self.batch_index[bp].shuffle()
            if none_each_epoch: return
            else: return self.batch(stage=stage, n_batch=n_batch, shuffle=shuffle, 
                                    drop_last=drop_last, none_each_epoch=none_each_epoch, restart=restart)
        
        ref_batch = batch_list[0]
        batch_values = []
        for values, flex in zip(zip(*[batch_item.values() for batch_item in batch_list]), ref_batch.flexible_batch):
            if isinstance(values[0], DataObject): values = [v.data for v in values]
            if isinstance(values[0], list): cated = sum(values, [])
            elif isinstance(values[0], tuple): cated = sum(values, tuple())
            elif isinstance(values[0], bt.Tensor):
                if values[0].has_batch: cated = bt.cat(values, {}, crop=True)
                else: cated = bt.stack(values, {}, crop=True)
            elif isinstance(values[0], bt.torch.Tensor): cated = bt.cat(values, 0)
            else: cated = list(values) # The list consists one-elemental objects
            if flex:
                if isinstance(cated, (list, tuple)) and len(cated) == 1: cated = cated * n_batch
                elif isinstance(cated, bt.Tensor) and cated.n_batch == 1: cated = cated.amplify(n_batch, {})
                elif isinstance(cated, bt.torch.Tensor) and cated.size(0) == 1: cated = cated.repeat(n_batch, *((1,) * (cated.ndim()-1)))
            # else: raise TypeError(f"Cannot perform BatchObject concatenation of values in type {type(values[0])}. ")
            batch_values.append(cated)
        return BatchObject(batch_list[0].keys(), batch_values)
        
        # selected_items = []
        # batch_list = []
        # new_epoch = False
        # release_list = []
        # key_index = None
        # for i, bp in enumerate(self.batch_pattern):
        #     if bp == ...:
        #         assert key_index is not None, "Only one key index is allowed for batch patterns. "
        #         key_index = i; continue
        #     candidates = self.batch_index[bp].filter(subset)
        #     cur_pointer = 0 if restart else self.batch_pointer[stage].get(bp, 0)
        #     if cur_pointer == 0 and self.use_slice_elements: cur_pointer = 0, 0
        #     batch_sids = []
        #     if bp.main_key == ... and selected_items:
        #         pairs = []
        #         if not self.use_slice_elements:
        #             for k in [random.choice(candidates[img_key]) for img_key in selected_items]:
        #                 x = self.data[k]
        #                 if not x.is_loaded: x.load()
        #                 release_list.append(x)
        #                 pairs.append((x, k))
        #         else:
        #             for k in [random.choice(candidates[img_key]).with_item(i_slice=slc) for img_key, slc in selected_items]:
        #                 pairs.append((self.data[k].slice(k.i_slice), k))
        #         batch_list.append(pairs)
        #     else:
        #         cum_n_batch = 0
        #         if not self.use_slice_elements:
        #             new_pointer = cur_pointer + n_batch
        #             selected_items = candidates[cur_pointer: new_pointer]
        #             if len(selected_items) == 0: new_pointer = 0; new_epoch = True; break
        #             pairs = []
        #             for k in [random.choice(candidates[img_key]) for img_key in selected_items]:
        #                 x = self.data[k]
        #                 if not x.is_loaded: x.load()
        #                 release_list.append(x)
        #                 pairs.append((x, k))
        #             batch_list.append(pairs)
        #             self.batch_pointer[stage][bp] = new_pointer
        #         else: 
        #             cur_pointer, slice_pointer = cur_pointer
        #             batched_slices = []
        #             for i in range(n_batch):
        #                 image_key = random.choice(candidates[cur_pointer])
        #                 sample_image = self.data[image_key]
        #                 sample_image.load()
        #                 self.append_n_subimage_record(sample_image.n_slice)
        #                 n_remain_batch = n_batch - len(batched_slices)
        #                 new_slice_pointer = slice_pointer + n_remain_batch
        #                 selected_items.extend([(image_key, i) for i in range(slice_pointer, min(new_slice_pointer, sample_image.n_slice))])
        #                 batched_slices.extend([(sample_image.slice(i), image_key.with_item(i_slice=i)) for i in range(slice_pointer, min(new_slice_pointer, sample_image.n_slice))])
        #                 if new_slice_pointer >= sample_image.n_slice: new_slice_pointer = 0; cur_pointer += 1; release_list.append(sample_image)
        #                 if len(batched_slices) >= n_batch: break
        #             batch_list.append(batched_slices)
        #             if len(selected_items) == 0: cur_pointer = 0; new_slice_pointer = 0; new_epoch = True; break
        #             self.batch_pointer[stage][bp] = cur_pointer, new_slice_pointer

        # if key_index is not None:
        #     if not self.use_slice_elements: keys = [(k, k) for k in selected_items]
        #     else: keys = [(k.main_key, k.with_item(i_slice=i)) for k, i in selected_items]
        #     batch_list = batch_list[:key_index] + [keys] + batch_list[key_index:]
        # is_partial_batch = any(len(x) < n_batch for x in batch_list)
        # if is_partial_batch and drop_last: new_epoch = True
        # if new_epoch: 
        #     self.batch_pointer = {'training': {}, 'validation': {}, 'testing': {}}
        #     if shuffle:
        #         for bp in self.batch_pattern:
        #             if bp == ...: continue
        #             self.batch_index[bp].shuffle()
        #     for r in release_list: r.release()
        #     if none_each_epoch: return
        #     else: return self.batch(stage=stage, n_batch=n_batch, shuffle=shuffle, 
        #                             drop_last=drop_last, none_each_epoch=none_each_epoch, restart=restart)
        # return_value = self.get_batch(batch_list)
        # for r in release_list: r.release()
        # return return_value
    
    # def get_batch(self, batch_list):
    #     """
    #     The function that creates tuple of batch by structured batch data. 

    #     Args:
    #         batch_list (list): a list of length equal to *.batch_pattern; with each element a list of length n_batch. 
    #             The nested elements are DataObjects (Slicer output in major function would result in DataObjects with i_slice). 
    #     """
    #     res_list = []
    #     for output_batch, bp in zip(batch_list, self.batch_pattern):
    #         if isinstance(output_batch[0][0], DataObject):
    #             if self.preprocess is None: prepcs = lambda data, k: data[k]
    #             else: prepcs = self.preprocess
    #             output_batch = [prepcs(self.data, k.data if bp.is_object else k.to_property(bp)) for x, k in output_batch]
    #             if all([x is None for x in output_batch]): raise TypeError(f"Remember to return data[k] by default in the 'preprocess' function for {self.name} dataset.")
    #         else: output_batch = [x for x, k in output_batch]
    #         ref = output_batch[0]
    #         if isinstance(ref, bt.Tensor): ...
    #         else: res_list.append(output_batch); continue
    #         avouch(isinstance(ref, bt.Tensor))
    #         if ref.has_batch: res_list.append(bt.cat(output_batch, {}, crop=True).detach())
    #         else: res_list.append(bt.stack(output_batch, {}, crop=True).detach())
    #     if self.batch_names is None: return BatchObject(res_list)
    #     return BatchObject(self.batch_names, res_list)

class LossCollector:
    
    def __init__(self):
        self.collection = {}
    
    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            self.collection.setdefault(k, bt.zeros(0))
            self.collection[k] = bt.cat(self.collection[k], bt.to_bttensor(v, device=self.collection[k].device))
            
    def __str__(self):
        return ', '.join([f"{k} [{v.mean_std}]" for k, v in self.collection.items()])

# class MedicalSubImage(Dataset):
    
#     def __getitem__(self, k):
#         try:
#             if isinstance(k, Subject): return self.data[k.subimage]
#         except KeyError: raise KeyError(f"Invalid key {k}.")
#         return super().__getitem__(k)

#     def __setitem__(self, k, v):
#         if isinstance(k, Subject):
#             self.data.setdefault(k.subimage, v)
#         else: self.data[k] = v

#     @property
#     def IMG(self):
#         if len(self.data) == 1:
#             return self.data[next(self.data.keys())].load()
    
# class MedicalImage(Dataset):
    
#     def __getitem__(self, k):
#         try:
#             if isinstance(k, Subject): return self.data[k.image][k.subimage]
#         except KeyError as e: raise KeyError(eval(str(e)).strip('.') + f" in {k}.")
#         return super().__getitem__(k)

#     def __setitem__(self, k, v):
#         if isinstance(k, Subject):
#             self.data.setdefault(k.image, MedicalSubImage())
#             self.data[k.image].setdefault(k.subimage, v)
#         else: self.data[k] = v

#     @property
#     def IMG(self):
#         if len(self.data) == 1:
#             return self.data[next(self.data.keys())].load()

# class MedicalDataset(Dataset):

#     def __init__(self, *args, **kwargs):
#         """
#         register data type. 
        
#         Args:
#             paths (str): The folders with data. Dataset will browse all the files within and 
#                 use the decorated function to change a path into a Subject/ImageObject structure. 
#                 The arguments for Subject/ImageObject are their 'info'.
#             paired_subimage (bool): whether the images inside are paired, the 'paired' images 
#                 will not split in traing/validation/test datasets. Defaults to True.
        
#         Examples::
#             >>> from micomputing.data import MedicalDataset, Subject, Image
#             >>> @MedicalDataset("folderpath1", "folderpath2")
#             ... def datasetname(path):
#             ...     return Subject(patientID = path.split()[1]) \
#             ...           .Image(modality = path.name)
#             ... 
#             >>> datasetname
#             datasetname Dataset (121 images): 
#             ==================================================
#             patientID = 152
#             || modality = MR
#             || modality = CT
#             ...
#             patientID = 174
#             || modality = CT
#         """
#         super().__init__(*args, **kwargs)
#         self.paired_subimage = kwargs.get('paired_subimage', True)
#         self.template = None

#     def __str__(self) -> str:
#         self.check_data()
#         str_print = SPrint()
#         s = "s" if len(self.data) > 1 else ''
#         str_print(f"{self.name} Dataset ({len(self.data)} subject{s}): ")
#         str_print('=' * 50)
#         omit = len(self.data) > 8
#         start = lambda i: ' |  ' * i
#         for i, (sb, sbd) in enumerate(self.data.items()):
#             if not omit or i < 2 or i >= len(self.data) - 2:
#                 str_print('\n'.join([f"{k} = {v}" for k, v in sb.items()]) + ':')
#                 for im, imd in sbd.data.items():
#                     str_print('\n'.join([f"{start(1)}{k} = {v}" for k, v in im.items()]) + ':')
#                     if not isinstance(imd, Dataset):
#                         str_print(start(2), imd)
#                         continue
#                     for si, sid in imd.data.items():
#                         str_print('\n'.join([f"{start(2)}{k} = {v}" for k, v in si.items()]) + ':')
#                         str_print(start(3), sid)
#             if omit and i == 2: str_print('...')
#         return str_print.text

#     def __getitem__(self, k):
#         try:
#             if isinstance(k, Subject): return self.data[k.subject][k.image][k.subimage]
#         except KeyError: raise KeyError(f"Invalid key {k}.")
#         if k in self.data: return self.data[k]
#         if isinstance(k, (dict, Info)): return self.data[self.get_info(**k)]
#         return self.data[self.get_info(k)]

#     def __setitem__(self, k, v):
#         if isinstance(k, Subject):
#             self.data.setdefault(k.subject, MedicalImage())
#             self[k.subject].setdefault(k.image, MedicalSubImage())
#             self[k.subject][k.image].setdefault(k.subimage, v)
#         else: self.data[k] = v
    
#     def sort_files(self):
#         for d in self.directories:
#             f = next(d.iter_files())
#             if f | 'dcm' or f | 'ima': search_list = d.iter_subdirs()
#             else: search_list = d.iter_files()
#             for f in search_list:
#                 info = self.map_info(f)
#                 if info is None: continue
#                 info.Image(path=f)
#                 if self.template is None: self.template = info
#                 else:
#                     avouch(self.template.subject.keys() == info.subject.keys() and self.template.image.keys() == info.image.keys())
#                     if len(self.template.keys()) < len(info.keys()): self.template = info
#                     info.SubImage(**{k: '' for k in self.template.keys()[len(info.keys()):]})
#                 self[info] = ImageObject(f)
#         self.data.sort()
#         self.split_datasets(training=0.7, validation=0.2)

#     def select(self, func=None, **kwargs):
#         """
#         Select elements in the data. 
        
#         Note: One can use decorator `@datasetname.select` of a select function to perform an in-place select or 
#             use function `datasetname.select(some_property_need_filtering=filter_value)` to create a new Dataset.
            
#         Args:
#             func (callable): A filter that accepts `subject_info` and `data_for_subject`.
#                 subject_info (dict): The dict consist of all info for subject.
#                 data_for_subject (dict): The 'info' of all child images, as key to find them, as well as 
#                     information needed for filtering.

#         Examples::
#             >>> @datasetname.select
#             >>> def patientID(subject_info, data_for_subject):
#             ...     \"\"\"data_for_subject: SortedDict containing info: data objects. \"\"\"
#             ...     if subject_info['patientID'] == "72": return False # bad data
#             ...     all_modalities = [i['modality'] for i in data_for_subject]
#             ...     return 'CT' in all_modalities and 'MR' in all_modalities
#             ... 
#             >>> datasetname.select(modality='CT&MR')
#             datasetname Dataset (111 images): 
#             ==================================================
#             patientID = 152
#             || modality = MR
#             || modality = CT
#             ...
#             patientID = 162
#             || modality = MR
#             || modality = CT
#         """
#         self.check_data()
#         inplace = True
#         if func is None:
#             def selected(info, data):
#                 def sat(x, y):
#                     get_y = lambda u, v: touch(lambda: type(u)(v), v)
#                     if not isinstance(y, str): return any([i == get_y(i, y) for i in x])
#                     return any([(all([any([i == get_y(i, d) for i in x]) for d in c.split('&')]), print(c))[0] for c in y.split('|')])
#                 all_info = [info.copy().Image(**k.dict()) for k in data.keys()]
#                 if len(all_info) == 0:
#                     return all([sat([info[k]], v) for k, v in kwargs.items() if k in info])
#                 return all([sat([i[k] for i in all_info], v) for k, v in kwargs.items() if k in all_info[0]])
#             func = selected
#             inplace = False
#         to_delete = []
#         for info in map(Subject, self.data.keys()):
#             if not func(info, self.data[info.subject].data): to_delete.append(info.subject)
#         if inplace:
#             for i in to_delete[::-1]: self.data.pop(i)
#             self.data.sort()
#             self.split_datasets(training=0.7, validation=0.2)
#         else:
#             data = self.data.copy()
#             for i in to_delete[::-1]: data.pop(i)
#             return MedicalDataset(data, name=self.name + '.selected')
    
#     def to_subject(self, **kwargs):
#         avouch(self.template is not None)
#         return Subject(**{k: kwargs.get(k, '') for k in self.template.subject.keys()}) \
#               .Image(**{k: kwargs.get(k, '') for k in self.template.image.keys()}) \
#               .SubImage(**{k: kwargs.get(k, '') for k in self.template.subimage.keys()})
    
#     def subimage_infos(self):
#         keys = []
#         for sb, sbd in self.data.items():
#             im, imd = sbd.first()
#             if not isinstance(imd, Dataset): continue
#             n_subimage = im['n_subimage']
#             for l in range(n_subimage):
#                 keys.append(Subject(**sb).Image(n_subimage=n_subimage).to_subimage(l))
#         return keys
    
#     def pair_infos(self):
#         if self.paired_subimage: infos = self.subimage_infos()
#         else: infos = map(Subject, self.data.keys())
#         return infos

#     def randomly_pick_infos(self, n):
#         self.check_data()
#         picked = []
#         infos = self.pair_infos()
#         for _ in range(n): picked.append(infos[random.randint(0, len(infos) - 1)])
#         return picked
    
#     def randomly_pick(self, n):
#         return self.get_batch(self.randomly_pick_infos(n))

#     def split_datasets(self, training = None, validation = None, testing = None):
#         """
#         Split dataset to training, validation and test sets by ratio. e.g. split_datasets(training=0.8, validation = 0.1)
#         """
#         self.check_data()
#         if validation is None and (training is None or testing is None): validation = 0
#         if testing is None and training is None: testing = 0
#         if training is None: training = 1 - validation - testing
#         elif validation is None: validation = 1 - testing - training
#         elif testing is None: testing = 1 - training - validation
#         avouch(training + testing + validation == 1, "Invalid ratios for function 'split_datasets' (Sum is not 1). ")
#         infos = list(self.data.keys())
#         n = len(infos)
#         n_train = int(training * n)
#         n_valid = int(validation * n)
#         random.shuffle(infos)
#         def add_subimage(x):
#             ret = []
#             for s in x:
#                 im, imd = None, None
#                 max_subimage = 0
#                 for img, imgd in self.data[s].items():
#                     if img['n_subimage'] > max_subimage:
#                         max_subimage = img['n_subimage']
#                         im, imd = img, imgd
#                 if not isinstance(imd, Dataset): continue
#                 n_subimage = im['n_subimage']
#                 for l in range(n_subimage):
#                     ret.append(Subject(**s).Image(n_subimage=n_subimage).to_subimage(l))
#             return ret
#         if not self.paired_subimage: add_subimage = lambda x: x
#         self._training_set = add_subimage(infos[:n_train])
#         self._validation_set = add_subimage(infos[n_train:n_train + n_valid])
#         self._testing_set = add_subimage(infos[n_train + n_valid:])
#         return self

#     @alias("train_batch")
#     def training_batch(self, n, **kwargs): return self.batch('training', n, **kwargs)

#     @alias("valid_batch")
#     def validation_batch(self, n, **kwargs): return self.batch('validation', n, **kwargs)

#     @alias("test_batch")
#     def testing_batch(self, n, **kwargs): return self.batch('testing', n, **kwargs)
    
#     def batch(self, stage='training', n_batch=4, shuffle='each_epoch', drop_last=True, none_each_epoch=True, restart=False):
#         self.check_data()
#         if not self._training_set: self.split_datasets(training=0.7, validation=0.2)
#         stage = stage.lower()
#         if stage == 'training': subset = self._training_set
#         elif stage == 'validation': subset = self._validation_set
#         elif stage == 'testing': subset = self._testing_set
#         p = self.batch_pointer[stage]
#         if restart: p = self.batch_pointer[stage] = 0
#         done = False
#         if p < len(subset): 
#             if shuffle: random.shuffle(subset)
#             info_batch = subset[p:p+n]
#             if len(info_batch) == n or not drop_last:
#                 self.batch_pointer[stage] += len(info_batch)
#                 done = True
#         if not done:
#             random.shuffle(subset)
#             self.batch_pointer[stage] = 0
#             if none_each_epoch: return None
#             info_batch = subset[p:p+n]
#             self.batch_pointer[stage] += len(info_batch)
#         return self.get_batch(info_batch)
            
#     def get_batch(self, info_batch):
#         data_arrays = []
#         for info in info_batch:
#             data = self[info.subject]
#             arrays = self.create_batch_func(info.Image(**data.first()[0]), data)
#             if len(arrays) == 0 or arrays is None: return
#             if not data_arrays: data_arrays = [[a] for a in arrays]; continue
#             for i, a in enumerate(arrays):
#                 data_arrays[i].append(a)
#         return tuple((bt.cat(da, {}, crop=True).detach() if da[0].has_batch else bt.stack(da, {}, crop=True).detach()) if isinstance(da[0], bt.Tensor) else da for da in data_arrays)

#     def create_batch(self, func):
#         """
#         Create batch from a subject (e.g. of a patient).
        
#         Args:
#             func (callable): decorated function taking arguments:
#                 subject_info (dict): the info consisting info of the current subject.
#                 group (Dataset Object): the dataset containing the data of this subject.
#                 Returns: torch array without batch dimension, can be a tuple of such arrays. 

#         Examples::
#             >>> @datasetname.create_batch
#             >>> def _(subject_info, group):
#             ...     return group['CT'].to_tensor(), group['MR'].to_tensor() - group['CT'].to_tensor()
#         """
#         self.create_batch_func = func
