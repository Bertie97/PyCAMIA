
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
from .stdio import IMG
from .zxhtools.TRS import TRS

with __info__:
    import batorch as bt
    from pycamia import alias, Path, SPrint, ByteSize
    from pycamia import avouch, touch, to_tuple, arg_tuple, execblock
    from pyoverload import sequence

__all__ = """
    Info
    Subject
    ImageObject
    Dataset
    MedicalDataset
""".split()

class SortedDict(dict):
    
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], dict):
            return SortedDict.__new__dict(*args, **kwargs)
        if len(args) > 1 and not isinstance(args[0], str):
            return SortedDict.__new__keyvalue(*args, **kwargs)
        return SortedDict.__new__default(*args, **kwargs)

    @classmethod
    def __new__keyvalue(cls, keys:list=[], values:list=[]):
        self = super().__new__(cls, zip(keys, values))
        self.key_order = keys
        return self

    @classmethod
    def __new__dict(cls, dic:dict, order=None):
        self = super().__new__(cls, dic)
        if order is None: self.key_order = list(dic.keys())
        else: self.key_order = order
        return self

    @classmethod
    def __new__default(cls, *args, **kwargs):
        self = super().__new__(cls, kwargs)
        if len(args) == 0: self.key_order = list(kwargs.keys())
        else: self.key_order = list(arg_tuple(args))
        return self

    def sort(self): self.key_order.sort()
        
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

    def pop(self, k):
        super().pop(k)
        self.key_order.remove(k)

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

class Info:
    def __init__(self, **kwargs):
        self._keys = []
        self._values = []
        self._hidden_keys = []
        self._hidden_values = []
        for k, v in kwargs.items():
            if k.startswith('_'): 
                self._hidden_keys.append(k[1:])
                self._hidden_values.append(v)
                continue
            self._keys.append(k)
            self._values.append(v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._keys:
                self._values[self._keys.index(k)] = v
                continue
            if k.startswith('_') and k[1:] in self._hidden_keys:
                self._hidden_values[self._hidden_keys.index(k[1:])] = v
                continue
            if k.startswith('_'): 
                self._hidden_keys.append(k[1:])
                self._hidden_values.append(v)
                continue
            self._keys.append(k)
            self._values.append(v)

    def tuple(self):
        return tuple(self._values)
    
    def visible(self):
        return self._keys

    def keys(self):
        return self._keys + ['_'+k for k in self._hidden_keys]

    def values(self):
        return self._values + self._hidden_values
    
    def items(self): return zip(self.keys(), self.values())
    
    def get(self, k, v): return self[k] if k in self else v
    
    def dict(self): return dict(self.items())
        
    def copy(self): return Info(**self.dict())

    def __getitem__(self, k):
        if isinstance(k, int): return self.tuple()[k]
        elif k in self._keys: return self._values[self._keys.index(k)]
        elif k in self._hidden_keys: return self._hidden_values[self._hidden_keys.index(k)]
        elif k.startswith('_') and k[1:] in self._hidden_keys: return self._hidden_values[self._hidden_keys.index(k[1:])]
        else: raise IndexError(f"No '{k}' in Info object. ")

    def __setitem__(self, k, v):
        if k in self._keys: self._values[self._keys.index(k)] = v
        elif k in self._hidden_keys: self._hidden_values[self._hidden_keys.index(k)] = v
        elif k.startswith('_') and k[1:] in self._hidden_keys: self._hidden_values[self._hidden_keys.index(k[1:])] = v
        elif k.startswith('_'): self._hidden_keys.append(k[1:]); self._hidden_values.append(v)
        else: self._keys.append(k); self._values.append(v)

    def __len__(self): return len(self._keys) + len(self._hidden_keys)

    def __contains__(self, k): return k in self._keys or k in self._hidden_keys or k.startswith('_') and k[1:] in self._hidden_keys

    for __op__ in [f'__{op}__' for op in "lt le gt ge eq".split()]:
        execblock(f"""
        def {__op__}(x, y):
            return x.tuple().{__op__}(getattr(y, 'tuple', lambda: tuple(y))())
        """)

    def __hash__(self): return hash(self.tuple())
    
    @alias('__repr__')
    def __str__(self):
        return '{' + ', '.join([f"{k}: {repr(v)}" for k, v in zip(self._keys, self._values)]) + \
            ('; ' + ', '.join([f"{k}: {repr(v)}" for k, v in zip(self._hidden_keys, self._hidden_values)]) if len(self._hidden_keys) > 0 else '') + '}'

class Subject:
    def __init__(self, x=None, **kwargs):
        if x is not None and isinstance(x, (dict, Info)):
            kwargs = x
        self.subject = Info(**kwargs)
        self.image = Info(_n_subimage = 0)
        self.subimage = Info(subimage_id = 'whole')

    def Image(self, **kwargs):
        if 'n_subimage' in kwargs: kwargs['_n_subimage'] = kwargs.pop('n_subimage')
        if 'path' in kwargs: kwargs['_path'] = kwargs.pop('path')
        self.image.update(**kwargs)
        return self

    def SubImage(self, *args, **kwargs):
        if len(args) == 1:
            self.Image(n_subimage = args[0])
            return self
        elif len(args) == 0:
            self.subimage.update(**kwargs)
            return self
        else: raise TypeError()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.subject: self.subject[k] = v
            elif k in self.image: self.image[k] = v
            elif k in self.subimage: self.subimage[k] = v
        return self

    def get_subimage(self, i):
        return Info(subimage_id = f"%0{len(str(self.image['n_subimage']))}d"%i)

    def to_subimage(self, i):
        ret = self.copy()
        if i < 0: ret.update(subimage_id = 'whole')
        else: ret.update(subimage_id = f"%0{len(str(self.image['n_subimage']))}d"%i)
        return ret
    
    def to_wholeimage(self):
        return self.to_subimage(-1)

    def tuple(self):
        return self.subject.tuple() + self.image.tuple() + self.subimage.tuple()

    def keys(self):
        return self.subject.keys() + self.image.keys() + self.subimage.keys()
        
    def copy(self):
        subject = Subject(**self.subject.dict())
        subject.image = self.image.copy()
        subject.subimage = self.subimage.copy()
        return subject

    def __contains__(self, k):
        if isinstance(k, int): return k < len(self)
        elif k in self.subject: return True
        elif k in self.image: return True
        elif k in self.subimage: return True
        else: return False

    def __getitem__(self, k):
        if isinstance(k, int): return self.tuple()[k]
        elif k in self.subject: return self.subject[k]
        elif k in self.image: return self.image[k]
        elif k in self.subimage: return self.subimage[k]
        else: raise IndexError(f"No '{k}' in micomputing.Subject-Image-SubImage-Info object. ")

    def __setitem__(self, k, v):
        if k in self.subject: self.subject[k] = v
        elif k in self.image: self.image[k] = v
        elif k in self.subimage: self.subimage[k] = v
        else: raise IndexError(f"No '{k}' in micomputing.Subject-Image-SubImage-Info object. ")

    def __len__(self): return len(self.subject) + len(self.image) + len(self.subimage)

    for __op__ in [f'__{op}__' for op in "lt le gt ge eq".split()]:
        execblock(f"""
        def {__op__}(x, y):
            return x.tuple().{__op__}(getattr(y, 'tuple', lambda: tuple(y))())
        """)

    def __hash__(self): return hash(self.tuple())
    
    @alias('__repr__')
    def __str__(self): return '{' + ' | '.join([str(self.subject).strip('{}'), str(self.image).strip('{}'), str(self.subimage).strip('{}')]) + '}'
    
    def __call__(self, **kwargs):
        subject = self.copy()
        for k, v in kwargs.items(): subject[k] = v
        return subject
    
class ImageObject:
    def __init__(self, file=None):
        self.file = file
        self.data = None
        
    @property
    def is_loaded(self): return self.data is not None
        
    @property
    def path(self): return self.file

    def load(self, loader=None):
        if self.is_loaded: return self.data
        file = Path(self.file)
        if loader is None:
            for loader in (IMG, TRS.load, lambda x: x.open().read(), lambda x: open(x).read()):
                try: self.data = loader(file.abs); break
                except Exception as e:
                    if isinstance(e, TypeError): pass
                    elif isinstance(e, FileNotFoundError): raise e
                    elif 'DecodeError' in str(type(e)): pass
                    else: raise e
        else: self.data = loader(file.abs)
        if self.data is None: raise TypeError(f"Cannot open file {file.abs} yet, please contact the developpers (Error Code: D651). ")
        return self.data

    def release(self):
        del self.data
        gc.collect()
        self.data = None

    @alias('__repr__')
    def __str__(self):
        return f"<ImageObject at {self.file}>"

class Dataset:
    
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            x = args[0]
            if isinstance(x, dict): self.__init__dict(*args, **kwargs)
            elif isinstance(x, str): self.__init__str(*args, **kwargs)
            else: self.__init__sequence(*args, **kwargs)
        else: self.__init__str(*args, **kwargs)

    def __init__dict(self, x: SortedDict, name = None):
        self.name = name
        self.data = x
        self.batch_pointer = {'training': 0, 'validation': 0, 'testing': 0}
        self.directories = []
        for k in self.data.keys():
            if 'path' not in k: continue
            if k.path.ref not in self.directories: self.directories.append(k.path.ref)
        self._cache = []

    def __init__sequence(self, x: sequence, name = None):
        self.__init__(*x, name=name)

    def __init__str(self, *directories: str, name = None):
        self.name = name
        self.data = SortedDict()
        self.batch_pointer = {'training': 0, 'validation': 0, 'testing': 0}
        self.directories = map(Path, directories)
        self._cache = []

    def __call__(self, func):
        self.map_info = func
        if self.name is None:
            self.name = func.__name__
        self.sort_files()
        return self
    
    def __len__(self):
        self.check_data()
        return len(self.data)

    def seed(self, s): random.seed(s)
    
    def check_data(self): avouch(len(self.data) > 0, "Dataset not created yet. Use `@Dataset(directory_paths)` in front of a function mapping a path to an info structure to create Dataset. ")
    
    def cache(self, k, v=None):
        if v is None: return self._cache[k]
        self._cache[k] = v
        
    def byte_size(self):
        total_size = 0
        for ele in self.data.values():
            if isinstance(ele, Dataset):
                total_size += ele.byte_size()
            elif isinstance(ele, ImageObject):
                if isinstance(ele.data, bt.Tensor):
                    total_size += ele.data.byte_size()
            elif isinstance(ele, bt.Tensor):
                total_size += ele.byte_size()
        return ByteSize(total_size)

    def sort_files(self):
        for d in self.directories:
            for f in d.iter_files():
                info = self.map_info(f)
                if info is None: continue
                self[info] = f
        self.data.sort()
        self.split_datasets(training=0.7, validation=0.2)

    def __str__(self) -> str:
        self.check_data()
        self.data.sort()
        str_print = SPrint()
        s = "s" if len(self.data) > 1 else ''
        if self.name is None:
            str_print("SubDataset", f"({len(self.data)}", f"subject{s}): ")
        else:
            str_print(self.name, "Dataset", f"({len(self.data)}", f"subject{s}): ")
        str_print('=' * 50)
        if hasattr(self, "_training_set"):
            str_print(f"{len(self._training_set)} training, {len(self._validation_set)} validation subject(s), {len(self._testing_set)} test subject(s).")
            str_print('=' * 50)
        cur_info = None
        level1s = set([d[0] for d in self.data.keys()])
        omit = len(level1s) > 8
        count = 0
        k = level1s.pop()
        main_level = len(k) - 1
        start = lambda i: (' |  ' * i if i <= main_level else (' |  ' * main_level + ' ‖ ' + ' |  ' * (i - main_level - 1)))
        for d, v in self.data.items():
            if cur_info is None:
                cur_info = d
                for i, k in enumerate(d.visible()):
                    if i < len(d.tuple()) - 1: str_print(start(i), f"{k} = {cur_info[k]}", sep='')
                    else:
                        skey = f"{k} = {cur_info[k]}: "
                        str_v = str(v).split('=' * 10)[-1].lstrip('=').strip()
                        str_print(start(i), skey, str_v.replace('\n', '\n' + start(i) + ' ' * len(skey)), sep='')
                continue
            for ik in range(len(cur_info)):
                if d[ik] != cur_info[ik]: break
            cur_info = d
            if omit:
                if ik == 0: count += 1
                if 2 <= count < len(level1s) - 1: continue
                if count == 1:
                    if ik == 0: str_print('...'); continue
                    else: continue
            for i in range(ik, len(cur_info.tuple())):
                if i < len(cur_info.tuple()) - 1: str_print(start(i), f"{cur_info.keys()[i]} = {cur_info[i]}", sep='')
                else:
                    skey = f"{cur_info.keys()[i]} = {cur_info[i]}: "
                    str_v = str(v).split('=' * 10)[-1].lstrip('=').strip()
                    str_print(start(i), skey, str_v.replace('\n', '\n' + start(i) + ' ' * len(skey)), sep='')
        return str_print.text

    def setdefault(self, k, v):
        if k not in self.data: self.data[k] = v

    def get_info(self, *args, **kwargs):
        avouch(len(args) > 0 and len(kwargs) == 0 or len(args) == 0 and len(kwargs) > 0)
        if len(args) > 0:
            k = to_tuple(args)
            if len(k) == 1 and isinstance(k[0], (list, tuple)): k = k[0]
        candidates = []
        for info in self.data.keys():
            if len(args) > 0:
                if all([x in info.tuple() for x in k]):
                    candidates.append(info)
            else:
                if all([info.get(k, None) == kwargs[k] for k in kwargs]):
                    candidates.append(info)
        if len(candidates) == 1: return candidates[0]
        raise TypeError(f"Cannot use part of keys {args if len(args) > 0 else kwargs} to find non-unique or non-existed items. Number of matches: {len(candidates)} ")
    
    def get(self, *args, **kwargs):
        avouch(len(args) > 0 and len(kwargs) == 0 or len(args) == 0 and len(kwargs) > 0)
        if len(args) > 0:
            k = args[0]
            v = None
            if len(args) > 1: v = args[1]
            if k not in self.data: return v
            return self.data[k]
        try: return self.data[self.get_info(**kwargs)]
        except TypeError: raise KeyError(f"Invalid key {kwargs}.")

    def items(self): return self.data.items()
    
    def first(self): return self.data.first()

    def __setitem__(self, k, v):
        self.data[k] = v

    def __getitem__(self, k):
        if isinstance(k, Info) and k == Info(): return self
        if k in self.data: return self.data[k]
        try:
            if isinstance(k, (dict, Info)): return self.data[self.get_info(**k)]
            return self.data[self.get_info(k)]
        except TypeError: raise KeyError(f"Invalid key {k}.")
    
    def __getattr__(self, k):
        try: return getattr(super(), k, self[k])
        except KeyError as e: raise AttributeError(str(e))

    def __iter__(self):
        return iter(self.data)

class MedicalSubImage(Dataset):
    
    def __getitem__(self, k):
        try:
            if isinstance(k, Subject): return self.data[k.subimage]
        except KeyError: raise KeyError(f"Invalid key {k}.")
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, Subject):
            self.data.setdefault(k.subimage, v)
        else: self.data[k] = v
    
class MedicalImage(Dataset):
    
    def __getitem__(self, k):
        try:
            if isinstance(k, Subject): return self.data[k.image][k.subimage]
        except KeyError as e: raise KeyError(eval(str(e)).strip('.') + f" in {k}.")
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, Subject):
            self.data.setdefault(k.image, MedicalSubImage())
            self.data[k.image].setdefault(k.subimage, v)
        else: self.data[k] = v

class MedicalDataset(Dataset):

    def __init__(self, *args, **kwargs):
        """
        register data type. 
        
        Examples:
        ----------
        >>> @Dataset("folderpath1", "folderpath2")
        ... def datasetname(path):
        ...     return Subject(patientID = path.split()[1]) \
        ...           .Image(modality = path.name)
        ... 
        >>> datasetname
        datasetname Dataset (121 images): 
        ==================================================
        patientID = 152
         || modality = MR
         || modality = CT
        ...
        patientID = 174
         || modality = CT
        """
        super().__init__(*args, **kwargs)
        self.paired_subimage = kwargs.get('paired_subimage', True)
        self.template = None

    def __str__(self) -> str:
        self.check_data()
        str_print = SPrint()
        s = "s" if len(self.data) > 1 else ''
        str_print(f"{self.name} Dataset ({len(self.data)} subject{s}): ")
        str_print('=' * 50)
        omit = len(self.data) > 8
        start = lambda i: ' |  ' * i
        for i, (sb, sbd) in enumerate(self.data.items()):
            if not omit or i < 2 or i >= len(self.data) - 2:
                str_print('\n'.join([f"{k} = {v}" for k, v in sb.items()]) + ':')
                for im, imd in sbd.data.items():
                    str_print('\n'.join([f"{start(1)}{k} = {v}" for k, v in im.items()]) + ':')
                    if not isinstance(imd, Dataset):
                        str_print(start(2), imd)
                        continue
                    for si, sid in imd.data.items():
                        str_print('\n'.join([f"{start(2)}{k} = {v}" for k, v in si.items()]) + ':')
                        str_print(start(3), sid)
            if omit and i == 2: str_print('...')
        return str_print.text

    def __getitem__(self, k):
        try:
            if isinstance(k, Subject): return self.data[k.subject][k.image][k.subimage]
        except KeyError: raise KeyError(f"Invalid key {k}.")
        if k in self.data: return self.data[k]
        if isinstance(k, (dict, Info)): return self.data[self.get_info(**k)]
        return self.data[self.get_info(k)]

    def __setitem__(self, k, v):
        if isinstance(k, Subject):
            self.data.setdefault(k.subject, MedicalImage())
            self[k.subject].setdefault(k.image, MedicalSubImage())
            self[k.subject][k.image].setdefault(k.subimage, v)
        else: self.data[k] = v
    
    def sort_files(self):
        for d in self.directories:
            f = next(d.iter_files())
            if f | 'dcm' or f | 'ima': search_list = d.iter_subdirs()
            else: search_list = d.iter_files()
            for f in search_list:
                info = self.map_info(f)
                if info is None: continue
                info.Image(path=f)
                if self.template is None: self.template = info
                else:
                    avouch(self.template.subject.keys() == info.subject.keys() and self.template.image.keys() == info.image.keys())
                    if len(self.template.keys()) < len(info.keys()): self.template = info
                    info.SubImage(**{k: '' for k in self.template.keys()[len(info.keys()):]})
                self[info] = ImageObject(f)
        self.data.sort()
        self.split_datasets(training=0.7, validation=0.2)

    def select(self, func=None, **kwargs):
        """
        Select elements in the data. 
        
        Note: One can use decorator `@datasetname.select` of a select function to perform an in-place select or 
            use function `datasetname.select` to create a new Dataset.

        Examples:
        ----------
        >>> @datasetname.select
        >>> def patientID(subject_info, data_for_subject):
        ...     \"\"\"data_for_subject: SortedDict containing info: data objects. \"\"\"
        ...     if subject_info['patientID'] == "72": return False # bad data
        ...     all_modalities = [i['modality'] for i in data_for_subject]
        ...     return 'CT' in all_modalities and 'MR' in all_modalities
        ... 
        >>> datasetname.select(modality='CT&MR')
        datasetname Dataset (111 images): 
        ==================================================
        patientID = 152
         || modality = MR
         || modality = CT
        ...
        patientID = 162
         || modality = MR
         || modality = CT
        """
        self.check_data()
        inplace = True
        if func is None:
            def selected(info, data):
                def sat(x, y):
                    get_y = lambda u, v: touch(lambda: type(u)(v), v)
                    if not isinstance(y, str): return any([i == get_y(i, y) for i in x])
                    return any([(all([any([i == get_y(i, d) for i in x]) for d in c.split('&')]), print(c))[0] for c in y.split('|')])
                all_info = [info.copy().Image(**k.dict()) for k in data.keys()]
                if len(all_info) == 0:
                    return all([sat([info[k]], v) for k, v in kwargs.items() if k in info])
                return all([sat([i[k] for i in all_info], v) for k, v in kwargs.items() if k in all_info[0]])
            func = selected
            inplace = False
        to_delete = []
        for info in map(Subject, self.data.keys()):
            if not func(info, self.data[info.subject].data): to_delete.append(info.subject)
        if inplace:
            for i in to_delete[::-1]: self.data.pop(i)
            self.data.sort()
            self.split_datasets(training=0.7, validation=0.2)
        else:
            data = self.data.copy()
            for i in to_delete[::-1]: data.pop(i)
            return MedicalDataset(data, name=self.name + '.selected')
    
    def to_subject(self, **kwargs):
        avouch(self.template is not None)
        return Subject(**{k: kwargs.get(k, '') for k in self.template.subject.keys()}) \
              .Image(**{k: kwargs.get(k, '') for k in self.template.image.keys()}) \
              .SubImage(**{k: kwargs.get(k, '') for k in self.template.subimage.keys()})
    
    def subimage_infos(self):
        keys = []
        for sb, sbd in self.data.items():
            im, imd = sbd.first()
            if not isinstance(imd, Dataset): continue
            n_subimage = im['n_subimage']
            for l in range(n_subimage):
                keys.append(Subject(**sb).Image(n_subimage=n_subimage).to_subimage(l))
        return keys
    
    def pair_infos(self):
        if self.paired_subimage: infos = self.subimage_infos()
        else: infos = map(Subject, self.data.keys())
        return infos

    def randomly_pick_infos(self, n):
        self.check_data()
        picked = []
        infos = self.pair_infos()
        for _ in range(n): picked.append(infos[random.randint(0, len(infos) - 1)])
        return picked
    
    def randomly_pick(self, n):
        return self.get_batch(self.randomly_pick_infos(n))

    def split_datasets(self, training = None, validation = None, testing = None):
        """
        Split dataset to training, validation and test sets by ratio. e.g. split_dataset(training=0.8, validation = 0.1)
        """
        self.check_data()
        if validation is None and (training is None or testing is None): validation = 0
        if testing is None and training is None: testing = 0
        if training is None: training = 1 - validation - testing
        elif validation is None: validation = 1 - testing - training
        elif testing is None: testing = 1 - training - validation
        avouch(training + testing + validation == 1, "Invalid ratios for function 'split_datasets' (Sum is not 1). ")
        infos = list(self.data.keys())
        n = len(infos)
        n_train = int(training * n)
        n_valid = int(validation * n)
        random.shuffle(infos)
        def add_subimage(x):
            ret = []
            for s in x:
                im, imd = None, None
                max_subimage = 0
                for img, imgd in self.data[s].items():
                    if img['n_subimage'] > max_subimage:
                        max_subimage = img['n_subimage']
                        im, imd = img, imgd
                if not isinstance(imd, Dataset): continue
                n_subimage = im['n_subimage']
                for l in range(n_subimage):
                    ret.append(Subject(**s).Image(n_subimage=n_subimage).to_subimage(l))
            return ret
        if not self.paired_subimage: add_subimage = lambda x: x
        self._training_set = add_subimage(infos[:n_train])
        self._validation_set = add_subimage(infos[n_train:n_train + n_valid])
        self._testing_set = add_subimage(infos[n_train + n_valid:])
        return self

    @alias("train_batch")
    def training_batch(self, n, **kwargs): return self.batch('training', n, **kwargs)

    @alias("valid_batch")
    def validation_batch(self, n, **kwargs): return self.batch('validation', n, **kwargs)

    @alias("test_batch")
    def testing_batch(self, n, **kwargs): return self.batch('testing', n, **kwargs)
    
    def batch(self, stage='training', n=4, shuffle=False, drop_last=True, none_each_epoch=True, restart=False):
        self.check_data()
        if not hasattr(self, '_training_set'): self.split_datasets(training=0.7, validation=0.2)
        stage = stage.lower()
        if stage == 'training': subset = self._training_set
        elif stage == 'validation': subset = self._validation_set
        elif stage == 'testing': subset = self._testing_set
        p = self.batch_pointer[stage]
        if restart: p = self.batch_pointer[stage] = 0
        done = False
        if p < len(subset): 
            if shuffle: random.shuffle(subset)
            info_batch = subset[p:p+n]
            if len(info_batch) == n or not drop_last:
                self.batch_pointer[stage] += len(info_batch)
                done = True
        if not done:
            random.shuffle(subset)
            self.batch_pointer[stage] = 0
            if none_each_epoch: return None
            info_batch = subset[p:p+n]
            self.batch_pointer[stage] += len(info_batch)
        return self.get_batch(info_batch)
            
    def get_batch(self, info_batch):
        data_arrays = []
        for info in info_batch:
            data = self[info.subject]
            arrays = self.create_batch_func(info.Image(**data.first()[0]), data)
            if len(arrays) == 0 or arrays is None: return
            if not data_arrays: data_arrays = [[a] for a in arrays]; continue
            for i, a in enumerate(arrays):
                data_arrays[i].append(a)
        return tuple((bt.cat(da, []).detach() if da[0].has_batch else bt.stack(da, []).detach()) if isinstance(da[0], bt.Tensor) else da for da in data_arrays)

    def create_batch(self, func):
        """
        Create batch from a subject (e.g. of a patient).

        Examples:
        ----------
        >>> @datasetname.create_batch
        >>> def _(group):
        ...     return group['CT'], group['MR'] - group['CT']
        """
        self.create_batch_func = func
        