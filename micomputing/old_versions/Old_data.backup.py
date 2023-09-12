
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

import random
from collections import OrderedDict
from .stdio import IMG
from .zxhtools.TRS import AFF, FFD

with __info__:
    import batorch as bt
    from pycamia import alias, Path, SPrint, avouch, touch, unique
    from pyoverload import overload

__all__ = """
    Dataset
    DatasetKey
""".split()

class DatasetKey:
    def __init__(self, **kwargs):
        self.keys = kwargs.keys()
    
class Dataset:
    
    def __init__(self, *directories, name = None, info_keys = None, main_key = None):
        """
        register data type. 
        
        Examples:
        ----------
        >>> @Dataset("folderpath1", "folderpath2")
        ... def datasetname(path):
        ...     info_from_path = OrderedDict(
        ...         _patientID = path.split()[1], # main key: `patientID`, necessary for paired data.
        ...         modality = path.name,
        ...     )
        ...     return info_from_path
        """
        self.name = name
        self.info_keys = info_keys
        self.main_key = main_key
        self.data = None
        self.batch_pointer = {'training': 0, 'validation': 0, 'testing': 0}
        self._cache = []
        if len(directories) == 1 and isinstance(directories[0], (list, tuple)):
            directories = directories[0]
            if len(directories) > 0 and not isinstance(directories[0], str) or len(directories) == 0:
                self.data = OrderedDict(directories)
        elif len(directories) == 1 and isinstance(directories[0], OrderedDict):
            self.data = directories[0]
        if self.data is None: self.directories = map(path, directories); return
        self.directories = []
        for _, p in self.data.items():
            if p.ref not in self.directories: self.directories.append(p.ref)
        random.seed(1234)
    
    def __call__(self, func):
        self.map_info = func
        if self.name is None:
            self.name = func.__name__
        self._wash()
        return self
    
    def _info2tuple(self, info):
        return tuple(info.get(k, None) for k in self.info_keys)
    
    def _tuple2info(self, t):
        return dict(zip(self.info_keys, t))
    
    def _main_info(self, d):
        level = self.info_keys.index(self.main_key) + 1
        return d[:level]
    
    def _wash(self):
        catch_info_keys = True
        if self.info_keys: catch_info_keys = False
        self.info_keys = []
        data_pairs = []
        for d in self.directories:
            f = next(d.iter_files())
            if f | 'dcm' or f | 'ima': search_list = d.iter_subdirs()
            else: search_list = d.iter_files()
            for f in search_list:
                info = self.map_info(f)
                if info is None: continue
                if catch_info_keys:
                    for k in info.keys():
                        if k.startswith('_'):
                            k = k[1:]
                            if self.main_key is None: self.main_key = k
                            if k != self.main_key: raise TypeError(f"'Dataset' object with multiple main keys: {self.main_key} and {k[1:]}. Note that only main key can begin with '_' in an info dictionary. ")
                        if k not in self.info_keys: self.info_keys.append(k)
                info[self.main_key] = info.pop('_' + self.main_key)
                data_pairs.append((info, f))
        if self.main_key is None: self.main_key = self.info_keys[-1]
        data = [(self._info2tuple(info), f) for info, f in data_pairs]
        data.sort(key=lambda x: x[0])
        self.data = OrderedDict(data)
        self.split_datasets(training=0.7, validation=0.2)
        
    def __str__(self) -> str:
        self.check_data()
        str_print = SPrint()
        str_print(f"{self.name} Dataset ({len(self.data)} images): ")
        str_print('=' * 50)
        cur_info = None
        n_level1 = len(set([d[0] for d, _ in self.data.items()]))
        omit = n_level1 > 5
        count = 0
        main_level = self.info_keys.index(self.main_key)
        start = lambda i: (' |  ' * i if i <= main_level else ' |  ' * main_level + ' ‖ ' + ' |  ' * (i - main_level - 1))
        for d, v in self.data.items():
            if cur_info is None:
                cur_info = d
                for i, k in enumerate(self.info_keys):
                    if i < len(self.info_keys) - 1: str_print(start(i), f"{k} = {cur_info[i]}", sep='')
                    else: str_print(start(i), f"{k} = {cur_info[i]}: {v}", sep='')
                continue
            for ik in range(len(self.info_keys)):
                if d[ik] != cur_info[ik]: break
            cur_info = d
            if omit:
                if ik == 0: count += 1
                if 2 <= count < n_level1 - 1: continue
                if count == 1:
                    if ik == 0: str_print('...'); continue
                    else: continue
            for i in range(ik, len(self.info_keys)):
                if i < len(self.info_keys) - 1: str_print(start(i), f"{self.info_keys[i]} = {cur_info[i]}", sep='')
                else: str_print(start(i), f"{k} = {cur_info[i]}: {v}", sep='')
        return str_print.text
    
    def __len__(self):
        self.check_data()
        return len(self.data)
    
    def select(self, func=None, **kwargs):
        """
        Select elements in the data. 
        
        Note: One can use decorator `@datasetname.select` of a select function to perform an in-place select or 
            use function `datasetname.select` to create a new Dataset.

        Examples:
        ----------
        >>> @datasetname.select
        >>> def patientID(info_list):
        ...     if info_list[0]['patientID'] == "72": return False # bad data
        ...     all_modalities = [i['modality'] for i in sub_info_list]
        ...     return 'CT' in all_modalities and 'MR' in all_modalities
        ... 
        >>> datasetname.select('patientID', modality='CT')
        datasetname Dataset (111 images): 
        ==================================================
        patientID = 152
         || modality = MR
         || modality = CT
        ...
        patientID = 174
         || modality = CT
        >>> datasetname.select(modality='CT')
        datasetname Dataset (111 images): 
        ==================================================
        patientID = 152
         || modality = CT
        ...
        patientID = 174
         || modality = CT
        """
        self.check_data()
        inplace = True
        if func is None:
            name = self.main_key
            for k in kwargs:
                avouch(k in self.info_keys, f"Key {k} not recognized in dataset {self.name}")
            def iskwarg(k, d):
                if k not in kwargs: return True
                if not isinstance(d, tuple): d = (d,)
                v = kwargs[k]
                return any([x == touch(lambda: type(x)(v), v) for x in d])
            func = lambda info_list: any([all(iskwarg(k, i[k]) for k in self.info_keys) for i in info_list])
            inplace = False
        else: name = func.__name__
        avouch(name in self.info_keys)
        n_level = self.info_keys.index(name) + 1
        collected = []
        to_delete = []
        to_keep = []
        cur_selected = None
        cur_indices = []
        for i, (d, _) in enumerate(self.data.items()):
            if cur_selected is None:
                cur_selected = d[:n_level]
            elif d[:n_level] != cur_selected:
                if not func(collected): to_delete.extend(cur_indices)
                else: to_keep.extend(cur_indices)
                collected.clear()
                cur_indices.clear()
                cur_selected = d[:n_level]
            collected.append(self._tuple2info(d))
            cur_indices.append(d)
            if i == len(self.data) - 1:
                if not func(collected): to_delete.extend(cur_indices)
                else: to_keep.extend(cur_indices)
        if inplace:
            for i in to_delete[::-1]: self.data.pop(i)
        else:
            data = self.data.copy()
            for i in to_delete[::-1]: data.pop(i)
            return Dataset(data, name=self.name + '.selected', info_keys=self.info_keys, main_key=self.main_key)
        
    def seed(self, s): random.seed(s)
    
    def check_data(self): avouch(self.data is not None, "Dataset not created yet. Use `@Dataset(directory_paths)` in front of a function mapping a path to an info structure to create Dataset. ")
    
    def cache(self, k, v=None):
        if v is None: return self._cache[k]
        self._cache[k] = v

    @overload
    @alias('__getitem__')
    def get(self, x: (tuple, dict)):
        self.check_data()
        avouch(len(x) == len(self.info_keys))
        if isinstance(x, dict): x = self._info2tuple(x)
        return self.data[x]

    @overload
    def get(self, **kwargs):
        return self.get(kwargs)

    @overload
    def load(self, x: (tuple, dict)):
        self.check_data()
        avouch(len(x) == len(self.info_keys))
        if isinstance(x, dict): x = self._info2tuple(x)
        p = self.data[x]
        if isinstance(p, Path):
            r = self.preprocessor(self._tuple2info(x))
            if r is None: p = [p]
            else:
                if isinstance(r, (tuple, list)):
                    if r[0] != p: r = type(r)([p]) + r
                    r = {x: r}
                if isinstance(r, dict):
                    for k in r:
                        if isinstance(k, dict):
                            t = self._info2tuple(k)
                            r[t] = r.pop(k)
                    self.data.update(r)
                    for k in r: self.load(k)
                    return
        if isinstance(p, list):
            avouch(isinstance(p[0], Path), "Procedures in Dataset should be a list started with a `path`. ")
            data = None
            for loader in (self.cache, IMG, AFF, FFD, lambda x: x.open().read(), lambda x: open(x).read()):
                try: data = loader(p[0].abs); break
                except Exception as e:
                    if 'DecodeError' not in str(type(e)): raise e
            if data is None: raise TypeError(f"Cannot open file {p[0]} yet, please contact the developpers. ")
            self.cache(p[0].abs, data)
            for f in p[1:]: data = f(data)
            self.data[x + ('data',)] = data
            return data
        return self.data[x]

    @overload
    def load(self, **kwargs):
        return self.load(kwargs)
    
    def main_infos(self):
        return unique([self._main_info(d) for d in self.data.keys()])
    
    def randomly_pick_infos(self, n):
        self.check_data()
        picked = []
        keys = self.main_infos()
        for _ in range(n): picked.append(keys[random.randint(0, len(keys) - 1)])
        return picked
    
    def randomly_pick(self, n):
        return self.create_batch(self.randomly_pick_infos(n))

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
        infos = self.main_infos()
        n = len(infos)
        n_train = int(training * n)
        n_valid = int(validation * n)
        random.shuffle(infos)
        self._training_set = infos[:n_train]
        self._validation_set = infos[n_train:n_train + n_valid]
        self._testing_set = infos[n_train + n_valid:]
        return self

    @alias("train_batch")
    def training_batch(self, n, **kwargs): return self.batch('training', n, **kwargs)

    @alias("valid_batch")
    def validation_batch(self, n, **kwargs): return self.batch('validation', n, **kwargs)

    @alias("test_batch")
    def testing_batch(self, n, **kwargs): return self.batch('testing', n, **kwargs)
    
    def batch(self, stage='training', n=4, shuffle=False, drop_last=True, none_each_epoch=True):
        self.check_data()
        stage = stage.lower()
        if stage == 'training': subset = self._training_set
        elif stage == 'validation': subset = self._validation_set
        elif stage == 'testing': subset = self._testing_set
        p = self.batch_pointer[stage]
        if p < len(subset): 
            if shuffle: random.shuffle(subset)
            info_batch = subset[p:p+n]
            if len(info_batch) == n or not drop_last:
                self.batch_pointer[stage] += len(info_batch)
                return self.create_batch(info_batch)
        random.shuffle(subset)
        self.batch_pointer[stage] = 0
        if none_each_epoch: return None
        info_batch = subset[p:p+n]
        self.batch_pointer[stage] += len(info_batch)
        return self._create_batch(info_batch)
        
    def _create_batch(self, info_batch):
        self.check_data()
        mainset = [[]] * len(info_batch)
        for k in self.data.keys():
            if self._main_info(k) in info_batch:
                mainset[info_batch.index(self._main_info(k))].append(k)
        batch_data = []
        for subset in mainset:
            group = Dataset.Group()
            for info in subset:
                data = self.load(info)
                group.append(info, data)
            self.organize_batch(group)
            batch_data.append(data)

    def create_batch(self, func):
        """
        Create batch from a group of data (e.g. of a patient).

        Examples:
        ----------
        >>> @datasetname.create_batch
        >>> def _(group):
        ...     return group['CT'], group['MR'] - group['CT']
        """
        self.organize_batch = func

    def preprocess(self, func):
        """
        The preprocess function.

        Examples:
        ----------
        >>> @datasetname.preprocess
        >>> def _(info):
        ...     return (Dataset.minmax(0, 500) if info['modality'] == 'MR' else Dataset.minmax(0, 1024)), Dataset.resample(1), Dataset.crop_as(256, 256)
        """
        self.preprocessor = func
                
    class Group(OrderedDict):
        def append(self, k, v): super().__setitem__(k, v)
        @alias('__setattr__')
        def __setitem__(self, k, v): super().__setitem__(k, v)
        @alias('__getattr__')
        def __getitem__(self, k):
            if k in self.keys(): return super().__getitem__(k)
            res = []
            more_poss = []
            if isinstance(k, dict): k = tuple(k.values())
            if not isinstance(k, tuple): k = (k,)
            for key in super().keys():
                if all([x in key for x in k]):
                    res.append(key)
                    if 'data' in key: more_poss.append(key)
            if len(more_poss) > 1: raise TypeError("Cannot use part of keys to find a group item as it is not unique. ")
            if len(more_poss) == 1: return more_poss[0]
            if len(res) > 1: raise TypeError("Cannot use part of keys to find a group item as it is not unique. ")
            if len(res) == 0: raise TypeError("Cannot use part of keys to find a group item as it is not existed. ")
            return res[0]
        
        
        
        
        
from collections import OrderedDict
from micomputing import Dataset, Subject
import batorch as bt

data_paths = open("../DATA_PATH.txt").read().strip().split('\n')

@Dataset(data_paths)
def LFD(p):
    if p.ref.name == "Liver_Fibrosis_PMX":
        if 'T1_DYN' not in p.name: return
        pid, name = p.parent.filename.split('_')
        ctype, _, phase = p.name.replace('_DYN', '').split('_')
        return Subject(
            patient_id = pid,
            patient_name = name
        ).Image(
            contrast_type = ctype,
            modality = 'T1',
            phase = phase
        ).SubImage(
            patch_id = 0,
        )
    elif p.ref.name == "Liver_Fibrosis_PMX_FLD":
        pid, name = p.parent.filename.split('_')
        ctype, phase = p.name.split('_')
        mod = p.ext.upper()
        if not pid.startswith('*'): return
        pid = pid[1:]
        return OrderedDict(
            patient_id = pid,
            _patch_id = 0,
            patient_name = name,
            contrast_type = ctype,
            modality = mod,
            phase = phase
        )
        
@LFD.select
def phase(sub_info_list):
    return sub_info_list[0]['phase'] != '1'
        
@LFD.select
def patient_id(sub_info_list):
    pairs = dict(img=set(), fld=set())
    for i in sub_info_list:
        p = (i['contrast_type'], i['phase'])
        if i['modality'] == 'T1': pairs['img'].add(p)
        if i['modality'] == 'FLD': pairs['fld'].add(p)
    return len(pairs['img']) >= 9 and len(pairs['fld']) >= 8

@LFD.preprocess
def _(info, path):
    
    return {info: path}

@LFD.create_batch
def _(group):
    msk = group['T1', 'PT', 'MSK']
    dyn = bt.stack(group['T1', 'PT', 'ART'], group['T1', 'PT', 'POR'], group['T1', 'PT', 'DEL'], {})
    pt = bt.cat(msk, (dyn - msk) / msk, {})
    aff = bt.stack(group['AFF', 'PT', 'ART'], group['AFF', 'PT', 'POR'], group['AFF', 'PT', 'DEL'], {})
    ffd = bt.stack(group['FFD', 'PT', 'ART'], group['FFD', 'PT', 'POR'], group['FFD', 'PT', 'DEL'], {})
    return pt, aff, ffd, group['T1', 'PMX', 'HBP']

LFD.load(('346', 0, 'zhangnaping', 'PMX', 'T1', 'ART'))
print(LFD)

