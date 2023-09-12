
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File of reading & writing medical files. ",
    help = "Use `NII` and `DCM` to cope with files.",
    requires = ["nibabel", "pydicom"]
)

__all__ = """
    IMG
    dcm2nii
    nii2dcm
""".split()

import os
from datetime import datetime as dt

with __info__:
    import SimpleITK as sitk
    import batorch as bt
    import numpy as np
    from pycamia import Path, Workflow, SPrint, Error
    from pycamia import no_print, touch, avouch, alias, get_environ_vars
    from pycamia import to_tuple, to_list, get_alphas, get_digits, arg_tuple
    from pycamia import item, argmax, sublist, map_ele
    from pycamia.numpyop import toI
    from pyoverload import overload, callable

available_modalities = ['PNG'] # TODO: MHA
sitk_supported_modalities = ['NII', 'NRRD']
sitk_default_exts = {'NII': '.nii.gz', 'NRRD': '.nrrd'}
sitk_modality_exts = {'.nii': 'NII', '.nii.gz': 'NII', '.nrrd': 'NRRD'}
available_modalities.extend(sitk_supported_modalities)

try: import nibabel as nib; import nrrd # pip install pynrrd
except ModuleNotFoundError: pass
try: import pydicom as dcm; available_modalities.append('DCM')
except [ImportError, ModuleNotFoundError]: pass

run = Workflow(*available_modalities, verbose=False)
cdate = lambda: dt.now().strftime("%Y%m%d")
ctime = lambda: dt.now().strftime("%H%M%S")
cmilitime = lambda: dt.now().strftime("%H%M%S.%f")[:-3]
cstamp = lambda: dt.now().strftime("%Y%m%d%H%M%S")
cmilistamp = lambda: dt.now().strftime("%Y%m%d%H%M%S%f")[:-2]
get_uid7 = lambda: "1.2.100.100000.100000.111." + cstamp()
get_uid10 = lambda: "1.3.10.100000.11.10000.1.0.1000." + cmilistamp()

basic_tags = {
    "0010|0010": "Unknown", # Patient Name
    "0010|0020": "00000001", # Patient ID
    "0010|0030": "20000101", # Patient Birth Date
    "0010|0040": 'M', # Patient Sex
    "0010|1010": "020Y", # Patient Age
    "0010|1020": "100", # Patient Height
    "0010|1030": "200", # Patient Weight
    "0008|0020": cdate(), # Study Date
    "0008|0030": ctime(), # Study Time
    "0008|0050": cdate() + "00010001", # Accession Number
    "0008|0060": "Unknown",  # Modality
    "0020|000D": get_uid7(), # Study Instance UID, for machine consumption
    "0020|0010": "111111111", # Study ID, for human consumption
    "0020|0052": get_uid10(), # Frame of Reference UID
    # "0028|1050": '197.64', # Window Center
    # "0028|1051": '375.23', # Window Width
}

with run("DCM", "NII"), run.all_tags:
    def dcm2nii(file1, file2):
        IMG(file1).astype(IMG.nii).save(file2)

    def nii2dcm(file1, file2):
        IMG(file1).astype(IMG.dcm).save(file2)
        
class Spacing(tuple): ...

class IMG:
    
    with run("PNG"), run.use_tag: png = PNG = 'PNG'
    with run("DCM"), run.use_tag: dcm = DCM = 'DCM'
    with run(sitk_supported_modalities), run.all_tags:
        for m in sitk_supported_modalities:
            exec(f"{m.lower()} = {m.upper()} = '{m.upper()}'")
    
    basic_info = "SeriesNumber SeriesDescription SeriesTime Shape".split()

    @overload
    def __init__(self, path: str, **kwargs):
        if path.__class__ != Path: path = Path(path)
        self.unresolved_warning = False
        self.component_seperated = False
        self.mismatch_seperated = False
        done = False
        if path.is_file():
            with run(sitk_supported_modalities), run.any_tag:
                if any([path.endswith(e) for e in sitk_modality_exts.keys()]): self.__init_sitk__(path, **kwargs); done =True
            with run("DCM"), run.use_tag:
                if path.endswith('.dcm') or path.endswith('.ima'): self.__init_dcm__(path.parent, **kwargs); done =True
        elif path.is_dir():
            with run("DCM"), run.use_tag:
                if len([x for x in os.listdir(path) if x.endswith('.dcm') or x.endswith('.ima')]) > 0: self.__init_dcm__(path, **kwargs); done =True
        if not done:
            if not path.exists(): raise FileNotFoundError(f"Cannot find file/folder {path}. ")
            self.ftype = kwargs.get('astype', None)
            if self.ftype is None: raise TypeError(f"micomputing.IMG failed to recognize type of {path}. Use `astype=IMG.nii` or other image format to identify it. ")
            with run(sitk_supported_modalities), run.any_tag:
                if self.ftype in sitk_supported_modalities: self.__init_sitk__(path, **kwargs); done =True
            with run("DCM"), run.use_tag:
                if self.ftype == "DCM": self.__init_dcm__(path, **kwargs); done =True
        if len(self.basics) <= 0: self.basics = basic_tags.copy()

    @overload
    def __init__(self, x: sitk.Image, **kwargs):
        self.unresolved_warning = False
        self.has_series = False
        self.ftype = kwargs.get('ftype', None)
        self.sid = kwargs.get('sid', get_uid10())
        self.image = x
        self.basics = kwargs.get('basics', basic_tags.copy())
        self.file_list = kwargs.get('file_list', [])
        self.info = kwargs.get('info', {})
        if 'Shape' not in self.info:
            if x.GetDepth(): self.info['Shape'] = x.GetSize() + (x.GetDepth(),)
            else: self.info['Shape'] = x.GetSize()

    @overload
    def __init__(self, x: bt.Tensor, **kwargs):
        x = x.squeeze()
        if not x.dtype.is_floating_point: x = x.type(bt.int16)
        if x.n_space >= 3: x = bt.permute_space(x, 2, 1, 0, *range(3, x.n_space))
        else: x = x.transpose(0, 1)
        x = x.detach().cpu()
        self.unresolved_warning = False
        self.has_series = x.has_special
        self.ftype = kwargs.get('ftype', None)
        self.basics = kwargs.get('basics', basic_tags)
        if self.has_series:
            if x.has_batch and x.has_channel: x = x.standard().flatten(0, 1).batch_dimension_(0)
            if x.has_batch: series = x.split(1, [])
            else: series = x.split(1, {})
            self.images = {}
            self.file_lists = {}
            self.series_infos = {}
            self.series_IDs = []
            for i, s in enumerate(series):
                s = s.squeeze()
                sid = get_uid10()
                self.series_IDs.append(sid)
                self.images[sid] = sitk.GetImageFromArray(s)
                self.file_lists[sid] = []
                self.series_infos[sid] = dict(
                    SeriesNumber = str(i + 1), 
                    SeriesDescription = f"Series {i + 1}", 
                    Shape = s.shape
                )
            return
        self.sid = kwargs.get('sid', get_uid10())
        self.image = sitk.GetImageFromArray(x)
        self.file_list = kwargs.get('file_list', [])
        self.info = kwargs.get('info', {})
        if 'Shape' not in self.info:
            self.info['Shape'] = x.space

    with run(sitk_supported_modalities), run.any_tag:
        def __init_sitk__(self, path, **kwargs):
            self.has_series = False
            self.ftype = item([t for e, t in sitk_modality_exts.items() if path.endswith(e)])
            self.sid = ''
            self.image = sitk.ReadImage(path)
            self.file_list = [path]
            self.basics = kwargs.get('basics', {})
            self.info = kwargs.get('info', {})
            self.info['Shape'] = self.image.GetSize()
    
    with run("DCM"), run.use_tag:       
        def __init_dcm__(self, path, **kwargs):
            """
            Initialize a dicom data structure

            Args:
                path (str): the path to the DICOM directory. 
                is_valid_series_ID (callable): a function that accepts a series id and returns whether it is valid.
                    Use `lambda x: x == a` to select one series and `lambda x: x != a` to exclude one. 
                is_valid_series (callable): a function that accepts a dictionary with keywords 'SeriesNumber', 'SeriesDescription', 
                    and 'Shape'. One should use them to identify whether the series ought to be preserved. 

            Raises:
                TypeError: Invalid DICOM files. 
            """
            self.has_series = True
            self.ftype = 'DCM'
            self.images = {}
            self.file_lists = {}
            self.series_infos = {}
            self.basics = {}
            self.series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
            if len(self.series_IDs) <= 0: raise TypeError(f"Not dicom directory at {path}.")
            is_valid_series_ID = kwargs.get('is_valid_series_ID', lambda x: True)
            if isinstance(is_valid_series_ID, str): is_valid_series_ID = lambda x: x == is_valid_series_ID
            if isinstance(is_valid_series_ID, list): is_valid_series_ID = lambda x: x in is_valid_series_ID
            self.series_IDs = [i for i in self.series_IDs if is_valid_series_ID(i)]
            if len(self.series_IDs) > 1:
                is_valid_series = kwargs.get('is_valid_series', lambda x: True)
                if isinstance(is_valid_series, str): is_valid_series = lambda x: is_valid_series in x.values()
                if isinstance(is_valid_series, list): is_valid_series = lambda x: any([i in is_valid_series for i in x.values()])
            else: is_valid_series = lambda x: True
            ids_to_delete = []
            for sid in self.series_IDs.copy():
                file_list = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, seriesID = sid))
                with dcm.dcmread(file_list[0], stop_before_pixels=True, force=True) as dcm_header:
                    if len(self.basics) <= 0: self.basics = {t: touch(lambda: str(dcm_header[tuple(eval('0x'+u) for u in t.split('|'))].value), "") for t in basic_tags}
                    info = dict(SeriesNumber = dcm_header.SeriesNumber, SeriesDescription = dcm_header.SeriesDescription, SeriesTime = dcm_header.SeriesTime, Shape = (dcm_header.Rows, dcm_header.Columns, len(file_list)))
                    if not is_valid_series(info): ids_to_delete.append(sid); continue
                seperated = self._read_dcm(sid, file_list, info)
                if seperated: ids_to_delete.append(sid)
            for sid in ids_to_delete:
                self.series_IDs.remove(sid)
            self.series_IDs.sort(key=lambda x: tuple(map(eval, x.split('.'))))
            if len(self.series_IDs) == 1:
                self.has_series = False
                self.sid = self.series_IDs[0]
                self.image = self.images[self.sid]
                self.file_list = self.file_lists[self.sid]
                self.info = self.series_infos[self.sid]
                
        def _handle_dcm_size_exception(self, file_list, *_):
            sizes = []
            files = []
            for file_path in file_list:
                with dcm.dcmread(file_path, stop_before_pixels=True, force=True) as dcm_header:
                    size = (dcm_header.Rows, dcm_header.Columns)
                    if size in sizes: files[sizes.index(size)].append(file_path)
                    else: sizes.append(size); files.append([file_path])
            self.mismatch_seperated = True
            return [({'Shape': s + (len(f),)}, f) for s, f in zip(sizes, files)]
        
        def _handle_dcm_non_uniform_exception(self, file_list, series_name):
            all_med_params = {}
            med_param_keys = []
            med_param_names = []
            med_param_values = []
            for file_path in file_list:
                with dcm.dcmread(file_path, stop_before_pixels=True, force=True) as dcm_header:
                    if len(med_param_keys) == 0:
                        for k in list(dcm_header.keys()):
                            if '0018,' not in str(k) and str(k) != "(0008, 0008)": continue
                            med_param_keys.append(k)
                            med_param_names.append(dcm_header[k].name)
                            med_param_values.append(set())
                            
                    all_med_params.setdefault(file_path, [])
                    for i, k in enumerate(med_param_keys):
                        v = dcm_header[k].value
                        if not isinstance(v, (str, int, float, tuple)): v = str(v)
                        all_med_params[file_path].append(v)
                        med_param_values[i].add(v)
            critical_pos = argmax(list(map(lambda x: len(x) > 1, med_param_values)))
            if len(critical_pos) == len(med_param_keys):
                file_list.sort()
                print(f"Non uniform sampling in {series_name}")
                print(f"Files: {file_list[0]}--{os.path.basename(file_list[-1])}")
                print("Failed to seperate: same medical parameters ")
                return
            med_names = sublist(med_param_names, critical_pos)
            partition = {}
            for file_path in file_list:
                value_set = tuple(sublist(all_med_params[file_path], critical_pos))
                value_set = to_tuple(value_set)
                partition.setdefault(value_set, [])
                partition[value_set].append(file_path)
            value_sets = sorted(list(partition.keys()))
            return [(dict(zip(med_names, vs)), partition[vs]) for vs in value_sets]
                    
        def _read_dcm(self, sid, file_list, info):
            series_name = f"[{info['SeriesTime']}] Series {info['SeriesDescription']}<{info['SeriesNumber']}> {info['Shape']}"
            with dcm.dcmread(file_list[1 if len(file_list) > 1 else 0], stop_before_pixels=True, force=True) as dcm_header: pS = getattr(dcm_header, 'ImagePositionPatient', [0])[-1]
            with dcm.dcmread(file_list[-2 if len(file_list) > 1 else -1], stop_before_pixels=True, force=True) as dcm_header: pE = getattr(dcm_header, 'ImagePositionPatient', [0])[-1]
            if pS > pE: file_list = file_list[::-1]
            with no_print as rec:
                try:
                    self.series_infos[sid] = info
                    self.file_lists[sid] = file_list
                    series_reader = sitk.ImageSeriesReader()
                    series_reader.SetFileNames(file_list)
                    self.images[sid] = series_reader.Execute()
                    res = "DONE"
                except RuntimeError as e:
                    error = str(e)
                    if 'Size mismatch' in error or 'requested regionRequested' in error:
                        res = self._handle_dcm_size_exception(file_list, series_name)
                    else:
                        file_list.sort()
                        self.unresolved_warning = True
                        raise RuntimeError(f"Failure in {series_name} Files: {file_list[0]}--{os.path.basename(file_list[-1])}\n" + error)
            if "WARNING" in rec.string():
                if "Non uniform sampling" in str(rec):
                    res = self._handle_dcm_non_uniform_exception(file_list, series_name)
                else:
                    file_list.sort()
                    print(rec)
                    print(f"in {series_name} Files: {file_list[0]}--{os.path.basename(file_list[-1])}")
                    self.unresolved_warning = True; return False
            if res is None: self.unresolved_warning = True; return False
            if isinstance(res, str): return False
            self.series_infos.pop(sid)
            self.file_lists.pop(sid)
            if sid in self.images: self.images.pop(sid)
            for i, (extra_info, files) in enumerate(res):
                key = sid + f'.{i+1}.0.0'
                new_info = info.copy()
                new_info.update(extra_info)
                if 'Shape' in extra_info:
                    d = extra_info['Shape'][-1]
                    if d <= 3: continue
                    new_info['Depth'] = d
                sep = self._read_dcm(key, files, new_info)
                if not sep: self.series_IDs.append(key)
            return True

    @alias("__array__")
    def to_numpy(self, sid=None):
        """
        Get the array data item of IMG.
        Note: It is transposed for 3D arrays as the direct array by sitk is in size (n_z, n_x, n_y)
        """
        if self.has_series:
            if sid is None: raise TypeError("Converting micomputing.IMG with series to numpy. Please Identify Series ID. ")
            image = self.images[sid]
        else: image = self.image
        image_data = np.array(sitk.GetArrayFromImage(image))
        if image_data.ndim >= 3: return image_data.transpose(2, 1, 0, *range(3, image_data.ndim))
        else: return image_data.transpose(0, 1)

    @alias("__tensor__")
    def to_tensor(self, sid=None):
        """
        Get the tensor item of IMG.
        Note: It is transposed for 3D arrays as the direct array by sitk is in size (n_z, n_x, n_y)
        """
        return bt.tensor(toI(self.to_numpy(sid=sid)))

    def from_array(self, array):
        array = bt.tensor(array).squeeze()
        if not array.dtype.is_floating_point: array = array.type(bt.int16)
        if array.n_space >= 3: array = bt.permute_space(array, 2, 1, 0, *range(3, array.n_space))
        else: array = array.transpose(0, 1)
        array = array.detach().cpu()
        old_image = self.image
        self.image = sitk.GetImageFromArray(array)
        self.image.SetDirection(old_image.GetDirection())
        self.image.SetSpacing(old_image.GetSpacing())
        self.image.SetOrigin(old_image.GetOrigin())
        return self
        
    def __str__(self):
        str_print = SPrint()
        if self.has_series:
            str_print(f"<micomputing.IMG with {len(self.series_IDs)} series, ftype={repr(self.ftype)}, dtype={repr(self.images[self.series_IDs[0]].GetPixelIDTypeAsString())}>")
            for sid in self.series_IDs:
                info = self.series_infos[sid]
                extra = info.copy()
                for k in IMG.basic_info: extra.pop(k)
                if len(extra) > 0: str_print(f"\t[{info['SeriesTime']}] Series {sid}: {info['SeriesDescription']}<{info['SeriesNumber']}>, Size: {info['Shape']}, {str(extra)}.")
                else: str_print(f"\t[{info['SeriesTime']}] Series {sid}: {info['SeriesDescription']}<{info['SeriesNumber']}>, Size: {info['Shape']}.")
        else: str_print(f"<micomputing.IMG object, Size: {self.info['Shape']}, ftype={repr(self.ftype)}, dtype={repr(self.image.GetPixelIDTypeAsString())}>")
        return str_print.text
        
    def split(self):
        if not self.has_series: raise TypeError("micomputing.IMG without series is not iterable. ")
        collection = []
        for sid in self.series_IDs:
            collection.append(self[sid])
        return collection
    
    def __getitem__(self, sid):
        if not self.has_series: raise TypeError("micomputing.IMG without series is not iterable. ")
        if not isinstance(sid, str): sid = str(sid)
        if '.' not in sid:
            is_series_num = touch(lambda: int(sid), False)
            if is_series_num: sid = item([x for x in self.series_IDs if self.series_infos[x]['SeriesNumber'] == sid])
            else: sid = item([x for x in self.series_IDs if self.series_infos[x]['SeriesDescription'] == sid])
        return IMG(self.images[sid], sid=sid, info=self.series_infos[sid], file_list=self.file_lists[sid], ftype=self.ftype)
    
    def __iter__(self):
        if not self.has_series: raise TypeError("micomputing.IMG without series is not iterable. ")
        for sid in self.series_IDs:
            yield self[sid]

    def __enter__(self): return self

    def __exit__(self, *args): return False

    def astype(self, ftype): self.ftype = ftype; return self
    
    def copy(self):
        if self.has_series:
            tmp_image = self.images[self.series_IDs[0]]
            output = IMG(tmp_image, ftype=self.ftype, basics=self.basics, file_list={}, info={})
            output.unresolved_warning = False
            output.has_series = True
            output.images = self.images.copy()
            output.file_lists = self.file_lists.copy()
            output.series_infos = self.series_infos.copy()
            output.basics = self.basics.copy()
            output.series_IDs = self.series_IDs.copy()
            return output
        return IMG(self.image, ftype=self.ftype, sid=self.sid, basics=self.basics, file_list=self.file_list, info=self.info)
    
    @property
    def path(self):
        if len(self.file_list) == 1: return self.file_list[0]
        return self.file_list[0].dirname
    
    @property
    @alias("affine_of")
    def affine(self, image=None):
        """
        Return the Niftii affine matrix that converts image indices to world coordinates. 
        Note: sitk contains direction in DICOM format with +x means left and +y means posterior.
        Hence, Niftii format of IMG requires the first two rows of the affine matrix being inversed. 
        """
        if image is None:
            if self.has_series: raise Error('Property')("No property 'affine' for micomputing.IMG with series. ")
            image = self.image
        if self.has_series and isinstance(image, str): image = self.images[image]
        dir = np.array(image.GetDirection())
        ndim = image.GetDimension()
        avouch(ndim ** 2 == len(dir))
        A = dir.reshape((ndim, ndim)) @ np.diag(np.array(image.GetSpacing()))
        b = np.array(image.GetOrigin()).reshape((ndim, 1))
        c = np.concatenate((np.zeros((1, ndim)), np.ones((1, 1))), 1)
        return np.diag([-1, -1] + [1] * (ndim - 1)) @ np.concatenate((np.concatenate((A, b), 1), c), 0)

    @affine.setter
    @alias("set_affine")
    def affine(self, aff, spacing=None, spacing2one=False):
        if self.has_series: raise Error('Property')("Cannot set property 'affine' for micomputing.IMG with series. ")
        ndim = aff.shape[-1] - 1
        try: aff = aff.reshape((ndim + 1, ndim + 1))
        except ValueError: raise Error('Value')(f"Cannot set affine matrix to image: expected shape ({ndim + 1}, {ndim + 1}) but got {aff.shape}")
        flipped_aff = np.diag([-1, -1] + [1] * (ndim - 1)) @ aff
        A = flipped_aff[:ndim, :ndim]
        b = flipped_aff[:ndim, -1]
        self.image.SetOrigin(b.tolist())
        if spacing2one:
            avouch(spacing is None)
            self.image.SetSpacing((1,) * ndim)
        elif spacing is not None:
            if isinstance(spacing, (int, float)): spacing = (spacing,) * ndim
            avouch(len(spacing) == ndim)
            self.image.SetSpacing(spacing)
            A = A @ np.diag(1 / np.array(spacing))
        else: A = A @ np.diag(1 / np.array(self.image.GetSpacing()))
        self.image.SetDirection(A.flatten().tolist())
    
    @property
    @alias("origin_of")
    def origin(self, image=None):
        if self.has_series:
            if isinstance(image, str): image = self.images[image]
            raise Error('Property')("No property 'origin' for micomputing.IMG with series. ")
        if image is None: image = self.image
        avouch(isinstance(image, sitk.Image), "Only Image type has origin.")
        return tuple(np.array(image.GetOrigin()) * np.array([-1, -1] + [1] * (image.GetDimension() - 2)))

    @origin.setter
    @alias("set_origin")
    def origin(self, *org):
        org = arg_tuple(org)
        if len(org) == 1: org = org * self.ndim
        if self.has_series: raise Error('Property')("Cannot set property 'origin' for micomputing.IMG with series. ")
        self.image.SetOrigin(tuple(s * x for s, x in zip([-1, -1, 1], org)))
    
    @property
    @alias("spacing_of")
    def spacing(self, image=None):
        if self.has_series:
            if isinstance(image, str): image = self.images[image]
            raise Error('Property')("No property 'spacing' for micomputing.IMG with series. ")
        if image is None: image = self.image
        avouch(isinstance(image, sitk.Image), "Only Image type has spacing.")
        return tuple(np.array(self.image.GetSpacing()))

    @spacing.setter
    @alias("set_spacing")
    def spacing(self, *sp):
        sp = arg_tuple(sp)
        if len(sp) == 1: sp = sp * self.ndim
        if self.has_series: raise Error('Property')("Cannot set property 'spacing' for micomputing.IMG with series. ")
        self.image.SetSpacing(sp)
    
    @property
    @alias("ndim_of")
    def ndim(self, image=None):
        if self.has_series:
            if isinstance(image, str): image = self.images[image]
            raise Error('Property')("No property 'ndim' for micomputing.IMG with series. ")
        if image is None: image = self.image
        avouch(isinstance(image, sitk.Image), "Only Image type has ndim.")
        return self.image.GetDimension()
    
    @property
    def shape(self):
        if self.has_series: raise Error('Property')("No property 'shape' for micomputing.IMG with series. ")
        return tuple(self.image.GetSize())
    
    @property
    def orientation(self):
        if self.has_series: raise Error('Property')("No property 'orientation' for micomputing.IMG with series. ")
        affine = self.affine
        dim_ord = np.abs(affine[:-1, :-1]).argmax(0).tolist()
        try: dx = np.array([affine[dim_ord[d], d] for d in range(len(dim_ord))])
        except ValueError: raise TypeError(f"Invalid affine matrix: {affine}")
        return ''.join([['RL', 'AP', 'SI'][i][j] for i, j in zip(dim_ord, dx > 0)])

    def reorient(self, orient='LPI'):
        if self.has_series: raise Error('Property')("Cannot reorient for micomputing.IMG with series. ")
        axis = {'L':'LR', 'R':'LR', 'A':'AP', 'P':'AP', 'I':'SI', 'S':'SI'}
        from .funcs import reorient
        from .trans import Translation, Reflection, DimPermutation, ComposedTransformation
        orient_axis = [axis[i] for i in self.orientation]
        permutation = [orient_axis.index(axis[i]) for i in orient]
        if len(permutation) < len(orient_axis):
            permutation.extend(list(set(range(len(orient_axis))) - set(permutation)))
        new_orient = [self.orientation[i] for i in permutation]
        reflect_dims = [i for i, (a, b) in enumerate(zip(new_orient, orient)) if a != b]
        data = self.to_tensor()
        output_data = reorient(data, from_orient=self.orientation, to_orient=orient)
        if output_data.n_space >= 3: output_data = bt.permute_space(output_data, 2, 1, 0, *range(3, output_data.n_space))
        else: output_data = output_data.transpose(0, 1)
        output_image = sitk.GetImageFromArray(output_data)
        for t in basic_tags: output_image.SetMetaData(t, self.basics[t])
        n_dim = len(permutation)
        spacing = tuple(self.image.GetSpacing()[i] for i in permutation)
        # correct_trans = Translation([(s - 1) * n for s, n in zip(spacing, data.shape)]).backward_(False)
        reflect = Reflection(reflect_dims, image_size=data).backward_(False)
        permute = DimPermutation(permutation).backward_(False)
        trans = ComposedTransformation(permute, reflect)
        affine = (np.diag([-1, -1] + [1] * (n_dim - 1)) @ self.affine @ trans.affine(n_dim).detach().cpu().numpy()).squeeze()
        output_image.SetDirection((affine[:n_dim, :n_dim] @ np.diag([1/s for s in spacing])).flatten().tolist())
        output_image.SetOrigin(affine[:n_dim, -1].tolist())
        output_image.SetSpacing(spacing)
        output = self.copy()
        output.image = output_image
        return output
    
    @property
    def patient_info(self):
        return dict(
            name = self.basics["0010|0010"],
            sex = self.basics["0010|0040"],
            age = self.basics["0010|1010"],
            height = self.basics["0010|1020"],
            weight = self.basics["0010|1030"],
            birthday = self.basics["0010|0030"]
        )
    
    @property
    def patient_name(self): return self.basics["0010|0010"]
    
    @property
    def patient_sex(self): return self.basics["0010|0040"]
    
    @property
    def patient_age(self): return self.basics["0010|1010"]
    
    @property
    def modality(self): return self.basics["0008|0060"]
    
    def resample(self, *new_spacing, new_size=None, mode='nearest'):
        if len(new_spacing) == 1 and isinstance(new_spacing[0], (tuple, list)): new_spacing = new_spacing[0]
        new_spacing = to_list(new_spacing)
        if len(new_spacing) == 1: new_spacing *= self.ndim
        if new_size is not None: new_size = to_list(new_size)
        def resample(image, new_size=new_size, mode=mode):
            resampler = sitk.ResampleImageFilter()
            if new_size is None: 
                old_spacing = image.GetSpacing()
                old_size = image.GetSize()
                new_size = [int(a * b / c) for a, b, c in zip(old_size, old_spacing, new_spacing)]
            if mode.lower() == 'linear': method = sitk.sitkLinear
            elif mode.lower() == 'nearest': method = sitk.sitkNearestNeighbor
            elif mode.lower() == 'bspline': method = sitk.sitkBSpline
            else: raise TypeError("Unrecognized argument 'mode'. ")
            resampler.SetReferenceImage(image)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(method)
            return resampler.Execute(image), new_size
        if self.has_series:
            for sid in self.series_IDs:
                self.images[sid], self.series_infos[sid]['Shape'] = resample(self.images[sid])
        else: self.image, self.info['Shape'] = resample(self.image)
        return self
    
    def _compose(self, *images):
        # Not sure if this is correct. [TODO: Check]
        if len(images) == 1 and isinstance(images[0], (list, tuple)): images = images[0]
        if len(images) <= 5:
            composer = sitk.JoinSeriesImageFilter()
            try: return composer.Execute(*images)
            except: pass
        images_origin = [x.GetOrigin() for x in images]
        images_data = map(sitk.GetArrayFromImage, images)
        template = images[0]
        spacing = template.GetSpacing()
        affine = np.array(template.GetDirection())
        m = np.sqrt(len(affine))
        affine = affine.reshape((m, m))
        if len(set(images_origin)) == 1:
            output_data = np.stack(images_data)
            output_data = output_data.transpose(*range(1, output_data.ndim), 0)
        else:
            origins = np.array(images_origin)
            coords = np.linalg.inv(affine) @ origins.T
            vec_dim = item(argmax(np.abs(coords[:, 0] - coords[:, -1]).tolist()))
            spacing_z = np.sqrt(np.square(origins[1:] - origins[:-1]).sum(1)).mean()
            output_data = np.concatenate(list(images_data))
            output_data = output_data.transpose(*range(2, 4 - vec_dim), 0, *range(4 - vec_dim, output_data.ndim), 1)
            spacing = spacing[:output_data.ndim-2] + (spacing_z,) + spacing[output_data.ndim-1:]
        output_image = sitk.GetImageFromArray(output_data)
        for t in basic_tags: output_image.SetMetaData(t, self.basics[t])
        output_image.SetDirection(affine.flatten().tolist())
        output_image.SetOrigin(images_origin[0])
        output_image.SetSpacing(spacing)
        return output_image
    
    def gather_multiple_components(self):
        categories = {}
        for sid in self.series_IDs:
            segs = sid.split('.')
            if len(segs) <= 10: continue
            pid = '.'.join(segs[:10])
            categories.setdefault(pid, [])
            categories[pid].append(sid)
        for pid in categories:
            if len(categories[pid]) <= 1: continue
            images = []
            for sid in sorted(categories[pid], key=lambda x: tuple(map(eval, x.split('.')))):
                images.append(self.images.pop(sid))
                if pid not in self.series_IDs:
                    self.series_infos[pid] = self.series_infos[sid]
                    self.series_IDs.append(pid)
                self.series_IDs.remove(sid)
            self.images[pid] = self._compose(images)
            self.series_infos[pid]['Shape'] = self.images[pid].GetSize()
        self.series_IDs.sort(key=lambda x: tuple(map(eval, x.split('.'))))
    
    def bundle(self):
        if self.has_series: ref_file = self.file_lists[0][0]
        ref_file = self.file_list[0]
        
        with run("NII"), run.use_tag:
            if self.ftype.upper() == 'NII': return nib.load(ref_file).header
        
        with run("DCM"), run.use_tag:
            if self.ftype.upper() == 'DCM': return dcm.dcmread(ref_file)

    def save_as_dtype(self, dtype: bt.dtype, path: (str, callable), as_type=None):
        if isinstance(dtype, str): dtype = getattr(bt, dtype)
        if self.has_series:
            img = self.copy()
            for sid in img.series_IDs:
                image_data = np.array(sitk.GetArrayFromImage(img.images[sid]))
                array = bt.tensor(toI(image_data)).type(dtype)
                old_image = img.images[sid]
                new_image = sitk.GetImageFromArray(array)
                new_image.SetDirection(old_image.GetDirection())
                new_image.SetSpacing(old_image.GetSpacing())
                new_image.SetOrigin(old_image.GetOrigin())
                img.images[sid] = new_image
            img.save(path, as_type=as_type)
        else:
            img = self.copy()
            image_data = np.array(sitk.GetArrayFromImage(img.image))
            array = bt.tensor(toI(image_data)).type(dtype)
            old_image = img.image
            new_image = sitk.GetImageFromArray(array)
            new_image.SetDirection(old_image.GetDirection())
            new_image.SetSpacing(old_image.GetSpacing())
            new_image.SetOrigin(old_image.GetOrigin())
            img.image = new_image
            img.save(path, as_type=as_type)

    @overload
    def save(self, data: bt.Tensor, path: (str, callable), as_type=None, **header):
        img = self.copy()
        img.from_array(data)
        for k, v in header.items():
            if not hasattr(img, k): continue
            setattr(img, k, v)
        img.save(path, as_type=as_type)

    @overload
    def save(self, img: 'IMG', path: (str, callable), as_type=None, **header):
        old_values = {}
        for k, v in header.items():
            if not hasattr(img, k): continue
            old_values[k] = getattr(img, k)
            setattr(img, k, v)
        img.save(path, as_type=as_type)
        for k, v in old_values.items(): setattr(img, k, v)

    @overload
    def save(self, trans: 'Transformation', path: (str, callable), as_type=None):
        img = self.copy()
        img.from_array(trans.toDDF(self.shape).squeeze())
        for k, v in header.items():
            if not hasattr(img, k): continue
            setattr(img, k, v)
        img.save(path, as_type=as_type)

    @overload
    def save(self, trans_file: 'TRS', path: (str, callable), as_type=None):
        img = self.copy()
        img.from_array(trans_file.trans.toDDF(self.shape).squeeze())
        for k, v in header.items():
            if not hasattr(img, k): continue
            setattr(img, k, v)
        img.save(path, as_type=as_type)

    @overload
    def save(self, path: (str, callable), as_type=None):
        if as_type is None:
            ftype_out = self.ftype
            if isinstance(path, str):
                with run(sitk_supported_modalities), run.any_tag:
                    ftype_out_list = [t for e, t in sitk_modality_exts.items() if path.endswith(e)]
                    if len(ftype_out_list) == 1: ftype_out = ftype_out_list[0]
                with run("DCM"), run.use_tag:
                    if isinstance(path, str) and (path.endswith('.dcm') or path.endswith('.ima')): ftype_out = 'DCM'
        else: ftype_out = as_type
        if ftype_out is None: raise Error("Unidentified")("Cannot save micomputing.IMG without identifing the `ftype`. ")

        if self.has_series: ids = self.series_IDs; images = self.images; infos = self.series_infos
        else: ids = [get_uid10()]; images = {ids[0]: self.image}; infos = {ids[0]: self.info}

        def get_path(sid):
            if callable(path): p = path(sid, infos)
            elif isinstance(path, str): p = path
            else: raise TypeError(f"Invalid saving path for micomputing.IMG: {path}.")
            return p
        isdir = lambda p: os.extsep not in os.path.basename(p) or len(os.path.basename(p).split(os.extsep)[-1]) > 6

        with run(sitk_supported_modalities), run.any_tag:
            if ftype_out.upper() in sitk_supported_modalities:
                ext = sitk_default_exts[ftype_out.upper()]
                used_path = []
                for sid in ids:
                    info = infos[sid]
                    p = get_path(sid)
                    if p is None: continue
                    if isdir(p):
                        if not os.path.exists(p): os.mkdir(p)
                        if self.has_series:
                            extra = info.copy()
                            for k in IMG.basic_info: extra.pop(k)
                            if len(extra) > 0: extra_str = ''.join([f"_{get_alphas(k)}{get_digits(str(v)) if get_digits(str(v)) else get_alphas(str(v))}" for k, v in extra.items()])
                            else: extra_str = ''
                            default_name = f"{info['SeriesTime']}_{info['SeriesDescription']}_{info['SeriesNumber']}{extra_str}{ext}"
                            p = os.path.join(p, default_name)
                        else: p = os.path.join(p, ext)
                    if p in used_path: print(f"Warning: overwriting file {p} as a same path was given for multiple series. ")
                    sitk.WriteImage(images[sid], p)
                    used_path.append(p)
        with run("DCM"), run.use_tag:
            if ftype_out.upper() == 'DCM':
                used_path = []
                count = {}
                for sid in ids:
                    info = infos[sid]
                    p = get_path(sid)
                    if p is None: continue
                    if isdir(p):
                        if not os.path.exists(p): os.makedirs(p)
                        if p not in used_path: count[p] = 1
                        image = images[sid]
                        slices = []
                        ndim = image.GetDimension()
                        if ndim == 3:
                            for z in range(image.GetDepth()):
                                slices.append(image[:, :, z])
                        elif ndim == 2:
                            slices.append(image)
                        series_writer = sitk.ImageFileWriter()
                        series_writer.KeepOriginalImageUIDOn()
                        aff = self.affine_of(image)
                        orientation = np.array(image.GetDirection()).reshape((ndim, ndim)).T.flatten()[:ndim * (ndim - 1)].tolist()
                        origin = list(image.GetOrigin())[:2]
                        spacing = list(image.GetSpacing())
                        for z, image_slice in enumerate(slices):
                            slice_loc = (aff @ np.array([[0], [0], [z], [1]]))[2, 0]
                            for t, v in basic_tags.items(): image_slice.SetMetaData(t, self.basics.get(t, v))
                            image_slice.SetMetaData("0020|0013", str(z+1))
                            image_slice.SetMetaData("0020|1041", str(-slice_loc))
                            image_slice.SetMetaData("0020|000e", sid)
                            image_slice.SetMetaData("0008|0021", cdate())
                            image_slice.SetMetaData("0008|0031", cmilitime())
                            image_slice.SetMetaData("0008|0012", cdate())
                            image_slice.SetMetaData("0008|0013", cmilitime())
                            image_slice.SetMetaData("0020|0011", str(info.get('SeriesNumber', '1')))
                            image_slice.SetMetaData("0008|103e", info.get('SeriesDescription', ''))
                            file_path = os.path.join(p, "slice%04d.dcm"%count[p])
                            series_writer.SetFileName(file_path)
                            series_writer.Execute(image_slice)
                            with dcm.dcmread(file_path) as image_again:
                                image_again.ImagePositionPatient = origin + [slice_loc] # 0020|0032
                                image_again.ImageOrientationPatient = orientation # 0020|0037
                                image_again.PixelSpacing = spacing[:2] # 0020|0030
                                image_again.SpacingBetweenSlices = str(spacing[-1]) # 0018|0088
                                image_again.SliceThickness = str(float(int(spacing[-1] - 0.1))) # 0018|0050
                                image_again.ImageType = ['DERIVED'] # 0008|0008
                                image_again.PatientSize = '200' # 0010|1020
                                image_again.save_as(file_path)
                            count[p] += 1
                        used_path.append(p)
                    else: raise TypeError("Please identify a directory when saving a DICOM format micomputing.IMG. ")
        with run("PNG"), run.use_tag:
            if ftype_out.upper() == 'PNG':
                used_path = []
                count = {}
                for sid in ids:
                    info = infos[sid]
                    p = get_path(sid)
                    if p is None: continue
                    if isdir(p):
                        if not os.path.exists(p): os.mkdir(p)
                        if p not in used_path: count[p] = 0
                        image = images[sid]
                        image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
                        file_names = [os.path.join(p, "slice%04d.png"%(count[p] + z)) for z in range(image.GetDepth())]
                        count[p] += image.GetDepth()
                        series_writer = sitk.ImageSeriesWriter()
                        series_writer.SetFileNames(file_names)
                        series_writer.Execute(image)
                        used_path.append(p)
                    else: raise TypeError("Please identify a directory when saving a DICOM format micomputing.IMG. ")
                    
