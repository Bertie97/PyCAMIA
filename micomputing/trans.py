
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File to transform image space.",
    help = "Use `from micomputing import *`.",
    requires = "batorch"
).check()

__all__ = """
    Transformation
    SpatialTransformation
    ImageTransformation
    ComposedTransformation CompoundTransformation
    Identity Id
    Rotation90
    Rotation180
    Rotation270
    Reflect Reflection
    Permutedim DimPermutation
    Rescale Rescaling
    Translate Translation
    Rigid Rig
    Affine Aff
    PolyAffine logEu
    LocallyAffine LARM
    FreeFormDeformation FFD
    DenseDisplacementField DDF
    MultiLayerPerception MLP
    
    Normalize
    
    resample
    interpolation
    interpolation_forward
    
    Affine2D2Matrix
    Quaterns2Matrix
    Matrix2Quaterns
""".split()

import json, math
from typing import Iterable
from .stdio import IMG

with __info__:
    import numpy as np
    import batorch as bt
    from pycamia import to_tuple, to_list, arg_tuple, avouch, prod, Path
    from pycamia import SPrint, Error, alias, get_environ_vars, Version
    from pyoverload import callable
    
eps = 1e-6
    
def batorch_tensor(x):
    if not isinstance(x, bt.torch.Tensor): return bt.tensor(x)
    elif type(x) != bt.Tensor: return x.as_subclass(bt.Tensor)
    return x
    
def is_spatial_transformation(x):
    return isinstance(x, SpatialTransformation) or isinstance(x, ComposedTransformation) and x.mode == "spatial"

class Transformation:
    
    def __init__(self, *params, **kwparams):
        self.batch_param = []
        self.backward = False
        self.params = params
        self.kwparams = kwparams
        self.n_dim = None
        self.n_batch = None
        self.reshape = [(1,)] # [(s1, s2, s3), (d1, d2), (d3, d4)]: transpose d1 with d2; d3 with d4; and then scale with s1, s2, s3
        if len(params) > 0 and isinstance(params[0], bt.Tensor):
            self.main_param = params[0]
            if self.main_param.has_batch:
                self.n_batch = self.main_param.n_batch
        self.require_grad_params = []
        for p in self.params:
            if isinstance(p, bt.Tensor) and (p.is_floating_point() or p.is_complex()):
                if not p.requires_grad:
                    self.require_grad_params.append(p.clone().requires_grad_(True))
                else: self.require_grad_params.append(p)
    
    def __call__(self, X): return X

    @alias("__repr__")
    def __str__(self):
        d = "backward" if self.backward else "forward"
        if getattr(self, 'main_param', None) is None: return f"<[{d}] {self.__class__.__name__.split('.')[-1]} transformation>"
        return f"<[{d}] {self.__class__.__name__.split('.')[-1]} transformation with param size: " + \
            f"({str(self.main_param.shape).split('(')[-1]}>"

    def __getitem__(self, i):
        if self.n_batch is None: raise TypeError(f"Cannot subscript a {self.__class__} transformation without batch dimension. ")
        from copy import copy
        clone = copy(self)
        if isinstance(i, int): i = slice(i, i + 1); n_count = 1
        elif isinstance(i, slice): n_count = (i.stop - i.start) // i.step
        else: i = batorch_tensor(i).long(); n_count = i.n_ele
        clone.params = tuple(p[i] for p in self.params)
        clone.n_batch = n_count
        if len(clone.params) > 0 and isinstance(clone.params[0], bt.Tensor): clone.main_param = clone.params[0]
        clone.require_grad_params = [p[i] for p in self.require_grad_params]
        for p in self.batch_param: setattr(clone, p, getattr(self, p)[i])
        return clone

    def __matmul__(x, y):
        return ComposedTransformation(x, y)
    
    def parameters(self):
        for p in self.require_grad_params: yield p
        
    def detach(self):
        self.params = tuple(x.detach() if isinstance(x, bt.torch.Tensor) else x for x in self.params)
        return self.__class__(*self.params, **self.kwparams)
    
    def to_dict(self):
        return dict(
            type = self.__class__.__name__,
            n_dim = self.n_dim,
            n_batch = self.n_batch,
            reshape = self.reshape,
            params = self.params,
            kwparams = self.kwparams
        )

    def from_dict(self, d):
        self.params = d['params']
        self.kwparams = d['kwparams']
        self.n_dim = d['n_dim']
        self.n_batch = d['n_batch']
        self.reshape = d['reshape']

    @staticmethod
    def obj_json(obj):
        if obj is None: return 'None'
        elif isinstance(obj, dict):
            key_remap = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    key_remap[k] = f'[{k.__class__.__name__}]:{k}'
                obj[k] = Transformation.obj_json(v)
            for k, new_k in key_remap.items(): obj[new_k] = obj.pop(k)
            return {'type': 'dict', 'value': obj}
        elif isinstance(obj, (list, tuple)):
            return {'type': obj.__class__.__name__, 'value': list(Transformation.obj_json(o) for o in obj)}
        elif isinstance(obj, bt.torch.Tensor):
            return {'type': 'bt.tensor', 'dtype': str(obj.dtype).replace('torch', 'bt'), 'value': obj.tolist()}
        elif isinstance(obj, np.ndarray):
            return {'type': 'np.array', 'dtype': 'np.' + str(obj.dtype), 'value': obj.tolist()}
        elif isinstance(obj, str): pass
        elif isinstance(obj, Iterable) and not isinstance(obj, str):
            return {'type': obj.__class__.__name__, 'value': list(obj)}
        return {'type': obj.__class__.__name__, 'value': obj}

    @staticmethod
    def json_obj(__obj, **kwargs):
        if isinstance(__obj, str):
            if __obj == 'None': return None
            __obj = json.loads(__obj)
        avouch('type' in __obj and 'value' in __obj)
        obj_type = eval(__obj['type'])
        if obj_type == dict:
            __dic = __obj['value']
            __key_remap = {}
            for __k, __v in __dic.items():
                if __k.startswith('[') and ':' in __k:
                    __cls, __value = __k.split(':')
                    __cls = __cls.strip().strip('[]')
                    __value = __value.strip()
                    __key_remap[__k] = eval(__cls)(__value)
                __dic[__k] = Transformation.json_obj(__v, **kwargs)
            for __k, __new_k in __key_remap.items(): __dic[__new_k] = __dic.pop(__k)
            return __dic
        elif obj_type in (list, tuple):
            return obj_type([Transformation.json_obj(x, **kwargs) for x in __obj['value']])
        elif 'dtype' in __obj:
            return obj_type(__obj['value'], dtype=eval(__obj['dtype']))
        return obj_type(__obj['value'])

    def save(self, p:str):
        p = Path(p)
        if not p.is_filepath(): p = p // 'trs'
        with p.open('w') as fp: json.dump(Transformation.obj_json(self.to_dict()), fp, ensure_ascii=False)
            
    @staticmethod
    def dict_trans(dic):
        if dic['type'] == 'ComposedTransformation':
            trans_list = [Transformation.dict_trans(d) for d in dic['trans_list']]
            trans = ComposedTransformation(*trans_list)
        else: trans = eval(dic['type'])(*dic['params'], **dic['kwparams'])
        for key in ('n_dim', 'n_batch', 'reshape'):
            if getattr(trans, key) is None: setattr(trans, key, dic[key])
            elif dic[key] is not None: avouch(getattr(trans, key) == dic[key])
        return trans

    @staticmethod
    def load(p:str):
        p = Path(p)
        vars = get_environ_vars()
        if not p.is_filepath(): raise TypeError(f"mc.Transformation accept *.txt/trs/AFF/FFD/FLD files only, not '{p}'.")
        if p | 'trs':
            with p.open() as fp: dic = Transformation.json_obj(fp.read(), **vars.all)
            return Transformation.dict_trans(dic)
        elif p | 'txt':
            with p.open() as fp:
                txt = fp.read()
            avouch(txt.startswith("#Insight Transform File V1.0"))
            all_trans = []
            for i, t in enumerate(txt.split("#Transform ")[1:]):
                index, trans_type, params, fixed_params = t.strip().split('\n')
                avouch(int(index.strip()) == i, f"Incorrect format of an ITK transform file type: Transform {i} not available. ")
                _, dtype, n_dim, _ = trans_type.split('_')
                n_dim = int(n_dim)
                raw_data = bt.cat(bt.tensor([float(x.strip()) for x in params.split(':')[-1].split()]).view([1], n_dim + 1, n_dim).T, bt.cat(bt.zeros(n_dim), bt.ones(1)).view([1], 1, -1), -2)
                if hasattr(raw_data, dtype): raw_data = getattr(raw_data, dtype)()
                all_trans.append(Affine(raw_data))
            if len(all_trans) == 1: return all_trans[0]
            return ComposedTransformation(*all_trans)
        else:
            from zxhtools.TRS import TRS
            return TRS.load(p).trans
    
    def fake_inv(self):
        self.backward = not self.backward
        return self
    
    def forward_(self, is_forward = True):
        self.backward = not is_forward
        return self
    
    def backward_(self, is_backward = True):
        self.backward = is_backward
        return self
    
    def as_spatial_trans(self):
        return ComposedTransformation(self, mode='spatial')
    
    def as_image_trans(self):
        return ComposedTransformation(self, mode='image')

@alias("CompoundTransformation")
class ComposedTransformation(Transformation):
    
    @staticmethod
    def __flat__(trans_list):
        flat_trans_list = []
        for t in trans_list:
            if not isinstance(t, ComposedTransformation): flat_trans_list.append(t)
            else: flat_trans_list.extend(ComposedTransformation.__flat__(t.trans_list))
        return flat_trans_list

    def __init__(self, *trans_list, mode = None):
        """
        Create a composed transformation. 

        Note: The composed transformation consist of a transformation list. 
            Remember that the spatial transformations are backward transformations by default, 
            which means they map y to x when x from I1 and y from I2 in transformation from source image I1 to target I2. 

        Params:
            trans_list [list or tuple]: The list of transformations, operating one by one onto an image. 
                A series of backward transformations would be equivalent to a composition of functions from right to left.
            mode [str: image|spatial]: Telling which kind of transformation the composed transformation is regarded as. 
                It is commonly auto-selected but one is required to specify it if warnings occur. 
        
        Param for call:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n@1, n@2, ..., n@n_dim)
                OR Images to be transformed. 
                size: ([n_batch:optional], {n_channel:optional}, n@1, n@2, ..., n@n_dim)
        """
        super().__init__()
        n_dims = set([t.n_dim for t in trans_list if t.n_dim is not None])
        n_batches = set([t.n_batch for t in trans_list if t.n_batch is not None])
        if 1 in n_batches: n_batches.remove(1)
        avouch(len(n_dims) <= 1, "Composition Failed: All components should have a same dimension. ")
        avouch(len(n_batches) <= 1, "Composition Failed: All components should have a same batch size. ")
        self.trans_list = ComposedTransformation.__flat__(trans_list)

        r = [(1,)]
        for t in self.trans_list:
            s, *pairs = t.reshape
            if len(s) == 1: s = tuple(x * y for x, y in zip(r[0], s * len(r[0])))
            else:
                s = list(s)
                for p, q in r[-1:0:-1]: s[p], s[q] = s[q], s[p]
                r[0] += (1,) * (len(s) - len(r[0]))
                s = tuple(x * y for x, y in zip(r[0], s))

            pairs = r[1:] + pairs
            r = [s]
            forward = {}
            backward = {}
            for p, q in pairs:
                u, v = backward.get(p, p), backward.get(q, q)
                forward[u] = q; forward[v] = p
                backward[q] = u; backward[p] = v
            visited = []
            for p in forward.keys():
                if p in visited: continue
                while True:
                    visited.append(p)
                    q = forward[p]
                    if q in visited: break
                    r.append((p, q))
                    p = q
            
        self.reshape = r

        self.n_dim = None if len(n_dims) == 0 else n_dims.pop()
        self.n_batch = None if len(n_dims) == 0 else n_batches.pop()
        self.is_spatial = all([isinstance(t, SpatialTransformation) for t in self.trans_list])
        if self.is_spatial: self.backward = True
        if mode == 'spatial': avouch(self.is_spatial)
        self.mode = mode
    
    def __len__(self): return len(self.trans_list)
    
    def __getitem__(self, i): return self.trans_list[i]
    
    def __str__(self):
        return f"<Composed transformation of [{', '.join([str(t).strip('<>') if isinstance(t, ComposedTransformation) else t.__class__.__name__.split('.')[-1] for t in self.trans_list])}]>"

    def inv(self):
        if not all([hasattr(t, 'inv') for t in self.trans_list]):
            if any([isinstance(t, ImageTransformation) and not hasattr(t, 'inv') for t in self.trans_list]):
                raise TypeError("Composed transformation not invertable: Not all image transformation components invertable. ")
            print("Warning: Composed transformation not invertable: Not all components invertable. Using forward transformation.")
        return ComposedTransformation(*[getattr(x, 'inv', x.fake_inv)() for x in self.trans_list[::-1]], mode = self.mode)
    
    def force_inv(self, *size):
        if all([hasattr(t, 'inv') for t in self.trans_list]): return self.inv()
        if any([isinstance(t, ImageTransformation) and not hasattr(t, 'inv') for t in self.trans_list]):
            raise TypeError("Composed transformation not invertable: Not all image transformation components invertable. ")
        return ComposedTransformation(*[getattr(x, 'inv', lambda: x.force_inv(*size))() for x in self.trans_list[::-1]], mode = self.mode)

    def __call__(self, X, mode = None, **kwargs):
        """
        Perform composed transformation. 
        
        Params:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n@1, n@2, ..., n@n_dim)
                OR Images to be transformed. 
                size: ([n_batch:optional], {n_channel:optional}, n@1, n@2, ..., n@n_dim)
            mode [str: image|spatial]: Telling which kind of transformation is needed. 
                It is commonly auto-selected but one is required to specify it if warnings occur. 
        """
        if mode is not None: mode = mode.lower()
        if mode not in ('image', 'spatial'): mode = None
        if mode is None:
            if self.mode is not None: mode = self.mode
            elif not X.has_channel: mode = 'image'
            elif self.n_dim is not None:
                if X.n_channel == self.n_dim and X.n_space == self.n_dim: pass
                elif X.n_channel != self.n_dim: mode = 'image'
                else: mode = 'spatial'
            if mode is None:
                if any([not isinstance(t, SpatialTransformation) for t in self.trans_list]): mode = 'image'
                else: mode = 'spatial'
                # Deeprecated Warning.
                # print(f"Warning: Ambiguous composed transformation for Tensor({str(X.shape).split('(')[-1]}. " + 
                #         f"Performing spatial transformation for coordinates with {self}. Use `mode='image'` for image transformation. ")
        if mode == 'image':
            X = ImageTransformation.__call__(self, X)
            cur_spatial_trans = []
            for t in self.trans_list + [lambda x: x]:
                if isinstance(t, SpatialTransformation) and t.backward:
                    cur_spatial_trans.append(t); continue
                if cur_spatial_trans:
                    trans = ComposedTransformation(*cur_spatial_trans[::-1], mode = 'spatial')
                    X = interpolation(X, trans, **kwargs)
                    cur_spatial_trans.clear()
                if isinstance(t, SpatialTransformation):
                    if hasattr(trans, 'inv'): X = interpolation(X, trans.inv().fake_inv(), **kwargs)
                    else: X = interpolation_forward(X, trans, **kwargs)
                else: X = t(X)
            return X
        else:
            avouch(self.is_spatial, "Cannot perform transformation of mode `spatial` by composed transformation with non-spatial transformation. ")
            X = SpatialTransformation.__call__(self, X)
            for t in self.trans_list[::-1]:
                if t.backward: X = t(X)
                else: X = t.force_inv()(X)
            return X
        
    def to(self, space = 'world', source_affine = bt.eye(4), target_affine = bt.eye(4)):
        return SpatialTransformation.to(self, space, source_affine, target_affine)

    def to_image_space(self, source, target):
        return SpatialTransformation.to_image_space(self, source, target)

    @alias('toDDF')
    def to_DDF(self, *args, **kwargs):
        avouch(self.is_spatial, "Cannot get `DDF` from composed transformation with non-spatial transformation. ")
        return SpatialTransformation.to_DDF(self, *args, **kwargs)
    
    def save_as_nii(self, nii_path, target=None):
        avouch(self.is_spatial, "Cannot get `DDF` and save it as Nifti file from composed transformation with non-spatial transformation. ")
        avouch(target is not None, "Method 'save_as_nii' needs a Nifti image 'target' as a keyword argument to identify the physical space. ")
        if isinstance(target, str): target = IMG(target)
        target.save(self.to_DDF(target.shape).movedim({}, -1), nii_path)

    def affine(self, n_dim=None):
        avouch(self.is_spatial, "Cannot get `affine` from composed transformation with non-spatial transformation. ")
        comp = self.compose()
        avouch(len(comp) == 1, "Error in calling `affine`: Only the composition of linear transformations can result in an affine matrix.")
        trans = comp.trans_list[0]
        return trans.affine(n_dim)

    def compose(self):
        out_list = []
        n_dim = self.n_dim
        cur_affine = None
        for t in self.trans_list:
            if not isinstance(t, SpatialTransformation): out_list.append(t); continue
            aff = t.affine(n_dim)
            if aff is None and cur_affine is not None:
                out_list.extend([Affine(cur_affine), t])
                cur_affine = None
            elif aff is None: out_list.append(t)
            elif aff.space != (n_dim+1, n_dim+1):
                raise TypeError(f"Unconsistent transformation with affine of size {aff.space} in ComposedTransformation. ")
            elif cur_affine is None: cur_affine = aff
            else: cur_affine = cur_affine @ aff
        if cur_affine is not None: out_list.append(Affine(cur_affine))
        return ComposedTransformation(*out_list, mode = self.mode)

    def to_dict(self):
        dic = super().to_dict()
        dic['trans_list'] = [t.to_dict() for t in self.trans_list]
        return dic

########### Spatial Transformations ###########

class SpatialTransformation(Transformation):
    
    def __init__(self, *params, **kwparams):
        super().__init__(*params, **kwparams)
        self.backward = True
        
    def __call__(self, X):
        """
        X [bt.Tensor]: Coordinates to be transformed.
            size: ([n_batch: optional], {n_dim}, n_1, n_2, ..., n_r)
        output [bt.Tensor]: The transformed coordinates.
            size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        """
        avouch(X.has_channel, f"Please use batorch tensor of size \
            ([n_batch:optional], {{n_dim}}, n_1, n_2, ..., n_r) \
                for {self.__class__.__name__.split('.')[-1]} Transformation, instead of {X.shape}. ")
        if self.n_dim is not None:
            avouch(X.n_channel == self.n_dim, f"{self.n_dim}D {self.__class__.__name__.split('.')[-1]} Transformation \
                does not take coordinates of size {X.shape}")
        if X.has_batch: avouch(self.n_batch is None or self.n_batch == 1 or X.n_batch == 1 or X.n_batch == self.n_batch, 
            f"{self.n_dim}D {self.__class__.__name__.split('.')[-1]} Transformation with batch size {self.n_batch} \
                does not take coordinates with wrong batch size. Current size: {X.shape}.")
        Y = X.float().clone()
        if not Y.has_batch:
            if self.n_batch is not None: Y = Y.multiply(self.n_batch, [])
            else: Y = Y.unsqueeze([])
        return Y

    def __matmul__(x, y):
        if is_spatial_transformation(x) and is_spatial_transformation(y):
            return ComposedTransformation(x, y, mode="spatial")
        return ComposedTransformation(x, y)
    
    def affine(self, n_dim=None):
        return None
    
    @alias('toDDF')
    def to_DDF(self, *shape):
        shape = arg_tuple(shape)
        grid = bt.image_grid(*shape).unsqueeze([]).float()
        return self(grid) - grid
    
    def save_as_nii(self, nii_path, target=None):
        avouch(target is not None, "Method 'save_as_nii' needs a Nifti image 'target' as a keyword argument to identify the physical space. ")
        if isinstance(target, str): target = IMG(target)
        target.save(self.to_DDF(target.shape).movedim({}, -1), nii_path)

    def to_image_space(self, source, target):
        if isinstance(source, str): source = IMG(source)
        if isinstance(target, str): target = IMG(target)
        if isinstance(source, IMG): source = source.affine
        if isinstance(target, IMG): target = target.affine
        return self.to('image', source_affine=source, target_affine=target)

    def to_world_space(self, source, target):
        if isinstance(source, str): source = IMG(source)
        if isinstance(target, str): target = IMG(target)
        if isinstance(source, IMG): source = source.affine
        if isinstance(target, IMG): target = target.affine
        return self.to('world', source_affine=source, target_affine=target)

    def to(self, space = "world", source_affine = bt.eye(4), target_affine = bt.eye(4)):
        target_affine = batorch_tensor(target_affine).float()
        source_affine = batorch_tensor(source_affine).float()
        if target_affine.ndim <= 2: target_affine = bt.unsqueeze(target_affine)
        if source_affine.ndim <= 2: source_affine = bt.unsqueeze(source_affine)
        if space.lower() == "world" or space.lower() == "physical": taffine = bt.inv(target_affine); saffine = source_affine
        elif space.lower() == "image" or space.lower() == "index": taffine = target_affine; saffine = bt.inv(source_affine)
        else: raise TypeError("Invalid space for method 'to', use 'world'/'physical' or 'image'/'index' instead.")
        return ComposedTransformation(Affine(saffine), self, Affine(taffine), mode='spatial')
    
    def num_inv(self, *size, iterate=True, verbose=False):
        from .funcs import bending
        size = arg_tuple(size)
        n_dim = len(size)
        X = bt.image_grid(*size).unsqueeze([]).float()
        Jac = bt.grad_image(self(X)) # ([n_batch], n_dim, {n_dim}, *n_data)
        scale = ((Jac ** 2).sum(1) * n_dim).sqrt().max({}).values # ([n_batch], *n_data)
        sigma = min((scale / 2.5).quantile(0.9).item(), 1.5)
        is_backward = False
        if self.backward: self.forward_(); is_backward = True
        inv_disp = interpolation_forward(X, self, target_space=size, fill='nearest', sigma=sigma) - X
        if is_backward: self.backward_()
        inv_disp = bt.conv(inv_disp.mergedims({}, []), bt.gaussian_kernel(n_dim, kernel_size=3)).view_as(inv_disp)
        inv_disp = - self.to_DDF(*size).clone()
        if iterate:
            inv_disp = inv_disp.detach().requires_grad_(True)
            optimizer = bt.Optimization(bt.optim.Adam, [inv_disp], lr = 1e-2)
            nodrop_count = 0
            prev_loss = None
            for i in range(400):
                invtrans = DenseDisplacementField(inv_disp)
                loss = bt.norm2(invtrans(self(X)) - X).mean() + 1e-3 * bending(inv_disp)
                optimizer.minimize(loss)
                if verbose: print(f"Iteratively inverse transformation: [iter {i+1}] loss = {loss.item()}")
                if loss.item() < 1e-2:
                    if verbose: print(f"Stop at iter {i+1} due to small loss")
                    break
                if nodrop_count >= 4:
                    if verbose: print(f"Stop at iter {i+1} due to no dropping. ")
                    break
                if prev_loss is not None and loss.item() >= prev_loss: nodrop_count += 1
                else: nodrop_count = 0
        return DenseDisplacementField(inv_disp)
    
    def force_inv(self, *size):
        if hasattr(self, 'inv'): return self.inv()
        else: return self.num_inv(*size)

@alias("Id")
class Identity(SpatialTransformation):

    def __init__(self):
        '''
        Identity transformation.
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates. (Same as X for Identity)
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        super().__init__()
        
    def affine(self, n_dim=None):
        if n_dim is None: return
        return bt.eye(n_dim + 1).unsqueeze([])

    def inv(self): return Identity().backward_(self.backward)

class Rotation90(SpatialTransformation):
    def __init__(self, dim1, dim2, image_size=None, resize_image=True):
        '''
        Transformation that rotates an image of `image_size` by 90 degrees.
        
        Note: The rotation is for coordinates, hence the image rotates clockwise. 
        
        Params:
            dim1, dim2 [int]: The plane we rotate on. Direction of the rotation is from `dim1` to `dim2`.
                i.e. counter-clockwise rotation with dim1 as x-axis and dim2 as y-axis: [dim1, dim2] coordinates (x, y) becomes (ymax-y, x).
            image_size [tuple or bt.Tensor]: The size of the image, or the image itself. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        super().__init__(dim1=dim1, dim2=dim2, image_size=image_size, resize_image=resize_image)

        self.dim1, self.dim2 = dim1, dim2
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)
        if resize_image: self.reshape = [(1,), (dim1, dim2)]
        self.resize_image = resize_image

    def __call__(self, X):
        X = super().__call__(X)
        dim1, dim2 = self.dim1, self.dim2
        select1 = (slice(None),) * X.channel_dim + (dim1,)
        select2 = (slice(None),) * X.channel_dim + (dim2,)
        if self.image_size is None: max_range = X[select2].max()
        else: max_range = self.image_size[dim2]
        X[select1] = X[select1] + max_range - X[select2]
        X[select2] = X[select1] + X[select2] - max_range
        X[select1] = X[select1] - X[select2]
        return X
        
    def affine(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        dim1, dim2 = self.dim1, self.dim2
        aff = bt.eye(n_dim + 1)
        aff[dim1][dim1] = 0.
        aff[dim1][dim2] = -1.
        aff[dim1][-1] = float(self.image_size[dim2])
        aff[dim2][dim2] = 0.
        aff[dim2][dim1] = 1.
        if not self.backward: aff = aff.inv()
        return aff.unsqueeze([])
    
    def inv(self): return Rotation270(self.dim1, self.dim2, image_size = self.image_size, resize_image = self.resize_image).backward_(self.backward)

class Rotation270(SpatialTransformation):
    def __init__(self, dim1, dim2, image_size=None, resize_image=True):
        '''
        Transformation that rotates an image by 270 degrees.
        
        Note: The rotation is for coordinates, hence the image rotates clockwise. 
        
        Params:
            dim1, dim2 [int]: The plane we rotate on. Direction of the rotation is from `dim1` to `dim2`.
                i.e. counter-clockwise rotation with dim1 as x-axis and dim2 as y-axis: [dim1, dim2] coordinates (x, y) becomes (y, xmax-x).
            image_size [tuple or bt.Tensor]: The size of the image, or the image itself. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        super().__init__(dim1=dim1, dim2=dim2, image_size=image_size, resize_image=resize_image)

        self.dim1, self.dim2 = dim1, dim2
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)
        if resize_image: self.reshape = [(1,), (dim1, dim2)]
        self.resize_image = resize_image

    def __call__(self, X):
        X = super().__call__(X)
        dim1, dim2 = self.dim1, self.dim2
        select1 = (slice(None),) * X.channel_dim + (dim1,)
        select2 = (slice(None),) * X.channel_dim + (dim2,)
        if self.image_size is None: max_range = X[select1].max()
        else: max_range = self.image_size[dim1]
        X[select2] = X[select2] + max_range - X[select1]
        X[select1] = X[select2] + X[select1] - max_range
        X[select2] = X[select2] - X[select1]
        return X
        
    def affine(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        dim1, dim2 = self.dim1, self.dim2
        aff = bt.eye(n_dim + 1)
        aff[dim1][dim1] = 0.
        aff[dim1][dim2] = 1.
        aff[dim2][dim2] = 0.
        aff[dim2][dim1] = -1.
        aff[dim2][-1] = float(self.image_size[dim1])
        if not self.backward: aff = aff.inv()
        return aff.unsqueeze([])
    
    def inv(self): return Rotation90(self.dim1, self.dim2, image_size = self.image_size, resize_image = self.resize_image).backward_(self.backward)

class Rotation180(SpatialTransformation):
    def __init__(self, dim1, dim2, image_size=None):
        '''
        Transformation that rotates an image by 180 degrees.
        
        Params:
            dim1, dim2 [int]: The plane we rotate on. It is also the central symmetry.
                i.e. rotation with dim1 as x-axis and dim2 as y-axis: [dim1, dim2] coordinates (x, y) becomes (xmax-x, ymax-y).
            image_size [tuple or bt.Tensor]: The size of the image, or the image itself. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        super().__init__(dim1=dim1, dim2=dim2, image_size=image_size)

        self.dim1, self.dim2 = dim1, dim2
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)

    def __call__(self, X):
        X = super().__call__(X)
        dim1, dim2 = self.dim1, self.dim2
        select1 = (slice(None),) * X.channel_dim + (dim1,)
        select2 = (slice(None),) * X.channel_dim + (dim2,)
        if self.image_size is None: max_range1, max_range2 = X[select1].max(), X[select2].max()
        else: max_range1, max_range2 = self.image_size[dim1], self.image_size[dim2]
        X[select1] = max_range1 - X[select1]
        X[select2] = max_range2 - X[select2]
        return X
        
    def affine(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        dim1, dim2 = self.dim1, self.dim2
        aff = bt.eye(n_dim + 1)
        aff[dim1][dim1] = -1.
        aff[dim1][-1] = float(self.image_size[dim1])
        aff[dim2][dim2] = -1.
        aff[dim2][-1] = float(self.image_size[dim2])
        if not self.backward: aff = aff.inv()
        return aff.unsqueeze([])

    def inv(self): return Rotation180(self.dim1, self.dim2, image_size = self.image_size).backward_(self.backward)

@alias("Reflect")
class Reflection(SpatialTransformation):
    def __init__(self, *dim, image_size=None):
        '''
        Transformation that reflects an image along dimension dim.
        
        Params:
            dim [int]: The dimension we reflect the image along. 
                i.e. reflection on dim: [dim] coordinate x becomes xmax-x.
            image_size [tuple or bt.Tensor]: The size of the image, or the image itself. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        super().__init__(dims=dim, image_size=image_size)

        if len(dim) == 1 and isinstance(dim[0], (list, tuple)): dim = dim[0]
        self.dims = dim
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)

    def __call__(self, X):
        X = super().__call__(X)
        dims = self.dims
        for dim in dims:
            select = (slice(None),) * X.channel_dim + (dim,)
            if self.image_size is None: max_range = X[select].max()
            else: max_range = self.image_size[dim]
            X[select] = max_range - X[select]
        return X
        
    def affine(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        aff = bt.eye(n_dim + 1)
        for dim in self.dims:
            aff[dim][dim] = -1.
            aff[dim][-1] = float(self.image_size[dim])
        if not self.backward: aff = aff.inv()
        return aff.unsqueeze([])

    def inv(self): return Reflect(*self.dims, image_size = self.image_size).backward_(self.backward)

@alias("Permutedim")
class DimPermutation(SpatialTransformation):
    def __init__(self, *dims, resize_image=True):
        '''
        Permute the dimensions for an image, similar to np.transpose or torch/batorch.permute.
        
        Params:
            dims [list or tuple or bt.Tensor]: The dimension permuation. 
                size: length(n_dim) or ([n_batch:optional], {n_dim}).

        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        if len(dims) == 1: dims = dims[0]
        if isinstance(dims, tuple): dims = list(dims)
        dims = batorch_tensor(dims).squeeze()
        dims = dims.long()
        if dims.n_dim <= 1: dims = dims.unsqueeze([])
        if dims.n_dim == 2:
            if not dims.has_batch: dims.batch_dimension = 0
            if not dims.has_channel: dims.channel_dimension = 0 if dims.batch_dimension > 0 else 1
        avouch(dims.has_batch and dims.has_channel, f"Please use batorch tensor of size \
            ([n_batch:optional], {{n_dim}}) for Translation parameters, instead of {dims.shape}. ")
        super().__init__(dims, resize_image=resize_image)

        self.dims = dims
        self.n_dim = dims.n_channel
        self.resize_image = resize_image
        if resize_image:
            if dims.n_batch > 1:
                dims_cap = dims.sample(random=False, dim=[])
                avouch(bt.norm(dims - dims_cap).sum() < 1e-4, "Cannot resize image when different permutation done for different batch members. ")
            dims = dims.pick(0, []).tolist()
            visited = []
            self.reshape = [(1,)]
            for p in range(self.n_dim):
                if p in visited: continue
                while True:
                    visited.append(p)
                    q = dims[p]
                    if q in visited: break
                    self.reshape.append((p, q))
                    p = q

    def __call__(self, X):
        X = super().__call__(X)
        return X.gather(X.channel_dimension, self.dims.expand_to(X))
        
    def affine(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        if n_dim is None: n_dim = self.n_dim
        n_batch = self.n_batch
        if n_batch is None: n_batch = 1
        aff = bt.diag(bt.one_hot(-1, n_dim + 1).float().multiply(n_batch, []))
        b = bt.batch_tensor(bt.arange(n_batch)).expand_to(self.dims)
        i = bt.arange(n_dim).multiply(n_batch, [])
        aff[b, i, self.dims] = 1.
        if not self.backward: aff = aff.inv()
        return aff
    
    # @classmethod
    # def random_init_params(cls, *shape, n_batch=2):
    #     shape = arg_tuple(shape)
    #     n_dim = len(shape)
    #     coeffs = bt.rand([n_batch], {n_dim})
    #     perms = bt.zeros([n_batch], {n_dim}).int()
    #     for i in range(n_dim):
    #         idx = coeffs.argmax({})
    #         ibatch = bt.batch_tensor(bt.arange(n_batch))
    #         perms[ibatch, idx] = i
    #         coeffs[ibatch, idx] = 0
    #     return perms

    def inv(self):
        n_dim = self.n_dim
        new_permute = (self.dims == bt.arange(n_dim).view([1], n_dim, {1})).float().argmax(-1).channel_dimension_(-1)
        return DimPermutation(new_permute, resize_image = self.resize_image).backward_(self.backward)

@alias("Rescale")
class Rescaling(SpatialTransformation):
    def __init__(self, *scale, resize_image=True):
        '''
        Scale an image.
        
        Note: The scaling is for coordinates, hence the image would shrink if scale > 1. 
        
        Params:
            scale [float or tuple or bt.Tensor]: The scaling for all dimensions (float) 
                or for each dimension (tuple). >1 means enlarging the coordinates.
                size: ([n_batch:optional], {n_dim}) for bt.Tensor.
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        if len(scale) == 1 and isinstance(scale[0], (int, float)): scale = scale[0] * bt.ones([1], {1})
        else:
            if len(scale) == 1: scale = scale[0]
            if isinstance(scale, tuple): scale = list(scale)
            if not isinstance(scale, bt.Tensor): scale = batorch_tensor(scale).squeeze()
            if scale.n_dim <= 1: scale = scale.unsqueeze([])
            if scale.n_dim == 2:
                if not scale.has_batch: scale.batch_dimension = 0
                if not scale.has_channel: scale.channel_dimension = 0 if scale.batch_dimension > 0 else 1
        avouch(scale.has_batch and scale.has_channel, f"Please use batorch tensor of size \
            ([n_batch:optional], {{n_dim}}) for Scaling parameters, instead of {scale.shape}. ")
        bt.to_device(scale)
        super().__init__(scale, resize_image=resize_image)

        self.scale = scale
        if self.scale.n_channel > 1: self.n_dim = self.scale.n_channel
        if resize_image:
            avouch((scale - scale[:1]).abs().sum().sum() < eps, "The batch should have a common rescaler if 'resize_image=True', they are being reshaped uniformly.")
            self.reshape = [tuple((1/scale[0]).tolist())]
        self.resize_image = resize_image
        self.batch_param.append('scale')

    def __call__(self, X):
        X = super().__call__(X)
        scale = self.scale
        return X * scale
    
    def affine(self, n_dim=None):
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        scale = self.scale
        if isinstance(scale, (int, float)): return scale * bt.eye(n_dim + 1)
        aff = bt.diag(bt.cat(scale, bt.ones([scale.n_batch], 1), 1))
        if not self.backward: aff = aff.inv()
        return aff
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, std=1e-2):
        shape = arg_tuple(shape)
        return std * bt.randn({len(shape)}).multiply(n_batch, []) + 1,

    def inv(self): return Rescaling(1/self.scale, resize_image = self.resize_image).backward_(self.backward)

@alias("Translate")
class Translation(SpatialTransformation):
    def __init__(self, *translation):
        '''
        Translate an image.
        
        Note: The translation is for coordinates, hence the image would go in the opposite direction. 
        
        Params:
            translation [tuple or bt.Tensor]: The translation of the coordinates.
                size: length(n_dim) or ([n_batch:optional], {n_dim})
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch:optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        if len(translation) == 1: translation = translation[0]
        if isinstance(translation, tuple): translation = list(translation)
        translation = batorch_tensor(translation).squeeze()
        if translation.n_dim <= 1: translation = translation.unsqueeze([])
        if translation.n_dim == 2:
            if not translation.has_batch: translation.batch_dimension = 0
            if not translation.has_channel: translation.channel_dimension = 0 if translation.batch_dimension > 0 else 1
        avouch(translation.has_batch and translation.has_channel, f"Please use batorch tensor of size \
            ([n_batch:optional], {{n_dim}}) for Translation parameters, instead of {translation.shape}. ")
        super().__init__(translation)

        self.translation = translation
        self.n_dim = self.translation.n_channel
        self.batch_param.append('translation')

    def __call__(self, X):
        X = super().__call__(X)
        return X + self.translation
    
    def affine(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        if n_dim is None: n_dim = self.n_dim
        n_batch = self.translation.n_batch
        aff = bt.cat(bt.cat(bt.eye(self.n_dim).multiply(n_batch, []), self.translation.with_channeldim(None).unsqueeze(-1), -1), bt.one_hot(-1, n_dim+1).unsqueeze(0).multiply(n_batch, []), 1)
        if not self.backward: aff = aff.inv()
        return aff
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, std=1):
        shape = arg_tuple(shape)
        return std * bt.randn([n_batch], {len(shape)}),

    def inv(self): return Translation(-self.translation).backward_(self.backward)

@alias("Rig")
class Rigid(SpatialTransformation):
    def __init__(self, angle, axis=None, translation=None, center=None, trans_stretch=None, spacing=1):
        '''
        Rigid transformation with respect to parameters.
        
        Params Set 1:
            angle [bt.Tensor or np.numpy]: the [clockwise] rotation angles about the axises (or z direction for 2D). It is counter-clockwise for image.
                size: ([n_batch],)
            axis [bt.Tensor or np.numpy]: the rotation axises, normalized vectors, None for 2D. 
                size: ([n_batch], {n_dim})
            translation [bt.Tensor or np.numpy]: the translations after the rotation, zeros by default. 
                size: ([n_batch], {n_dim})
            center [bt.Tensor or np.numpy]: the center for the rotations, zeros by default. 
                size: ([n_batch], {n_dim})
            trans_stretch [int]: only used for iterative parameter training, 1 by default. 20 seems to be a good choice. 
            spacing [tuple]: the spacing of the transformed 
        
        Params Set 2:
            matrix [bt.Tensor or np.numpy]: the affine matrix, it should be orthogonal or will be projected by Procrustes Problem. 
                size: ([n_batch], n_dim + 1, n_dim + 1)
            trans_stretch [int]: only used for iterative parameter training, 1 by default. 20 seems to be a good choice. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        angle = batorch_tensor(angle)
        if angle.n_dim <= 0 and not angle.has_batch: angle = angle.unsqueeze([])
        if angle.n_dim == 1 and not angle.has_batch: angle = angle.with_batchdim(0)
        if angle.n_dim == 2 and not angle.has_batch: angle = angle.unsqueeze([])
        if angle.n_dim == 3 and not angle.has_batch and angle.shape[1] == angle.shape[2]: angle.batch_dimension = 0
        avouch(angle.has_batch, f"Please use batorch tensor of size ([n_batch],) for Rigid rotation angles, instead of {angle.shape}. ")
        n_batch = angle.n_batch
        n_dim = None
        if angle.n_dim >= 2:
            matrix = angle
            n_dim = matrix.size(-1) - 1
            center = bt.zeros([n_batch], {n_dim})
            translation = matrix[..., :-1, -1].with_channeldim(1)
            A = matrix[..., :-1, :-1]
            avouch(bt.norm(A.T @ A - bt.eye(A)).sum() < 1e-4, "Please make sure that matrix input for Rigid is orthogonal. Use Affine instead if it is not. ")
            # I = bt.eye(A)
            # Z_left = (A - I)[..., :-1] # ([n_batch], n_dim, n_dim-1)
            # Z_right = (A - I)[..., -1] # ([n_batch], n_dim)
            # ZTZ = Z_left.T @ Z_left
            # if n_dim == 2:
            #     axis = None
            #     angle = bt.where(bt.abs(bt.det(ZTZ)) < 1e-6, bt.zeros([n_batch]), bt.acos(1 - ZTZ / 2).squeeze(-1, -1))
            # elif n_dim == 3:
            #     invZTZ = bt.inv(bt.where(bt.abs(bt.det(ZTZ)).unsqueeze(-1, -1) < 1e-6, bt.eye(ZTZ), ZTZ))
            #     vector = bt.cat(- invZTZ @ Z_left.T @ Z_right, bt.ones([n_batch], 1), 1).with_channeldim(1)
            #     angle = bt.where(bt.abs(bt.det(ZTZ)) < 1e-6, bt.zeros([n_batch]), vector.norm())
            #     axis = bt.where((bt.abs(bt.det(ZTZ)) < 1e-6).multiply(n_dim, {}), bt.channel_tensor(bt.one_hot(0, n_dim)).multiply(n_batch, []), vector / angle)
            if n_dim == 2:
                axis = None
                angle = bt.acos(A[..., 0, 0]) * bt.sign(A[..., 0, 1])
            elif n_dim == 3:
                anti_sym = (A - A.T) / 2
                sym = (A + A.T) / 2 - bt.eye(A)
                angle = bt.acos(((anti_sym @ anti_sym) / sym).mean() - 1)
                axis = bt.uncross_matrix(anti_sym)
                axis /= bt.sin(angle)
            else:
                raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support {n_dim} dimensional transforms (2 & 3D only). " + 
                                            "Please contact the developers if there are feasible algorithms (Error Code: T333). Thank you. ")

        if translation is not None:
            translation = batorch_tensor(translation)
            if translation.n_dim <= 1 and not translation.has_batch: translation = translation.unsqueeze([])
            if translation.n_dim == 2 and not translation.has_batch: translation = translation.with_batchdim((1 - translation.channel_dimension) if translation.has_channel else 0)
            if translation.n_dim == 2 and not translation.has_channel: translation = translation.with_channeldim(1 - translation.batch_dimension)
            avouch(translation.has_batch and translation.has_channel, f"Please use batorch tensor of size ([n_batch], {{n_dim}}) for Rigid translation, instead of {translation.shape}. ")
            if n_batch == 1 and translation.n_batch > 1: n_batch = translation.n_batch
            if n_dim is None: n_dim = translation.n_channel
            else: avouch(n_dim == translation.n_channel, "Systematic error. Please contact the developers for details (Error Code: T332). ")
            n_dim = translation.n_channel
        if center is not None:
            center = batorch_tensor(center)
            if center.n_dim <= 1 and not center.has_batch: center = center.unsqueeze([])
            if center.n_dim == 2 and not center.has_batch: center = center.with_batchdim((1 - center.channel_dimension) if center.has_channel else 0)
            if center.n_dim == 2 and not center.has_channel: center = center.with_channeldim(1 - center.batch_dimension)
            avouch(center.has_batch and center.has_channel, f"Please use batorch tensor of size ([n_batch], {{n_dim}}) for Rigid center, instead of {center.shape}. ")
            if n_batch == 1 and center.n_batch > 1: n_batch = center.n_batch
            if n_dim is None: n_dim = center.n_channel
            else: avouch(n_dim == center.n_channel, f"Center({center.n_channel}D) and translation({n_dim}D) in trans.Rigid should have the same dimension. ")
            n_dim = center.n_channel
        if axis is not None:
            axis = batorch_tensor(axis)
            if axis.n_dim <= 1 and not axis.has_batch: axis = axis.unsqueeze([])
            if axis.n_dim == 2 and not axis.has_batch: axis = axis.with_batchdim((1 - axis.channel_dimension) if axis.has_channel else 0)
            if axis.n_dim == 2 and not axis.has_channel: axis = axis.with_channeldim(1 - axis.batch_dimension)
            avouch(axis.has_batch and axis.has_channel, f"Please use batorch tensor of size ([n_batch], {{n_dim}}) for Rigid axises, instead of {axis.shape}. ")
            if ((axis.norm() - 1) ** 2).sum().sum() >= 1e-2:
                print("warning: param. axises for 'Rigid' transformation should be norm-1 vectors, auto-normalization would be performed. Please contact the developers if it is necessary to be non-unit vectors (Error Code: T331). ")
                axis = axis / axis.norm()
            if n_batch == 1 and axis.n_batch > 1: n_batch = axis.n_batch
            if n_dim is None: n_dim = axis.n_channel
            else: avouch(n_dim == axis.n_channel, f"Translation({n_dim}D) and axises({axis.n_channel}D)) in trans.Rigid should have the same dimension. ")
        if n_dim is None: n_dim = 2 if axis is None else 3
        if translation is None: translation = bt.zeros([n_batch], {n_dim})
        if center is None: center = bt.zeros([n_batch], {n_dim})
        
        # Now we create the matrix
        angle = angle.float()
        center = center.float()
        translation = translation.float()
        if n_dim == 2:
            A = bt.stack(bt.cos(angle), bt.sin(angle), -bt.sin(angle), bt.cos(angle), -1).splitdim(-1, 2, 2)
        elif n_dim == 3:
            axis = axis.float()
            A = bt.eye([n_batch], n_dim) + (1 - bt.cos(angle)) * bt.cross_matrix(axis) @ bt.cross_matrix(axis) + bt.sin(angle) * bt.cross_matrix(axis)
        else:
            raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support {n_dim} dimensional transforms (2 & 3D only). " + 
                                        "Please contact the developers if there are feasible algorithms (Error Code: T333). Thank you. ")
        matrix = bt.cat(bt.cat(A, ((translation + center).with_channeldim(None) - A @ center.with_channeldim(None)).unsqueeze(-1), -1), 
                        bt.cat(bt.zeros(n_dim), bt.ones(1)).unsqueeze([], 1), 1)
        if trans_stretch is not None: matrix[..., :n_dim, -1] *= trans_stretch
        if not isinstance(spacing, tuple): spacing = to_tuple(spacing)
        if len(spacing) == 1: spacing = spacing * n_dim
        A = bt.diag(batorch_tensor(list(spacing) + [1.])).unsqueeze([]).astype(matrix)
        super().__init__(angle, axis, translation, center=center, trans_stretch=trans_stretch, spacing=spacing)

        self.n_dim = n_dim
        self.trans_stretch = trans_stretch
        self.spacing = spacing
        self.matrix = A.inv() @ matrix @ A
        self.batch_param.append('matrix')

    def __call__(self, X):
        X = super().__call__(X)
        matrix = self.matrix
        n_dim = self.n_dim
        A = matrix[:, :n_dim, :n_dim]
        b = matrix[:, :n_dim, n_dim]
        shape = X.shape
        Y = (A @ X.flatten().channel_dim_(None) + b.unsqueeze(-1)).view(shape)
        return Y
    
    def affine(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        return self.matrix if self.backward else self.matrix.inv()
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, std=1):
        shape = arg_tuple(shape)
        n_dim = len(shape)
        if n_dim == 2:
            return std * bt.randn([n_batch]), None, std * bt.randn([n_batch], {n_dim}), bt.channel_tensor(shape).multiply(n_batch, []) / 2
        if n_dim == 3:
            ax = bt.randn([n_batch], {n_dim})
            return std * bt.randn([n_batch]), ax / ax.norm(), std * bt.randn([n_batch], {n_dim}), bt.channel_tensor(shape).multiply(n_batch, []) / 2
        else:
            raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support {n_dim} dimensional transforms (2 & 3D only). " + 
                                        "Please contact the developers if there are feasible algorithms (Error Code: T334). Thank you. ")

    def inv(self):
        A = bt.diag(batorch_tensor(list(self.spacing) + [1.])).unsqueeze([]).astype(self.matrix)
        matrix = A @ bt.inv(self.matrix) @ A.inv()
        if self.trans_stretch is not None: matrix[..., :self.n_dim, -1] /= self.trans_stretch
        return Rigid(matrix, trans_stretch = self.trans_stretch, spacing=self.spacing).backward_(self.backward)

@alias("Aff")
class Affine(SpatialTransformation):
    def __init__(self, matrix, trans_stretch=None, spacing=1):
        '''
        Affine transformation with respect to transformation matrix.
        
        Params:
            matrix [bt.Tensor or np.numpy]: the affine matrix. 
                size: ([n_batch], n_dim + 1, n_dim + 1)
            trans_stretch [int]: only used for iterative parameter training, 1 by default. 20 seems to be a good choice. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        matrix = batorch_tensor(matrix)
        n_dim = matrix.size(-1) - 1
        if matrix.n_dim <= 2 and not matrix.has_batch: matrix = matrix.unsqueeze([])
        if matrix.n_dim == 3 and not matrix.has_batch and matrix.shape[1] == matrix.shape[2]: matrix.batch_dimension = 0
        avouch(matrix.has_batch, f"Please use batorch tensor of size ([n_batch], n_dim + 1, n_dim + 1) for Affine parameters, instead of {matrix.shape}. ")
        if trans_stretch is not None: matrix[..., :n_dim, -1] *= trans_stretch
        if not isinstance(spacing, tuple): spacing = to_tuple(spacing)
        if len(spacing) == 1: spacing = spacing * n_dim
        A = bt.diag(batorch_tensor(list(spacing) + [1.])).unsqueeze([]).astype(matrix)
        super().__init__(matrix, trans_stretch=trans_stretch, spacing=spacing)

        self.n_dim = matrix.size(-1) - 1
        self.trans_stretch = trans_stretch
        self.spacing = spacing
        self.matrix = A.inv() @ matrix @ A
        self.batch_param.append('matrix')

    def __call__(self, X):
        X = super().__call__(X)
        matrix = self.matrix.float()
        n_dim = self.n_dim
        A = matrix[:, :n_dim, :n_dim]
        b = matrix[:, :n_dim, n_dim]
        shape = X.shape
        Y = (A @ X.flatten().channel_dim_(None) + b.unsqueeze(-1)).view(shape)
        return Y
    
    def affine(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        return self.matrix if self.backward else self.matrix.inv()
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, std=1e-2, trans_std=1):
        shape = arg_tuple(shape)
        n_dim = len(shape)
        affmat = bt.eye([n_batch], n_dim + 1, n_dim + 1)
        affmat[..., :n_dim, :n_dim] += std * bt.randn([n_batch], n_dim, n_dim)
        affmat[..., :n_dim, -1] += trans_std * bt.randn([n_batch], n_dim)
        return affmat,

    def inv(self):
        A = bt.diag(batorch_tensor(list(self.spacing) + [1.])).unsqueeze([]).astype(self.matrix)
        matrix = A @ bt.inv(self.matrix) @ A.inv()
        if self.trans_stretch is not None: matrix[..., :self.n_dim, -1] /= self.trans_stretch
        return Affine(matrix, trans_stretch = self.trans_stretch, spacing=self.spacing).backward_(self.backward)

@alias("logEu")
class PolyAffine(SpatialTransformation):
    def __init__(self, dmatrices, masks, order=2, is_inv=False, trans_stretch=None):
        '''
        Poly affine transformation with respect to transformation matrices [1].
        Note that dmatrices for this tranformation IS NOT the actual affine matrices, but IS a differentiation instead.
        
        Params:
            dmatrices [bt.Tensor or np.numpy]: One affine matrix for each region. 
                size: ([n_batch], {n_region}, n_dim + 1, n_dim + 1)
            masks [bt.Tensor or np.numpy]: One 0-1 mask for each region. 
                size: ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
            order [int]: the order of interpolation coefficient. The influence of an affine decays at a rate of 1 / distance^order.
            trans_stretch [int]: only used for iterative parameter training, 1 by default. 20 seems to be a good choice.  
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n@1, n@2, ..., n@n_dim)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)

        [1] Arsigny V , Commowick O , Ayache N , et al. A Fast and Log-Euclidean Polyaffine Framework for Locally 
            Linear Registration[J]. Journal of Mathematical Imaging & Vision, 2009, 33(2):222-238.
        '''
        from .funcs import distance_map
        if not isinstance(dmatrices, bt.Tensor): dmatrices = bt.tensor(dmatrices)
        if dmatrices.n_dim <= 3 and not dmatrices.has_batch: dmatrices = dmatrices.unsqueeze([])
        if dmatrices.n_dim <= 3 and not dmatrices.has_channel: dmatrices = dmatrices.unsqueeze({})
        if dmatrices.n_dim == 4 and dmatrices.shape[2] == dmatrices.shape[3]:
            if not dmatrices.has_batch: dmatrices.batch_dimension = 0
            if not dmatrices.has_channel: dmatrices.channel_dimension = 1
        avouch(dmatrices.has_batch and dmatrices.has_channel, "Please use batorch tensor of size ([n_batch], {n_region}," +
               f"n_dim + 1, n_dim + 1) for PolyAffine parameters, instead of {dmatrices.shape}. ")
        n_dim = dmatrices.size(-1) - 1
        if trans_stretch is not None: dmatrices[..., :n_dim, -1] *= trans_stretch
        if not isinstance(masks, bt.Tensor): masks = bt.tensor(masks)
        if masks.n_dim <= n_dim + 1 and not masks.has_batch: masks = masks.unsqueeze([])
        if masks.n_dim <= n_dim + 1 and not masks.has_channel: masks = masks.unsqueeze({})
        if masks.n_dim == n_dim + 2 and not masks.has_batch: masks.batch_dimension = 0
        if masks.n_dim == n_dim + 2 and not masks.has_channel: masks.channel_dimension = 1
        avouch(masks.has_batch and masks.has_channel, "Please use batorch tensor of size ([n_batch], {n_region}," +
               f"n@1, n@2, ..., n@n_dim) for PolyAffine parameters, instead of {masks.shape}. ")

        # preprocess masks ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        n_batch = masks.n_batch
        n_region = masks.n_channel
        dis = distance_map(masks)
        dis = (dis + 1).clamp(0)
        # import micomputing.plot as plt
        # plt.subplots(2)
        # plt.imsshow(dis[0, 0], dis[0, 1])
        # plt.show()
        weights = 1 / (dis ** order + 1e-5)
        weights = weights / weights.sum({}) # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        
        # # deprecated preprocess of masks
        # masks = masks.numpy().astype(np.int)
        # n_batch, n_region, *_ = masks.shape
        # _dis_map = bt.zeros(*masks.shape)
        # for i in range(n_batch):
        #     for j in range(n_region):
        #         mask_image = sitk.GetImageFromArray(masks[i, j], isVector = False)
        #         dis_map = sitk.GetArrayViewFromImage(sitk.SignedMaurerDistanceMap(mask_image, squaredDistance = False, useImageSpacing = False))
        #         dis_map = np.array(dis_map).astype(np.float)
        #         _dis_map[i, j] = bt.tensor(dis_map * (dis_map > 0).astype(np.float))
        # k = 2
        # invpowk_dis_map = 1 / (_dis_map ** k + 1e-5)
        # sum_dis_map = invpowk_dis_map.sum(1, keepdim = True)
        # weights = invpowk_dis_map / sum_dis_map
        # from matplotlib import pyplot as plt
        # plt.subplot(121); plt.imshow(weights[0, 0])
        # plt.subplot(122); plt.imshow(weights[0, 1])
        # plt.show()
        # if trans_stretch is not None: dmatrices = dmatrices * bt.tensor([1.] * n_dim + [trans_stretch]).unsqueeze(0, 0, 0)

        super().__init__(dmatrices, masks=masks, order=order, is_inv=is_inv, trans_stretch=trans_stretch)
        self.n_batch = n_batch
        self.n_dim = n_dim
        self.masks = masks
        self.weights = weights
        self.trans_stretch = trans_stretch
        self.dmatrices = dmatrices
        self.is_inv = is_inv
        self.order = order
        
    def update_masks(self, masks):
        from .funcs import distance_map
        self.masks = masks
        dis = distance_map(masks)
        dis = (dis + 1).clamp(0)
        weights = 1 / (dis ** self.order + 1e-5)
        weights = weights / weights.sum({}) 
        self.weights = weights

    def __call__(self, X):
        X = super().__call__(X)
        dmatrices = self.dmatrices # ([n_batch], {n_region}, n_dim + 1, n_dim + 1)
        weights = self.weights # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        n_dim = self.n_dim
        n_region = dmatrices.n_channel
        Xs = X.multiply(n_region, {}) # ([n_batch], {n_region}, n_dim, n@1, n@2, ..., n@n_dim)
        A = dmatrices[..., :n_dim, :n_dim]
        b = dmatrices[..., :n_dim, n_dim:]
        Y = (A @ Xs.flatten(3) + b).view_as(Xs)
        D = (Y * weights.unsqueeze()).sum({}).with_channeldim(1) - X
        if self.is_inv: D = -D
        trans = DenseDisplacementField(D)
        trans2 = DenseDisplacementField(trans(trans(X)) - X)
        trans4 = DenseDisplacementField(trans2(trans2(X)) - X)
        trans8 = DenseDisplacementField(trans4(trans4(X)) - X)
        trans16 = DenseDisplacementField(trans8(trans8(X)) - X)
        trans32 = DenseDisplacementField(trans16(trans16(X)) - X)
        trans64 = DenseDisplacementField(trans32(trans32(X)) - X)
        return trans64(X)
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, n_region=2, std=1e-4, trans_std=1e-2):
        shape = arg_tuple(shape)
        n_dim = len(shape)
        affmat = bt.eye([n_batch], {n_region}, n_dim + 1, n_dim + 1)
        affmat[..., :n_dim, :n_dim] += std * bt.randn([n_batch], {n_region}, n_dim, n_dim)
        affmat[..., :n_dim, -1] += trans_std * bt.randn([n_batch], {n_region}, n_dim)
        centers = (bt.rand([n_batch], n_region, {n_dim}) * bt.channel_tensor(shape)).int()
        level = (((centers.unsqueeze(1) - centers.unsqueeze(2)) ** 2).sum({}).sqrt() + 1e4 * bt.eye([n_batch], n_region, n_region)).min() / 3
        masks = ((bt.image_grid(*shape).multiply(n_region, 0).multiply(n_batch, []) - centers) ** 2).sum({}).sqrt() < level
        return affmat, masks.with_channeldim(1)

    def inv(self):
        new_matrices = self.dmatrices.inv()
        n_batch = self.dmatrices.n_batch
        new_masks = interpolation(self.masks.mergedims([], {}), Affine(bt.pow(new_matrices.mergedims([], {}), 64))).splitdim([], [n_batch], {-1})
        if self.trans_stretch is not None: new_matrices[..., :self.n_dim, -1] /= self.trans_stretch
        return PolyAffine(new_matrices, new_masks, order=self.order, is_inv=self.is_inv, trans_stretch = self.trans_stretch).backward_(self.backward)
        # n_batch = self.dmatrices.n_batch
        # affs = bt.inv(self.dmatrices)
        # masks = interpolation(self.masks.mergedims([], {}), Affine(bt.matpow(affs.mergedims([], {}), 64))).splitdim([], [n_batch], {-1})
        # return PolyAffine(affs, masks = masks, trans_stretch = self.trans_stretch).backward_(self.backward)

@alias("LARM")
class LocallyAffine(SpatialTransformation):
    def __init__(self, matrices, masks, order=2, trans_stretch=None, avoid_conflict=True):
        '''
        Locally affine transformation with respect to transformation matrices [1].
        
        Params:
            matrices [bt.Tensor or np.numpy]: One affine matrix for each region. 
                size: ([n_batch], {n_region}, n_dim + 1, n_dim + 1)
            masks [bt.Tensor or np.numpy]: One 0-1 mask for each region. 
                size: ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
            order [int]: the order of interpolation coefficient. The influence of an affine decays at a rate of 1 / distance^order.
            trans_stretch [int]: only used for iterative parameter training, 1 by default. 20 seems to be a good choice.  
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n@1, n@2, ..., n@n_dim)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)

        [1] Zhuang X , Rhode K , Arridge S , et al. An Atlas-Based Segmentation Propagation Framework Using Locally Affine 
            Registration - Application to Automatic Whole Heart Segmentation[J]. Springer, Berlin, Heidelberg, 2008.
        '''
        from .funcs import dilate, distance_map
        matrices = batorch_tensor(matrices)
        if matrices.n_dim <= 3 and not matrices.has_batch: matrices = matrices.unsqueeze([])
        if matrices.n_dim <= 3 and not matrices.has_channel: matrices = matrices.unsqueeze({})
        if matrices.n_dim == 4 and matrices.shape[2] == matrices.shape[3]:
            if not matrices.has_batch: matrices.batch_dimension = 0
            if not matrices.has_channel: matrices.channel_dimension = 1
        avouch(matrices.has_batch and matrices.has_channel, "Please use batorch tensor of size ([n_batch], {n_region}," +
               f"n_dim + 1, n_dim + 1) for LocallyAffine parameters, instead of {matrices.shape}. ")
        n_dim = matrices.size(-1) - 1
        if trans_stretch is not None: matrices[..., :n_dim, -1] *= trans_stretch
        masks = batorch_tensor(masks).int()
        if masks.n_dim <= n_dim + 1 and not masks.has_batch: masks = masks.unsqueeze([])
        if masks.n_dim <= n_dim + 1 and not masks.has_channel: masks = masks.unsqueeze({})
        if masks.n_dim == n_dim + 2 and not masks.has_batch: masks.batch_dimension = 0
        if masks.n_dim == n_dim + 2 and not masks.has_channel: masks.channel_dimension = 1
        avouch(masks.has_batch and masks.has_channel, "Please use batorch tensor of size ([n_batch], {n_region}," + 
               f"n@1, n@2, ..., n@n_dim) for LocallyAffine parameters, instead of {masks.shape}. ")

        # preprocess masks
        n_batch = matrices.n_batch
        n_region = matrices.n_channel
        if avoid_conflict:
            Gi = Affine(matrices.mergedims({}, []))
            GiVi = interpolation(masks.mergedims({}, []), Gi.inv(), method='Nearest').splitdim([], [n_batch], n_region) # ([n_batch], n_region, n@1, n@2, ..., n@n_dim)
            GiVijoinGkVk = (GiVi.unsqueeze(1) * GiVi.unsqueeze(2)).mergedims(1, []).with_channeldim(1) # ([n_batch x n_region], {n_region}, n@1, n@2, ..., n@n_dim)
            Rik = interpolation(GiVijoinGkVk, Gi, method='Nearest').splitdim([], [n_batch], n_region) # ([n_batch], n_region, {n_region}, n@1, n@2, ..., n@n_dim)
            URik = (Rik.sum({}).with_channeldim(1) - masks > 0.1).float() # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
            URik_plus = dilate(URik.mergedims({}, []), 1).splitdim([], [n_batch], n_region) # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
            masks = (masks - URik_plus > 0.1).float() # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)

        weights = 1 / (distance_map(masks) ** order + 1e-5)
        weights = weights / weights.sum({}) # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        # if trans_stretch is not None: matrices = matrices * batorch_tensor([1.] * n_dim + [trans_stretch]).unsqueeze(0, 0, 0)

        super().__init__(matrices, masks=masks, order=order, trans_stretch=trans_stretch)
        self.avoid_conflict = avoid_conflict
        self.order = order
        self.n_batch = n_batch
        self.n_dim = n_dim
        self.masks = masks
        self.weights = weights
        self.trans_stretch = trans_stretch
        self.matrices = matrices
        self.batch_param.extend(['matrices', 'weights', 'masks'])
        
    def update_masks(self, masks):
        from .funcs import dilate, distance_map
        # preprocess masks
        n_batch = self.n_batch
        n_region = self.matrices.n_channel
        if self.avoid_conflict:
            Gi = Affine(self.matrices.mergedims({}, []))
            GiVi = interpolation(masks.mergedims({}, []), Gi.inv(), method='Nearest').splitdim([], [n_batch], n_region) # ([n_batch], n_region, n@1, n@2, ..., n@n_dim)
            GiVijoinGkVk = (GiVi.unsqueeze(1) * GiVi.unsqueeze(2)).mergedims(1, []).with_channeldim(1) # ([n_batch x n_region], {n_region}, n@1, n@2, ..., n@n_dim)
            Rik = interpolation(GiVijoinGkVk, Gi, method='Nearest').splitdim([], [n_batch], n_region) # ([n_batch], n_region, {n_region}, n@1, n@2, ..., n@n_dim)
            URik = (Rik.sum({}).with_channeldim(1) - masks > 0.1).float() # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
            URik_plus = dilate(URik.mergedims({}, []), 1).splitdim([], [n_batch], n_region) # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
            masks = (masks - URik_plus > 0.1).float() # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)

        self.masks = masks
        weights = 1 / (distance_map(masks) ** self.order + 1e-5)
        weights = weights / weights.sum({}) # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        self.weights = weights

    def __call__(self, X):
        X = super().__call__(X)
        matrices = self.matrices
        masks = self.masks # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        weights = self.weights # ([n_batch], {n_region}, n@1, n@2, ..., n@n_dim)
        n_dim = self.n_dim
        n_region = matrices.n_channel
        Xs = X.multiply(n_region, {}) # ([n_batch], {n_region}, n_dim, n@1, n@2, ..., n@n_dim)
        A = matrices[..., :n_dim, :n_dim]
        b = matrices[..., :n_dim, n_dim:]
        Y = (A @ Xs.flatten(3) + b).view_as(Xs)
        return (weights.unsqueeze() * Y).sum({}).with_channeldim(1) * (1 - masks.sum({}).unsqueeze({})) + (Y * masks.unsqueeze()).sum({}).with_channeldim(1)
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, n_region=2, std=1e-2, trans_std=1):
        shape = arg_tuple(shape)
        n_dim = len(shape)
        affmat = bt.eye([n_batch], {n_region}, n_dim + 1, n_dim + 1)
        affmat[..., :n_dim, :n_dim] += std * bt.randn([n_batch], {n_region}, n_dim, n_dim)
        affmat[..., :n_dim, -1] += trans_std * bt.randn([n_batch], {n_region}, n_dim)
        centers = (bt.rand([n_batch], n_region, {n_dim}) * bt.channel_tensor(shape)).int()
        level = (((centers.unsqueeze(1) - centers.unsqueeze(2)) ** 2).sum({}).sqrt() + 1e4 * bt.eye([n_batch], n_region, n_region)).min() / 3
        masks = ((bt.image_grid(*shape).multiply(n_region, 0).multiply(n_batch, []) - centers) ** 2).sum({}).sqrt() < level
        return affmat, masks.with_channeldim(1)

    def inv(self):
        new_matrices = self.matrices.inv()
        n_batch = self.matrices.n_batch
        new_masks = interpolation(self.masks.mergedims([], {}), Affine(new_matrices.mergedims([], {}))).splitdim([], [n_batch], {-1})
        if self.trans_stretch is not None: new_matrices[..., :self.n_dim, -1] /= self.trans_stretch
        return LocallyAffine(new_matrices, masks = new_masks, trans_stretch = self.trans_stretch).backward_(self.backward)

@alias("FFD")
class FreeFormDeformation(SpatialTransformation):

    def __init__(self, offsets, spacing=1, origin=0):
        '''
        Free Form Deformation (FFD) transformation [1].
        
        Params:
            offsets [bt.Tensor]: the FFD offsets. 
                size: ([n_batch], {n_dim}, m@1, m@2, ..., m@n_dim)
                for m@1 x m@2 x ... x m@n_dim grid of Δcontrol points
            spacing [int or tuple]: FFD spacing; spacing between FFD control points. 
            origin [int or tuple]: FFD origin; coordinate for the (0, 0, 0) control point. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n@1, n@2, ..., n@n_dim)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
                
        [1] Rueckert D , Sonoda L I , Hayes C , et al. Nonrigid registration using free-form deformations: 
            application to breast MR images[J]. IEEE Transactions on Medical Imaging, 1999(8).
        '''
        offsets = batorch_tensor(offsets)
        if not offsets.has_channel:
            if offsets.size(0) == offsets.n_dim - 1:
                n_dim = offsets.size(0)
                offsets.channel_dimension = 0
                offsets = offsets.unsqueeze([])
            elif offsets.size(1) == offsets.n_dim - 2:
                n_dim = offsets.size(1)
                offsets.channel_dimension = 1
            else: raise TypeError(f"FFD parameters with size {offsets.shape} donot match ([n_batch], {{n_dim}}, m_1, m_2, ..., m_r) [r=n_dim]. ")
        if not offsets.has_batch:
            n_dim = offsets.n_channel
            if offsets.n_dim <= n_dim + 1: offsets = offsets.unsqueeze([])
            else: offsets.batch_dimension = 0
        avouch(offsets.has_batch and offsets.has_channel, f"Please use batorch tensor of size \
            ([n_batch], {{n_dim}}, m_1, m_2, ..., m_r) [r=n_dim] for FFD parameters, instead of {offsets.shape}. ")
        n_dim = offsets.n_channel
        spacing = to_tuple(spacing)
        origin = to_tuple(origin)
        if len(spacing) == 1: spacing *= n_dim
        if len(origin) == 1: origin *= n_dim
        super().__init__(offsets, spacing=spacing, origin=origin)

        self.n_dim = n_dim
        self.offsets = offsets
        self.spacing = spacing
        self.origin = origin
        self.batch_param.append('offsets')

    @property
    def displacements(self): return self.offsets
    
    @displacements.setter
    def displacements(self, value): self.offsets = value
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, spacing=None, std=1e-2):
        shape = arg_tuple(shape)
        n_dim = len(shape)
        if spacing is None: spacing = 1 << (int(math.log2(min(shape))) - 3)
        if not isinstance(spacing, (tuple, list)): spacing = (spacing,)
        if len(spacing) == 1: spacing = spacing * n_dim
        ffd_gridsize = [l // s + 1 for l, s in zip(shape, spacing)]
        return std * bt.randn([n_batch], {n_dim}, *ffd_gridsize)
    
    def __call__(self, X):
        X = super().__call__(X)
        shape = X.shape
        n_dim = self.n_dim
        offsets = self.offsets.float()
        spacing = self.spacing
        n_batch = offsets.n_batch
        # X: ([n_batch], {n_dim}, n_data = n_1 x n_2 x ... x n_r)
        X = X.flatten()
        X -= bt.channel_tensor(self.origin)
        n_data = X.size(-1)
        size = bt.channel_tensor(offsets.space)
        # Normalize X in the domain (m_1, m_2, ..., m_{n_dim}).
        FFDX = X / bt.channel_tensor(spacing).float()
        iX = bt.floor(FFDX).float(); uX = FFDX - iX
        # Compute the weights. W: (4, [n_batch], {n_dim}, n_data = n_1 x n_2 x ... x n_r)
        i = bt.arange(-1, 3).expand_to(4, [n_batch], {n_dim}, n_data, axis=0)
        W = Bspline(i, uX.multiply(4, 0))
        "Compute FFD Transformation"
        output = bt.zeros_like(X)
        # Loop in the space {-1, 0, 1, 2} ^ n_dim; G is in (0, 1, 2, 3)
        for G in bt.image_grid([4]*n_dim).flatten().transpose(0, 1):
            G = bt.channel_tensor(G)
            # Weights for each point: [product of W[G[D], t, D, x] for D in range(n_dim)] for point x and batch t.
            # Wg: ([n_batch], {1}, n_data = n_1 x n_2 x ... x n_r)
            Wg = W.gather(0, G.expand_to((1,) + W.shape[1:])).squeeze(0).prod({}, keepdim=True)
            # Compute the indices of related control points. Ind: ([n_batch], {n_dim}, n_data = n_1 x n_2 x ... x n_r)
            Ind = bt.clamp(iX.long() + G - 1, min=0)
            Ind = bt.min(Ind, (size - 1).expand_to(Ind))
            # Convert the indices to 1 dimensional. Dot: ([n_batch], n_data = n_1 x n_2 x ... x n_r)
            Dot = Ind[:, 0]
            for r in range(1, n_dim): Dot *= size[r]; Dot += Ind[:, r]
            # Obtain the coordinates of the control points. CPoints: ([n_batch], {n_dim}, n_data = n_1 x n_2 x ... x n_r)
            CPoints = offsets.flatten().gather(-1, Dot.long().expand_to(Ind)).float()
            # Add the weighted control coordinates to the output coordinates.
            output += (Wg * CPoints).view_as(X)
        # Denormalize the outputs.
        output += X
        output += bt.channel_tensor(self.origin)
        return output.view(shape)

@alias("DDF")
class DenseDisplacementField(SpatialTransformation):
    def __init__(self, displacements, shape=None, interpolate=False):
        '''
        Dense Displacement Field (DDF) transformation.
        
        Params:
            displacements [bt.Tensor]: the displacement of each voxel. 
                size: ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
            shape [bt.Size or tuple]: the shape of displacement (needed if input displacement is a transformation). 
            interpolate [bool]: Whether to force interpolation in apply. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n@1, n@2, ..., n@n_dim)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
        '''
        if isinstance(displacements, SpatialTransformation): displacements = displacements.toDDF(shape)
        displacements = batorch_tensor(displacements)
        if not displacements.has_channel:
            if displacements.size(0) == displacements.n_dim - 1:
                n_dim = displacements.size(0)
                displacements.channel_dimension = 0
                displacements = displacements.unsqueeze([])
            elif displacements.size(1) == displacements.n_dim - 2:
                n_dim = displacements.size(1)
                displacements.channel_dimension = 1
            else: raise TypeError(f"DDF parameters with size {displacements.shape} donot match ([n_batch], {{n_dim}}, n@1, n@2, ..., n@n_dim). ")
        if not displacements.has_batch:
            n_dim = displacements.n_channel
            if displacements.n_dim <= n_dim + 1: displacements = displacements.unsqueeze([])
            else: displacements.batch_dimension = 0
        displacements = displacements.float()
        avouch(displacements.has_batch and displacements.has_channel, f"Please use batorch tensor of size \
            ([n_batch], {{n_dim}}, n@1, n@2, ..., n@n_dim) for DDF parameters, instead of {displacements.shape}. ")
        super().__init__(displacements, shape=shape, interpolate=interpolate)
        self.n_dim = displacements.n_channel
        self.displacements = displacements
        self.interpolate = interpolate
        self.batch_param.append('displacements')

    @property
    def offsets(self): return self.displacements
    
    @offsets.setter
    def offsets(self, value): self.displacements = value
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=2, std=1e-2):
        shape = arg_tuple(shape)
        n_dim = len(shape)
        return std * bt.randn([n_batch], {n_dim}, *shape)

    def __call__(self, X):
        X = super().__call__(X)
        displacements = self.displacements
        n_dim = self.n_dim
        if X.n_space == 0: X = X.unsqueeze(-1)
        if not self.interpolate and X.space == displacements.space and X.n_channel == displacements.n_channel: return X + displacements
        else: return X + interpolation(displacements, target_space=X).channel_dimension_(1)

@alias("MLP")
class MultiLayerPerception(SpatialTransformation):
    def __init__(self, weights, hidden_layers=[], active_function=None, trans_stretch=1):
        '''
        A transformation defined by a MLP. 
        
        Params:
            weights [bt.Tensor]: the weights for the perception network. 
                size: ([n_batch], n_dim * n_hl_1 + n_hl_1 + sum(i=1..k-1){n_hl_i * n_hl_{i+1} + n_hl_{i+1}} + n_hl_k * n_dim + n_dim)
                where k is the number of hidden layers and n_hl_i is the length of the i-th hidden layer. 
            hidden_layers [list of int]: the lengths for the hidden layers, i.e. [n_hl_1, n_hl_2, ..., n_hl_k]
                It is by default '[]' so that the default MLP is an affine transformation. 
            active_function [class]: the active_function. 
            trans_stretch [int]: 1 by default. 20 seems to be a good choice. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n@1, n@2, ..., n@n_dim)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
        '''
        self.hidden_layers = hidden_layers
        if not weights.has_batch:
            if weights.n_dim == 2: weights.batch_dim = 0
            else: weights = weights.unsqueeze([])
        self.weights = weights
        self.trans_stretch = trans_stretch
        super().__init__(weights, hidden_layers = hidden_layers, trans_stretch = trans_stretch)
        dim_const = weights.size(-1) - sum(hidden_layers) - sum(x * y for x, y in zip(hidden_layers[:-1], hidden_layers[1:]))
        dim_coeff = hidden_layers[0] + hidden_layers[-1] + 1
        avouch(dim_const % dim_coeff == 0, f"Wrong weight length for hidden layers of sizes {hidden_layers}, {dim_coeff} x n_dim + {dim_const} expected, but got {weights.size(-1)}.")
        self.n_dim = dim_const // dim_coeff
        self.n_batch = weights.n_batch
        self.layers = []
        p = 0
        for i in range(len(hidden_layers) + 1):
            in_features = self.n_dim if i == 0 else hidden_layers[i - 1]
            out_features = self.n_dim if i == len(hidden_layers) else hidden_layers[i]
            layer_weights = weights[..., p:p+out_features*in_features].view([self.n_batch], out_features, in_features)
            p += out_features * in_features
            layer_bias = weights[..., p:p+out_features].view([self.n_batch], out_features)
            p += out_features
            self.layers.append((layer_weights, layer_bias))
        self.active_function = active_function
        if self.active_function is None: self.active_function = bt.nn.ReLU
    
    @classmethod
    def get_weight_length(cls, n_dim, *hidden_layers):
        hidden_layers = arg_tuple(hidden_layers)
        return n_dim * (hidden_layers[0] + hidden_layers[-1] + 1) \
            + sum(hidden_layers) + sum(x * y for x, y in zip(hidden_layers[:-1], hidden_layers[1:]))
    
    @property
    def n_weight_length(self):
        return MultiLayerPerception.get_weight_length(self.n_dim, self.hidden_layers)
    
    @classmethod
    def random_init_params(cls, n_dim, *hidden_layers, n_batch=2, std=1e-4):
        hidden_layers = arg_tuple(hidden_layers)
        n_length = MultiLayerPerception.get_weight_length(n_dim, hidden_layers)
        return std * bt.randn([n_batch], n_length)
    
    def __call__(self, X):
        X = super().__call__(X)
        if X.n_space == 0: X = X.unsqueeze(-1)
        Y = X.flatten().channel_dim_(None)
        for i, (weights, bias) in enumerate(self.layers):
            Y = weights @ Y
            if i < len(self.layers) - 1: Y = self.active_function()(Y + bias.unsqueeze(-1))
            else: Y += self.trans_stretch * bias.unsqueeze(-1)
        return X + Y.view_as(X).channel_dimension_(1)
    
def _init_interpolation(image, trans, fill, target_space):
    image = bt.to_device(batorch_tensor(image))
    shape_out = image.shape
    if trans is None or trans.n_dim is None:
        if not image.has_batch:
            image = image.unsqueeze([])
        if not image.has_channel:
            image = image.standard().unsqueeze({1})
        n_dim = image.n_space # Get the spatial rank.
    else:
        n_dim = trans.n_dim
        if image.n_dim == n_dim:
            if image.has_special:
                print(f"Warning: 'interpolation' trying to transform {image.n_space}+{image.n_special}D image (with batch or channel) by {n_dim}D transformation, auto-removing special dimensions.")
                image.remove_special_()
            image = image.unsqueeze([]).unsqueeze_({})
        elif image.n_dim == n_dim + 1:
            if not image.has_batch:
                if image.has_channel: image = image.unsqueeze([])
                else: image = image.with_batchdim(0).unsqueeze({1})
            elif not image.has_channel:
                image = image.unsqueeze({})
            else:
                print(f"Warning: 'interpolation' trying to transform {image.n_space}+{image.n_special}D image (with batch or channel) by {n_dim}D transformation, auto-removing the channel dimensions.")
                image = image.with_channeldim(None).unsqueeze({})
        elif image.n_dim == n_dim + 2:
            # _channal/batch dimensions used here as they are n_dim when not existed. 
            if image.n_special == 1:
                print(f"Warning: 'interpolation' trying to transform {image.n_space}+1D image (with batch or channel) by {n_dim}D transformation, auto-inserting new special dimension.")
            if not image.has_batch: image.batch_dimension = 0 if image._channel_dimension > 0 else 1
            if not image.has_channel: image.channel_dimension = 0 if image._batch_dimension > 0 else 1
    avouch(image.has_batch and image.has_channel, "Please use batorch tensor of size " +
            "([n_batch], {n_channel/n_feature:optional}, m_1, m_2, ..., m_r) [r=n_dim] for " + 
            f"data to be spatially interpolated, instead of {image.shape}. ")
    if trans is not None:
        avouch(image.n_batch == 1 or trans.n_batch in (None, image.n_batch, 1), "Please use transformation of a " +
            f"suitable n_batch to transform image with batchsize {image.n_batch}, currently {trans.n_batch}.")

    # Deal with the shape of input `image`
    n_batch = image.n_batch
    if n_batch == 1 and trans is not None and trans.n_batch is not None and trans.n_batch > 1: n_batch = trans.n_batch
    if n_batch == 1 and isinstance(target_space, bt.Tensor) and target_space.has_batch and target_space.n_batch > 1: n_batch = target_space.n_batch
    if image.n_batch == 1: image = image.repeated(n_batch, [])
    n_feature = image.n_channel
    size = bt.channel_tensor(image.space).int()
    if n_batch > 1 and not shape_out.has_batch: shape_out = bt.Size([n_batch]) + shape_out
    
    # Deal with input  `target_space`
    if target_space is None:
        scale, *pairs = trans.reshape
        if len(scale) == 1: scale *= n_dim
        target_space = [int(x * y) for x, y in zip(image.space, scale)]
        for p, q in pairs: target_space[p], target_space[q] = target_space[q], target_space[p]
        target_space = tuple(target_space)
        shape_out = shape_out.reset_space(target_space)
    if isinstance(target_space, tuple) and len(target_space) == n_dim: pass
    elif isinstance(target_space, bt.torch.Tensor): pass
    else: raise TypeError(f"Wrong target space for interpolation: {target_space}. ")
    if isinstance(target_space, tuple): 
        # Create a grid X with size ({n_dim}, size_1, size_2, ..., size_r) [r=n_dim].
        X = bt.image_grid(target_space).float() # + bt.channel_tensor([float(a-b)/2 for a, b in zip(image.space, target_space)])
        # Compute the transformed coordinates. Y: ([n_batch], {n_dim}, size_1, size_2, ..., size_r) [r=n_dim].
        if trans is None: trans = Identity()
        Y = trans(X)
        if not Y.has_batch: Y = Y.multiply(n_batch, [])
        if Y.n_batch == 1: Y = Y.repeated(n_batch, [])
        Y = Y.amplify(n_feature, [])
        shape_out = shape_out.reset_space(target_space)
    else:
        target_space = batorch_tensor(target_space)
        if not target_space.has_batch:
            if target_space.size(0) == n_batch and n_batch != n_dim or len([x for x in target_space.shape if x == n_dim]) >= 2:
                target_space.with_batchdim(0)
            else: target_space = target_space.unsqueeze([])
        if not target_space.has_channel:
            if target_space.batch_dimension != 0 and target_space.size(0) == n_dim: target_space.with_channeldim(0)
            elif target_space.batch_dimension != 1 and target_space.size(1) == n_dim: target_space.with_channeldim(1)
            elif target_space.batch_dimension != target_space.n_dim - 1 and target_space.size(-1) == n_dim: target_space.with_channeldim(-1)
        avouch(target_space.has_channel and target_space.n_channel == n_dim, "'target_space' for interpolation should have a channel dimension for coordinates. ")
        Y = target_space.repeated(n_batch // target_space.n_batch, []).amplify(n_feature, [])
        shape_out = shape_out.reset_space(target_space.space)
        
    image = image.mergedims({}, [])
    n_batch = image.n_batch

    if isinstance(fill, str):
        if fill.lower() == 'nearest': background = None
        elif fill.lower() == 'background':
            bk_value = bt.stack([image[(slice(None),) + tuple(g)] for g in (bt.image_grid([2]*n_dim) * bt.channel_tensor(size-1)).flatten().transpose(0,1)], 1).median(1).values
            background = bk_value
        elif fill.lower() == 'zero': background = 0
    else: background = fill
    n_data = Y.flatten().size(-1)

    return image, trans, background, shape_out, n_dim, n_batch, n_feature, n_data, size, Y

@alias("resample")
def interpolation(
        image: bt.Tensor, 
        trans: callable = None, 
        method: str = 'Linear', 
        target_space: tuple = None,
        fill: (str, int, float) = 0,
        derivative: bool = False
    ):
    '''
    Interpolate using backward transformation.
    i.e. Compute the image I s.t. trans(x) = y for x in I and y in input image using interpolation method:
        method = Linear: Bilinear interpolation
        method = Nearest [NO GRADIENT!!!]: Nearest interpolation

    Params:
        image [bt.Tensor]: The target image.
            size: ([n_batch:optional], {n_channel:optional}, m@1, m@2, ..., m@n_dim)
        trans [Function, micomputing.SpatialTransformation]: Transformation function, mapping
            size: ([n_batch:optional], {n_dim}, n@1, n@2, ..., n@n_dim) to ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
        method [str: linear|nearest]: The interpolation method. 
        target_space [tuple, bt.Tensor]:
            Size (tuple) of a target ROI at the center of image. 
            OR Transformed coordinate space (bt.Tensor) of the output image. 
            size: length(n_dim) or ([n_batch:optional], {n_dim:optional}, size@1, size@2, ..., size@r)
        fill [str: nearest|background, int/float(number)]: Indicate the way to fill background outside `Surrounding`. 
        derivative [bool]: Whether to return the gradient. One can omit it when using torch.autograd.

        output [bt.Tensor]: The transformed image.
            size: ([n_batch], {n_channel:optional}, m@1, m@2, ..., m@n_dim)
            or when `target_space` is defined by tensor. 
            size: ([n_batch], size@1, size@2, ..., size@n_dim)
            or the derivative for the interpolation. (if `derivative = True`)
            size: ([n_batch], {n_dim}, size@1, size@2, ..., size@n_dim)

    Examples:
    ----------
    >>> Image = bt.rand(3, 100, 120, 80)
    >>> AM = bt.rand(4, 4)
    >>> AM[3, :] = bt.one_hot(-1, 4)
    >>> interpolation(Image, Affine(AM), method='Linear')
    '''
    # Deal with input `trans` and special dimensions of `image`
    if trans is not None and not trans.backward:
        if hasattr(trans, 'inv'): trans = trans.inv().fake_inv()
        else:
            print("Warning: Forward transformation found in method `interpolation`. Using `interpolation_forward` instead. ")
            return interpolation_forward(image, trans, target_space = target_space, fill = fill)
    
    image, trans, background, shape_out, n_dim, n_batch, n_feature, n_data, size, Y = _init_interpolation(image, trans, fill, target_space)

    iY = bt.floor(Y).long() # Generate the integer part of Y
    if isinstance(method, str):
        if method.lower() == 'bspline':
            if derivative: raise TypeError("No derivatives for bspline interpolations are available so far. Please write it by yourself. ")
            # TODO: FFD
            raise TypeError("Bspline interpolation is not available so far. Please write it by yourself. ")
        if method.lower() == 'linear': fY = Y - iY.float() # The decimal part of Y.
        elif method.lower() == 'nearest': fY = bt.floor(Y - iY.float() + 0.5).long() # The decimal part of Y.
        else: raise TypeError(f"Unrecognized argument 'method': {method}. ")
    else: raise TypeError(f"Unrecognized argument 'method': {method}. ")
    W = bt.stack((1 - fY, fY), 1).view([n_batch], 2, {n_dim}, -1) # ([n_batch], 2, {n_dim}, n_data).

    # Prepare for the output space: n_batch, m_1, ..., m_s
    if derivative: output = bt.zeros([n_batch], {n_dim}, *shape_out.space)
    else: output = bt.zeros(shape_out)
    
    for G in bt.image_grid([2]*n_dim).flatten().transpose(0, 1):
        ## New version of interpolation with indexing.
        Ind = (iY + G).flatten() # ([n_batch], {n_dim}, n_data).
        condition = ((Ind < 0) + (Ind > (size - 1).unsqueeze_to(Ind))).sum({}) == 0 # Compute border: ([n_batch], n_data).
        Ind = bt.min(Ind.clamp(min=0), (size - 1).unsqueeze_to(Ind)) # Find nearest border pix. 
        coeff = bt.where(G.multiply(n_batch, []).multiply(n_data, -1) == 0, W[:, 0], W[:, 1]) # ([n_batch], {n_dim}, n_data)
        batch_dim = bt.batch_tensor(bt.arange(n_batch)).unsqueeze({0}).multiply(n_data, -1) # ({1}, [n_batch], n_data)
        Ind_with_batch = bt.cat(batch_dim, Ind.transpose(0, 1), {}) # ({1 + n_dim}, [n_batch], n_data)
        values = image[Ind_with_batch.split(1, {})].squeeze({}) # ([n_batch], n_data)
        if background is not None: values[~condition] = background
        if not derivative:
            output += (coeff.prod({}) * values).view(shape_out)
        else:
            coeff_tmpmat = coeff.multiply(n_dim, 1) # ([n_batch], n_dim, {n_dim}, n_data)
            coeff_tmpmat[:, bt.arange(n_dim), bt.arange(n_dim)] = 1
            dcoeff = coeff_tmpmat.prod(1) * (G * 2 - 1).float().unsqueeze_to(coeff)
            output += (dcoeff * values.unsqueeze({})).view_as(output)
    bt.torch.cuda.empty_cache()
    if - eps < image.min().item() < 0: output = output.clamp(min=0)
    if 1 < image.max().item() < 1 + eps: output = output.clamp(max=1)
    return output

@alias("resample_forward")
def interpolation_forward(
        image, 
        trans = None, 
        method = '', 
        target_space = None,
        fill = 'zero',
        sigma = 1,
        derivative = False,
    ):
    '''
    Interpolate using forward transformation. It is not yet implemented. 
    i.e. Compute the image I s.t. trans(x) = y for x in input image and y in the output I using interpolation method:
        method = Linear [NO GRADIENT!!!]: Bilinear interpolation
        method = Nearest [NO GRADIENT!!!]: Nearest interpolation

    Params:
        image [bt.Tensor]: The target image.
            size: ([n_batch:optional], {n_channel:optional}, m@1, m@2, ..., m@n_dim)
        trans [Function, micomputing.SpatialTransformation]: Transformation function, mapping
            size: ([n_batch:optional], {n_dim}, n@1, n@2, ..., n@n_dim) to ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
        method [str: linear|nearest]: The interpolation method, not yet available for forward transformations. 
        target_space [tuple]: Size (tuple) of a target ROI at the center of image.
        fill [str: nearest|background, int/float(number)]: Indicate the way to fill background outside `Surrounding`. 
        sigma [int, float]: The size of blurry window. The bigger it is, the more running time is needed and more blurry 
            image is generated. The smaller it is, the function runs faster and the image is sharper, except it may take 
            time to mend the wholes in the image. 
            One can set a smaller number for smooth transformations and a bigger one for folded ones.
        derivative [bool]: Whether to return the gradient. One can omit it when using torch.autograd. Not yet available 
            for forward transformations.

        output [bt.Tensor]: The transformed image.
            size: ([n_batch], {n_channel:optional}, m@1, m@2, ..., m@n_dim)
            or when `target_space` is defined by tensor. 
            size: ([n_batch], size@1, size@2, ..., size@n_dim)
            or the derivative for the interpolation. (if `derivative = True`)
            size: ([n_batch], {n_dim}, size@1, size@2, ..., size@n_dim)

    Examples:
    ----------
    >>> Image = bt.rand(3, 100, 120, 80)
    >>> AM = bt.rand(4, 4)
    >>> AM[3, :] = bt.one_hot(-1, 4)
    >>> interpolation(Image, Affine(AM), method='Linear')
    '''
    # Deal with input `trans` and special dimensions of `image`
    if trans is not None and trans.backward:
        if hasattr(trans, 'inv'): trans = trans.inv().fake_inv()
        else:
            print("Warning: Backward transformation found in method `interpolation_forward`. Using `interpolation` instead. ")
            return interpolation(image, trans, method = method, target_space = target_space, fill = fill)
    
    image, trans, background, shape_out, n_dim, n_batch, n_feature, n_data, size, Y = _init_interpolation(image, trans, fill, None)
    shape_out.reset_space(target_space)
    
    avouch(Y.space == image.space, "Forward interpolation needs 'target_space' the same as input 'image'.")
    n_window = max(int(3 * sigma), 1)
    iY = bt.floor(Y).long() # Generate the integer part of Y
    indices = []; weights = []; values = []
    for G in bt.image_grid([n_window]*n_dim).flatten().transpose(0, 1):
        gY = iY + G
        gY = bt.min(gY, bt.channel_tensor(Y.space).expand_to(gY) - 1).clamp(min=0)
        indices.append(gY.flatten(2)) # ([n_batch], {n_dim}, n_data)
        W = bt.exp(- ((Y - gY) ** 2).sum({}) / 2 / (sigma ** 2)) / (sigma ** n_dim) / (math.sqrt(2 * math.pi) ** n_dim) # ([n_batch], *n_data)
        weights.append(W.flatten(1)) # ([n_batch], n_data)
        values.append((W * image).flatten(1)) # ([n_batch], n_data)
    # ([n_batch], {1 + n_dim}, n_data x n_window ^ n_dim)
    Mindices = bt.cat(bt.arange(n_batch).with_batchdim(0).unsqueeze({}).multiply(len(indices) * n_data, -1), bt.cat(indices, -1), {})
    # ([n_batch], n_data x n_window ^ n_dim)
    Mweights = bt.cat(weights, -1)
    # ([n_batch], n_data x n_window ^ n_dim)
    Mvalues = bt.cat(values, -1)
    # ({1 + n_hist}, n_batch x n_data x 4 ^ n_hist)
    indices = Mindices.transpose(0, 1).flatten(1)
    # (n_batch x n_data x 4 ^ n_hist,)
    weights = Mweights.flatten(0)
    # (n_batch x n_data x 4 ^ n_hist,)
    values = Mvalues.flatten(0)
    
    if Version(bt.torch.__version__) >= Version("2"):
        creator = bt.torch.sparse_coo_tensor
    elif indices.device.type == 'cpu': creator = bt.torch.sparse.FloatTensor
    else: creator = bt.torch.cuda.sparse.FloatTensor

    collected_intensities = creator(indices, values, (n_batch,) + shape_out.space).to_dense()
    collected_intensities = bt.Tensor(collected_intensities, batch_dim=0) # ([n_batch], *n_data)
    collected_weights = creator(indices, weights, (n_batch,) + shape_out.space).to_dense()
    collected_weights = bt.Tensor(collected_weights, batch_dim=0) # ([n_batch], *n_data)
    
    neighbor_window = bt.ones((5,) * n_dim)
    ret = bt.where(collected_weights > 0, 
        bt.divide(collected_intensities, collected_weights, 0.),
        bt.divide(bt.conv(collected_intensities, neighbor_window), bt.conv(collected_weights, neighbor_window), 0.)
    )
    mask = ret == 0
    while True:
        values = bt.conv(mask.float() * ret, neighbor_window)
        weights = bt.conv(mask.float(), neighbor_window).int()
        ret = bt.where(mask, bt.where(weights > 0, values / weights, ret), ret)
        new_mask = ret == 0
        if (new_mask != mask).int().sum().sum().item() / mask.nele < 0.05: break
        mask = new_mask
    return ret.view(shape_out)

############ Image Transformations ############

class ImageTransformation(Transformation):
        
    def __call__(self, X):
        '''
        X [bt.Tensor]: Image to be transformed.
            size: ([n_batch:optional], {n_channel:optional}, n@1, n@2, ..., n@n_dim)
        output [bt.Tensor]: The transformed image.
            size: ([n_batch], {n_channel}, n@1, n@2, ..., n@n_dim)
        '''
        X = batorch_tensor(X)
        if self.n_dim is None:
            if not X.has_batch: X = X.unsqueeze([])
            if not X.has_channel: X = X.standard().unsqueeze({1})
        else:
            if X.n_dim == self.n_dim: X = X.remove_special_().unsqueeze([]).unsqueeze({1})
            elif X.n_dim == self.n_dim + 1:
                if X.has_batch: X.channel_dimension = None; X = X.unsqueeze({0 if X.batch_dimension > 0 else 1})
                elif X.has_channel: X = X.unsqueeze([])
                else: X = X.batch_dimension_(0).unsqueeze({1})
            elif X.n_dim == self.n_dim + 2:
                # _channal/batch dimensions used here as they are n_dim when not existed. 
                if not X.has_batch: X.batch_dimension = 0 if X._channel_dimension > 0 else 1
                if not X.has_channel: X.channel_dimension = 0 if X._batch_dimension > 0 else 1
        avouch(X.has_batch and X.has_channel, f"Please use batorch tensor of size \
            ([n_batch], {{n_channel/n_feature:optional}}, m_1, m_2, ..., m_r) [r=n_dim] for \
                {self.__class__.__name__.split('.')[-1]} Transformation, instead of {X.shape}. ")
        return X.clone()

class Normalize(ImageTransformation):
    def __init__(self, *_range):
        '''
        Normalize the intensity of an image.
        
        Params:
            _range = (low, high) [int or float or bt.Tensor]: The lowest (and highest) intensity. 
                size: length(2) or ([n_batch], {2})
            
        When it is called:
            X [bt.Tensor]: Image to be transformed.
                size: ([n_batch:optional], {n_channel:optional}, n@1, n@2, ..., n@n_dim)
            output [bt.Tensor]: The transformed image.
                size: ([n_batch], {n_channel}, n@1, n@2, ..., n@n_dim)
        '''
        if len(_range) == 0: _range = None
        elif len(_range) == 1 and isinstance(_range[0], (list, tuple)): _range = batorch_tensor(list(_range[0]))
        elif len(_range) == 1 and isinstance(_range[0], bt.Tensor): _range = _range[0]
        elif len(_range) == 1: _range = bt.channel_tensor((0, _range))
        elif len(_range) == 2: _range = bt.channel_tensor(_range)
        else: raise TypeError(f"Invalid range for Normalize: {_range}. ")
        if _range is None: pass
        else:
            if _range.n_dim < 2: _range = _range.unsqueeze([])
            if not _range.has_batch: _range.batch_dimension = 0
            if not _range.has_channel: _range.channel_dimension = 1
            avouch(_range.has_batch and _range.has_channel, f"Please use batorch tensor of size \
                ([n_batch:optional], {{2}}) for Normalizing parameters, instead of {_range.shape}. ")
        super().__init__(_range)
        self.range = _range
        self._range = None

    def __call__(self, X):
        X = super().__call__(X)
        _range = self.range
        if _range is None:
            _range = bt.quantile(X.flatten().float(), batorch_tensor([0.025, 0.975]), axis=-1).movedim(0, -1).special_from_(X)
            self._range = _range
        return ((X - _range[..., 0]) / (_range[..., 1] - _range[..., 0])).clamp(0., 1.)
    
    def inv(self):
        if self.range is not None: _range = self.range
        elif self._range is not None: _range = self._range
        else: return Normalize(None)
        den = _range[..., 1] - _range[..., 0]
        _range = bt.stack(-_range[..., 0], 1-_range[..., 0], -1) / den
        return Normalize(_range)

############# Supporting Functions ############

def Bspline(i, U):
    """
    Cubic B-spline function. 
    Note: As long as i and U have the same size, any shape of tensors would do.
    
    Params:
        i [bt.Tensor]: the index of segment function of B-spline.
            The value of each element can be chosen in (-1, 0, 1, 2). 
        U [bt.Tensor]: the decimal argument of B-spline function. It should be within range [0, 1).
    """
    i = batorch_tensor(i); U = batorch_tensor(U)
    return (
        bt.where(i == -1, (1 - U) ** 3 / 6,
        bt.where(i == 0, U ** 3 / 2 - U * U + 2 / 3,
        bt.where(i == 1, (- 3 * U ** 3 + 3 * U * U + 3 * U + 1) / 6,
        bt.where(i == 2, U ** 3 / 6,
        bt.zeros_like(U)))))
    )

def dBspline(i, U):
    """
    The derivative of B-spline function, with respect to U. 
    Note: As long as i and U have the same size, any shape of tensors would do.
    
    Params:
        i [bt.Tensor]: the index of segment function of B-spline.
            The value of each element can be chosen in (-1, 0, 1, 2). 
        U [bt.Tensor]: the decimal argument of B-spline function. It should be within range [0, 1).
    """
    i = batorch_tensor(i); U = batorch_tensor(U)
    return (
        bt.where(i == -1, - 3 * (1 - U) ** 2 / 6,
        bt.where(i == 0, 3 * U ** 2 / 2 - 2 * U,
        bt.where(i == 1, (- 3 * U ** 2 + 2 * U + 1) / 2,
        bt.where(i == 2, 3 * U ** 2 / 6,
        bt.zeros_like(U)))))
    )

def fBspline(c, x):
    c = batorch_tensor(c); x = batorch_tensor(x)
    d = x - c
    return (
        bt.where((-2 <= d) * (d < -1), d ** 3 + 6 * d ** 2 + 12 * d + 8,
        bt.where((-1 <= d) * (d < 0), - 3 * d ** 3 - 6 * d ** 2 + 4,
        bt.where((0 <= d) * (d < 1), 3 * d ** 3 - 6 * d ** 2 + 4,
        bt.where((1 <= d) * (d < 2), - d ** 3 + 6 * d ** 2 - 12 * d + 8,
        bt.zeros_like(d))))) / 6
    )

def Affine2D2Matrix(params):
    """
    t1, t2, θ, s1, s2, ρ1, ρ2 in size: ([n_batch], {7})
    t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2 in size: ([n_batch], {9})
    output in size: ([n_batch], 3, 3)
    """
    params = batorch_tensor(params)
    if params.n_dim <= 1 and not params.has_batch: params = params.unsqueeze([])
    if params.n_dim <= 1 and not params.has_channel: params = params.unsqueeze({1})
    if params.n_dim == 2 and not params.has_batch: params.batch_dimension = 0
    if params.n_dim == 2 and not params.has_channel: params.channel_dimension = 1
    avouch(params.has_batch, f"Please use batorch tensor of size ([n_batch], {7 or 9}) \
        for Affine parameters, instead of {params.shape}. ")
    n_batch = params.n_batch
    if params.size(1) == 7:
        t1, t2, θ, s1, s2, ρ1, ρ2 = params.split()
        c1 = bt.zeros([n_batch], 1); c2 = bt.zeros([n_batch], 1)
    if params.size(1) == 9:
        t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2 = params.split()
    a = (ρ1 * ρ2 + 1) * s1 * bt.cos(θ) + ρ1 * s2 * bt.sin(θ)
    b = - (ρ1 * ρ2 + 1) * s1 * bt.sin(θ) + ρ1 * s2 * bt.cos(θ)
    c = ρ2 * s1 * bt.cos(θ) + s2 * bt.sin(θ)
    d = - ρ2 * s1 * bt.sin(θ) + s2 * bt.cos(θ)
    return bt.cat(
        bt.cat((a, b, t1 - a * c1 - b * c2 + c1, c, d, t2 - c * c1 - d * c2 + c2), {}).view([n_batch], 2, 3), 
        bt.one_hot(-1, 3).multiply(n_batch, []).view([n_batch], 1, 3), 1
    )

def Quaterns2Matrix(params):
    """
        Quatern: qb, qc, qd, px, py, pz in size: ([n_batch], {6})
        Matrix: ([n_batch], 4, 4)
    """
    params = batorch_tensor(params)
    if params.n_dim <= 1 and not params.has_batch: params = params.unsqueeze([])
    if params.n_dim <= 1 and not params.has_channel: params = params.unsqueeze({1})
    if params.n_dim == 2 and not params.has_batch: params.batch_dimension = 0
    if params.n_dim == 2 and not params.has_channel: params.channel_dimension = 1
    avouch(params.n_dim == 2 and params.has_batch and params.has_channel, 
           f"Please use batorch tensor of size ([n_batch], {{6}}) for Affine parameters, instead of {params.shape}. ")
    n_batch = params.n_batch
    b, c, d, x, y, z = params.split()
    a = bt.sqrt((1-b*b-c*c-d*d).clamp(0))
    R11 = a*a+b*b-c*c-d*d
    R12 = 2*b*c-2*a*d
    R13 = 2*b*d+2*a*c
    R21 = 2*b*c+2*a*d
    R22 = a*a+c*c-b*b-d*d
    R23 = 2*c*d-2*a*b
    R31 = 2*b*d-2*a*c
    R32 = 2*c*d+2*a*b
    R33 = a*a+d*d-c*c-b*b
    return bt.cat(
        bt.cat((R11, R12, R13, x, R21, R22, R23, y, R31, R32, R33, z), 1).view([n_batch], 3, 4),
        bt.one_hot(-1, 4).multiply(n_batch, []).view([n_batch], 1, 4), 1
    )

def Matrix2Quaterns(params):
    """
        Matrix: ([n_batch], 4, 4)
        Quatern: qb, qc, qd, px, py, pz in size: ([n_batch], {6})
    """
    params = batorch_tensor(params)
    if params.n_dim <= 2 and not params.has_batch: params = params.unsqueeze([])
    if params.n_dim == 3 and not params.has_batch: params.batch_dimension = 0
    if params.n_dim == 3 and params.has_channel: params.channel_dimension = None
    avouch(params.n_dim == 3 and params.has_batch and not params.has_channel, 
           f"Please use batorch tensor of size ([n_batch], 4, 4) for Affine matrix, instead of {params.shape}. ")
    n_batch = params.n_batch
    x, y, z = params[..., :3, -1].channel_dim_(1).split(1, 1)
    a2 = (bt.diag(params).sum().unsqueeze({}) + 1) / 4
    a = bt.sqrt(a2)
    b2 = a2 - (params[..., 1, 1] + params[..., 2, 2]) / 2
    c2 = a2 - (params[..., 2, 2] + params[..., 0, 0]) / 2
    d2 = a2 - (params[..., 0, 0] + params[..., 1, 1]) / 2
    D = params - params.T
    b = bt.sign(D[..., 2, 1]) * bt.sqrt(b2)
    c = - bt.sign(D[..., 2, 0]) * bt.sqrt(c2)
    d = bt.sign(D[..., 1, 0]) * bt.sqrt(d2)
    return bt.cat(b, c, d, x, y, z, {})
