
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File of functions for medical images.",
    help = "Use `from micomputing import *`."
)

__all__ = """
    reorient
    rescale
    dilate
    blur
    bending
    distance_map
    registration
    local_prior
    center_of_gravity
""".split()

import os
import math
import ctypes
import numpy as np
import SimpleITK as sitk
from .stdio import IMG, affine2spacing, affine2orient
from .trans import *
from .metrics import *

eps = 1e-6

with __info__:
    from pyoverload import overload, array
    from pycamia import alias, avouch, to_tuple, to_list, prod, Version, Path
    from batorch import torch as torch
    import batorch as bt

for trial in range(2):
    try:
        dll = ctypes.cdll.LoadLibrary(Path(__file__).parent/'micfunctions.so')
        distance_map_func = dll.distance_map
        distance_map_func.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS"),
                                      np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS"),
                                      ctypes.c_int,
                                      np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS"),
                                      np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS")]
        break
    except OSError: # Try to compile dll or warn the user
        if trial == 0: os.system("g++ -o micfunctions.so -shared -fPIC micfuncions.cpp")
        else: print("Unable to compile C++ DLL: micfunctions, 'micomputing.dilate' is not functional, please install command 'g++' or use 'micomputing.dilate_sitk' instead.")

@overload
@bt.batorch_wrapper
def reorient(data: array, from_orient='LPI', to_orient='RAI'):
    axis = {'L':'LR', 'R':'LR', 'A':'AP', 'P':'AP', 'I':'IS', 'S':'IS'}
    orient_axis = [axis[i] for i in from_orient]
    permutation = [orient_axis.index(axis[i]) for i in to_orient]
    if len(permutation) < len(orient_axis):
        permutation.extend(list(set(range(len(orient_axis))) - set(permutation)))
    new_orient = [from_orient[i] for i in permutation]
    data = bt.permute_space(data, *permutation)
    if data.has_channel: data = data.pick(bt.tensor(permutation, device=data.device), []).with_channel_dim(data.channel_dim)
    flip_dims = [i + data.space_start for i, (a, b) in enumerate(zip(new_orient, to_orient)) if a != b]
    if flip_dims:
        data = data.flip(*flip_dims)
        if data.has_channel: data = bt.channel_tensor([1 if a == b else -1 for a, b in zip(new_orient, to_orient)], device=data.device) * data
    return data

@overload
def reorient(data: IMG, to_orient='RAI'):
    return data.reorient(to_orient)

@bt.batorch_wrapper
def rescale(data: bt.Tensor, scaling):
    return interpolation(data, Rescale(scaling).inv())

@bt.batorch_wrapper
def rescale_to(data: bt.Tensor, size):
    return bt.crop_as(rescale(data, tuple(x // y for x, y in zip(size, data.space))), size)

# @bt.batorch_wrapper
# def reflect(data: bt.Tensor, *dims):
#     if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
#     ind = [slice(None),] * data.ndim
#     for d in dims:
#         if d < 0: d += data.ndim
#         ind[d] = bt.arange(data.size(d)-1, -1, -1).to(data.device)
#         data = data[ind]
#         ind[d] = slice(None)
#     return data

@bt.batorch_wrapper
def blur(data: bt.Tensor, kernel_size = 3):
    """
    blur(data, kernel_size: int) -> Tensor

    Blur the image by Gaussian kernel. 
    """
    kernel = bt.gaussian_kernel(data.n_space_dim, kernel_size=kernel_size)
    return bt.conv(data, kernel)
    # conv = eval("bt.nn.functional.conv%dd"%data.n_space_dim)
    # cdata = data.clone()
    # if not cdata.has_batch: cdata = cdata.unsqueeze([])
    # if cdata.has_channel: cdata = cdata.mergedims({}, [])
    # result = conv(cdata.unsqueeze(1), bt.unsqueeze(kernel, 0, 1), padding=kernel_size // 2).squeeze(1)
    # result = bt.crop_as(result, cdata.shape)
    # if data.has_channel: result = result.view_as(data)
    # return result

@bt.batorch_wrapper
def center_of_gravity(image: bt.Tensor):
    image = image.float().normalize()
    n_batch = None
    if not image.has_batch: image = image.unsqueeze({})
    if image.has_channel: n_batch = image.n_batch; image = image.mergedims([], {})
    standard_grid = bt.image_grid(*image.space, device=image.device).unsqueeze({}).float()
    center = bt.where(image.sum(...).abs() < bt.eps, standard_grid.mean(...), (image * standard_grid).sum(...) / image.sum(...)).with_channel_dim(None)
    if n_batch is not None: return center.splitdim({}, {n_batch}, [-1])
    return center

# @bt.batorch_wrapper
# def distance_map(masks, spacing = 1):
#     """
#     The signed distance map of masks (outside positive) [NO Gradient!]. 

#     Args:
#         masks [bt.tensor]: ([n_batch:optional], {n_feature:optional}, n@1, ..., n@n_dim)

#     Returns: ([n_batch:optional], {n_feature:optional}, n@1, ..., n@n_dim)
#     """
#     spacing = to_tuple(spacing)
#     if len(spacing) == 1: spacing *= masks.n_space_dim
#     squeeze_batch, squeeze_channel = False, False
#     if not masks.has_batch:
#         squeeze_batch = True
#         masks = masks.unsqueeze([])
#     if not masks.has_channel:
#         squeeze_channel = True
#         masks = masks.unsqueeze({})
#     n_batch = masks.n_batch
#     n_feature = masks.n_channel
#     output = bt.zeros_like(masks).float()
#     masks = masks.detach().cpu()
#     for b in range(n_batch):
#         for j in range(n_feature):
#             mask_image = sitk.GetImageFromArray(masks[b, j].numpy().astype(np.int32), isVector = False)
#             mask_image.SetSpacing(spacing)
#             dis_map = sitk.GetArrayViewFromImage(sitk.SignedMaurerDistanceMap(mask_image, insideIsPositive = False, squaredDistance = False, useImageSpacing = True))
#             dis_map = np.array(dis_map).astype(np.float)
#             output[b, j] = bt.tensor(dis_map)
#     if squeeze_batch: output.squeeze([])
#     if squeeze_channel: output.squeeze({})
#     return output

@alias("distance_map_cpp")
@bt.batorch_wrapper
def distance_map(masks: bt.Tensor, spacing = 1):
    """
    The signed distance map of masks (outside positive). 

    Args:
        masks (bt.tensor): ({n_batch:optional}, [n_feature:optional], n_1, ..., n_r) [r=n_dim]

    Returns: in shape ({n_batch:optional}, [n_feature:optional], n_1, ..., n_r) [r=n_dim]
    """
    n_dim = masks.n_space_dim
    spacing = to_tuple(spacing)
    if len(spacing) == 1: spacing *= n_dim
    squeeze_batch, squeeze_channel = False, False
    if not masks.has_batch:
        squeeze_batch = True
        masks = masks.unsqueeze({})
    if not masks.has_channel:
        squeeze_channel = True
        masks = masks.unsqueeze([])
    n_batch = masks.n_batch
    n_feature = masks.n_channel
    size = masks.space
    n_data = prod(size)
    mask_sent = masks.mergedims([], {}).detach().cpu().numpy().flatten().astype(np.int32)
    dismap_get = np.zeros_like(mask_sent).flatten().astype(np.float32)
    size_in = np.array((n_batch * n_feature,) + size).astype(np.int32)
    spacing_in = np.array(spacing).astype(np.float32)
    
    distance_map_func(mask_sent, dismap_get, n_dim + 1, size_in, spacing_in)
    dismap = dismap_get.reshape(size_in)
    # mask_sent_ = np.ascontiguousarray(mask_sent, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # dismap_get_ = np.ascontiguousarray(dismap_get, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # size_in_ = np.ascontiguousarray(size_in, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # spacing_in_ = np.ascontiguousarray(spacing_in, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # dll.distance_map(mask_sent_, dismap_get_, n_dim + 1, size_in_, spacing_in_)
    # casted_dismap_buffer = ctypes.cast(dismap_get_, ctypes.POINTER(ctypes.c_int * (n_batch * n_feature * n_data))).contents
    # dismap = np.frombuffer(casted_dismap_buffer, dtype=np.float32, count=n_batch * n_feature * n_data).reshape(size_in)
    
    output = bt.tensor(dismap).splitdim(0, {n_batch}, [n_feature])
    if squeeze_batch: output.squeeze({})
    if squeeze_channel: output.squeeze([])
    return output

from math import sqrt
@bt.batorch_wrapper
def distance_map_python(masks: bt.Tensor, spacing=1):
    n_dim = masks.n_space_dim
    spacing = to_tuple(spacing)
    if len(spacing) == 1: spacing *= n_dim
    squeeze_batch, squeeze_channel = False, False
    if not masks.has_batch:
        squeeze_batch = True
        masks = masks.unsqueeze({})
    if not masks.has_channel:
        squeeze_channel = True
        masks = masks.unsqueeze([])
    n_batch = masks.n_batch
    n_feature = masks.n_channel
    size = masks.space
    n_data = prod(size)
    anchors = {}
    inf = int(2 * max(size) * sqrt(n_dim))
    dismap = inf * bt.ones_like(masks) - 2 * inf * masks
    for b in range(n_batch):
        for f in range(n_feature):
            for order in range(6):
                coords = bt.image_grid(size).flatten(1)
                if order % 2 == 1: coords = coords[..., ::-1]
                visited = bt.zeros_like(masks[b][f])
                for x in coords.split(1,1):
                    x = tuple(x.squeeze().tolist())
                    for d in range(n_dim):
                        for p in range(-1, 2, 2):
                            y = list(x).copy()
                            y[d] += p
                            y = tuple(y)
                            if not 0 <= y[d] < size[d]: continue
                            if not visited[y]: continue
                            x_in_mask = masks[b][f][x] != 0
                            y_in_mask = masks[b][f][y] != 0
                            if x_in_mask and not y_in_mask or not x_in_mask and y_in_mask:
                                anchors[x] = y
                                neighbor_dis = 1 if not x_in_mask else -1
                            neighbor_anchor = anchors.get(y, None)
                            if neighbor_anchor is None: continue
                            neighbor_dis = sqrt(sum((a - b) ** 2 for a, b in zip(x, neighbor_anchor)))
                            if x_in_mask:
                                if -neighbor_dis > dismap[b][f][x]:
                                    dismap[b][f][x] = -neighbor_dis
                                    anchors[x] = y
                            elif neighbor_dis < dismap[b][f][x]:
                                dismap[b][f][x] = neighbor_dis
                                anchors[x] = y
                    visited[x] = 1
    return dismap
    

@bt.batorch_wrapper
def dilate(mask, distance = 0, spacing = 1, class_labels = None):
    """mask: ({n_batch}, n_1, ..., n_r) [r=n_dim]"""
    if not class_labels: class_labels = sorted(mask.unique().tolist())[1:]
    if len(class_labels) == 0: return mask
    spacing = to_tuple(spacing)
    if len(spacing) == 1: spacing *= mask.n_space_dim
    squeeze_batch = False
    if not mask.has_batch:
        squeeze_batch = True
        mask = mask.unsqueeze({})
    min_values = distance_map_cpp(bt.stack([mask == l for l in class_labels], []), spacing=spacing).min([])
    res = bt.tensor(class_labels)[min_values.indices].float() * (min_values <= distance).float()
    if squeeze_batch: res.squeeze_({})
    return res

@bt.batorch_wrapper
def dilate_sitk(mask, distance = 0, spacing = 1, class_labels = None):
    """mask: ({n_batch}, n_1, ..., n_r) [r=n_dim]"""
    if not class_labels: class_labels = sorted(mask.unique().tolist())[1:]
    if len(class_labels) == 0: return mask
    spacing = to_tuple(spacing)
    if len(spacing) == 1: spacing *= mask.n_space_dim
    squeeze_batch = False
    if not mask.has_batch:
        squeeze_batch = True
        mask = mask.unsqueeze({})
    min_values, indices = distance_map(bt.stack([mask == l for l in class_labels], {}), spacing=spacing).min({})
    res = bt.tensor(class_labels)[indices].float() * (min_values <= distance).float()
    if squeeze_batch: res.squeeze_({})
    return res

@bt.batorch_wrapper
def dilate_python(mask, distance = 0, spacing = 1, class_labels = None):
    """mask: ({n_batch}, n_1, ..., n_r) [r=n_dim]"""
    if not class_labels: class_labels = sorted(mask.unique().tolist())[1:]
    if len(class_labels) == 0: return mask
    spacing = to_tuple(spacing)
    if len(spacing) == 1: spacing *= mask.n_space_dim
    squeeze_batch = False
    if not mask.has_batch:
        squeeze_batch = True
        mask = mask.unsqueeze({})
    min_values, indices = distance_map_python(bt.stack([mask == l for l in class_labels], {}), spacing=spacing).min({})
    res = bt.tensor(class_labels)[indices].float() * (min_values <= distance).float()
    if squeeze_batch: res.squeeze_({})
    return res
    # distance_map(mask, spacing)
    # n_batch = mask.n_batch
    # dilated = bt.zeros_like(mask)
    # masks = masks.cpu().detach()
    # for b in range(n_batch):
    #     mask_image = sitk.GetImageFromArray(masks[b].numpy().astype(np.int), isVector = False)
    #     mask_image.SetSpacing(spacing)
    #     canvas = np.zeros((len(class_labels),) + mask.space)
    #     i = 0
    #     for l in class_labels:
    #         dis_map = sitk.GetArrayViewFromImage(sitk.SignedMaurerDistanceMap(mask_image == l, insideIsPositive = False, squaredDistance = False, useImageSpacing = True))
    #         dis_map = np.array(dis_map).astype(np.float)
    #         if l > 0:
    #             canvas[i] = np.where(dis_map <= distance, dis_map, np.inf)
    #             i += 1
    #     output = np.zeros(mask.space).astype(np.int)
    #     label_map = np.argmin(canvas, 0)
    #     i = 0
    #     for l in class_labels:
    #         if l == 0: continue
    #         output[label_map == i] = l
    #         i += 1
    #     output[np.min(canvas, 0) == np.inf] = 0
    #     dilated[b] = bt.tensor(output)
    # if squeeze_batch: dilated.squeeze([])
    # return dilated

@bt.batorch_wrapper
def bending(disp: bt.Tensor, mask=None):
    """disp: ({n_batch}, [n_dim], n_1, ..., n_r) [r=n_dim]"""
    avouch(disp.has_channel and disp.has_batch)
    n_dim = disp.n_channel
    if mask is None: mask = bt.ones({disp.n_batch}, *disp.space)
    Jac = bt.grad_image(disp) # ({n_batch}, [n_dim, n_dim], n_1-dx, n_2-dx, ..., n_r-dx) [r=n_dim]
    Hes = bt.grad_image(Jac.flatten([])) # ({n_batch}, [n_dim, n_dim x n_dim], n_1-2dx, n_2-2dx, ..., n_r-2dx) [r=n_dim]
    size = Hes.shape[3:]
    Hes = Hes.splitdim([1], [n_dim, n_dim])
    masked_Hes_sum = (((Hes ** 2).sum([0], [1]) + eps).sqrt().mean([]) * bt.crop_as(mask, size)).sum(...)
    mask_volume = mask.sum(...)
    return bt.divide(masked_Hes_sum, mask_volume, 0.0)

class local_prior:
    
    def __init__(self, method = "isometry", spacing = 1, **kwargs):
        super().__init__()
        self.method = method
        self.spacing_source = kwargs.get('spacing_source', spacing)
        self.spacing_target = kwargs.get('spacing_target', spacing)
        self.source_affine = kwargs.get('source_affine')
        self.target_affine = kwargs.get('target_affine')
        self.R = self.b = None
    
    def __call__(self, trans, source_mask = None, target_mask = None, n_label_source_per_batch = None, n_label_target_per_batch = None, **kwargs):
        '''
        trans [Function or micomputing.Transformation]: the transformation from target (fixed) to source (moving). 
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) to ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        source_mask: ({n_label_source}, n_1, n_2, ..., n_r) [r=n_dim]
        target_mask: ({n_label_target}, n_1, n_2, ..., n_r) [r=n_dim]
        OR: 
        source_mask: ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        target_mask: ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        '''
        for k, v in kwargs.items(): setattr(self, k, v)
        if source_mask is not None and not isinstance(source_mask, bt.Tensor): source_mask = bt.tensor(source_mask)
        if target_mask is not None and not isinstance(target_mask, bt.Tensor): target_mask = bt.tensor(target_mask)
        avouch(source_mask is None or source_mask.has_batch, "Please input source/target masks with batch for 'rigidity'. ")
        avouch(target_mask is None or target_mask.has_batch, "Please input source/target masks with batch for 'rigidity'. ")
        if source_mask is not None and source_mask.has_channel:
            n_batch = source_mask.n_batch; n_label_source_per_batch = [source_mask.n_channel] * n_batch;
            source_mask = source_mask.mergedims({}, [])
        if target_mask is not None and target_mask.has_channel:
            n_batch = target_mask.n_batch; n_label_target_per_batch = [target_mask.n_channel] * n_batch;
            target_mask = target_mask.mergedims({}, [])
        if source_mask is None and target_mask is None: return bt.tensor(0.0)
        if source_mask is not None and target_mask is None: n_label_source = source_mask.n_batch; size = source_mask.space; n_label_target = 0
        elif source_mask is None and target_mask is not None: n_label_target = target_mask.n_batch; size = target_mask.space; n_label_source = 0
        else:
            n_label_source = source_mask.n_batch; size = source_mask.space
            n_label_target = target_mask.n_batch; size_ = target_mask.space
            if size != size_: raise TypeError("Wrong size of inputs (mismatch of spaces of source and target masks) for rigidity constraint. ")
        n_label = n_label_source + n_label_target
        n_dim = len(size)
        displacement = trans.toDDF(*size)
        n_batch, n_dim_, size_ = displacement.n_batch, displacement.n_channel, displacement.space
        if source_mask is not None and n_label_source_per_batch is None: n_label_source_per_batch = [1] * n_batch
        if target_mask is not None and n_label_target_per_batch is None: n_label_target_per_batch = [1] * n_batch
        n_label_source_per_batch = to_list(n_label_source_per_batch)
        n_label_target_per_batch = to_list(n_label_target_per_batch)
        if n_dim == n_dim_ and size == size_: pass
        elif not n_label_source_per_batch or len(n_label_source_per_batch) == n_batch and sum(n_label_source_per_batch) == n_label_source: pass
        elif not n_label_target_per_batch or len(n_label_target_per_batch) == n_batch and sum(n_label_target_per_batch) == n_label_target: pass
        else: raise TypeError("Wrong size of transformation for rigidity constraint. ")
        standard_grid = bt.image_grid(size).multiply(n_batch, []).float()
        
        if isinstance(self.spacing_source, int): self.spacing_source = (self.spacing_source,) * n_dim
        if isinstance(self.spacing_target, int): self.spacing_target = (self.spacing_target,) * n_dim
        avouch(len(self.spacing_source) == len(self.spacing_target) == n_dim, 
               f"Wrong spacing dimension for local_prior('{self.method}'):\
               source spacing of {len(self.spacing_source)}D and target spacing \
               of {len(self.spacing_target)}D and transformation of {n_dim}D.")

        if source_mask is not None:
            index = bt.zeros([n_label_source])
            idones = bt.cumsum(bt.tensor(n_label_source_per_batch), 0).long()
            index[idones[idones < n_label_source]] = 1
            index = bt.cumsum(index, 0).long()

            if self.target_affine: target_world_mesh = self.target_affine(standard_grid)
            else: target_world_mesh = standard_grid * bt.channel_tensor(self.spacing_target)
            image_trans = trans
            if self.target_affine: image_trans = trans @ self.target_affine
            if self.source_affine: image_trans = self.source_affine.inv() @ image_trans
            source_world_mesh = trans(target_world_mesh)
            X_source = source_world_mesh[index]
            Y_source = target_world_mesh[index]
            g_source = interpolation(source_mask, trans[index], method='Nearest')
        else:
            X_source = bt.tensor([])
            Y_source = bt.tensor([])
            g_source = bt.tensor([])

        if target_mask is not None:
            index = bt.zeros([n_label_target])
            idones = bt.cumsum(bt.tensor(n_label_target_per_batch), 0).long()
            index[idones[idones < n_label_target]] = 1
            index = bt.cumsum(index, 0).long()
            
            if self.target_affine: target_world_mesh = self.target_affine(standard_grid)
            else: target_world_mesh = standard_grid * bt.channel_tensor(self.spacing_target)
            source_world_mesh = trans(target_world_mesh)
            X_target = target_world_mesh[index]
            Y_target = source_world_mesh[index]
            g_target = target_mask
        else:
            X_target = bt.tensor([])
            Y_target = bt.tensor([])
            g_target = bt.tensor([])

        X = bt.cat(X_source, X_target, [])
        Y = bt.cat(Y_source, Y_target, [])
        g = bt.cat(g_source, g_target, [])
        '''
        X: ([n_label], {n_dim}, n@1, n@2, ..., n@n_dim)
        Y: ([n_label], {n_dim}, n@1, n@2, ..., n@n_dim)
        g: ([n_label], n@1, n@2, ..., n@n_dim)
        '''
        self.g_mask = g
        if self.method != "FLIRT":
            self.init_coords = X.clone().flatten().with_channel_dim(None), Y.clone().flatten().with_channel_dim(None)
            def mask_encode(X, g):
                '''
                X: ([n_batch], {n_dim}, n@1, ..., n@n_dim)
                g >= 0, ([n_batch], n@1, ..., n@n_dim)
                return X diag g^{1/2} (I - (g^{1/2} g^T^{1/2}) / (1^T g))
                '''
                sqrt_g = bt.sqrt(g.flatten().unsqueeze({})) # ([n_batch], {1}, n_data)
                Y = X.flatten(2) * sqrt_g # ([n_batch], {n_dim}, n_data)
                Z = (Y * sqrt_g).sum(-1, keepdim = True) * sqrt_g
                return (Y - bt.divide(Z, g.sum(), 0.0)).view_as(X)
            X = mask_encode(X, g)
            Y = mask_encode(Y, g)
            self.Aux = None
            self.method = self.method.replace(' ', '_')
            if not hasattr(self, "local_" + self.method):
                raise TypeError(f"Unknown method ({self.method}) for local prior constraint. All availables are: affinity, rigidity, affinity with rigid penalty, isometry, rotation3D, FLIRT, Jacobian. ")
            Del = getattr(self, "local_" + self.method)(X, Y)
            cons = 0
            if Del is not None: cons += bt.divide(bt.norm2(Del) / Del.size(1), g.sum(), 0.0)
            if self.Aux is not None: cons += self.Aux
        else:
            if source_mask is not None: raise TypeError("'FLIRT' method does not receive masks on the source/moving image.")
            cons = self.local_FLIRT(X, Y, g)
        return cons.mean()
    
    def set_matrix(self, R):
        if R is None: return
        self.R = R
        X, Y = self.init_coords
        self.b = (((Y - R@X) * self.g_mask.flatten().unsqueeze(1)).sum(-1) / self.g_mask.sum(-1, keepdim=True)).with_channel_dim(1)
        
    def with_param(self, key, value):
        setattr(self, key, value)
        return self

    def local_affinity(self, X, Y):
        X = X.flatten().with_channel_dim(None); Y = Y.flatten().with_channel_dim(None)
        R = Y @ X.T @ bt.inv(X @ X.T)
        self.set_matrix(R)
        return Y - R @ X

    def local_affinity_with_rigid_penalty(self, X, Y):
        X = X.flatten().with_channel_dim(None); Y = Y.flatten().with_channel_dim(None)
        R = Y @ X.T @ bt.inv(X @ X.T)
        self.Aux = bt.Fnorm2(R.T @ R - bt.eye(R))
        self.set_matrix(R)
        return Y - R @ X
    
    def local_rigidity(self, X, Y):
        # X, Y, g = torch_tensor(X), torch_tensor(Y), torch_tensor(self.g_mask)
        # n_label = X.size(0)
        # n_dim = X.size(1)
        # size = X.shape[2:]
        # X_ = torch.cat((X, torch.ones(n_label, 1, *size)), 1)
        # Y_ = torch.cat((Y, torch.ones(n_label, 1, *size)), 1)
        # if n_dim == 2:
        #     X = torch.cat((X_.flatten(2), torch.multiple(torch.unsqueeze(torch.tensor([0., 0., 2.]), -1), n_label)), 2)
        #     Y = torch.cat((Y_.flatten(2), torch.multiple(torch.unsqueeze(torch.tensor([0., 0., 2.]), -1), n_label)), 2)
        #     self.g_mask = torch.cat((g.flatten(1), torch.multiple(torch.tensor([1.0]), n_label)), 1)
        #     X_old = torch.cat((torch.cat((X_old, torch.ones(n_label, 1, *size)), 1).flatten(2), torch.multiple(torch.unsqueeze(torch.tensor([0., 0., 2.]), -1), n_label)), 2)
        #     Y_old = torch.cat((torch.cat((Y_old, torch.ones(n_label, 1, *size)), 1).flatten(2), torch.multiple(torch.unsqueeze(torch.tensor([0., 0., 2.]), -1), n_label)), 2)
        # elif n_dim == 3:
        #     X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
        # A = X @ T(Y) + Y @ T(X)
        # A = A - torch.unsqueeze(torch.trace(A), [-1, -1]) * torch.eye(A)
        # b = torch.uncross_matrix(Y @ T(X)) - torch.uncross_matrix(X @ T(Y))
        # thetas = []
        # vectors = []
        # for t in range(A.size(0)):
        #     if torch.Fnorm2(A)[t] < 1e-4: thetas.append(torch.tensor(0.0)); vectors.append(torch.tensor([0., 0., 0.])); continue
        #     L, P = torch.linalg.eig(A[t])
        #     l = L[:, 0]
        #     c = torch.squeeze(T(P) @ torch.unsqueeze(b[t], -1), -1)
        #     f = ~ torch.equals(c ** 2, 0)
        #     if sum(f) >=1:
        #         coeff2 = torch.divide(l.prod(), l, 0.0).sum() - (c ** 2).sum()
        #         coeff1 = ((c ** 2) * (l.sum() - l)).sum() - l.prod()
        #         coeff0 = - ((c ** 2) * torch.divide(l.prod(), l, 0.0)).sum()
        #         p = np.poly1d([1, - l.sum().item(), coeff2.item(), coeff1.item(), coeff0.item()])
        #     else: thetas.append(torch.tensor(0.0)); vectors.append(torch.tensor([0., 0., 0.])); continue
        #     proots = torch.tensor(np.real(p.roots[np.abs(np.imag(p.roots)) < 1e-4])).to(X.device)
        #     if proots.numel() == 0: thetas.append(torch.tensor(0.0)); vectors.append(torch.tensor([0., 0., 0.])); continue
        #     mu = torch.unsqueeze(proots, [-1, -1]) * torch.eye(torch.unsqueeze(A[t].to(torch.float64)))
        #     v = - torch.inverse(torch.unsqueeze(A[t].to(torch.float64)) - mu) @ torch.unsqueeze(b[t].to(torch.float64), [0, -1])
        #     v = v.float()
        #     vTb = (torch.T(v) @ torch.unsqueeze(b[t], -1)).squeeze()
        #     if not any(torch.equals(proots, vTb)): thetas.append(torch.tensor(0.0)); vectors.append(torch.tensor([0., 0., 0.])); continue
        #     v = v[torch.equals(proots, vTb)]
        #     theta = 2 * torch.atan(torch.norm2(v))
        #     loss = torch.unsqueeze(1 + torch.cos(theta), [-1, -1]) * ((torch.T(v) @ torch.unsqueeze(A[t]) @ v) / 2 + torch.T(v) @ torch.unsqueeze(b[t], -1))
        #     tmax = torch.argmax(loss)
        #     thetas.append(theta[tmax])
        #     vectors.append(v[tmax].squeeze(-1))
        # th = torch.stack(thetas)
        # vec = torch.stack(vectors)
        # vx = torch.cross_matrix(vec)
        # R = torch.unsqueeze(1 + torch.cos(th), [-1, -1]) * (vx @ vx + vx) + torch.eye(vx)
        # Del = Y - R.detach() @ X
        n_label = X.n_batch
        n_dim = X.n_channel
        size = X.space
        if n_dim == 2:
            X_ = bt.cat(X, bt.ones([n_label], {1}, *size), {})
            Y_ = bt.cat(Y, bt.ones([n_label], {1}, *size), {})
            X = bt.cat(X_.flatten(), bt.channel_tensor([0, 0, 2]), -1)
            Y = bt.cat(Y_.flatten(), bt.channel_tensor([0, 0, 2]), -1)
            self.g_mask = bt.cat(self.g_mask.flatten(), bt.tensor([1.0]).multiply(n_label, []), 1)
            self.init_coords = tuple(bt.cat(bt.cat(x, bt.ones([n_label], 1, x.size(-1)), 1).with_channel_dim(1), bt.channel_tensor([0, 0, 2]), -1).with_channel_dim(None) for x in self.init_coords)
        elif n_dim == 3:
            X = X.flatten(); Y = Y.flatten()
        X.with_channel_dim(None)
        Y.with_channel_dim(None)
        A = X @ Y.T + Y @ X.T
        A = A - bt.unsqueeze(bt.trace(A), -1, -1) * bt.eye(A)
        b = (bt.uncross_matrix(Y @ X.T) - bt.uncross_matrix(X @ Y.T)).with_channel_dim(None)
        thetas = []
        vectors = []
        for t in range(A.n_batch):
            if bt.Fnorm2(A)[t] < 1e-4: thetas.append(bt.tensor(0.0)); vectors.append(bt.zeros(3)); continue
            if bt.__torchversion__ >= Version('1.10'): L, P = bt.linalg.eig(A[t])
            else:
                K, V = bt.eig(A[t], eigenvectors=True)
                L = bt.complex(K[:, 0], K[:, 1])
                Vr = bt.where((K[:, 1] < 0).reshape((1, -1)), bt.cat((V[:, :1], V[:, :-1]), 1), V)
                Vi = (K[:, 1] > 0).reshape((1, -1)) * bt.cat((V[:, 1:], V[:, -1:]), 1) - (K[:, 1] < 0).reshape((1, -1)) * V
                P = bt.complex(Vr, Vi)
            l = L.real
            c = bt.squeeze(P.real.T @ bt.unsqueeze(b[t], -1), -1)
            f = ~ bt.equals(c ** 2, 0)
            if sum(f) >= 1:
                coeff2 = bt.divide(l.prod(), l, 0.0).sum() - (c ** 2).sum()
                coeff1 = ((c ** 2) * (l.sum() - l)).sum() - l.prod()
                coeff0 = - ((c ** 2) * bt.divide(l.prod(), l, 0.0)).sum()
                p = np.poly1d([1, - l.sum().item(), coeff2.item(), coeff1.item(), coeff0.item()]) # p: np.array
            else: thetas.append(bt.tensor(0.0)); vectors.append(bt.zeros(3)); continue
            proots = bt.batch_tensor(np.real(p.roots[np.abs(np.imag(p.roots)) < 1e-4]))
            if proots.numel() == 0: thetas.append(bt.tensor(0.0)); vectors.append(bt.zeros(3)); continue
            mu = bt.unsqueeze(proots, -1, -1) * bt.eye(bt.unsqueeze(A[t].to(bt.float64), []))
            v = - bt.inv(bt.unsqueeze(A[t].to(bt.float64), []) - mu) @ bt.unsqueeze(b[t].to(bt.float64), [], -1)
            v = v.float()
            vTb = (v.T @ bt.unsqueeze(b[t], -1)).squeeze()
            if not any(bt.equals(proots, vTb)): thetas.append(bt.tensor(0.0)); vectors.append(bt.channel_tensor([0, 0, 0])); continue
            v = v[bt.equals(proots, vTb)]
            theta = 2 * bt.atan(bt.norm(v))
            loss = bt.unsqueeze(1 + bt.cos(theta), -1, -1) * ((v.T @ bt.unsqueeze(A[t], []) @ v) / 2 + v.T @ bt.unsqueeze(b[t], -1))
            tmax = bt.argmax(loss)
            thetas.append(theta[tmax])
            vectors.append(v[tmax].squeeze(-1).with_batch_dim(None).with_channel_dim(0))
        th = bt.stack(thetas, [])
        vec = bt.stack(vectors, []).with_channel_dim(1)
        vx = bt.cross_matrix(vec)
        R = bt.unsqueeze(1 + bt.cos(th), -1, -1) * (vx @ vx + vx) + bt.eye(vx)
        R = R.detach()
        self.set_matrix(R)
        return Y - R @ X
    
    def local_isometry(self, X, Y):
        n_dim = X.n_channel
        Xdissq, Ydissq = [], []
        for d in range(n_dim):
            u = (slice(None),) * (2 + d) + (slice(1, None),) + (slice(None),) * (n_dim - d - 1)
            l = (slice(None),) * (2 + d) + (slice(None, -1),) + (slice(None),) * (n_dim - d - 1)
            Xdissq.append(bt.where(X[u] * X[l] == 0, bt.zeros_like(X[u]), (X[u] - X[l]) ** 2).sum(1).flatten(1))
            Ydissq.append(bt.where(Y[u] * Y[l] == 0, bt.zeros_like(Y[u]), (Y[u] - Y[l]) ** 2).sum(1).flatten(1))
        Xdissq = bt.cat(Xdissq, 1)
        Ydissq = bt.cat(Ydissq, 1)
        Aux = 1e-2 * (bt.Fnorm2(bt.unsqueeze(Xdissq[Xdissq > 0] - 1, [])) + bt.Fnorm2(bt.unsqueeze(Ydissq[Ydissq > 0] - 1, [])))
        
        Xdissq, Ydissq = [], []
        for d in grid(*((2,) * n_dim)):
            sl = (slice(None, -1), slice(1, None))
            u = (slice(None), slice(None)) + tuple(sl[i] for i in d)
            l = (slice(None), slice(None)) + tuple(sl[1-i] for i in d)
            Xdissq.append(bt.where(X[u] * X[l] == 0, bt.zeros_like(X[u]), (X[u] - X[l]) ** 2).sum(1).flatten(1))
            Ydissq.append(bt.where(Y[u] * Y[l] == 0, bt.zeros_like(Y[u]), (Y[u] - Y[l]) ** 2).sum(1).flatten(1))
        Xdissq = bt.cat(Xdissq, 1)
        Ydissq = bt.cat(Ydissq, 1)
        Aux += 1e-3 * (bt.Fnorm2(bt.unsqueeze(Xdissq[Xdissq > 0] - n_dim, [])) + bt.Fnorm2(bt.unsqueeze(Ydissq[Ydissq > 0] - n_dim, [])))
    
    def local_rotation3D(self, X, Y):
        X = X.flatten().with_channel_dim(None); Y = Y.flatten().with_channel_dim(None)
        A = Y @ X.T @ bt.inv(X @ X.T)
        omega = bt.uncross_matrix(A - T(A))
        norm = bt.norm2(omega)
        omega = omega / norm
        theta = bt.asin(norm / 2)
        wx = bt.cross_matrix(omega)
        R = (1 - bt.cos(theta)) * (wx @ wx) + bt.sin(theta) * wx + bt.eye(wx)
        self.set_matrix(R)
        return Y - R @ X
    
    def local_FLIRT(self, X, Y, g):
        '''
        X: ([n_label], {n_dim}, n@1, n@2, ..., n@n_dim)
        Y: ([n_label], {n_dim}, n@1, n@2, ..., n@n_dim)
        g: ([n_label], n@1, n@2, ..., n@n_dim)
        '''
        n_label = X.n_batch
        n_dim = X.n_channel
        g = g.detach()
        dt = getattr(self, 'scale', 1)
        avouch(X.space == Y.space == g.space, "'local_FLIRT' needs a same image space for inputs.")
        if dt > (min(g.space) - 1) // 2: raise TypeError(f"'scale' being {dt} for local prior is too big for image size {g.space}.")
        JacOfPoints = torch.Jacobian(X, Y, dt=dt) # dy_i/dx_j: (n_label, n_dim, n_dim, n@1-dt, n@2-dt, ..., n@n_dim-dt)
        flatJacOfPoints = JacOfPoints.flatten(1, 2) # (n_label, n_dim x n_dim, n@1-dt, n@2-dt, ..., n@n_dim-dt)
        dJac = torch.Jacobian(torch.crop_as(X, flatJacOfPoints.shape[2:], n_keepdim=2), flatJacOfPoints, dt=dt) # (n_label, n_dim x n_dim, n_dim, n@1-2dt, n@2-2dt, ..., n@n_dim-2dt)
        HesOfPoints = (dJac.view(n_label, n_dim, n_dim, n_dim, -1) ** 2).sum(1) # ||dy||^2/dx1_i/dx2_j: (n_label, n_dim, n_dim, n_data(-2dt))
        JacOfPoints = torch.movedim(JacOfPoints.flatten(3), 3, 1).flatten(0, 1) # (n_label x n_data(-dt), n_dim, n_dim)
        HesOfPoints = torch.movedim(HesOfPoints, 3, 1) # (n_label, n_data(-2dt), n_dim, n_dim)
        order0 = ((torch.det(JacOfPoints).view(n_label, -1) - 1) ** 2) # (n_label, n_data(-dt))
        order1 = ((JacOfPoints.transpose(1, 2) @ JacOfPoints - torch.eye(JacOfPoints)) ** 2).sum([1, 2]).view(n_label, -1)
        order2 = (HesOfPoints ** 2).sum([2, 3])
        gshape_dt = tuple(x - dt for x in g.space)
        gshape_2dt = tuple(x - 2 * dt for x in g.space)
        return bt.divide((((order0 + order1).view(n_label, *gshape_dt) * torch.crop_as(g, gshape_dt, n_keepdim=1)).flatten(1).sum(1) + (order2.view(n_label, *gshape_2dt) * torch.crop_as(g, gshape_2dt, n_keepdim=1)).flatten(1).sum(1)), g.flatten(1).sum(1)).as_subclass(bt.Tensor).with_batch_dim(0)
    
    def local_Jacobian(self, X, Y):
        n_label = X.n_batch
        size = X.space
        X = bt.pad(bt.cat(X, bt.ones([n_label], {1}, *size), {}))
        Y = bt.pad(bt.cat(Y, bt.ones([n_label], {1}, *size), {}))
        slicer = lambda n: {i: slice(n-i-1, None if i == 0 else -i) for i in range(n)}
        X = bt.stack([X[(slice(None), slice(None)) + tuple(slicer(3)[t] for t in g)] for g in grid(*((3,) * n_dim))], 2)
        Y = bt.stack([Y[(slice(None), slice(None)) + tuple(slicer(3)[t] for t in g)] for g in grid(*((3,) * n_dim))], 2)
        # X, Y: ([n_label], {n_dim + 1}, 3 ^ n_dim, n@1, ..., n@n_dim)
        Nx = X.flatten(3).mergedims(3, [])
        Ny = Y.flatten(3).mergedims(3, [])
        val = bt.det(Nx @ Nx.T) > 1e-2
        Nx = Nx[val].with_channel_dim(None)
        Ny = Ny[val].with_channel_dim(None)
        # Nx, Ny: ([n_label x n_data], {n_dim + 1}, 3 ^ n_dim)
        NR = Ny @ Nx @ inv(Nx @ Nx.T)
        # NR: ([n_label x n_data], n_dim + 1, n_dim + 1)
        dis = bt.Fnorm2(NR.T @ NR - bt.eye(NR)).splitdim([], [n_label], -1)
        self.Aux = 1e-2 * dis.mean()

class registration:
    """
    Registration function. 

    Args:
        trans_cls [SpatialTransformation]: The transformation applied. 
        source [bt.Tensor]: The source/moving image. 
        target [bt.Tensor]: The target/fixed image. 
        down_sample [int or tuple]: Down sample the image by a factor of `1/down_sample` (in unit mm). Defaults to 6.
        pre_align [Transformation or str]: Specify a pre-align strategy. Defaults to 'center', meaning alignment by gray-scale centers. 
        FFD_spacing [int or tuple]: Spacing between FFD control points. Defaults to 20.
        max_iter [int]: Number of iterations. Defaults to 200.
        step_length [float or callable]: Learning rate, use a function of iteration i to perform dynamic weight. Defaults to 1e-3.
        loss [str]: Loss function, in the attached list. Defaults to 'NMI'.
        verbose [bool]: Whether to print the loss during iteration, or not. Defaults to True.

    Returns:
        ComposedTransformation: A transformation that can convert source to target-like. 
        Use `ret[0]` to obtain the corresponding SpatialTransformation, and `ret(image)` to perform image transformation. 
        
    Attached:
        List
        ----------
        MI = MutualInformation, NMI = NormalizedMutualInformation, KL = KLDivergence, 
        CLE = CorrelationOfLocalEstimation, NVI = NormalizedVectorInformation,
        SSD = SumSquaredDifference, MSE = MeanSquaredErrors, PSNR = PeakSignalToNoiseRatio,
        CE = CrossEntropy, CC = CrossCorrelation, NCC = NormalizedCrossCorrelation, SSIM = StructuralSimilarity,
        DSC = LabelDiceScore, JCD = LabelJaccardCoefficient, VS = LabelVolumeSimilarity,
        FP = LabelFalsePositive, FN = LabelFalseNegative, HD = LabelHausdorffDistance,
        MdSD = LabelMedianSurfaceDistance, ASD = LabelAverageSurfaceDistance, MSD = LabelAverageSurfaceDistance,
        divSD = LabelDivergenceOfSurfaceDistance, stdSD = LabelDivergenceOfSurfaceDistance
    """
    def __init__(self, source, target, *, source_range=None, target_range=None, pre_align='center', verbose=True):
        source_aff = None
        target_aff = None
        if isinstance(source, str): source = IMG(source)
        if isinstance(target, str): target = IMG(target)
        if isinstance(source, IMG):
            source_aff = bt.tensor(source.affine).with_n_feat_dim(2).float()
            source = source.to_tensor()
            if source.has_feature: source = source.pick(1, [])
        if isinstance(target, IMG):
            target_aff = bt.tensor(target.affine).with_n_feat_dim(2).float()
            target = target.to_tensor()
            if target.has_feature: target = target.pick(1, [])
        
        if source_aff is None: source_aff = bt.eye({source.n_batch}, [n_dim + 1])
        if target_aff is None: target_aff = bt.eye({target.n_batch}, [n_dim + 1])
        
        avouch(source.n_space_dim == target.n_space_dim)
        if not source.has_batch: source = source.unsqueeze({})
        if not target.has_batch: target = target.unsqueeze({})
        if not source.has_channel: source = source.unsqueeze([])
        if not target.has_channel: target = target.unsqueeze([])
        
        if source_range is not None: normalize = Normalize(*source_range)
        else: normalize = Normalize()
        source = normalize(source)
        if target_range is not None: normalize = Normalize(*target_range)
        else: normalize = Normalize()
        target = normalize(target)
        
        if pre_align == 'image':
            source_center = bt.channel_tensor([(sp - 1) / 2 for sp in source.space])
            target_center = bt.channel_tensor([(sp - 1) / 2 for sp in target.space])
            pre_align = CenterAlign(source_center, target_center, domain='physical-space', source=source_aff, target=target_aff)
        elif pre_align == 'physical': pre_align = Identity(domain='physical-space', source=source_aff, target=target_aff)
        elif pre_align == 'center':
            source_center = bt.center_gravity(source)
            target_center = bt.center_gravity(target)
            pre_align = CenterAlign(source_center, target_center, domain='physical-space', source=source_aff, target=target_aff)
        
        avouch(isinstance(pre_align, Transformation), TypeError("Argument 'pre_align' for registrion can only be 'image', 'physical', 'center' or mc.Transformation objects. "))
        
        self.source = source
        self.target = target
        self.verbose = verbose
        self.trans = pre_align
        self.source_aff = source_aff
        self.target_aff = target_aff
        
    def align(self, trans_cls: Transformation, down_sample=6, max_iter=200, step_length=1e-3, loss='NMI', FFD_spacing=20):
        source = self.source
        target = self.target
        verbose = self.verbose
        source_aff = self.source_aff
        target_aff = self.target_aff
        
        n_dim = source.n_space_dim
        down_sample = to_tuple(down_sample)
        if len(down_sample) == 1: down_sample *= n_dim
        
        source_spacing = affine2spacing(source_aff)
        source_factors = [int(ds // sp) for ds, sp in zip(down_sample, source_spacing)]
        source_scalemat = bt.channel_tensor(source_factors + [1.]).unsqueeze([0])
        source_aff *= source_scalemat
        source_idx = (slice(None),) * source.space_domain[0] + tuple(slice(None, None, f) for f in source_factors) + (slice(None),) * (source.n_dim - source.space_domain[1])
        source = source[source_idx]
        target_spacing = affine2spacing(target_aff)
        target_factors = [int(ds // sp) for ds, sp in zip(down_sample, target_spacing)]
        target_scalemat = bt.channel_tensor(target_factors + [1.]).unsqueeze([0])
        target_aff *= target_scalemat
        target_idx = (slice(None),) * target.space_domain[0] + tuple(slice(None, None, f) for f in target_factors) + (slice(None),) * (target.n_dim - target.space_domain[1])
        target = target[target_idx]
        
        pre_align = self.trans.between_spaces(source_aff, target_aff)
        
        specific_params = {}
        if trans_cls == FreeFormDeformation:
            FFD_spacing = to_tuple(FFD_spacing)
            if len(FFD_spacing) == 1: FFD_spacing *= n_dim
            FFD_spacing = tuple(fsp / sp for fsp, sp in zip(FFD_spacing, target_spacing))
            specific_params['spacing'] = FFD_spacing
        if trans_cls in [Affine, Rigid, PolyAffine, LocallyAffine]:
            specific_params['trans_std'] = 1 / max(down_sample)
        params = trans_cls.random_init_params(*target.space, n_batch=target.n_batch, std=1e-3, **specific_params)
        for p in params: p.requires_grad = True
    
        optimizer = bt.Optimization(bt.optim.Adam, params, lr=step_length)
        for i in range(max_iter):
            raw_trans = trans_cls(*params, domain='physical', use_implementation='image-space', source=source_aff, target=target_aff)
            trans = ComposedTransformation(raw_trans, pre_align)
            warped = trans(source, to_space=target.space, domain='non-spatial')
            nmi = metric(loss)(warped, target).mean()
            optimizer.maximize(nmi)
            if verbose: print(f"{trans_cls.__name__} Registration: iteration = {i+1}, NMI = {nmi.item()}")
        
        trans.source /= source_scalemat
        trans.target /= target_scalemat
        self.trans = trans.compose().detach()

# def registration(trans_cls: Transformation, source, target, down_sample=6, pre_align='center', 
#                  source_range=None, target_range=None,
#                  FFD_spacing=20, max_iter=200, step_length=1e-3, loss='NMI', verbose=True):
#     """
#     Registration function. 

#     Args:
#         trans_cls [SpatialTransformation]: The transformation applied. 
#         source [bt.Tensor]: The source/moving image. 
#         target [bt.Tensor]: The target/fixed image. 
#         down_sample [int or tuple]: Down sample the image by a factor of `1/down_sample` (in unit mm). Defaults to 6.
#         pre_align [Transformation or str]: Specify a pre-align strategy. Defaults to 'center', meaning alignment by gray-scale centers. 
#         FFD_spacing [int or tuple]: Spacing between FFD control points. Defaults to 20.
#         max_iter [int]: Number of iterations. Defaults to 200.
#         step_length [float or callable]: Learning rate, use a function of iteration i to perform dynamic weight. Defaults to 1e-3.
#         loss [str]: Loss function, in the attached list. Defaults to 'NMI'.
#         verbose [bool]: Whether to print the loss during iteration, or not. Defaults to True.

#     Returns:
#         ComposedTransformation: A transformation that can convert source to target-like. 
#         Use `ret[0]` to obtain the corresponding SpatialTransformation, and `ret(image)` to perform image transformation. 
        
#     Attached:
#         List
#         ----------
#         MI = MutualInformation, NMI = NormalizedMutualInformation, KL = KLDivergence, 
#         CLE = CorrelationOfLocalEstimation, NVI = NormalizedVectorInformation,
#         SSD = SumSquaredDifference, MSE = MeanSquaredErrors, PSNR = PeakSignalToNoiseRatio,
#         CE = CrossEntropy, CC = CrossCorrelation, NCC = NormalizedCrossCorrelation, SSIM = StructuralSimilarity,
#         DSC = LabelDiceScore, JCD = LabelJaccardCoefficient, VS = LabelVolumeSimilarity,
#         FP = LabelFalsePositive, FN = LabelFalseNegative, HD = LabelHausdorffDistance,
#         MdSD = LabelMedianSurfaceDistance, ASD = LabelAverageSurfaceDistance, MSD = LabelAverageSurfaceDistance,
#         divSD = LabelDivergenceOfSurfaceDistance, stdSD = LabelDivergenceOfSurfaceDistance
#     """
#     source_aff = None
#     target_aff = None
#     if isinstance(source, str): source = IMG(source)
#     if isinstance(target, str): target = IMG(target)
#     if isinstance(source, IMG):
#         source_aff = bt.tensor(source.affine).with_n_feat_dim(2).float()
#         source = source.to_tensor()
#     if isinstance(target, IMG):
#         target_aff = bt.tensor(target.affine).with_n_feat_dim(2).float()
#         target = target.to_tensor()
#     if source_aff is None: source_aff = bt.eye({n_batch}, [n_dim + 1])
#     if target_aff is None: target_aff = bt.eye({n_batch}, [n_dim + 1])
    
#     avouch(source.n_space_dim == target.n_space_dim)
#     if not source.has_batch: source = source.unsqueeze({})
#     if not target.has_batch: target = target.unsqueeze({})
#     if not source.has_channel: source = source.unsqueeze([])
#     if not target.has_channel: target = target.unsqueeze([])
    
#     if source_range is not None: normalize = Normalize(*source_range)
#     else: normalize = Normalize()
#     source = normalize(source)
#     if target_range is not None: normalize = Normalize(*target_range)
#     else: normalize = Normalize()
#     target = normalize(target)
    
#     n_dim = source.n_space_dim
#     down_sample = to_tuple(down_sample)
#     if len(down_sample) == 1: down_sample *= n_dim
    
#     source_spacing = affine2spacing(source_aff)
#     source_factors = [int(ds // sp) for ds, sp in zip(down_sample, source_spacing)]
#     source_scalemat = bt.channel_tensor(source_factors + [1.]).unsqueeze([0])
#     source_aff *= source_scalemat
#     source_idx = (slice(None),) * source.space_domain[0] + tuple(slice(None, None, f) for f in source_factors) + (slice(None),) * (source.n_dim - source.space_domain[1])
#     source = source[source_idx]
#     target_spacing = affine2spacing(target_aff)
#     target_factors = [int(ds // sp) for ds, sp in zip(down_sample, target_spacing)]
#     target_scalemat = bt.channel_tensor(target_factors + [1.]).unsqueeze([0])
#     target_aff *= target_scalemat
#     target_idx = (slice(None),) * target.space_domain[0] + tuple(slice(None, None, f) for f in target_factors) + (slice(None),) * (target.n_dim - target.space_domain[1])
#     target = target[target_idx]
    
#     shape = target.space
#     n_batch = target.n_batch
    
#     if pre_align == 'image':
#         source_center = bt.channel_tensor([(sp - 1) / 2 for sp in source.space])
#         target_center = bt.channel_tensor([(sp - 1) / 2 for sp in target.space])
#         pre_align = CenterAlign(source_center, target_center, domain='physical-space', source=source_aff, target=target_aff)
#     elif pre_align == 'physical': pre_align = Identity(domain='physical-space')
#     elif pre_align == 'center':
#         source_center = (bt.image_grid(source.space) * source).sum(...) / source.sum(...)
#         target_center = (bt.image_grid(target.space) * target).sum(...) / target.sum(...)
#         pre_align = CenterAlign(source_center, target_center, domain='physical-space', source=source_aff, target=target_aff)
    
#     avouch(isinstance(pre_align, Transformation), TypeError("Argument 'pre_align' for registrion can only be 'image', 'physical', 'center' or mc.Transformation objects. "))
    
#     specific_params = {}
#     if trans_cls == FreeFormDeformation:
#         FFD_spacing = to_tuple(FFD_spacing)
#         if len(FFD_spacing) == 1: FFD_spacing *= n_dim
#         FFD_spacing = tuple(fsp / sp for fsp, sp in zip(FFD_spacing, target_spacing))
#         specific_params['spacing'] = FFD_spacing
#     params = trans_cls.random_init_params(*shape, n_batch=n_batch, std=1e-3, **specific_params)
#     for p in params: p.requires_grad = True
    
#     optimizer = bt.Optimization(bt.optim.Adam, params, lr=step_length)
#     for i in range(max_iter):
#         raw_trans = trans_cls(*params, domain='physical', use_implementation='image-space', source=source_aff, target=target_aff)
#         trans = ComposedTransformation(raw_trans, pre_align)
#         warped = trans(source, to_space=target.space, domain='non-spatial')
#         nmi = metric(loss)(warped, target).mean()
#         optimizer.maximize(nmi)
#         if verbose: print(f"{trans_cls.__name__} Registration: iteration = {i+1}, NMI = {nmi.item()}")
    
#     trans.source /= source_scalemat
#     trans.target /= target_scalemat
#     return trans

# def registration(trans_cls, source, target, spacing=1, sub_space=6, 
#                  FFD_spacing=20, max_iter=200, step_length=1e-3, loss='NMI', verbose=True):
#     """
#     Registration function. 

#     Args:
#         trans_cls [SpatialTransformation]: The transformation applied. 
#         source [bt.Tensor]: The source/moving image. 
#         target [bt.Tensor]: The target/fixed image. 
#         spacing [int or tuple]: The spacing of source and target images. Defaults to 1.
#         sub_space [int or tuple]: Down sample the image by a factor of `1/sub_space`. Defaults to 6.
#         FFD_spacing [int or tuple]: Spacing between FFD control points. Defaults to 20.
#         max_iter [int]: Number of iterations. Defaults to 200.
#         step_length [float or callable]: Learning rate, use a function of iteration i to perform dynamic weight. Defaults to 1e-3.
#         loss [str]: Loss function, in the attached list. Defaults to 'NMI'.
#         verbose [bool]: Whether to print the loss during iteration, or not. Defaults to True.

#     Returns:
#         ComposedTransformation: A transformation that can convert source to target-like. 
#         Use `ret[0]` to obtain the corresponding SpatialTransformation, and `ret(image)` to perform image transformation. 
        
#     Attached:
#         List
#         ----------
#         MI = MutualInformation, NMI = NormalizedMutualInformation, KL = KLDivergence, 
#         CLE = CorrelationOfLocalEstimation, NVI = NormalizedVectorInformation,
#         SSD = SumSquaredDifference, MSE = MeanSquaredErrors, PSNR = PeakSignalToNoiseRatio,
#         CE = CrossEntropy, CC = CrossCorrelation, NCC = NormalizedCrossCorrelation, SSIM = StructuralSimilarity,
#         DSC = LabelDiceScore, JCD = LabelJaccardCoefficient, VS = LabelVolumeSimilarity,
#         FP = LabelFalsePositive, FN = LabelFalseNegative, HD = LabelHausdorffDistance,
#         MdSD = LabelMedianSurfaceDistance, ASD = LabelAverageSurfaceDistance, MSD = LabelAverageSurfaceDistance,
#         divSD = LabelDivergenceOfSurfaceDistance, stdSD = LabelDivergenceOfSurfaceDistance
#     """
#     if isinstance(source, IMG): source = source.to_tensor()
#     if isinstance(target, IMG): target = target.to_tensor()
#     avouch(source.n_space_dim == target.n_space_dim)
#     if not source.has_batch: source = source.unsqueeze([])
#     if not target.has_batch: target = target.unsqueeze([])
#     if not source.has_channel: source = source.unsqueeze({1})
#     if not target.has_channel: target = target.unsqueeze({1})
#     normalize = Normalize()
#     source = normalize(source)
#     target = normalize(target)
#     size = source.space
#     n_batch = source.n_batch
#     n_dim = len(size)
#     sub_space = to_tuple(sub_space)
#     if len(sub_space) == 1: sub_space *= n_dim
#     spacing = to_tuple(spacing)
#     if len(spacing) == 1: spacing *= n_dim
#     FFD_spacing = to_tuple(FFD_spacing)
#     if len(FFD_spacing) == 1: FFD_spacing *= n_dim

#     idx = bt.meshgrid(*tuple(bt.arange(0, s, sub / sp).int() for s, sub, sp in zip(size, sub_space, spacing)))
#     idx = (slice(None),) * (source.n_dim - len(idx)) + idx
#     source = source[idx]
#     target = target[idx]
#     FFD_spacing = tuple(fsp/sp for fsp, sp in zip(FFD_spacing, spacing))
    
#     if trans_cls == Rescale:
#         params = bt.ones([n_batch], n_dim)
#     elif trans_cls == Affine:
#         params = bt.eye([n_batch], n_dim + 1)
#         class StretchedAffine(Affine):
#             def __init__(self, matrix): super().__init__(matrix, trans_stretch=20)
#         trans_cls = StretchedAffine
#     elif trans_cls == FreeFormDeformation:
#         params = bt.zeros([n_batch], {n_dim}, *tuple(math.ceil(s / fsp) for s, fsp in zip(size, FFD_spacing)))
#         class FreeFormDeformationField(FreeFormDeformation):
#             def __init__(self, offsets): super().__init__(offsets, FFD_spacing)
#         trans_cls = FreeFormDeformationField
#     elif trans_cls == DenseDisplacementField:
#         params = bt.zeros([n_batch], {n_dim}, *size)
#     params.requires_grad = True
        
#     optimizer = bt.Optim(bt.optim.Adam, params, lr=step_length)
        
#     for i in range(max_iter):
#         trans = trans_cls(params)
#         transformed = interpolation(source, trans)
#         nmi = metric(loss)(transformed, target).mean()
#         optimizer.maximize(nmi)
#         if verbose: print(f"{trans_cls.__name__} Registration: iteration = {i+1}, NMI = {nmi.item()}")
    
#     return trans_cls(params)
