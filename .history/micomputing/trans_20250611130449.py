
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
    ComposedTransformation CompoundTransformation
    
    interpolation
    interpolation_forward
    
    Identity Id
    Rotation90
    Rotation180
    Rotation270
    Reflect                Reflection
    Permutedim             PermuteDim  DimPermutation
    Rescale                Rescaling   rand_Rescaling
    Translate              Translation rand_Translation            CenterAlign
    Rigid                  Rig         rand_Rigidity               rand_Rig
    Affine                 Aff         rand_Affinity               rand_Aff
    PolyAffine                         rand_PolyAffine
    logEuclidean           logEu       rand_logEuclidean           rand_logEu
    LocallyAffine          LARM        rand_LocallyAffine          rand_LARM
    FreeFormDeformation    FFD         rand_FreeFormDeformation    rand_FFD
    DenseDisplacementField DDF         rand_DenseDisplacementField rand_DDF
    VelocityField          VF          rand_VelocityField          rand_VF
    
    Normalize              Cropping
""".split()
#     FreeFormDeformation FFD
#     DenseDisplacementField DDF
#     MultiLayerPerception MLP
    
#     Normalize
    
#     resample
#     interpolation
#     interpolation_forward
    
#     Affine2D2Matrix
#     Quaterns2Matrix
#     Matrix2Quaterns
# """.split()
DenseDisplacementField = None

import json, math
from copy import copy
from typing import Iterable
from .stdio import IMG, affine2spacing
from functools import wraps

with __info__:
    import numpy as np
    import batorch as bt
    from pycamia import to_tuple, to_list, arg_tuple, avouch, prod, Path
    from pycamia import SPrint, Error, Warn, alias, get_environ_vars, Version
    from pyoverload import callable
    
eps = 1e-6
    
# def is_spatial_transformation(x):
#     return isinstance(x, (WorldCoordsTransformation, ImageCoordsTransformation)) \
#         or isinstance(x, ComposedTransformation) and x.mode == "spatial"

# def is_world_coords_transformation(x):
#     return isinstance(x, Transformation) or isinstance(x, ComposedTransformation) and x.mode == "spatial"

def perform_reshape(shape, reshape_args):
    if len(reshape_args) == 1: return tuple(reshape_args[0])
    n_dim = len(shape)
    padding, scale, *pairs = reshape_args
    if len(scale) == 1: scale *= n_dim
    if len(padding) == 1: padding *= n_dim
    padding = tuple(p if isinstance(p, tuple) else (p, p) for p in padding)
    target_space = [int(x * y + a + b) for x, y, (a, b) in zip(shape, scale, padding)]
    for p, q in pairs: target_space[p], target_space[q] = target_space[q], target_space[p]
    return tuple(target_space)

def req_spatial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        avouch(self.domain != 'non-spatial', f"Transformation.{func.__name__} is only available for spatial transformations, not {self.__name__}. ")
        avouch(any(d != 'non-spatial' for d in self.implemented_domains), f"Transformation.{func.__name__} is only available for spatial transformations, but only intensity transformation is implemented for {self.__name__}. ")
        return func(self, *args[1:], **kwargs)
    return wrapper

class Transformation:
    """
    The base class for Transformations. 
    Note that Transformation is not inheritted from 'torch.nn.Module' due to two reasons:
        1. 'Transformations' are basically not network structures or anything similar. 
        2. Using 'forward' for calling may cause confusion as there are forward/ backward transformations. 
        
    Backward transformations create warped image through resampling by transformed coordinates from target image space.
    Forward transformations map the coordinates from source image space to target and warp images accordingly. 
    Non-spatial image transformations are regarded as forward transformations. 
    
    Args: (See corresponding properties for details. )
        domain (str: physical_space | image_space | non_spatial);
        |___domain=physical_space___ backward (bool);
        |_______OR image_space______ source (IMG or None);
        |___________________________ target (IMG or None).
    
    Properties:
        params (tuple): The parameters of the transformation model. The first `params[0]`, namely `self.main_param`, is 
            the main parameter which identifies the batch size. 
        kwparams (dict): The parameters with names in initializations. 
        n_dim (int): The dimension of input images. Defaults to 'None' for arbitrary dimensions. 
        n_batch (int): The manually determined batch size. 
        reshape (list): A list recording the reshaping criteria of the transformation;
            e.g. [((dl1⁻, dl1⁺), (dl2⁻, dl2⁺), (dl3⁻, dl3⁺)), (s[0], s[1], s[2]), (d[0], d[1]), (d[2], d[3])] means:
                1) transpose dimensions d[2] and d[3]; followed by d[0] with d[1];
                2) rescale at dimension d with s[d];
                3) crop(* < 0) or pad(* > 0) a side by dl[d]♯,
                   where #=+ for the larger side and #=- for the lower side;
            The shape transformation goes from left to right. Only the firt two transformation can be a tuple with length other than 2, 
                meaning scaling and padding instead of transpose. Having a length 1, e.g. (*,), stands for isotropic scaling or padding;
                being (1,) means no scaling, (0,) means no padding.
        domain (str): The source/ target domain of the transformation. 
            candidates: "physical-space"; "image-space"; "non-spatial". 
            domain = 'physical-space'/ 'image-space': transformation maps coordinates to coordinates.
                input size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
            domain = 'non-spatial': transformation maps (a batched series of) images to images.
                input size: ({n_batch: optional}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
        backward (bool): Whether the transformation performs the operation on the target image (backward) or the source image (forward),
            spatial trans.'s (domain='physical-space'/ 'image-space') are backward trans.'s by default, who map the coordinates from the target image space to the source image. 
                i.e., T(I)(x) = I(y) = I∘T(x) for coordinates y = T(x). 
            image trans.'s (domain='non-spatial') are forward trans.'s by default, who map the source images to targets. 
                i.e., T(I)(x) = v = T(u) = T∘I(x) for intensities v = T(u=I(x)).
            The defults are equivalent with each other for resampling.
        Only activated for spatial transformations: 
        source (IMG or None): The source image of type IMG.
        target (IMG or None): The target image of type IMG.

    All possible usages are:
        [Notations:]
            T: The implemented tranformation, called as a target-to-source transformation.
            S: The implemented tranformation, called as a source-to-target transformation, equivalent to the inversed transformation of T theoretically. 
            Aₜ: The affine matrix for the target image, mapping the image coordinates to the physical space. 
            Aₛ: The affine matrix for the source image. 
            I: The input images (batched). 
            x: The input coordinates. 
        domain='physical-space'; backward= True; input=[x: phy. coords.| target, domain='physical-space']: output = T(x)            <= physical coords. in source image space. 
        domain='physical-space'; backward= True; input=[x: phy. coords.| target, domain='image-space']: output = Aₛ∘T∘Aₜ⁻¹(x)        <= physical coords. in source image space. 
        domain='physical-space'; backward= True; input=[I:      images | source, domain='non-spatial']: output[x] = I[Aₛ⁻¹∘T∘Aₜ(x)]  <= resampled images. 
        domain='physical-space'; backward=False; input=[x: phy. coords.| source, domain='physical-space']: output = S(x)            <= physical coords. in target image space. 
        domain='physical-space'; backward=False; input=[x: phy. coords.| source, domain='image-space']: output = Aₜ∘S∘Aₛ⁻¹(x)        <= physical coords. in target image space. 
        domain='physical-space'; backward=False; input=[I:      images | source, domain='non-spatial']: output[Aₜ⁻¹∘S∘Aₛ(x)] = I(x)  <= resampled images (by forward interpolation). 

        domain='image-space'; backward= True; input=[x: img. coords.| target, domain='physical-space']: output = Aₛ⁻¹∘T∘Aₜ(x)        <= image coords. in source image.
        domain='image-space'; backward= True; input=[x: img. coords.| target, domain='image-space']: output = T(x)                  <= image coords. in source image.
        domain='image-space'; backward= True; input=[I:      images | source, domain='non-spatial']: output[x] = I[T(x)]            <= resampled images. 
        domain='image-space'; backward=False; input=[x: img. coords.| source, domain='physical-space']: output = Aₜ⁻¹∘S∘Aₛ(x)        <= image coords. in target image.
        domain='image-space'; backward=False; input=[x: img. coords.| source, domain='image-space']: output = S(x)                  <= image coords. in target image. 
        domain='image-space'; backward=False; input=[I:      images | source, domain='non-spatial']: output[S(x)] = I(x)            <= resampled images (by forward interpolation). 

        domain='non-spatial'; backward= N/A ; input=[I: images | source     , domain='non-spatial']: output = T(I)                  <= transformed images. 
    
    NOTE: property 'backward' only takes effect in domain changes where the source/ target images are involved. 

    Methods: 
        Including basic operation of image transformations. 
        __call__ (x()): the call function of the object, with an input of image/ coordinates, and output the image/ coordinates. 
        __str__ (print(x)/ str(x)): indicate the print string for the transformation object. 
        __getitem__ (x[y]): find the i-th element in batch for each parameter, i.e. the i-th transformation in a batched transformation. 
        __matmul__ (x @ y): compose multiple transformations. 
        parameters: get the parameters of this model, just like an nn.Module object. 
        detach: detach all the parameters, just like an nn.Module object. 
        train/ eval: convert the transformation to traning/ evaluation mode, just like an nn.Module object. 
        to_dict/ from_dict: convert between the transformation and dictionary object.
        obj_json/ json_obj: convert between the transformation and json object, majorly designed for saving/ loading the transformation. 
        save/ load: save(load) the transformation to(from) local files. 
        is_forward/ is_backward: return the direction of special transformation. 
        as_forward/ as_backward: make the transformation a forward/ backward transformation. 
        direct_inv: inverse the transformation by changing the forward/backward direction. 
        
    A subclass requires methods:
        During the inheritance of Transformation, one needs to re-implement the following methods in the subclass:
            __call_image_space__: implementation of this function or any function (not named after resolved names) wrapped with 
                '@Transformation.for_image_space', results in defining a call of the transformation with the inputs of image-space coordinates. 
            __call_physical_space__: implementation of this or any function wrapped with '@Transformation.for_physical_space' defines a call with inputs of physical-space coordinates. 
            __call_non_spatial__: implementation of this or any function wrapped with '@Transformation.for_non_spatial' defines a call with inputs of images instead of coordinates. 
        Optional methods are:
            __affine__: implementation of it defines the function to get the equivalent affine matrix for the transformation. 
            __inv__: implementation of it defines the inverse transformation. 
            classmethod(random_init_params): implementation of it provides a class method creating random initial parameters for the transformation. 
                It returns a tuple of the same length as the input params. 
    """
    
    def __init__(self, *params, **kwparams):
        self.batch_param = []
        self.domain = kwparams.get('domain', ['image-space', 'physical-space'][kwparams.get('physical', False)])
        self.domain = {
            'physical coordinates': 'physical-space',
            'physical coords': 'physical-space',
            'physical space': 'physical-space',
            'physical-space coordinates': 'physical-space',
            'physical': 'physical-space',
            'phy': 'physical-space',
            'image coordinates': 'image-space',
            'image coords': 'image-space',
            'image space': 'image-space',
            'image-space coordinates': 'image-space',
            'img': 'image-space',
            'intensities': 'non-spatial',
            'whole image': 'non-spatial',
            'int': 'non-spatial'
        }.get(self.domain, self.domain)
        avouch(self.domain != 'image', "Ambiguous domain specifier 'image': use 'image coords/image space' for image-space, 'whole image' for non-spatial. ")
        avouch(self.domain in ('image-space', 'physical-space', 'non-spatial'), f"Unrecognized domain '{self.domain}'. ")
        self.backward = kwparams.get('backward', self.domain in ('physical-space', 'image-space'))
        self.use_implementation = kwparams.get('use_implementation', None)
        self.source = kwparams.get('source', None)
        self.target = kwparams.get('target', None)
        self.params = params
        self.kwparams = kwparams
        self.n_dim = None
        self.n_batch = None
        self.reshape = [(0,), (1,)]
        """
        e.g. [((dl[0]⁻, dl[0]⁺), (dl[1]⁻, dl[1]⁺), (dl[2]⁻, dl[2]⁺)), (s[0], s[1], s[2]), (d[0], d[1]), (d[2], d[3])] means:
            1) transpose dimensions d[2] and d[3]; followed by d[0] with d[1];
            2) rescale at dimension d with s[d];
            3) crop(* < 0) or pad(* > 0) a side by dl[d]♯,
                where #=+ for the larger side and #=- for the lower side;
        The shape transformation goes from left to right. Only the firt two transformation can be a tuple with length other than 2, 
            meaning scaling and padding instead of transpose. Having a length 1, e.g. (*,), stands for isotropic scaling or padding;
            being (1,) means no scaling, (0,) means no padding.
        """
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

    for_physical_space = for_phy = alias("__call_physical_space__")
    for_image_space = for_img = alias("__call_image_space__")
    for_non_spatial = for_non = alias("__call_non_spatial__")
    
    @property
    def implemented_domains(self):
        domains = ('image-space', 'physical-space', 'non-spatial')
        defined = [getattr(self, '__call_' + d.replace('-', '_') + '__') != NotImplemented for d in domains]
        # avouch(any(defined), NotImplementedError(f"{self.__name__} needs re-implementation of __call_physical_space/image_space/non_spatial__ during inheritance. "))
        return [d for d, y in zip(domains, defined) if y]
    
    @property
    def core_domain(self): return self.implemented_domains[0]
    
    def __call__(self, X, domain=None, use_implementation=None, to_space=None, **kwargs):
        """
        Please re-implement '__call__' by sub-methods '__call_physical_space__', '__call_image_space__', '__call_non_spatial__', instead of directly replacing '__call__'. 
            OR one may use '@Transformation.for_physical_space/ for_image_space/ for_non_spatial' for unreserved functions names. 
        domain = physical_space/ image_space::
            X (bt.Tensor): Coordinates to be transformed.
                size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
            Returns (bt.Tensor): The transformed coordinates.
                size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        domain = non_spatial::
            X (bt.Tensor): Images to be transformed.
                size: ({n_batch: optional}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
            Returns (bt.Tensor): The transformed images.
                size: ({n_batch}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
        domain = file::
            X (str): The path to the file to be transformed.
            output_path (str): The path to the output file. If not specified, the output file will be saved in the same directory as the input file.
            Returns: None
        
        Other Args:
            use_implementation (str): The implementation function used to perform transformation. Defaults to the implementation for self.domain. 
            to_space (tuple or str): The output shape (file path for reference output space) for the non-spatial transformation. 
        """
        # Deal with file transformation
        if isinstance(X, str):
            avouch(domain is None or domain == 'file', TypeError("Only file transformation accept image input of type 'str'. "))
            avouch(isinstance(to_space, str), TypeError(f"The argument to_space={to_space!r} should be string paths when performing file transformation. "))
            R = IMG(to_space)
            Rshape = R.shape
            X = IMG(X)
            Xaff = X.affine
            Xdata = X.to_tensor()
            prev_srctgt = self.source, self.target
            print(X.affine, R.affine)
            self.between_spaces(X, R)
            Ydata = self(Xdata, to_space=Rshape, domain='non-spatial', **kwargs)
            self.source, self.target = prev_srctgt
            if 'output_path' in kwargs:
                Y = kwargs.pop('output_path')
                avouch(isinstance(Y, str), TypeError(f"The argument output_path={Y!r} should be string paths when performing file transformation. "))
                R.save(Ydata, Y)
                return
            class Warped:
                def __init__(self, data, reference): self.data = data; self.reference = reference
                def __rshift__(self, output_path): self.reference.save(self.data, output_path)
            return Warped(Ydata, R)
        
        # We first determine core_domain: the domain of the transformation. domain: the domain of the input. 
        core_domain = self.domain
        if not (X.has_channel and (self.n_dim is None or X.n_channel == self.n_dim)): domain = 'non-spatial'
        if domain is None: domain = core_domain
        
        # Then, we find the mapping in between: the implementation in core_domain unless use_implementation is specified to select an implementation. 
        if use_implementation is None: use_implementation = self.use_implementation
        if use_implementation:
            func = getattr(self, '__call_' + use_implementation.replace('-', '_') + '__')
        else: func = getattr(self, '__call_' + core_domain.replace('-', '_') + '__')
        if func == NotImplemented:
            if core_domain == 'physical-space' and self.__class__ in [Affine]:
                func = self.__call_image_space__
            else: raise NotImplementedError(f"{self.__name__} does not implement {core_domain} transformation. ")
        
        ##  check & initialize the input ##
        ###################################
        no_space = False
        if domain == 'non-spatial': # non-spatial image transformation
            n_dim = X.n_space_dim
        else: # domain in ('physical-space', 'image-space')
            avouch(X.has_channel, f"Please use batorch tensor of size ({{n_batch:optional}}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim] for {self.__name__}, instead of {X.shape}. ")
            n_dim = X.n_channel
            if X.n_space_dim == 0: no_space = True; X = X.unsqueeze(-1)
            # avouch(X.n_channel == n_dim, f"{n_dim}D {self.__name__} does not take coordinates of size {X.shape}")
        if self.n_dim is not None:
            avouch(n_dim == self.n_dim, f"{self.n_dim}D {self.__name__} does not take inputs of size {X.shape}")
        if X.has_batch and self.n_batch is not None:
            avouch(self.n_batch == 1 or X.n_batch == 1 or X.n_batch == self.n_batch, 
            f"{self.n_dim}D {self.__name__} with batch size {self.n_batch} does not take inputs with wrong batch size. Current size: {X.shape}.")

        return_image = False
        if domain == 'non-spatial' and core_domain != 'non-spatial':
            return_image = True
            input_images = X
            if to_space is None: to_space = X.space
            X = bt.image_grid(*perform_reshape(to_space, self.reshape)).duplicate(self.n_batch if self.n_batch is not None else 1, {}).float()
            domain = 'image-space'

        Y = X.clone().float()
        if not Y.has_batch:
            if self.n_batch is not None: Y = Y.duplicate(self.n_batch, {})
            else: Y = Y.unsqueeze({})
        elif Y.n_batch == 1 and self.n_batch is not None: Y = Y.amplify(self.n_batch, {})
        
        ##    convert between domains    ##
        ###################################
        self.trans_record = []
        if domain == core_domain:
            R = func(Y)
            if isinstance(self, Translation): self.trans_record.append(f"- {self.__name__} Transformation with param {self.translation}")
            elif isinstance(self, (Affine, Rigid)): self.trans_record.append(f"- {self.__name__} Transformation with param:\n{self.params[0]}")
            else: self.trans_record.append(f"- {self.__name__} Transformation with param {self.params!r}")
            if not return_image:
                if no_space: return R.squeeze(...)
                return R
        else:
            if core_domain == 'non-spatial': raise TypeError("Cannot input coordinates for non-spatial transformations, please use keyword argument 'domain='non-spatial'' when calling to identify it. ")
            avouch(self.source is not None and self.target is not None, "Cannot perform image/physical-space transformation of image coordinates without source/target information. ")

            taffine = Transformation.__get_affine__(self.target).type(Y.dtype)
            saffine = Transformation.__get_affine__(self.source).type(Y.dtype)
            if self.backward: prev_affine, post_affine = taffine, saffine
            else: prev_affine, post_affine = saffine, taffine
            if domain == 'image-space': post_affine = post_affine.inv()
            else: prev_affine = prev_affine.inv()
            Y = (bt.matmul(prev_affine[..., :n_dim, :n_dim], Y.flatten(...).add_special_dim(-1, [])) + prev_affine[..., :n_dim, n_dim:]).view_as(Y)
            R = func(Y)
            R = (bt.matmul(post_affine[..., :n_dim, :n_dim], R.flatten(...).add_special_dim(-1, [])) + post_affine[..., :n_dim, n_dim:]).view_as(R)
            self.trans_record.append(f"- Affine Transformation with matrix:\n{prev_affine}")
            if isinstance(self, Translation): self.trans_record.append(f"- {self.__name__} Transformation with param {self.translation}")
            elif isinstance(self, (Affine, Rigid)): self.trans_record.append(f"- {self.__name__} Transformation with param:\n{self.params[0]}")
            else: self.trans_record.append(f"- {self.__name__} Transformation with param {self.params!r}")
            self.trans_record.append(f"- Affine Transformation with matrix:\n{post_affine}")

        if return_image:
            if self.backward: return interpolation(input_images, target_space=R, **kwargs)
            else: return interpolation_forward(input_images, target_space=R, **kwargs)
        if no_space: return R.squeeze(...)
        return R
    
    __call_physical_space__ = NotImplemented
    __call_image_space__ = NotImplemented
    __call_non_spatial__ = NotImplemented
    __affine__ = NotImplemented
    __inv__ = NotImplemented
    
    @staticmethod
    def __get_affine__(x):
        if isinstance(x, str): x = IMG(x)
        if isinstance(x, IMG): x = x.affine
        x = bt.to_bttensor(x).auto_device()
        avouch(2 <= x.n_dim <= 3, TypeError(f"Invalid affine matrix represented by shape {x.shape}. "))
        avouch(not x.has_func and not x.has_sequence, TypeError(f"Invalid affine matrix represented by shape {x.shape}, only batch/feature dimensions are allowed. "))
        if x.n_dim == 3: x.special_from(bt.Size({1}, [1, 1]))
        else: x.special_from(bt.Size([1, 1]))
        avouch(x.feature[0] == x.feature[1], TypeError(f"Invalid affine matrix represented by shape {x.shape}: should be of shape (n_dim+1, n_dim+1). "))
        return x
    
    @property
    def __name__(self):
        st = f" w/o ST" if self.source is None or self.target is None else f" w/ ST"
        return f"{self.__class__.__name__.split('.')[-1]} ({self.domain}{st}) transformation"

    @alias("__repr__")
    def __str__(self):
        d = ("backward" if self.backward else "forward") if self.domain in ('physical-space', 'image-space') else ''
        if getattr(self, 'main_param', None) is None: return f"<[{d}] {self.__name__}>"
        return f"<[{d}] {self.__name__} with param size: ({str(self.main_param.shape).split('(')[-1]}>"

    def __getitem__(self, i):
        if self.n_batch is None: raise TypeError(f"Cannot subscript a {self.__name__} without batch dimension. ")
        clone = copy(self)
        if isinstance(i, int): i = slice(i, i + 1); n_count = 1
        elif isinstance(i, slice): n_count = (i.stop - i.start) // i.step
        else: i = bt.to_bttensor(i).long(); n_count = i.n_ele
        clone.params = tuple(p[i] for p in self.params)
        clone.n_batch = n_count
        if len(clone.params) > 0 and isinstance(clone.params[0], bt.Tensor): clone.main_param = clone.params[0]
        clone.require_grad_params = [p[i] for p in self.require_grad_params]
        for p in self.batch_param: setattr(clone, p, getattr(self, p)[i])
        return clone

    def parameters(self):
        for p in self.require_grad_params: yield p

    def detach(self):
        trans_dict = self.to_dict()
        params = tuple(x.detach() if isinstance(x, (bt.torch.Tensor, Transformation)) else x for x in self.params)
        trans_dict['params'] = params
        detached = self.__class__(*params, **self.kwparams)
        detached.from_dict(trans_dict)
        return detached

    def train(self): return self
    def eval(self): return self
    
    @property
    def device(self):
        for p in self.params:
            if hasattr(p, 'device'): return p.device
    
    def to(self, target):
        result = copy(self)
        result.from_dict({k: v.to(target) if isinstance(v, (bt.torch.Tensor, Transformation)) else v for k, v in self.to_dict().items()})
        return result

    def to_dict(self):
        return dict(
            type = self.__class__.__name__,
            n_dim = self.n_dim,
            n_batch = self.n_batch,
            reshape = self.reshape,
            params = self.params,
            kwparams = self.kwparams,
            domain = self.domain,
            backward = self.backward,
            source = self.source,
            target = self.target
        )

    def from_dict(self, d):
        self.params = d['params']
        self.kwparams = d['kwparams']
        self.n_dim = d['n_dim']
        self.n_batch = d['n_batch']
        self.reshape = d['reshape']
        self.domain = d['domain']
        self.backward = d['backward']
        self.source = d['source']
        self.target = d['target']

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
        for key in ('n_dim', 'n_batch', 'reshape', 'domain', 'backward'):
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
        
    def is_in_domain(self, domain): return self.domain == domain
    
    def as_in_domain(self, domain): self.domain = domain

    def is_spatial_trans(self): return self.domain in ('physical-space', 'image-space')

    @alias('is_non_spatial_trans')
    def is_image_trans(self): return self.domain == 'non-spatial'

    def as_spatial_trans(self):
        if 'image-space' not in self.implemented_domains and 'physical-space' not in self.implemented_domains:
            raise NotImplementedError(f"Methods for image/physical coordinates are not defined in {self.__name__}, hence it can only be intensity transformation. ")
        if self.domain == 'non-spatial': self.domain = 'image-space'
        return self

    @alias('as_non_spatial_trans')
    def as_image_trans(self):
        if self.domain != 'non-spatial': self.domain = 'non-spatial'
        return self

    def __matmul__(self, other):
        """Operator: self @ other
        Arg other: 
            Transformation for composition of transformations;
            batorch.Tensor for image array (with space dimensions).
                            or affine transformation (with feature dimensions).
        """
        try:
            if not isinstance(other, bt.torch.Tensor): I = bt.tensor(other)
            elif not isinstance(other, bt.Tensor): I = other.as_subclass(bt.Tensor).init_special()
            if I.n_space_dim > 0:
                avouch(self.is_image_trans(), "Cannot conver image I by T(I) unless T is an image transformation. ")
                return self(I, domain='non-spatial')
            if I.n_feature_dim == 2: other = Affine(I, domain=self.domain)
        except TypeError: ...
        if isinstance(other, Transformation):
            return ComposedTransformation(self, other)
        raise TypeError(f"Cannot perform combination of {self} and {other}: only transformations or batorch images can be combined with Transformation. ")

    def __imatmul__(self, other):
        """In-place operator: self @= other
        Arg other: 
            Transformation for composition of transformations
            batorch.Tensor for affine transformation (with feature dimensions).
            [cannot be image tensor]
        """
        try:
            if not isinstance(other, bt.torch.Tensor): otr = bt.tensor(other)
            elif not isinstance(other, bt.Tensor): otr = other.as_subclass(bt.Tensor).init_special()
            if otr.n_feature_dim == 2: other = Affine(otr, domain=self.domain)
            else: raise TypeError(f"Cannot perform combination of {self} and {other}: only transformations can be combined with Transformation. ")
        except TypeError: ...
        if isinstance(other, Transformation):
            return ComposedTransformation(self, other)
        raise TypeError(f"Cannot perform combination of {self} and {other}: only transformations can be combined with Transformation. ")

    def __rmatmul__(self, other):
        """Operator: other @ self
        Arg other: 
            Transformation for composition of transformations;
            batorch.Tensor for image array (with space dimensions).
                            or affine transformation (with feature dimensions).
        """
        try:
            if not isinstance(other, bt.torch.Tensor): I = bt.tensor(other)
            elif not isinstance(other, bt.Tensor): I = other.as_subclass(bt.Tensor).init_special()
            if I.n_space_dim > 0:
                avouch(self.is_spatial_trans(), "Cannot conver image I by I∘T unless T is a spatial transformation. ")
                return self(I, domain='non-spatial')
            if I.n_feature_dim == 2: return Affine(I, domain=self.domain) @ self
        except TypeError: ...
        raise TypeError(f"Cannot perform combination of {other} and {self}: only transformations or batorch images can be combined with Transformation. ")
    
    #######################################
    ## Requires: spatial transformations ##
    #######################################

    @req_spatial
    def is_forward(self): return not self.backward

    @req_spatial
    def is_backward(self): return self.backward
    
    @req_spatial
    def backward_(self, backward_value):
        self.backward = backward_value
        return self

    @req_spatial
    def direct_inv(self):
        self.backward = not self.backward
        return self
    
    @req_spatial
    def inv_with_direct(self):
        return self.inv().direct_inv()

    @req_spatial
    def as_forward(self, is_forward = True):
        self.backward = not is_forward
        return self

    @req_spatial
    def as_backward(self, is_backward = True):
        self.backward = is_backward
        return self

    @req_spatial
    def to_forward(self, is_forward = True):
        if self.backward == is_forward:
            return self.inv().backward_(not is_forward)
        return self

    @req_spatial
    def to_backward(self, is_backward = True):
        if self.backward != is_backward:
            return self.inv().backward_(is_backward)
        return self

    @req_spatial
    def between_spaces(self, source, target):
        if isinstance(source, str): source = IMG(source)
        if isinstance(target, str): target = IMG(target)
        self.source = source
        self.target = target
        return self
    
    @req_spatial
    def is_in_image_space(self): return self.domain == 'image-space'
    
    @alias('in_world_space')
    @req_spatial
    def is_in_physical_space(self): return self.domain == 'physical-space'

    @req_spatial
    def as_in_image_space(self, **kwargs):
        avouch(self.domain == 'physical-space', "Only transformation between physical spaces can be regarded as image-space transformation by calling 'as_in_image_space'. ")
        self.source = kwargs.get('source', self.source)
        self.target = kwargs.get('target', self.target)
        # avouch(self.source is not None and self.target is not None, f"'source' and 'target' properties are needed in 'as_in_image_space', one may either use keyword arguments 'source=' and 'target=', or use method 'between_spaces' to record source and target spaces.")
        self.domain = 'image-space'
        return self

    @alias('as_in_world_space')
    @req_spatial
    def as_in_physical_space(self, **kwargs):
        avouch(self.domain == 'image-space', "Only transformation between image spaces can be regarded as physical-space transformation by calling 'as_in_physical_space'. ")
        self.source = kwargs.get('source', self.source)
        self.target = kwargs.get('target', self.target)
        # avouch(self.source is not None and self.target is not None, f"'source' and 'target' properties are needed in 'as_in_physical_space', one may either use keyword arguments 'source=' and 'target=', or use method 'between_spaces' to record source and target spaces.")
        self.domain = 'physical-space'
        return self
    
    @req_spatial
    def to_image_space(self, **kwargs):
        avouch(self.domain == 'physical-space', "Only transformation between physical spaces can be converted into image-space transformation. ")
        source = kwargs.get('source', self.source)
        target = kwargs.get('target', self.target)
        avouch(source is not None and target is not None, f"'source' and 'target' properties are needed in 'to_image_space', one may either use keyword arguments 'source=' and 'target=', or use method 'between_spaces' to record source and target spaces.")
        source_affine = Transformation.__get_affine__(self.source)
        target_affine = Transformation.__get_affine__(self.target)
        if self.backward:
            return ComposedTransformation(Affine(bt.inv(source_affine)), copy(self).as_in_image_space(), Affine(target_affine), domain='image-space', backward=self.backward)
        else:
            return ComposedTransformation(Affine(bt.inv(target_affine)), copy(self).as_in_image_space(), Affine(source_affine), domain='image-space', backward=self.backward)
    
    @alias('to_world_space')
    @req_spatial
    def to_physical_space(self, **kwargs):
        avouch(not self.physical, "Only transformation between image spaces can be converted into physical-space transformation. ")
        source = kwargs.get('source', self.source)
        target = kwargs.get('target', self.target)
        avouch(source is not None and target is not None, f"'source' and 'target' properties are needed in 'to_physical_space', one may either use keyword arguments 'source=' and 'target=', or use method 'between_spaces' to record source and target spaces.")
        source_affine = Transformation.__get_affine__(self.source)
        target_affine = Transformation.__get_affine__(self.target)
        if self.backward:
            return ComposedTransformation(Affine(source_affine), copy(self).as_in_physical_space(), Affine(bt.inv(target_affine)), domain='physical-space', backward=self.backward)
        else:
            return ComposedTransformation(Affine(target_affine), copy(self).as_in_physical_space(), Affine(bt.inv(source_affine)), domain='physical-space', backward=self.backward)

    @req_spatial
    def affine(self, n_dim=None, domain='image-space', backward=True):
        if self.__affine__ == NotImplemented: return
        
        aff = self.__affine__(n_dim)
        if self.backward != backward: aff = aff.inv()
        if self.domain != domain:
            avouch(self.source is not None and self.target is not None, "Cannot perform image/physical-space transformation of image coordinates without source/target information. ")
            taffine = Transformation.__get_affine__(self.target)
            saffine = Transformation.__get_affine__(self.source)
            if self.domain == 'image-space' and backward: aff = saffine @ aff @ taffine.inv()
            elif self.domain == 'image-space': aff = taffine @ aff @ saffine.inv()
            elif backward: aff = saffine.inv() @ aff @ taffine
            else: aff = taffine.inv() @ aff @ saffine
        return aff
    
    @alias("is_linear")
    @property
    def is_affine(self):
        return self.__affine__ != NotImplemented
    
    def inv(self):
        if self.domain in ('image-space', 'physical-space'): return self.__inv__().backward_(self.backward)
        else: return self.__inv__()

    @alias('toDDF')
    @req_spatial
    def to_DDF(self, *shape, domain='image-space', **kwargs):
        shape = arg_tuple(shape)
        image_grid = bt.image_grid(*shape).unsqueeze({}).float()
        if domain == 'physical-space':
            self.source = kwargs.get('source', self.source)
            self.target = kwargs.get('target', self.target)
            avouch(source is not None and target is not None, f"'source' and 'target' properties are needed in to_DDF(domain='physical-space'), one may either use keyword arguments 'source=' and 'target=', or use method 'between_spaces' to record source and target spaces.")

            if self.backward: affine = Transformation.__get_affine__(self.target)
            else: affine = Transformation.__get_affine__(self.source)
            physical_grid = bt.matmul(affine, image_grid)
            return self(physical_grid, domain='physical-space') - physical_grid
        return self(image_grid, domain='image-space') - image_grid

    @req_spatial
    def save_as_nii(self, nii_path: str, target=None):
        if target is None: target = self.target
        if isinstance(target, str): target = IMG(target)
        avouch(isinstance(target, IMG), "Method 'Transformation.save_as_nii' needs a Nifti image 'target' as a keyword argument to identify the physical space. ")
        target.save(self.to_DDF(*target.shape, domain='physical-space').movedim([], -1), nii_path)
    
    # def num_inv(self, *size, iterate=True, verbose=False):
    #     from .funcs import bending
    #     size = arg_tuple(size)
    #     n_dim = len(size)
    #     X = bt.image_grid(*size).unsqueeze([]).float()
    #     Jac = bt.grad_image(self(X)) # ({n_batch}, n_dim, [n_dim], *n_data)
    #     scale = ((Jac ** 2).sum(1) * n_dim).sqrt().max({}).values # ({n_batch}, *n_data)
    #     sigma = min((scale / 2.5).quantile(0.9).item(), 1.5)
    #     if not iterate:
    #         is_backward = False
    #         if self.backward: self.forward_(); is_backward = True
    #         inv_disp = interpolation_forward(X, self, target_space=size, fill='nearest', sigma=sigma) - X
    #         if is_backward: self.backward_()
    #         inv_disp = bt.conv(inv_disp.mergedims({}, []), bt.gaussian_kernel(n_dim, kernel_size=3)).view_as(inv_disp)
    #         inv_disp = [inv_disp]
    #     else:
    #         inv_disp = - self.to_DDF(*size).clone()
    #         inv_disp = [inv_disp.detach().requires_grad_(True)]
    #         optimizer = bt.Optimization(bt.optim.Adam, inv_disp, lr = 1e-2)
    #         nodrop_count = 0
    #         prev_loss = None
    #         for i in range(400):
    #             invtrans = DenseDisplacementField(inv_disp[0])
    #             loss = bt.norm2(invtrans(self.detach()(X)) - X).mean() + 1e-3 * bending(inv_disp[0])
    #             optimizer.minimize(loss)
    #             if verbose: print(f"Iteratively inverse transformation: [iter {i+1}] loss = {loss.item()}")
    #             if loss.item() < 1e-2:
    #                 if verbose: print(f"Stop at iter {i+1} due to small loss")
    #                 break
    #             if nodrop_count >= 4:
    #                 if verbose: print(f"Stop at iter {i+1} due to no dropping. ")
    #                 break
    #             if prev_loss is not None and loss.item() >= prev_loss: nodrop_count += 1
    #             else: nodrop_count = 0
    #     return DenseDisplacementField(inv_disp[0])
    
    def force_inv(self, *size):
        if self.inv != NotImplemented: return self.inv()
        else: return self.num_inv(*size)

@alias("CompoundTransformation")
class ComposedTransformation(Transformation):
    """
    Create a composed transformation. 

    Note: The composed transformation `T` consist of a list of transformations, `[T₁, T₂, ⋯, Tₖ]`. 
        If domain = image/physical-space, backward = True (represented by image-space transformations):
            T(X) = T₁(T₂( ⋯ (Tₖ(X)))) for coordinates X, i.e., I∘T = I∘T₁∘T₂∘ ⋯ ∘Tₖ.
        If domain = image/physical-space, backward = False (represented by image-space transformations):
            I₁(T₁(x)) = I(x); ⋯ ; Iₖ∘Tₖ(x) = Iₖ₋₁(x), i.e., Iₖ∘Tₖ∘Tₖ₋₁∘ ⋯ ∘T₁ = I; Iₖ∘T = I.
        If domain = non-spatial:
            T(X) = Tₖ(Tₖ₋₁( ⋯ (T₁(I)))) for images I, i.e., T∘I = Tₖ∘Tₖ₋₁∘ ⋯ ∘T₁∘I. 
        Always remember, transformation is performed in order T₁, T₂, ⋯ , Tₖ, 
            except continuous coordinate transformations go backward. 

    Args:
        trans_list (list or tuple): The list of transformations, operating one by one onto an image. 
            A series of backward transformations would be equivalent to a composition of functions from right to left.
    
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
            OR Images to be transformed. 
            size: ({n_batch:optional}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    @staticmethod
    def __flat__(trans_list):
        flat_trans_list = []
        for t in trans_list:
            if not isinstance(t, ComposedTransformation): flat_trans_list.append(t)
            else: flat_trans_list.extend(ComposedTransformation.__flat__(t.trans_list))
        return flat_trans_list

    def __init__(self, *trans_list, domain=None, backward=None, **kwparams):
        source = None; target = None
        if 'source' in kwparams: source = kwparams.pop('source')
        if 'target' in kwparams: target = kwparams.pop('target')
        super().__init__(*trans_list, **kwparams)
        
        ##  find uniform n_dim & n_batch ##
        ###################################
        n_dims = set([t.n_dim for t in trans_list if t.n_dim is not None])
        n_batches = set([t.n_batch for t in trans_list if t.n_batch is not None])
        if 1 in n_batches: n_batches.remove(1)
        avouch(len(n_dims) <= 1, "Composition Failed: All components should have a same dimension. ")
        avouch(len(n_batches) <= 1, "Composition Failed: All components should have a same batch size. ")
        self.n_dim = None if len(n_dims) == 0 else n_dims.pop()
        self.n_batch = None if len(n_dims) == 0 else n_batches.pop()
        
        ## find uniform domain & backward##
        ###################################
        self.trans_list = ComposedTransformation.__flat__(trans_list)
        if domain is None:
            if any(t.domain == 'non-spatial' for t in trans_list): domain = 'non-spatial'
            else:
                n_img = sum(t.domain == 'image-space' or (t.source is not None and t.target is not None) for t in trans_list)
                n_phy = sum(t.domain == 'physical-space' or (t.source is not None and t.target is not None) for t in trans_list)
                if n_img == len(trans_list) and n_phy == len(trans_list): domain = trans_list[-1].domain
                elif n_img == len(trans_list): domain = 'image-space'
                elif n_phy == len(trans_list): domain = 'physical-space'
                else: raise TypeError("Only spatial transformations in the same type of image space (image/physical) can be composed together, please add source/target to the elements so that they can be converted. ")
        self.domain = domain
        if domain != 'non-spatial' and backward is None:
            n_bac = sum(t.backward or t.inv != NotImplemented for t in trans_list)
            n_for = sum(not t.backward or t.inv != NotImplemented for t in trans_list)
            if n_bac == len(trans_list): backward = True
            elif n_for == len(trans_list): backward = False
            else: raise TypeError("Only spatial transformations in the same direction can be composed together, please ensure that they can be inversed so that they can be converted to the same direction. ")
        self.backward = backward

        ##  combine the reshape property ##
        ###################################
        accum = [(0,), (1,)]
        add_pad = lambda x, y: tuple(a+b for a, b in zip(x if isinstance(x, tuple) else (x, x), y if isinstance(y, tuple) else (y, y)))
        for t in self.trans_list:
            if len(t.reshape) == 1: accum = t.reshape; continue
            if len(accum) == 1: accum = [perform_reshape(accum[0], t.reshape)]; continue
            padding_a, scale_a, *pairs_a = accum
            padding_t, scale_t, *pairs_t = t.reshape

            if len(padding_a) == 1: padding_a *= len(padding_t)
            else:
                padding_a = list(padding_a)
                for p, q in pairs_t[::-1]: padding_a[p], padding_a[q] = padding_a[q], padding_a[p]
                if len(padding_t) == 1: padding_t *= len(padding_a)
            padding = tuple(add_pad(x, y) for x, y in zip(padding_t, padding_a))

            if len(scale_a) == 1: scale_a *= len(scale_t)
            else:
                scale_a = list(scale_a)
                for p, q in pairs_t[::-1]: scale_a[p], scale_a[q] = scale_a[q], scale_a[p]
                if len(scale_t) == 1: scale_t *= len(scale_a)
            scale = tuple(x * y for x, y in zip(scale_t, scale_a))

            pairs = pairs_t + pairs_a
            r_result = [padding, scale]
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
                    r_result.append((p, q))
                    p = q
            
        self.reshape = r_result
        if source is not None: self.source = source
        if target is not None: self.target = target
    
    def __len__(self): return len(self.trans_list)
    
    def __getitem__(self, i): return ComposedTransformation(*[t[i] for t in self.trans_list])
    
    def get_trans(self, i): return self.trans_list[i]
    
    @property
    def __name__(self):
        return f"Composed ({self.domain}) transformation"

    def __repr__(self):
        d = ("backward" if self.backward else "forward") if self.domain in ('physical-space', 'image-space') else ''
        return f"<[{d}]{self.__name__} of [{', '.join([repr(t).strip('<>') if isinstance(t, ComposedTransformation) else t.__name__ for t in self.trans_list])}]>"
    
    def __str__(self):
        inner_listed = ',\n'.join('    ' + str(t) for t in self.trans_list)
        return f"Composed transformation(\n{inner_listed}\n)"

    def inv(self):
        if not all([t.inv != NotImplemented for t in self.trans_list]):
            if any([t.domain == 'non-spatial' and t.inv == NotImplemented for t in self.trans_list]):
                raise TypeError("Composed transformation not invertable: Not all image transformation components invertable. ")
            Warn("Composed transformation not invertable: Not all spatial transformation components invertable. Using forward transformation instead.")
        return ComposedTransformation(*[t.inv() if t.inv != NotImplemented else t.direct_inv() for t in self.trans_list[::-1]], domain=self.domain, backward=self.backward)
    
    def force_inv(self, *size):
        if all([t.inv != NotImplemented for t in self.trans_list]): return self.inv()
        if any([t.domain == 'non-spatial' and t.inv == NotImplemented for t in self.trans_list]):
            raise TypeError("Composed transformation not invertable: Not all image transformation components invertable. ")
        return ComposedTransformation(*[t.inv() if t.inv != NotImplemented else t.force_inv(*size) for t in self.trans_list[::-1]], domain=self.domain, backward=self.backward)
    
    @property
    def implemented_domains(self):
        domains = ('image-space', 'physical-space', 'non-spatial')
        implemented = {
            'image-space': ['image-space', 'non-spatial'],
            'physical-space': ['physical-space'],
            'non-spatial': ['non-spatial'],
            'image-space-aff': ['image-space', 'physical-space', 'non-spatial'],
            'physical-space-aff': ['physical-space', 'image-space'],
            'non-spatial-aff': ['non-spatial']
        }
        defined = [all(d in sum([implemented[imp_dom] if t.source is None else implemented[imp_dom + '-aff'] for imp_dom in t.implemented_domains], []) for t in self.trans_list) for d in domains]
        avouch(any(defined), NotImplementedError(f"{self.__name__} needs re-implementation of __call_physical_space/image_space/non_spatial__ during inheritance. "))
        return [d for d, y in zip(domains, defined) if y]
    
    @property
    def source(self): return self.trans_list[0].source
    @source.setter
    def source(self, value):
        if hasattr(self, 'trans_list'):
            self.trans_list[0].source = value

    @property
    def target(self): return self.trans_list[-1].target
    @target.setter
    def target(self, value):
        if hasattr(self, 'trans_list'):
            self.trans_list[-1].target = value

    def __call__(self, X, domain=None, use_implementation=None, to_space=None, **kwargs):
        """
        Perform composed transformation. 
        
        domain = physical/image-space::
            X (bt.Tensor): Coordinates to be transformed.
                size: ({n_batch: optional}, [r], n_1, n_2, ..., n_r) [r=n_dim]
            Returns (bt.Tensor): The transformed coordinates.
                size: ({n_batch}, [r], n_1, n_2, ..., n_r) [r=n_dim]
        domain = non-spatial::
            X (bt.Tensor): Images to be transformed.
                size: ({n_batch: optional}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
            Returns (bt.Tensor): The transformed images.
                size: ({n_batch}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
        
        Other Args:
            use_implementation (str): The implementation function used to perform transformation. Defaults to the implementation for self.domain. 
            to_space (tuple): The output shape for the non-spatial transformation. 
        """
        # Deal with file transformation
        if isinstance(X, str):
            return super().__call__(X, domain=domain, use_implementation=use_implementation, to_space=to_space, **kwargs)
        
        if not (X.has_channel and (self.n_dim is None or X.n_channel == self.n_dim)): domain = 'non-spatial'
        if domain is None: domain = self.domain

        self.trans_record = []
        accum_spatial_trans = []
        for t in self.trans_list + [None]:
            if t is not None and t.domain != 'non-spatial' and (t.backward or t.inv != NotImplemented): accum_spatial_trans.append(t); continue
            if len(accum_spatial_trans) > 0:
                n_img = sum(t.domain == 'image-space' or (t.source is not None and t.target is not None) for t in accum_spatial_trans)
                n_phy = sum(t.domain == 'physical-space' or (t.source is not None and t.target is not None) for t in accum_spatial_trans)
                if n_img == len(accum_spatial_trans) and n_phy == len(accum_spatial_trans):
                    n_img = sum(t.domain == 'image-space' for t in accum_spatial_trans)
                    n_phy = sum(t.domain == 'physical-space' for t in accum_spatial_trans)
                    if n_img > n_phy: core_domain = 'image-space'
                    else: core_domain = 'physical-space'
                elif n_img == len(accum_spatial_trans): core_domain = 'image-space'
                elif n_phy == len(accum_spatial_trans): core_domain = 'physical-space'
                else: core_domain = domain

                return_image = False
                if domain == 'non-spatial':
                    return_image = True
                    input_images = X
                    if to_space is None: to_space = X.space
                    X = bt.image_grid(*perform_reshape(to_space, self.reshape)).duplicate(self.n_batch if self.n_batch is not None else 1, {}).float()
                    domain = 'image-space'
                
                if domain == core_domain:
                    for st in accum_spatial_trans[::-1]:
                        if st.backward: ct = st
                        else: ct = st.inv()
                        X = ct(X, domain=core_domain)
                        self.trans_record.extend(ct.trans_record)
                else:
                    source = self.source
                    target = self.target
                    avouch(source is not None and target is not None, "Cannot perform image/physical-space transformation for image coordinates without source/target information. ")

                    taffine = Transformation.__get_affine__(target).type(X.dtype)
                    saffine = Transformation.__get_affine__(source).type(X.dtype)
                    prev_affine, post_affine = taffine, saffine
                    if domain == 'image-space': post_affine = post_affine.inv()
                    else: prev_affine = prev_affine.inv()
                    
                    coord_str = lambda x: f"{x[0, 0]!r}".split('val=')[0] + "val=" + ', '.join([f"{x[0, i]!r}".rstrip('>').split('val=')[-1] for i in range(self.n_dim)]) + '>'
                    self.trans_record.append(f"[COORD.] {coord_str(X)}")
                    X = (bt.matmul(prev_affine[..., :n_dim, :n_dim], X.flatten(...).add_special_dim(-1, [])) + prev_affine[..., :n_dim, n_dim:]).view_as(X)
                    self.trans_record.append(f"- Affine Transformation with matrix:\n{prev_affine}")
                    self.trans_record.append(f"[COORD.] {coord_str(X)}")
                    for st in accum_spatial_trans[::-1]:
                        if st.backward: ct = st
                        else: ct = st.inv()
                        X = ct(X, domain=core_domain)
                        self.trans_record.extend(ct.trans_record)
                        self.trans_record.append(f"[COORD.] {coord_str(X)}")
                    X = (bt.matmul(post_affine[..., :n_dim, :n_dim], X.flatten(...).add_special_dim(-1, [])) + post_affine[..., :n_dim, n_dim:]).view_as(X)
                    self.trans_record.append(f"- Affine Transformation with matrix:\n{post_affine}")
                    self.trans_record.append(f"[COORD.] {coord_str(X)}")
                
                if return_image: X = interpolation(input_images, target_space=X, **kwargs)
            if t is not None:
                X = t(X, domain=domain)
        return X

    @req_spatial
    def affine(self, n_dim=None, **kwargs):
        # avouch(self.is_spatial, "Cannot get `affine` from composed transformation with non-spatial transformation. ")
        comp = self.compose()
        avouch(len(comp) == 1, "Error in calling `affine`: Only the composition of linear transformations can result in an affine matrix.")
        trans = comp.trans_list[0]
        return trans.affine(n_dim, **kwargs)
    
    @alias("is_linear")
    @property
    def is_affine(self):
        return all(t.is_affine for t in self.trans_list)

    def compose(self):
        out_list = []
        n_dim = self.n_dim
        domain = self.domain
        if domain == 'non-spatial': domain = 'image-space'
        cur_affine = []
        for t in self.trans_list + [None]:
            aff = None
            if t is not None and t.domain != 'non-spatial': aff = t.affine(n_dim)
            if aff is None:
                if cur_affine:
                    accum_aff = None
                    for a in cur_affine:
                        aff = a.affine(n_dim, domain=self.domain, backward=self.backward)
                        if aff.space == (n_dim+1, n_dim+1):
                            aff = aff.view(aff.shape[:aff.space_start] + bt.Size([n_dim+1, n_dim+1]) + aff.shape[aff.space_stop:])
                        if accum_aff is None: accum_aff = aff; continue
                        if self.backward: accum_aff = accum_aff @ aff
                        else: accum_aff = aff @ accum_aff
                    if self.backward: source = cur_affine[0].source; target = cur_affine[-1].target
                    else: source = cur_affine[-1].source; target = cur_affine[0].target
                    out_list.append(Affine(accum_aff, domain=self.domain, backward=self.backward, source=source, target=target))
                    cur_affine.clear()
                if t is not None: out_list.append(t)
                continue
            elif aff.space != (n_dim+1, n_dim+1) and aff.feature != (n_dim+1, n_dim+1):
                raise TypeError(f"Unconsistent transformation with affine of size {aff.shape} in ComposedTransformation: space or feature dimensions should be of size ({n_dim+1}, {n_dim+1}). ")
            else: cur_affine.append(t)
        return ComposedTransformation(*out_list, domain=self.domain, backward=self.backward, source=self.source, target=self.target)

    def to_dict(self):
        dic = super().to_dict()
        dic['trans_list'] = [t.to_dict() for t in self.trans_list]
        return dic

# ########### Spatial Transformations ###########

@alias("Id")
class Identity(Transformation):
    """
    Identity transformation.
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates. (Same as X for Identity)
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_implementation = 'image-space'

    def __affine__(self, n_dim=None):
        if n_dim is None: return
        return bt.eye([n_dim + 1]).unsqueeze({})

    def __call_image_space__(self, X): return X

    def __inv__(self): return Identity()

class Rotation90(Transformation):
    """
    Transformation that rotates an image of `image_size` by 90 degrees.
    
    Note: The rotation is for coordinates, hence the image rotates clockwise. 
    
    Args:
        dim1, dim2 (int): The plane we rotate on. Direction of the rotation is from `dim1` to `dim2`.
            i.e. counter-clockwise rotation with dim1 as x-axis and dim2 as y-axis: [dim1, dim2] coordinates (x, y) becomes (ymax-y, x).
        image_size (tuple or bt.Tensor): The size of the image, or the image itself. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """

    def __init__(self, dim1, dim2, image_size=None, resize_image=True, **kwargs):
        super().__init__(dim1=dim1, dim2=dim2, image_size=image_size, resize_image=resize_image, **kwargs)

        self.dim1, self.dim2 = dim1, dim2
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)
        if resize_image: self.reshape = [(0,), (1,), (dim1, dim2)]
        self.resize_image = resize_image

    def __call_image_space__(self, X):
        dim1, dim2 = self.dim1, self.dim2
        select1 = (slice(None),) * X.channel_dim + (dim1,)
        select2 = (slice(None),) * X.channel_dim + (dim2,)
        if self.image_size is None: max_range = X[select2].max()
        else: max_range = self.image_size[dim2]
        X[select1] = X[select1] + max_range - X[select2]
        X[select2] = X[select1] + X[select2] - max_range
        X[select1] = X[select1] - X[select2]
        return X
        
    def __affine__(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        dim1, dim2 = self.dim1, self.dim2
        aff = bt.eye([n_dim + 1])
        aff[dim1][dim1] = 0.
        aff[dim1][dim2] = -1.
        aff[dim1][-1] = float(self.image_size[dim2])
        aff[dim2][dim2] = 0.
        aff[dim2][dim1] = 1.
        return aff.unsqueeze({})
    
    def __inv__(self): return Rotation270(self.dim1, self.dim2, image_size = self.image_size, resize_image = self.resize_image)

class Rotation270(Transformation):
    """
    Transformation that rotates an image by 270 degrees.
    
    Note: The rotation is for coordinates, hence the image rotates clockwise. 
    
    Args:
        dim1, dim2 (int): The plane we rotate on. Direction of the rotation is from `dim1` to `dim2`.
            i.e. counter-clockwise rotation with dim1 as x-axis and dim2 as y-axis: [dim1, dim2] coordinates (x, y) becomes (y, xmax-x).
        image_size (tuple or bt.Tensor): The size of the image, or the image itself. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, dim1, dim2, image_size=None, resize_image=True, **kwargs):
        super().__init__(dim1=dim1, dim2=dim2, image_size=image_size, resize_image=resize_image, **kwargs)

        self.dim1, self.dim2 = dim1, dim2
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)
        if resize_image: self.reshape = [(0,), (1,), (dim1, dim2)]
        self.resize_image = resize_image

    def __call_image_space__(self, X):
        dim1, dim2 = self.dim1, self.dim2
        select1 = (slice(None),) * X.channel_dim + (dim1,)
        select2 = (slice(None),) * X.channel_dim + (dim2,)
        if self.image_size is None: max_range = X[select1].max()
        else: max_range = self.image_size[dim1]
        X[select2] = X[select2] + max_range - X[select1]
        X[select1] = X[select2] + X[select1] - max_range
        X[select2] = X[select2] - X[select1]
        return X
        
    def __affine__(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        dim1, dim2 = self.dim1, self.dim2
        aff = bt.eye([n_dim + 1])
        aff[dim1][dim1] = 0.
        aff[dim1][dim2] = 1.
        aff[dim2][dim2] = 0.
        aff[dim2][dim1] = -1.
        aff[dim2][-1] = float(self.image_size[dim1])
        return aff.unsqueeze({})
    
    def __inv__(self): return Rotation90(self.dim1, self.dim2, image_size = self.image_size, resize_image = self.resize_image)

class Rotation180(Transformation):
    """
    Transformation that rotates an image by 180 degrees.
    
    Args:
        dim1, dim2 (int): The plane we rotate on. It is also the central symmetry.
            i.e. rotation with dim1 as x-axis and dim2 as y-axis: [dim1, dim2] coordinates (x, y) becomes (xmax-x, ymax-y).
        image_size (tuple or bt.Tensor): The size of the image, or the image itself. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, dim1, dim2, image_size=None, **kwargs):
        super().__init__(dim1=dim1, dim2=dim2, image_size=image_size, **kwargs)

        self.dim1, self.dim2 = dim1, dim2
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)

    def __call_image_space__(self, X):
        dim1, dim2 = self.dim1, self.dim2
        select1 = (slice(None),) * X.channel_dim + (dim1,)
        select2 = (slice(None),) * X.channel_dim + (dim2,)
        if self.image_size is None: max_range1, max_range2 = X[select1].max(), X[select2].max()
        else: max_range1, max_range2 = self.image_size[dim1], self.image_size[dim2]
        X[select1] = max_range1 - X[select1]
        X[select2] = max_range2 - X[select2]
        return X
        
    def __affine__(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        dim1, dim2 = self.dim1, self.dim2
        aff = bt.eye([n_dim + 1])
        aff[dim1][dim1] = -1.
        aff[dim1][-1] = float(self.image_size[dim1])
        aff[dim2][dim2] = -1.
        aff[dim2][-1] = float(self.image_size[dim2])
        return aff.unsqueeze({})

    def __inv__(self): return Rotation180(self.dim1, self.dim2, image_size = self.image_size)

@alias("Reflect")
class Reflection(Transformation):
    """
    Transformation that reflects an image along dimension dim.
    
    Args:
        dims (int's): The dimension we reflect the image along. 
            i.e. reflection on dims: coordinate x at dim (in dims) becomes x_max-x.
        image_size (tuple or bt.Tensor): The size of the image, or the image itself. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, *dims, image_size=None, **kwargs):
        super().__init__(dims=dims, image_size=image_size, **kwargs)

        if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
        self.dims = dims
        if isinstance(image_size, bt.torch.Tensor): image_size = image_size.shape
        self.image_size = image_size
        if image_size is not None: self.n_dim = len(image_size)

    def __call_image_space__(self, X):
        dims = self.dims
        for dim in dims:
            select = (slice(None),) * X.channel_dim + (dim,)
            if self.image_size is None: max_range = X[select].max()
            else: max_range = self.image_size[dim]
            X[select] = max_range - X[select]
        return X

    def __inv__(self): return Reflect(*self.dims, image_size = self.image_size)
        
    def __affine__(self, n_dim=None):
        if self.image_size is None: return
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        aff = bt.eye([n_dim + 1])
        for dim in self.dims:
            aff[dim][dim] = -1.
            aff[dim][-1] = float(self.image_size[dim])
        return aff.unsqueeze({})

@alias("Permutedim", "PermuteDim")
class DimPermutation(Transformation):
    """
    Permute the dimensions for an image, similar to np.transpose or torch/batorch.permute.
    
    Args:
        dims (list or tuple or bt.Tensor): The dimension permuation. 
            size: length(n_dim) or ({n_batch:optional}, [n_dim]).

    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, *dims, resize_image=True, **kwargs):
        if len(dims) == 1: dims = dims[0]
        if isinstance(dims, tuple): dims = list(dims)
        dims = bt.channel_tensor(dims).squeeze().long()
        bt.input_shape().set(dims = "({n_batch}, [n_dim])")
        # if dims.n_dim <= 1: dims = dims.unsqueeze({})
        # if dims.n_dim == 2:
        #     if not dims.has_batch: dims.batch_dimension = 0
        #     if not dims.has_channel: dims.channel_dimension = 0 if dims.batch_dimension > 0 else 1
        # avouch(dims.has_batch and dims.has_channel, f"Please use batorch tensor of size \
        #     ({{n_batch:optional}}, [n_dim]) for Translation parameters, instead of {dims.shape}. ")
        super().__init__(dims, resize_image=resize_image, **kwargs)

        self.dims = dims
        self.n_dim = dims.n_channel
        self.resize_image = resize_image
        if resize_image:
            if dims.n_batch > 1:
                dims_cap = dims.sample(random=False, dim={})
                avouch(bt.norm(dims - dims_cap).sum() < 1e-4, "Cannot resize image when different permutation done for different batch members. ")
            dims = dims.pick(0, {}).tolist()
            visited = []
            self.reshape = [(0,), (1,)]
            for p in range(self.n_dim):
                if p in visited: continue
                while True:
                    visited.append(p)
                    q = dims[p]
                    if q in visited: break
                    self.reshape.append((p, q))
                    p = q

    def __call_image_space__(self, X):
        return X.gather(X.channel_dimension, self.dims.expand_to(X))

    def __affine__(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        if n_dim is None: n_dim = self.n_dim
        n_batch = self.n_batch
        if n_batch is None: n_batch = 1
        aff = bt.diag(bt.one_hot(-1, n_dim + 1).float().view({n_batch}, [n_dim + 1]))
        b = bt.arange({n_batch}).expand_to(self.dims)
        i = bt.arange(n_dim).duplicate(n_batch, {})
        aff[b, i, self.dims] = 1.
        return aff

    def __inv__(self):
        n_dim = self.n_dim
        new_permute = (self.dims == bt.arange(n_dim).view({1}, [n_dim, 1])).float().argmax(-1).channel_dimension_(-1)
        return DimPermutation(new_permute, resize_image = self.resize_image)
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        coeffs = bt.rand({n_batch}, [n_dim])
        perms = bt.zeros({n_batch}, [n_dim]).int()
        for i in range(n_dim):
            idx = coeffs.argmax([])
            ibatch = bt.arange({n_batch})
            perms[ibatch, idx] = i
            coeffs[ibatch, idx] = 0
        return perms,

@alias("Rescale")
class Rescaling(Transformation):
    """
    Scale an image.
    
    Note: The scaling is for image coordinates, hence the image would shrink if scale > 1. 
    
    Args:
        scale (float or tuple or bt.Tensor): The scaling for all dimensions (float) 
            or for each dimension (tuple). >1 means enlarging the coordinates.
            size: ({n_batch:optional}, [n_dim]) for bt.Tensor.
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, *scale, resize_image=True, **kwargs):
        if len(scale) == 1 and isinstance(scale[0], (int, float)): scale = scale[0] * bt.ones({1}, [1])
        else:
            if len(scale) == 1: scale = scale[0]
            if isinstance(scale, tuple): scale = list(scale)
            if not isinstance(scale, bt.Tensor): scale = bt.to_bttensor(scale).squeeze()
            bt.input_shape().set(scale = "({n_batch}, [n_dim])")
        #     if scale.n_dim <= 1: scale = scale.unsqueeze({})
        #     if scale.n_dim == 2:
        #         if not scale.has_batch: scale.batch_dimension = 0
        #         if not scale.has_channel: scale.channel_dimension = 0 if scale.batch_dimension > 0 else 1
        # avouch(scale.has_batch and scale.has_channel, f"Please use batorch tensor of size \
        #     ({{n_batch:optional}}, [n_dim]) for Scaling parameters, instead of {scale.shape}. ")
        # bt.to_device(scale)
        super().__init__(scale, resize_image=resize_image, **kwargs)

        self.scale = scale
        if self.scale.n_channel > 1: self.n_dim = self.scale.n_channel
        if resize_image:
            avouch((scale - scale[:1]).abs().sum() < eps, "The batch should have a common rescaler if 'resize_image=True', they are being reshaped uniformly.")
            self.reshape = [(0,), tuple((1/scale[0]).tolist())]
        self.resize_image = resize_image
        self.batch_param.append('scale')

    def __call_image_space__(self, X):
        scale = self.scale
        return X * scale
    
    def __affine__(self, n_dim=None):
        if n_dim is None and self.n_dim is None: return
        if n_dim is None: n_dim = self.n_dim
        avouch(self.n_dim is None or self.n_dim == n_dim)
        scale = self.scale
        if isinstance(scale, (int, float)): return bt.cat(bt.cat(scale * bt.eye([n_dim]), bt.zeros([n_dim, 1]), 1), bt.cat(bt.zeros([1, n_dim]), bt.ones([1, 1]), 1), 0)
        if scale.n_channel == 1: scale = scale.amplify(n_dim, {})
        aff = bt.diag(bt.cat(scale, bt.ones({scale.n_batch}, [1]), []))
        return aff

    def __inv__(self): return Rescaling(1/self.scale, resize_image = self.resize_image)
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, std=1e-2):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        return std * bt.randn([len(shape)]).duplicate(n_batch, {}) + 1,

def rand_Rescaling(image, std=1e-1, **kwargs):
    return Rescaling(*Rescaling.random_init_params(*image.space, n_batch=image.n_batch, std=std), **kwargs)

@alias("Translate", "Shift")
class Translation(Transformation):
    """
    Translate / shift an image.
    
    Note: The translation is for coordinates, hence the image would go in the opposite direction. 
    
    Args:
        translation (tuple or bt.Tensor): The translation of the coordinates.
            size: length(n_dim) or ({n_batch:optional}, [n_dim])
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, *translation, **kwargs):
        if len(translation) == 1: translation = translation[0]
        if isinstance(translation, tuple): translation = list(translation)
        translation = bt.to_bttensor(translation).squeeze()
        if translation.n_dim == 1: translation = translation.view([])
        bt.input_shape().set(translation = "({n_batch}, [n_dim])")
        # if translation.n_dim <= 1: translation = translation.unsqueeze({})
        # if translation.n_dim == 2:
        #     if not translation.has_batch: translation.batch_dimension = 0
        #     if not translation.has_channel: translation.channel_dimension = 0 if translation.batch_dimension > 0 else 1
        # avouch(translation.has_batch and translation.has_channel, f"Please use batorch tensor of size \
        #     ({{n_batch:optional}}, [n_dim]) for Translation parameters, instead of {translation.shape}. ")
        super().__init__(translation, **kwargs)

        self.translation = translation
        self.n_dim = self.translation.n_channel
        self.batch_param.append('translation')
        
        self.use_implementation = 'image-space'

    def __call_image_space__(self, X):
        return X + self.translation
    
    def __affine__(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        if n_dim is None: n_dim = self.n_dim
        n_batch = self.translation.n_batch
        aff = bt.cat(bt.cat(bt.eye([self.n_dim]).duplicate(n_batch, {}), self.translation.unsqueeze([-1]), -1), bt.one_hot(-1, n_dim+1).view([1, n_dim+1]).duplicate(n_batch, {}), 1)
        return aff

    def __inv__(self): return Translation(-self.translation)
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, std=1):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        return std * bt.randn({n_batch}, [len(shape)]),

def rand_Translation(image, std=5, **kwargs):
    return Translation(*Translation.random_init_params(*image.space, n_batch=image.n_batch, std=std), **kwargs)

class CenterAlign(Translation):
    """
    Shift an image to match a given coordinate (in image space) to another.
    
    Args:
        source (tuple or bt.Tensor): The anchor coordinates.
            size: length(n_dim) or ({n_batch:optional}, [n_dim])
        target (tuple or bt.Tensor): The target coordinates.
            size: length(n_dim) or ({n_batch:optional}, [n_dim])
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, source_center, target_center, **kwargs):
        source_center = bt.to_bttensor(source_center).squeeze()
        target_center = bt.to_bttensor(target_center).squeeze()
        if source_center.n_dim == 1: source_center = source_center.view([])
        if target_center.n_dim == 1: target_center = target_center.view([])
        bt.input_shape().set(source_center = "({n_batch}, [n_dim])", target_center = "({n_batch}, [n_dim])")
        
        super().__init__(source_center - target_center, **kwargs)
        self.params = source_center, target_center
        if self.domain == 'physical-space':
            avouch(self.source is not None and self.target is not None, "Cannot perform physical-space center alignment for transformation without source/target information. ")
            taffine = Transformation.__get_affine__(self.target)
            saffine = Transformation.__get_affine__(self.source)
            self.translation = Affine(saffine)(source_center) - Affine(taffine)(target_center)

@alias("Rig")
class Rigid(Transformation):
    """
    Rigid transformation with respect to parameters.
    !!Attention: It's domain is 'physical-space' by default. 
    
    Args (usage 1):
        angle (bt.Tensor or np.numpy): the [clockwise] rotation angles about the axises (or z direction for 2D). It is counter-clockwise for image.
            size: ({n_batch},)
        axis (bt.Tensor or np.numpy): the rotation axises, normalized vectors, None for 2D. 
            size: ({n_batch}, [n_dim])
        translation (bt.Tensor or np.numpy): the translations after the rotation, zeros by default. 
            size: ({n_batch}, [n_dim])
        center (bt.Tensor or np.numpy): the center for the rotations, zeros by default. 
            size: ({n_batch}, [n_dim])
        trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
    
    Args (usage 2):
        matrix (bt.Tensor or np.numpy): the affine matrix, it should be orthogonal or will be projected by Procrustes Problem. 
            size: ({n_batch}, [n_dim + 1, n_dim + 1])
        trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, angle, axis=None, translation=None, center=None, trans_stretch=None, **kwargs):
        angle = bt.to_bttensor(angle)
        bt.input_shape().usage(
            angle = "({n_batch},)", 
            axis = "({n_batch}, [n_dim]) [optional]", 
            translation = "({n_batch}, [n_dim])", 
            center = "({n_batch}, [n_dim])"
        ).usage(angle = "({n_batch}, [n_dim + 1, n_dim + 1])")
        if usage == 2:
            matrix = angle
            center = bt.zeros({n_batch}, [n_dim])
            translation = matrix[..., :-1, -1].with_channel_dim(1)
            A = matrix[..., :-1, :-1]
            avouch(bt.norm(A.T @ A - bt.eye(A)).sum() < 1e-4, "Please make sure that matrix input for Rigid is orthogonal. Use Affine instead if it is not. ")
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
                raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support [n_dim] dimensional transforms (2 & 3D only). " + 
                                            "Please contact the developers if there are feasible algorithms (Error Code: T333). Thank you. ")
        
        # if angle.n_dim <= 0 and not angle.has_batch: angle = angle.unsqueeze({})
        # if angle.n_dim == 1 and not angle.has_batch: angle = angle.with_batch_dim(0)
        # if angle.n_dim == 2 and not angle.has_batch: angle = angle.unsqueeze({})
        # if angle.n_dim == 3 and not angle.has_batch and angle.shape[1] == angle.shape[2]: angle.batch_dimension = 0
        # avouch(angle.has_batch, f"Please use batorch tensor of size ({{n_batch}},) for Rigid rotation angles, instead of {angle.shape}. ")
        # n_batch = angle.n_batch
        # n_dim = None
        # if angle.n_dim >= 2:
        #     matrix = angle
        #     n_dim = matrix.size(-1) - 1
        #     center = bt.zeros({n_batch}, [n_dim])
        #     translation = matrix[..., :-1, -1].with_channel_dim(1)
        #     A = matrix[..., :-1, :-1]
        #     avouch(bt.norm(A.T @ A - bt.eye(A)).sum() < 1e-4, "Please make sure that matrix input for Rigid is orthogonal. Use Affine instead if it is not. ")
        #     # I = bt.eye(A)
        #     # Z_left = (A - I)[..., :-1] # ({n_batch}, n_dim, n_dim-1)
        #     # Z_right = (A - I)[..., -1] # ({n_batch}, n_dim)
        #     # ZTZ = Z_left.T @ Z_left
        #     # if n_dim == 2:
        #     #     axis = None
        #     #     angle = bt.where(bt.abs(bt.det(ZTZ)) < 1e-6, bt.zeros({n_batch}), bt.acos(1 - ZTZ / 2).squeeze(-1, -1))
        #     # elif n_dim == 3:
        #     #     invZTZ = bt.inv(bt.where(bt.abs(bt.det(ZTZ)).unsqueeze(-1, -1) < 1e-6, bt.eye(ZTZ), ZTZ))
        #     #     vector = bt.cat(- invZTZ @ Z_left.T @ Z_right, bt.ones({n_batch}, 1), 1).with_channel_dim(1)
        #     #     angle = bt.where(bt.abs(bt.det(ZTZ)) < 1e-6, bt.zeros({n_batch}), vector.norm())
        #     #     axis = bt.where((bt.abs(bt.det(ZTZ)) < 1e-6).duplicate(n_dim, []), bt.channel_tensor(bt.one_hot(0, n_dim)).duplicate(n_batch, {}), vector / angle)
        #     if n_dim == 2:
        #         axis = None
        #         angle = bt.acos(A[..., 0, 0]) * bt.sign(A[..., 0, 1])
        #     elif n_dim == 3:
        #         anti_sym = (A - A.T) / 2
        #         sym = (A + A.T) / 2 - bt.eye(A)
        #         angle = bt.acos(((anti_sym @ anti_sym) / sym).mean() - 1)
        #         axis = bt.uncross_matrix(anti_sym)
        #         axis /= bt.sin(angle)
        #     else:
        #         raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support [n_dim] dimensional transforms (2 & 3D only). " + 
        #                                     "Please contact the developers if there are feasible algorithms (Error Code: T333). Thank you. ")

        # if translation is not None:
        #     translation = bt.to_bttensor(translation)
        #     if translation.n_dim <= 1 and not translation.has_batch: translation = translation.unsqueeze({})
        #     if translation.n_dim == 2 and not translation.has_batch: translation = translation.with_batch_dim((1 - translation.channel_dimension) if translation.has_channel else 0)
        #     if translation.n_dim == 2 and not translation.has_channel: translation = translation.with_channel_dim(1 - translation.batch_dimension)
        #     avouch(translation.has_batch and translation.has_channel, f"Please use batorch tensor of size ({{n_batch}}, [n_dim]) for Rigid translation, instead of {translation.shape}. ")
        #     if n_batch == 1 and translation.n_batch > 1: n_batch = translation.n_batch
        #     if n_dim is None: n_dim = translation.n_channel
        #     else: avouch(n_dim == translation.n_channel, "Systematic error. Please contact the developers for details (Error Code: T332). ")
        #     n_dim = translation.n_channel
        # if center is not None:
        #     center = bt.to_bttensor(center)
        #     if center.n_dim <= 1 and not center.has_batch: center = center.unsqueeze({})
        #     if center.n_dim == 2 and not center.has_batch: center = center.with_batch_dim((1 - center.channel_dimension) if center.has_channel else 0)
        #     if center.n_dim == 2 and not center.has_channel: center = center.with_channel_dim(1 - center.batch_dimension)
        #     avouch(center.has_batch and center.has_channel, f"Please use batorch tensor of size ({{n_batch}}, [n_dim]) for Rigid center, instead of {center.shape}. ")
        #     if n_batch == 1 and center.n_batch > 1: n_batch = center.n_batch
        #     if n_dim is None: n_dim = center.n_channel
        #     else: avouch(n_dim == center.n_channel, f"Center({center.n_channel}D) and translation([n_dim]D) in trans.Rigid should have the same dimension. ")
        #     n_dim = center.n_channel
        # if axis is not None:
        #     axis = bt.to_bttensor(axis)
        #     if axis.n_dim <= 1 and not axis.has_batch: axis = axis.unsqueeze({})
        #     if axis.n_dim == 2 and not axis.has_batch: axis = axis.with_batch_dim((1 - axis.channel_dimension) if axis.has_channel else 0)
        #     if axis.n_dim == 2 and not axis.has_channel: axis = axis.with_channel_dim(1 - axis.batch_dimension)
        #     avouch(axis.has_batch and axis.has_channel, f"Please use batorch tensor of size ({{n_batch}}, [n_dim]) for Rigid axises, instead of {axis.shape}. ")
        #     if ((axis.norm() - 1) ** 2).sum().sum() >= 1e-2:
        #         print("warning: param. axises for 'Rigid' transformation should be norm-1 vectors, auto-normalization would be performed. Please contact the developers if it is necessary to be non-unit vectors (Error Code: T331). ")
        #         axis = axis / axis.norm()
        #     if n_batch == 1 and axis.n_batch > 1: n_batch = axis.n_batch
        #     if n_dim is None: n_dim = axis.n_channel
        #     else: avouch(n_dim == axis.n_channel, f"Translation({n_dim}D) and axises({axis.n_channel}D)) in trans.Rigid should have the same dimension. ")
        # if n_dim is None: n_dim = 2 if axis is None else 3
        # if translation is None: translation = bt.zeros({n_batch}, [n_dim])
        # if center is None: center = bt.zeros({n_batch}, [n_dim])
        
        # Now we create the matrix
        angle = angle.float()
        center = center.float()
        translation = translation.float()
        if n_dim == 2:
            A = bt.stack(bt.cos(angle), bt.sin(angle), -bt.sin(angle), bt.cos(angle), -1).splitdim(-1, [2, 2])
        elif n_dim == 3:
            axis = axis.float()
            A = bt.eye({n_batch}, [n_dim]) + (1 - bt.cos(angle)) * bt.cross_matrix(axis) @ bt.cross_matrix(axis) + bt.sin(angle) * bt.cross_matrix(axis)
        else:
            raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support [n_dim] dimensional transforms (2 & 3D only). " + 
                                        "Please contact the developers if there are feasible algorithms (Error Code: T333). Thank you. ")
        matrix = bt.cat(bt.cat(A, (translation + center - A @ center).unsqueeze([-1]), -1), 
                        bt.cat(bt.zeros([n_dim]), bt.ones([1])).unsqueeze({}, [1]), 1)
        if trans_stretch is not None: matrix[..., :n_dim, -1] *= trans_stretch
        if 'domain' not in kwargs: kwargs['domain'] = 'physical-space'
        super().__init__(angle, axis, translation, center=center, trans_stretch=trans_stretch, **kwargs)

        self.n_dim = n_dim
        self.trans_stretch = trans_stretch
        self.matrix = matrix
        self.batch_param.append('matrix')

    def __call_physical_space__(self, X):
        matrix = self.matrix
        n_dim = self.n_dim
        A = matrix[:, :n_dim, :n_dim]
        b = matrix[:, :n_dim, n_dim]
        shape = X.shape
        Y = (A @ X.flatten(...).add_special_dim(-1, []) + b.unsqueeze([-1])).view(shape)
        return Y
    
    def __affine__(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        return self.matrix
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, std=1, trans_std=5):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        if n_dim == 2:
            return std * bt.randn({n_batch}), None, trans_std * bt.randn({n_batch}, [n_dim]), bt.channel_tensor(shape).duplicate(n_batch, {}) / 2
        if n_dim == 3:
            ax = bt.randn({n_batch}, [n_dim])
            return std * bt.randn({n_batch}), ax / ax.norm(), trans_std * bt.randn({n_batch}, [n_dim]), bt.channel_tensor(shape).duplicate(n_batch, {}) / 2
        else:
            raise Error("NotImplemented")(f"Rigid transformation in micomputing does not support [n_dim] dimensional transforms (2 & 3D only). " + 
                                        "Please contact the developers if there are feasible algorithms (Error Code: T334). Thank you. ")

    def __inv__(self):
        inv_matrix = bt.inv(self.matrix)
        #                   [A tb]             [A⁻¹ -tA⁻¹b]          [A⁻¹ -A⁻¹b]
        # Note: self.matrix=[0  1], inv_matrix=[0      1  ]; we need [0     1  ] for the inversed transformation. 
        if self.trans_stretch is not None: inv_matrix[..., :self.n_dim, -1] /= self.trans_stretch
        return Rigid(inv_matrix, trans_stretch = self.trans_stretch).backward_(self.backward)

@alias("rand_Rig")
def rand_Rigidity(image, std=1e-1, trans_std=5, **kwargs):
    return Rigid(*Rigid.random_init_params(*image.space, n_batch=image.n_batch, std=std, trans_std=trans_std), **kwargs)

@alias("Aff")
class Affine(Transformation):
    """
    Affine transformation with respect to transformation matrix.
    
    Args:
        matrix (bt.Tensor or np.numpy): the affine matrix. 
            size: ({n_batch}, [n_dim + 1, n_dim + 1])
        trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
        domain (str: 'image-space'|'physical-space'): the coordinate domain that the transformation is in. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, matrix, trans_stretch=None, **kwargs):
        matrix = bt.to_bttensor(matrix)
        bt.input_shape().set(matrix = "({n_batch}, [n_dim + 1, n_dim + 1])")
        if trans_stretch is not None: matrix[..., :n_dim, -1] *= trans_stretch
        super().__init__(matrix, trans_stretch=trans_stretch, **kwargs)

        self.n_dim = n_dim
        self.trans_stretch = trans_stretch
        self.matrix = matrix
        self.batch_param.append('matrix')
        
        self.use_implementation = 'image-space'

    def __call_image_space__(self, X):
        matrix = self.matrix.float()
        n_dim = self.n_dim
        A = matrix[:, :n_dim, :n_dim]
        b = matrix[:, :n_dim, n_dim]
        shape = X.shape
        Y = (A @ X.flatten(...).add_special_dim(-1, []) + b.unsqueeze([-1])).view(shape)
        return Y
    
    def __affine__(self, n_dim=None):
        avouch(n_dim is None or self.n_dim == n_dim)
        return self.matrix
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, std=1e-2, trans_std=5):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        affmat = bt.eye({n_batch}, [n_dim + 1])
        affmat[..., :n_dim, :n_dim] += std * bt.randn({n_batch}, [n_dim, n_dim])
        affmat[..., :n_dim, -1] += trans_std * bt.randn({n_batch}, [n_dim])
        return affmat,

    def __inv__(self):
        inv_matrix = bt.inv(self.matrix)
        #                   [A tb]             [A⁻¹ -tA⁻¹b]          [A⁻¹ -A⁻¹b]
        # Note: self.matrix=[0  1], inv_matrix=[0      1  ]; we need [0     1  ] for the inversed transformation. 
        if self.trans_stretch is not None: inv_matrix[..., :self.n_dim, -1] /= self.trans_stretch
        return Affine(inv_matrix, trans_stretch=self.trans_stretch, domain=self.domain, backward=self.backward)

@alias("rand_Aff")
def rand_Affinity(image, std=1e-1, trans_std=5, **kwargs):
    return Affine(*Affine.random_init_params(*image.space, n_batch=image.n_batch, std=std, trans_std=trans_std), **kwargs)

@alias("logEu", "logEuclidean")
class PolyAffine(Transformation):
    """
    Poly affine transformation with respect to transformation matrices [1].
    Note that dmatrices for this tranformation IS NOT the actual affine matrices, but IS a differentiation instead.
    
    Args:
        dmatrices (bt.Tensor or np.numpy): One affine matrix for each region. 
            size: ({n_batch}, [n_region], [n_dim + 1, n_dim + 1])
        masks (bt.Tensor or np.numpy): One 0-1 mask for each region. 
            size: ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        order (int): the order of interpolation coefficient. The influence of an affine decays at a rate of 1 / distanceᵒʳᵈᵉʳ. Defaults to 2. 
        level (int): the level of velocity integration. The dmatrix is regarded as 1 / 2ˡᵉᵛᵉˡ of true matrix. Defaults to 4. 
        trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]

    [1] Arsigny V , Commowick O , Ayache N , et al. A Fast and Log-Euclidean Polyaffine Framework for Locally 
        Linear Registration[J]. Journal of Mathematical Imaging & Vision, 2009, 33(2):222-238.
    """
    
    def __init__(self, dmatrices, masks, order=2, level=4, is_inv=False, trans_stretch=None, **kwargs):
        from .funcs import distance_map
        dmatrices = bt.to_bttensor(dmatrices)
        masks = bt.to_bttensor(masks).int()
        bt.input_shape().set(
            dmatrices = "({n_batch}, [n_region], [n_dim + 1, n_dim + 1])", 
            masks = "({n_batch}, [n_region], n_1, ..., n_r) [r=n_dim]"
        )
        # if dmatrices.n_dim <= 3 and not dmatrices.has_batch: dmatrices = dmatrices.unsqueeze({})
        # if dmatrices.n_feature_dim < 3 and not dmatrices.has_feature: dmatrices = dmatrices.unsqueeze([1])
        # if dmatrices.n_dim == 4 and dmatrices.shape[2] == dmatrices.shape[3]:
        #     if not dmatrices.has_batch: dmatrices.batch_dimension = 0
        #     if not dmatrices.has_feature: dmatrices.n_feature_dim = 3
        # avouch(dmatrices.has_batch and dmatrices.n_feature_dim == 3, "Please use batorch tensor of size ({n_batch}, [n_region]," +
        #        f"[n_dim + 1, n_dim + 1]) for PolyAffine parameters, instead of {dmatrices.shape}. ")
        # n_dim = dmatrices.size(-1) - 1
        # if trans_stretch is not None: dmatrices[..., :n_dim, -1] *= trans_stretch
        # masks = bt.to_bttensor(masks)
        # if masks.n_dim <= n_dim + 1 and not masks.has_batch: masks = masks.unsqueeze({})
        # if masks.n_dim <= n_dim + 1 and not masks.has_channel: masks = masks.unsqueeze([])
        # if masks.n_dim == n_dim + 2 and not masks.has_batch: masks.batch_dimension = 0
        # if masks.n_dim == n_dim + 2 and not masks.has_channel: masks.channel_dimension = 1
        # avouch(masks.has_batch and masks.has_channel, "Please use batorch tensor of size ({n_batch}, [n_region]," +
        #        f"n_1, n_2, ..., n_r) [r=n_dim] for PolyAffine parameters, instead of {masks.shape}. ")

        # preprocess masks ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        n_batch = masks.n_batch
        n_region = masks.n_channel
        dis = distance_map(masks)
        dis = (dis + 1).clamp(0)
        # import micomputing.plot as plt
        # plt.subplots(2)
        # plt.imsshow(dis[0, 0], dis[0, 1])
        # plt.show()
        weights = 1 / (dis ** order + 1e-5)
        weights = weights / weights.sum([]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        
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

        super().__init__(dmatrices, masks=masks, order=order, level=level, is_inv=is_inv, trans_stretch=trans_stretch, **kwargs)
        self.n_batch = n_batch
        self.n_dim = n_dim
        self.masks = masks
        self.weights = weights
        self.trans_stretch = trans_stretch
        self.dmatrices = dmatrices
        self.is_inv = is_inv
        self.order = order
        self.level = level
        
        self.use_implementation = 'image-space'
        
    def update_masks(self, masks):
        from .funcs import distance_map
        self.masks = masks
        dis = distance_map(masks)
        dis = (dis + 1).clamp(0)
        weights = 1 / (dis ** self.order + 1e-5)
        weights = weights / weights.sum([]) 
        self.weights = weights

    def __call_image_space__(self, X):
        dmatrices = self.dmatrices # ({n_batch}, [n_region], [n_dim + 1, n_dim + 1])
        weights = self.weights # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        n_dim = self.n_dim
        n_region = weights.n_channel
        Xs = X.duplicate(n_region, [1]) # ({n_batch}, [n_region], [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        A = dmatrices[..., :n_dim, :n_dim] # ({n_batch}, [n_region], [n_dim, n_dim])
        b = dmatrices[..., :n_dim, n_dim:] # ({n_batch}, [n_region], [n_dim, 1])
        Y = (A @ Xs.flatten(...).add_special_dim(-1, []) + b).view_as(Xs)
        velocities = (Y * weights.unsqueeze([2])).sum([0]) - X
        if self.is_inv: velocities = -velocities
        return VelocityField(velocities * 2 ** self.level, level=self.level, domain=self.domain)(X)
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, n_region=2, std=1e-4, trans_std=1e-2):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        affmat = bt.eye({n_batch}, [n_region], [n_dim + 1, n_dim + 1])
        affmat[..., :n_dim, :n_dim] += std * bt.randn({n_batch}, [n_region], [n_dim, n_dim])
        affmat[..., :n_dim, -1] += trans_std * bt.randn({n_batch}, [n_region], [n_dim])
        centers = (bt.rand({n_batch}, [n_region], [n_dim]) * bt.channel_tensor(shape).view({1}, [1], [-1])).int()
        dis_level = (((centers.unsqueeze([1]) - centers.unsqueeze([2])) ** 2).sum([-1]).sqrt() + 1e4 * bt.eye({n_batch}, [n_region])).min() / 3
        masks = ((bt.image_grid(*shape).duplicate(n_region, [0]).duplicate(n_batch, {}) - centers) ** 2).sum([-1]).sqrt() < dis_level
        return affmat, masks

    def __inv__(self):
        new_matrices = self.dmatrices.inv()
        n_batch = self.dmatrices.n_batch
        new_masks = interpolation(self.masks.mergedims([], {}), Affine(bt.pow(new_matrices.mergedims([0], {}), 2 ** self.level))).splitdim({}, {n_batch}, [-1])
        if self.trans_stretch is not None: new_matrices[..., :self.n_dim, -1] /= self.trans_stretch
        return PolyAffine(new_matrices, new_masks, order=self.order, level=self.level, is_inv=self.is_inv, trans_stretch = self.trans_stretch).backward_(self.backward)
        # n_batch = self.dmatrices.n_batch
        # affs = bt.inv(self.dmatrices)
        # masks = interpolation(self.masks.mergedims([], {}), Affine(bt.matpow(affs.mergedims([], {}), 64))).splitdim([], {n_batch}, {-1})
        # return PolyAffine(affs, masks = masks, trans_stretch = self.trans_stretch).backward_(self.backward)
        
@alias("rand_logEu", "rand_logEuclidean")
def rand_PolyAffine(image, n_region=3, std=1e-2, trans_std=1, **kwargs):
    return PolyAffine(*PolyAffine.random_init_params(*image.space, n_batch=image.n_batch, n_region=n_region, std=std, trans_std=trans_std), **kwargs)

@alias("LARM")
class LocallyAffine(Transformation):
    """
    Locally affine transformation with respect to transformation matrices [1].
    
    Args:
        matrices (bt.Tensor or np.numpy): One affine matrix for each region. 
            size: ({n_batch}, [n_region], [n_dim + 1, n_dim + 1])
        masks (bt.Tensor or np.numpy): One 0-1 mask for each region. 
            size: ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        order (int): the order of interpolation coefficient. The influence of an affine decays at a rate of 1 / distance^order.
        trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]

    [1] Zhuang X , Rhode K , Arridge S , et al. An Atlas-Based Segmentation Propagation Framework Using Locally Affine 
        Registration - Application to Automatic Whole Heart Segmentation[J]. Springer, Berlin, Heidelberg, 2008.
    """
    
    def __init__(self, matrices, masks, order=2, trans_stretch=None, avoid_conflict=True, **kwargs):
        from .funcs import dilate, distance_map
        matrices = bt.to_bttensor(matrices)
        masks = bt.to_bttensor(masks).int()
        bt.input_shape().set(
            matrices = "({n_batch}, [n_region], [n_dim + 1, n_dim + 1])", 
            masks = "({n_batch}, [n_region], n_1, ..., n_r) [r=n_dim]"
        )
        # if matrices.n_dim <= 3 and not matrices.has_batch: matrices = matrices.unsqueeze({})
        # if matrices.n_dim <= 3 and not matrices.has_channel: matrices = matrices.unsqueeze([1])
        # if matrices.n_dim == 4 and matrices.shape[2] == matrices.shape[3]:
        #     if not matrices.has_batch: matrices.batch_dimension = 0
        #     if not matrices.has_feature: matrices.n_feature_dim = 3
        # avouch(matrices.has_batch and matrices.n_feature_dim == 3, "Please use batorch tensor of size ({n_batch}, [n_region]," +
        #        f"[n_dim + 1, n_dim + 1]) for LocallyAffine parameters, instead of {matrices.shape}. ")
        # n_dim = matrices.size(-1) - 1
        # if trans_stretch is not None: matrices[..., :n_dim, -1] *= trans_stretch
        # masks = bt.to_bttensor(masks).int()
        # if masks.n_dim <= n_dim + 1 and not masks.has_batch: masks = masks.unsqueeze({})
        # if masks.n_dim <= n_dim + 1 and not masks.has_channel: masks = masks.unsqueeze([])
        # if masks.n_dim == n_dim + 2 and not masks.has_batch: masks.batch_dimension = 0
        # if masks.n_dim == n_dim + 2 and not masks.has_channel: masks.channel_dimension = 1
        # avouch(masks.has_batch and masks.has_channel, "Please use batorch tensor of size ({n_batch}, [n_region]," + 
        #        f"n_1, n_2, ..., n_r) [r=n_dim] for LocallyAffine parameters, instead of {masks.shape}. ")

        # preprocess masks
        n_batch = matrices.n_batch
        n_region = matrices.size(1)
        if avoid_conflict:
            Gi = Affine(matrices.mergedims([0], {}))
            GiVi = interpolation(masks.mergedims([], {}), Gi.inv(), method='Nearest').splitdim({}, {n_batch}, [n_region]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            GiVijoinGkVk = (GiVi.unsqueeze([1]) * GiVi.unsqueeze([2])).mergedims(1, {}) # ({n_batch x n_region}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            Rik = interpolation(GiVijoinGkVk, Gi, method='Nearest').splitdim({}, {n_batch}, [n_region]) # ({n_batch}, [n_region, n_region], n_1, n_2, ..., n_r) [r=n_dim]
            URik = (Rik.sum([1]) - masks > 0.1).float() # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            URik_plus = dilate(URik.mergedims([], {}), 1).splitdim({}, {n_batch}, [n_region]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            masks = (masks - URik_plus > 0.1).float() # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]

        weights = 1 / (distance_map(masks) ** order + 1e-5)
        weights = weights / weights.sum([]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        if trans_stretch is not None: matrices = matrices * bt.to_bttensor([1.] * n_dim + [trans_stretch]).unsqueeze(0, 0, 0)

        super().__init__(matrices, masks=masks, order=order, trans_stretch=trans_stretch, **kwargs)
        self.avoid_conflict = avoid_conflict
        self.order = order
        self.n_batch = n_batch
        self.n_dim = n_dim
        self.masks = masks
        self.weights = weights
        self.trans_stretch = trans_stretch
        self.matrices = matrices
        self.batch_param.extend(['matrices', 'weights', 'masks'])
        
        self.use_implementation = 'image-space'
        
    def update_masks(self, masks):
        from .funcs import dilate, distance_map
        # preprocess masks
        n_batch = self.n_batch
        n_region = self.matrices.n_channel
        if self.avoid_conflict:
            Gi = Affine(self.matrices.mergedims([], {}))
            GiVi = interpolation(masks.mergedims([], {}), Gi.inv(), method='Nearest').splitdim({}, {n_batch}, [n_region]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            GiVijoinGkVk = (GiVi.unsqueeze(1) * GiVi.unsqueeze(2)).mergedims(1, {}) # ({n_batch x n_region}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            Rik = interpolation(GiVijoinGkVk, Gi, method='Nearest').splitdim({}, {n_batch}, [n_region]) # ({n_batch}, [n_region, n_region], n_1, n_2, ..., n_r) [r=n_dim]
            URik = (Rik.sum([1]) - masks > 0.1).float() # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            URik_plus = dilate(URik.mergedims([], {}), 1).splitdim({}, {n_batch}, [n_region]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
            masks = (masks - URik_plus > 0.1).float() # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]

        self.masks = masks
        weights = 1 / (distance_map(masks) ** self.order + 1e-5)
        weights = weights / weights.sum([]) # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        self.weights = weights

    def __call_image_space__(self, X):
        matrices = self.matrices
        masks = self.masks # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        weights = self.weights # ({n_batch}, [n_region], n_1, n_2, ..., n_r) [r=n_dim]
        n_dim = self.n_dim
        n_region = matrices.size([0])
        Xs = X.duplicate(n_region, [1]) # ({n_batch}, [n_region, n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        A = matrices[..., :n_dim, :n_dim]
        b = matrices[..., :n_dim, n_dim:]
        Y = (A @ Xs.flatten(...).change_special_dim(-1, []) + b).view_as(Xs)
        return (weights.unsqueeze([2]) * Y).sum([0]) * (1 - masks.sum([], keepdim=True)) + (Y * masks.unsqueeze([2])).sum([0])

    @classmethod
    def random_init_params(cls, *shape, n_batch=1, n_region=2, std=1e-2, trans_std=1):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        affmat = bt.eye({n_batch}, [n_region], [n_dim + 1, n_dim + 1])
        affmat[..., :n_dim, :n_dim] += std * bt.randn({n_batch}, [n_region], [n_dim, n_dim])
        affmat[..., :n_dim, -1] += trans_std * bt.randn({n_batch}, [n_region], [n_dim])
        centers = (bt.rand({n_batch}, [n_region], [n_dim]) * bt.channel_tensor(shape).view({1}, [1], [-1])).int()
        dis_level = (((centers.unsqueeze([1]) - centers.unsqueeze([2])) ** 2).sum([-1]).sqrt() + 1e4 * bt.eye({n_batch}, [n_region])).min() / 3
        masks = ((bt.image_grid(*shape).duplicate(n_region, [0]).duplicate(n_batch, {}) - centers) ** 2).sum([-1]).sqrt() < dis_level
        return affmat, masks

    def __inv__(self):
        new_matrices = self.matrices.inv()
        n_batch = self.matrices.n_batch
        new_masks = interpolation(self.masks.mergedims([], {}), Affine(new_matrices.mergedims([0], {}))).splitdim({}, {n_batch}, [-1])
        if self.trans_stretch is not None: new_matrices[..., :self.n_dim, -1] /= self.trans_stretch
        return LocallyAffine(new_matrices, order=self.order, level=self.level, avoid_conflict=self.avoid_conflict, masks = new_masks, trans_stretch = self.trans_stretch).backward_(self.backward)
    
@alias("rand_LARM")
def rand_LocallyAffine(image, n_region=3, std=1e-1, trans_std=5, **kwargs):
    return LocallyAffine(*LocallyAffine.random_init_params(*image.space, n_batch=image.n_batch, n_region=n_region, std=std, trans_std=trans_std), **kwargs)

@alias("FFD")
class FreeFormDeformation(Transformation):
    """
    Free Form Deformation (FFD) transformation [1].
    
    Args:
        offsets (bt.Tensor): the FFD offsets. 
            size: ({n_batch}, [n_dim], m_1, m_2, ..., m_r) [r=n_dim]
            for m_1 x m_2 x ... x m_r grid of Δcontrol points
        control_spacing (int or tuple): FFD control spacing; spacing between FFD control points. Defaults to 10px/mm. 
        origin (int or tuple): FFD origin; coordinate for the (0, 0, 0) control point. Defaults to (0, 0, 0). 
        affine (None or bt.Tensor): FFD affine transformation matrix for the FFD grid. Defaults to None.
    
    Note: offsets, control_spacing and origin are in px for image-space transformations, and in mm for physical-space ones.
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
            
    [1] Rueckert D , Sonoda L I , Hayes C , et al. Nonrigid registration using free-form deformations: 
        application to breast MR images[J]. IEEE Transactions on Medical Imaging, 1999(8).
    """

    def __init__(self, offsets, control_spacing=10, origin=None, **kwargs):
        offsets = bt.to_bttensor(offsets)
        bt.input_shape().set(offsets = "({n_batch}, [n_dim], m_1, ..., m_r) [r=n_dim]")
        # if not offsets.has_channel:
        #     if offsets.size(0) == offsets.n_dim - 1:
        #         n_dim = offsets.size(0)
        #         offsets.channel_dimension = 0
        #         offsets = offsets.unsqueeze({})
        #     elif offsets.size(1) == offsets.n_dim - 2:
        #         n_dim = offsets.size(1)
        #         offsets.channel_dimension = 1
        #     else: raise TypeError(f"FFD parameters with size {offsets.shape} donot match ({{n_batch}}, [n_dim], m_1, m_2, ..., m_r) [r=n_dim]. ")
        # if not offsets.has_batch:
        #     n_dim = offsets.n_channel
        #     if offsets.n_dim <= n_dim + 1: offsets = offsets.unsqueeze({})
        #     else: offsets.batch_dimension = 0
        # avouch(offsets.has_batch and offsets.has_channel, f"Please use batorch tensor of size \
        #     ({{n_batch}}, [n_dim], m_1, m_2, ..., m_r) [r=n_dim] for FFD parameters, instead of {offsets.shape}. ")
        # n_dim = offsets.n_channel
        control_spacing = to_tuple(control_spacing)
        if len(control_spacing) == 1: control_spacing *= n_dim
        if origin is not None:
            origin = to_tuple(origin)
            if len(origin) == 1: origin *= n_dim
        super().__init__(offsets, control_spacing=control_spacing, origin=origin, **kwargs)

        self.n_dim = n_dim
        self.n_batch = offsets.n_batch
        self.offsets = offsets
        self.control_spacing = control_spacing
        self.origin = origin
        self.batch_param.append('offsets')
        
        # self.use_implementation = 'image-space'

    # parameter alias
    @property
    def displacements(self): return self.offsets
    
    @displacements.setter
    def displacements(self, value): self.offsets = value
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, spacing=1, control_spacing=None, std=1):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        if not isinstance(spacing, (tuple, list)): spacing = (spacing,)
        if len(spacing) == 1: spacing = spacing * n_dim
        if control_spacing is None: control_spacing = 1 << (int(math.log2(min([(l-1)*d for l, d in zip(shape, spacing)]))) - 3)
        if not isinstance(control_spacing, (tuple, list)): control_spacing = (control_spacing,)
        if len(control_spacing) == 1: control_spacing = control_spacing * n_dim
        ffd_gridsize = [math.ceil((l-1) * d / c) for l, d, c in zip(shape, spacing, control_spacing)]
        if n_batch is None: return std * bt.randn([n_dim], *ffd_gridsize)
        return std * bt.randn({n_batch}, [n_dim], *ffd_gridsize),
    
    def __call_image_space__(self, X):
        shape = X.shape
        n_dim = self.n_dim
        if self.origin is None: origin = bt.zeros(n_dim)
        offsets = self.offsets.float()
        control_spacing = self.control_spacing
        n_batch = self.n_batch
        origin = self.origin if isinstance(self.origin, bt.Tensor) else bt.channel_tensor(self.origin)
        # X: ({n_batch}, [n_dim], n_data = n_1 x n_2 x ... x n_r) [r=n_dim]
        X = X.flatten(...)
        X -= origin
        n_data = X.size(-1)
        size = bt.channel_tensor(offsets.space)
        # Normalize X in the domain (m_1, m_2, ..., m_[n_dim]).
        FFDX = X / bt.channel_tensor(control_spacing).float()
        iX = bt.floor(FFDX).float(); uX = FFDX - iX
        coord_str = lambda x: f"{x[0, 0]!r}".split('val=')[0] + "val=" + ', '.join([f"{x[0, i]!r}".rstrip('>').split('val=')[-1] for i in range(self.n_dim)]) + '>'
        # Compute the weights. W: ((4), {n_batch}, [n_dim], n_data = n_1 x n_2 x ... x n_r) [r=n_dim]
        i = bt.arange(-1, 3).with_func_dim(True).expand_to(bt.Size((4,), {n_batch}, [n_dim], n_data))
        W = Bspline(i, uX.duplicate(4, bt.FuncDim(0)))
        "Compute FFD Transformation"
        output = bt.zeros_like(X)
        # Loop in the space {-1, 0, 1, 2} ^ n_dim; G is in {0, 1, 2, 3} ^ n_dim
        for G in bt.image_grid([4]*n_dim).flatten(...).transpose(0, 1):
            # Weights for each point: [product of W[G[D], t, D, x] for D in range(n_dim)] for point x and batch t.
            # Wg: ({n_batch}, [1], n_data = n_1 x n_2 x ... x n_r) [r=n_dim]
            Wg = W.gather(0, G.expand_to(bt.Size(bt.FuncDim(1)) + W.shape[1:])).squeeze(0).prod([], keepdim=True)
            # Compute the indices of related control points. Ind: ({n_batch}, [n_dim], n_data = n_1 x n_2 x ... x n_r) [r=n_dim]
            Ind = bt.clamp(iX.long() + G - 1, min=0)
            Ind = bt.min(Ind, (size - 1).expand_to(Ind))
            # Convert the indices to 1 dimensional. Dot: ({n_batch}, n_data = n_1 x n_2 x ... x n_r) [r=n_dim]
            Dot = Ind[:, 0]
            for r in range(1, n_dim): Dot *= size[r]; Dot += Ind[:, r]
            # Obtain the coordinates of the control points. CPoints: ({n_batch}, [n_dim], n_data = n_1 x n_2 x ... x n_r) [r=n_dim]
            CPoints = offsets.flatten(...).gather(-1, Dot.long().expand_to(Ind)).float()
            # Add the weighted control coordinates to the output coordinates.
            output += (Wg * CPoints).view_as(X)
        # Denormalize the outputs.
        output += X
        output += origin
        return output.view(shape)
    
    def __call_physical_space__(self, phyX):
        n_dim = self.n_dim
        if self.backward:
            avouch(self.target is not None, "Cannot perform physical-space FFD transformation as FFD grid requires image coordinates to locate. ")
            prev_affine = Transformation.__get_affine__(self.target).type(phyX.dtype)
        else:
            avouch(self.source is not None, "Cannot perform physical-space FFD transformation as FFD grid requires image coordinates to locate. ")
            prev_affine = Transformation.__get_affine__(self.source).type(phyX.dtype)
        prev_affine = prev_affine.inv()
        imgX = (bt.matmul(prev_affine[..., :n_dim, :n_dim], phyX.flatten(...).add_special_dim(-1, [])) + prev_affine[..., :n_dim, n_dim:]).view_as(phyX)
        # coord_str = lambda x: f"{x[0, 0]!r}".split('val=')[0] + "val=" + ', '.join([f"{x[0, i]!r}".rstrip('>').split('val=')[-1] for i in range(self.n_dim)]) + '>'
        # print(coord_str(phyX))
        # print(coord_str(imgX))
        # print(self.control_spacing, affine2spacing(prev_affine))
        # print(self.offsets.shape, phyX.shape, imgX.shape)
        self.trans_record.append(f"- [In phy-FFD with origin {tuple(self.origin.round(decimals=4).tolist()) if isinstance(self.origin, bt.Tensor) else self.origin}]: Affine with matrix:\n{prev_affine}")
        if self.origin is not None:
            phy_origin = self.origin
            self.origin = (bt.matmul(prev_affine[..., :n_dim, :n_dim], bt.channel_tensor(self.origin)) + prev_affine[..., :n_dim, -1])
        else:
            phy_origin = None
            self.origin = bt.channel_tensor(bt.zeros(n_dim))
        phy_control_spacing = self.control_spacing
        self.control_spacing = tuple(pd * d for pd, d in zip(phy_control_spacing, affine2spacing(prev_affine)))
        res = self.__call_image_space__(imgX) - imgX - self.origin
        # print(1, coord_str(self.offsets))
        # print(2, coord_str(res))
        res = res + phyX
        self.control_spacing = phy_control_spacing
        if phy_origin is not None: self.origin = phy_origin
        return res

@alias("rand_FFD")
def rand_FreeFormDeformation(image, control_spacing=None, spacing=1, std=10, **kwargs):
    if not isinstance(spacing, (tuple, list)): spacing = (spacing,)
    if len(spacing) == 1: spacing = spacing * n_dim
    if control_spacing is None: control_spacing = 1 << (int(math.log2(min([(l-1)*d for l, d in zip(image.space, spacing)]))) - 3)
    return FreeFormDeformation(*FreeFormDeformation.random_init_params(*image.space, spacing=spacing, control_spacing=control_spacing, n_batch=image.n_batch, std=std), control_spacing=control_spacing, **kwargs)

@alias("DDF")
class DenseDisplacementField(Transformation):
    """
    Dense Displacement Field (DDF) transformation.
    
    Args:
        displacements (bt.Tensor): the displacement of each voxel (or pixel). 
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        shape (bt.Size or tuple): the shape of displacement (needed if input displacement is a transformation). 
        interpolate (bool): Whether to force interpolation in apply. 

    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, displacements, shape=None, interpolate=False, **kwargs):
        if isinstance(displacements, Transformation): displacements = displacements.toDDF(shape)
        displacements = bt.to_bttensor(displacements).float()
        bt.input_shape().set(displacements = "({n_batch}, [n_dim], n_1, ..., n_r) [r=n_dim]")
        # if not displacements.has_channel:
        #     if displacements.size(0) == displacements.n_dim - 1:
        #         n_dim = displacements.size(0)
        #         displacements.channel_dimension = 0
        #         displacements = displacements.unsqueeze({})
        #     elif displacements.size(1) == displacements.n_dim - 2:
        #         n_dim = displacements.size(1)
        #         displacements.channel_dimension = 1
        #     else: raise TypeError(f"DDF parameters with size {displacements.shape} donot match ({{n_batch}}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]. ")
        # if not displacements.has_batch:
        #     n_dim = displacements.n_channel
        #     if displacements.n_dim <= n_dim + 1: displacements = displacements.unsqueeze({})
        #     else: displacements.batch_dimension = 0
        # displacements = displacements.float()
        # avouch(displacements.has_batch and displacements.has_channel, f"Please use batorch tensor of size \
        #     ({{n_batch}}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim] for DDF parameters, instead of {displacements.shape}. ")
        super().__init__(displacements, shape=shape, interpolate=interpolate, **kwargs)
        self.n_dim = displacements.n_channel
        self.displacements = displacements
        self.interpolate = interpolate
        self.batch_param.append('displacements')
        
        self.use_implementation = 'image-space'

    # parameter alias
    @property
    def offsets(self): return self.displacements
    
    @offsets.setter
    def offsets(self, value): self.displacements = value
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, std=1):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        if n_batch is None: return std * bt.randn([n_dim], *shape)
        return std * bt.randn({n_batch}, [n_dim], *shape),

    def __call_image_space__(self, X):
        displacements = self.displacements
        n_dim = self.n_dim
        if not X.has_space: X = X.unsqueeze(-1)
        if not self.interpolate and X.space == displacements.space and X.n_channel == displacements.n_channel: return X + displacements
        else: return X + interpolation(displacements, target_space=X)

@alias("rand_DDF")
def rand_DenseDisplacementField(image, std=1, **kwargs):
    return DenseDisplacementField(*DenseDisplacementField.random_init_params(*image.space, n_batch=image.n_batch, std=std), **kwargs)

@alias("VF")
class VelocityField(Transformation):
    """
    Velocity Field (VF) transformation.
    
    Args:
        scaled_velocities (bt.Tensor): the velocity at each voxel (or pixel), scaled to displacement level. 
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        shape (bt.Size or tuple): the shape of velocity (needed if input is a transformation). 
        interpolate (bool): Whether to force interpolation in apply. 
        level (int): the level of velocity, the transformation in delta time 1 / 2ˡᵉᵛᵉˡ is,
            dT = V / 2ˡᵉᵛᵉˡ. Therefore, the total transformation is dT∘ ⋯ ∘dT for 2ˡᵉᵛᵉˡ times.

    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, scaled_velocities, shape=None, interpolate=False, level=4, **kwargs):
        if isinstance(scaled_velocities, Transformation): scaled_velocities = scaled_velocities.toDDF(shape)
        scaled_velocities = bt.to_bttensor(scaled_velocities).float()
        bt.input_shape().set(scaled_velocities = "({n_batch}, [n_dim], n_1, ..., n_r) [r=n_dim]")
        # if not scaled_velocities.has_channel:
        #     if scaled_velocities.size(0) == scaled_velocities.n_dim - 1:
        #         n_dim = scaled_velocities.size(0)
        #         scaled_velocities.channel_dimension = 0
        #         scaled_velocities = scaled_velocities.unsqueeze({})
        #     elif scaled_velocities.size(1) == scaled_velocities.n_dim - 2:
        #         n_dim = scaled_velocities.size(1)
        #         scaled_velocities.channel_dimension = 1
        #     else: raise TypeError(f"DDF parameters with size {scaled_velocities.shape} donot match ({{n_batch}}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]. ")
        # if not scaled_velocities.has_batch:
        #     n_dim = scaled_velocities.n_channel
        #     if scaled_velocities.n_dim <= n_dim + 1: scaled_velocities = scaled_velocities.unsqueeze({})
        #     else: scaled_velocities.batch_dimension = 0
        # scaled_velocities = scaled_velocities.float()
        # avouch(scaled_velocities.has_batch and scaled_velocities.has_channel, f"Please use batorch tensor of size \
        #     ({{n_batch}}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim] for DDF parameters, instead of {scaled_velocities.shape}. ")
        super().__init__(scaled_velocities, shape=shape, interpolate=interpolate, level=level, **kwargs)
        self.n_dim = scaled_velocities.n_channel
        self.scaled_velocities = scaled_velocities
        self.interpolate = interpolate
        self.level = level
        self.batch_param.append('scaled_velocities')
        
        self.use_implementation = 'image-space'
    
    @classmethod
    def random_init_params(cls, *shape, n_batch=1, std=1):
        shape = arg_tuple(shape)
        if n_batch is None and isinstance(shape, bt.Size): n_batch = shape.n_batch
        if n_batch is None: n_batch = 1
        n_dim = len(shape)
        if n_batch is None: return std * bt.randn([n_dim], *shape)
        return std * bt.randn({n_batch}, [n_dim], *shape),

    def __call_image_space__(self, X):
        level = self.level
        velocities = self.scaled_velocities / (1 << level)
        
        standard_grid = bt.image_grid(*velocities.space)
        for l in range(level):
            velocities += interpolation(velocities, target_space=standard_grid + velocities)
        displacements = velocities
        
        n_dim = self.n_dim
        if X.n_space_dim == 0: X = X.unsqueeze(-1)
        if not self.interpolate and X.space == displacements.space and X.n_channel == displacements.n_channel: return X + displacements
        else: return X + interpolation(displacements, target_space=X)

@alias("rand_VF")
def rand_VelocityField(image, std=10, level=4, **kwargs):
    return VelocityField(*VelocityField.random_init_params(*image.space, n_batch=image.n_batch, std=std), level=level, **kwargs)

@alias("MLP")
class MultiLayerPerception(Transformation):
    """
    A transformation defined by a MLP. 
    
    Args:
        weights (bt.Tensor): the weights for the perception network. 
            size: ({n_batch}, n_dim * n_hl_1 + n_hl_1 + sum(i=1..k-1){n_hl_i * n_hl_{i+1} + n_hl_{i+1}} + n_hl_k * n_dim + n_dim)
            where k is the number of hidden layers and n_hl_i is the length of the i-th hidden layer. 
        hidden_layers (list of int): the lengths for the hidden layers, i.e. [n_hl_1, n_hl_2, ..., n_hl_k]
            It is by default '[]' so that the default MLP is an affine transformation. 
        active_function (class): the active_function. 
        trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
        
    Args for __call__:
        X (bt.Tensor): Coordinates to be transformed.
            size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed coordinates.
            size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
    """
    
    def __init__(self, weights, hidden_layers=[], active_function=None, trans_stretch=1, residual=True):
        raise NotImplementedError("The new version of MLP is not implemented yet. ")
        self.hidden_layers = hidden_layers
        n_hl = sum(x * y for x, y in zip(hidden_layers[:-1], hidden_layers[1:])) + sum(hidden_layers)
        n_hl_1 = hidden_layers[0]
        n_hl_k = hidden_layers[-1]
        bt.input_shape(n_hl=n_hl, n_hl_1=n_hl_1, n_hl_k=n_hl_k).set(weights = "({n_batch}, n_dim * n_hl_1 + n_hl + n_hl_k * n_dim + n_dim)")
        # if not weights.has_batch:
        #     if weights.n_dim == 2: weights.batch_dim = 0
        #     else: weights = weights.unsqueeze([])
        self.weights = weights
        self.trans_stretch = trans_stretch
        super().__init__(weights, hidden_layers = hidden_layers, trans_stretch = trans_stretch, residual = residual)
        dim_const = weights.size(-1) - n_hl
        dim_coeff = n_hl_1 + n_hl_k + 1
        avouch(dim_const % dim_coeff == 0, f"Wrong weight length for hidden layers of sizes {hidden_layers}, {dim_coeff} x n_dim + {dim_const} expected, but got {weights.size(-1)}.")
        self.n_dim = dim_const // dim_coeff
        self.n_batch = weights.n_batch
        self.layers = []
        p = 0
        for i in range(len(hidden_layers) + 1):
            in_features = self.n_dim if i == 0 else hidden_layers[i - 1]
            out_features = self.n_dim if i == len(hidden_layers) else hidden_layers[i]
            layer_weights = weights[..., p:p+out_features*in_features].view({self.n_batch}, out_features, in_features)
            p += out_features * in_features
            layer_bias = weights[..., p:p+out_features].view({self.n_batch}, out_features)
            p += out_features
            self.layers.append((layer_weights, layer_bias))
        self.active_function = active_function
        self.residual = residual
        if self.active_function is None: self.active_function = bt.nn.ReLU
        
        self.use_implementation = 'image-space'
    
    @classmethod
    def get_weight_length(cls, n_dim, *hidden_layers):
        hidden_layers = arg_tuple(hidden_layers)
        return n_dim * (hidden_layers[0] + hidden_layers[-1] + 1) \
            + sum(hidden_layers) + sum(x * y for x, y in zip(hidden_layers[:-1], hidden_layers[1:]))
    
    @property
    def n_weight_length(self):
        return MultiLayerPerception.get_weight_length(self.n_dim, self.hidden_layers)
    
    @classmethod
    def random_init_params(cls, n_dim, *hidden_layers, n_batch=1, std=1e-4):
        hidden_layers = arg_tuple(hidden_layers)
        n_length = MultiLayerPerception.get_weight_length(n_dim, hidden_layers)
        return std * bt.randn({n_batch}, n_length),
    
    def __call__(self, X):
        if not X.has_space: X = X.unsqueeze(-1)
        Y = X.flatten(...).with_feature_dim(None)
        for i, (weights, bias) in enumerate(self.layers):
            Y = weights @ Y
            if i < len(self.layers) - 1: Y = self.active_function()(Y + bias.unsqueeze(-1))
            else: Y += self.trans_stretch * bias.unsqueeze(-1)
        if self.residual:
            return X + Y.view_as(X)
        else: return Y.view_as(X)

@alias("rand_MLP")
def rand_MultiLayerPerception(image, std=10, level=4, **kwargs):
    hl = [40, 40]
    return MultiLayerPerception(*MultiLayerPerception.random_init_params(image.n_dim, *hl, n_batch=image.n_batch, std=std), hidden_layers=hl, **kwargs)

# @alias("SAT")
# class SelfAttentionTransformation(Transformation):
#     def __init__(self, weights, hidden_layers=[], active_function=None, trans_stretch=1, residual=True):
#         """
#         A transformation defined by a Network with self-attention structure. 
        
#         Args:
#             weights (bt.Tensor): the weights for the perception network. 
#                 size: ({n_batch}, n_dim * n_hl_1 + n_hl_1 + sum(i=1..k-1){n_hl_i * n_hl_{i+1} + n_hl_{i+1}} + n_hl_k * n_dim + n_dim)
#                 where k is the number of hidden layers and n_hl_i is the length of the i-th hidden layer. 
#             hidden_layers (list of int): the lengths for the hidden layers, i.e. [n_hl_1, n_hl_2, ..., n_hl_k]
#                 It is by default '[]' so that the default MLP is an affine transformation. 
#             active_function (class): the active_function. 
#             trans_stretch (float): scaling of translation movement, commonly needed in iterative parameter training, 1 by default. 5 seems to be a good choice. 
            
#         Args for __call__:
#             X (bt.Tensor): Coordinates to be transformed.
#                 size: ({n_batch: optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
#             Returns (bt.Tensor): The transformed coordinates.
#                 size: ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
#         """
#         self.hidden_layers = hidden_layers
#         if not weights.has_batch:
#             if weights.n_dim == 2: weights.batch_dim = 0
#             else: weights = weights.unsqueeze([])
#         self.weights = weights
#         self.trans_stretch = trans_stretch
#         super().__init__(weights, hidden_layers = hidden_layers, trans_stretch = trans_stretch, residual = residual)
#         dim_const = weights.size(-1) - sum(hidden_layers) - sum(x * y for x, y in zip(hidden_layers[:-1], hidden_layers[1:]))
#         dim_coeff = hidden_layers[0] + hidden_layers[-1] + 1
#         avouch(dim_const % dim_coeff == 0, f"Wrong weight length for hidden layers of sizes {hidden_layers}, {dim_coeff} x n_dim + {dim_const} expected, but got {weights.size(-1)}.")
#         self.n_dim = dim_const // dim_coeff
#         self.n_batch = weights.n_batch
#         self.layers = []
#         p = 0
#         for i in range(len(hidden_layers) + 1):
#             in_features = self.n_dim if i == 0 else hidden_layers[i - 1]
#             out_features = self.n_dim if i == len(hidden_layers) else hidden_layers[i]
#             layer_weights = weights[..., p:p+out_features*in_features].view([self.n_batch], out_features, in_features)
#             p += out_features * in_features
#             layer_bias = weights[..., p:p+out_features].view([self.n_batch], out_features)
#             p += out_features
#             self.layers.append((layer_weights, layer_bias))
#         self.active_function = active_function
#         self.residual = residual
#         if self.active_function is None: self.active_function = bt.nn.ReLU
    
#     @classmethod
#     def get_weight_length(cls, n_dim, *hidden_layers):
#         hidden_layers = arg_tuple(hidden_layers)
#         return n_dim * (hidden_layers[0] + hidden_layers[-1] + 1) \
#             + sum(hidden_layers) + sum(x * y for x, y in zip(hidden_layers[:-1], hidden_layers[1:]))
    
#     @property
#     def n_weight_length(self):
#         return MultiLayerPerception.get_weight_length(self.n_dim, self.hidden_layers)
    
#     @classmethod
#     def random_init_params(cls, n_dim, *hidden_layers, n_batch=1, std=1e-4):
#         hidden_layers = arg_tuple(hidden_layers)
#         n_length = MultiLayerPerception.get_weight_length(n_dim, hidden_layers)
#         return std * bt.randn({n_batch}, n_length)
    
#     def __call__(self, X):
#         if not X.has_space: X = X.unsqueeze(-1)
#         Y = X.flatten(...).channel_dim_(None)
#         for i, (weights, bias) in enumerate(self.layers):
#             Y = weights @ Y
#             if i < len(self.layers) - 1: Y = self.active_function()(Y + bias.unsqueeze(-1))
#             else: Y += self.trans_stretch * bias.unsqueeze(-1)
#         if self.residual:
#             return X + Y.view_as(X).channel_dimension_(1)
#         else: return Y.view_as(X).channel_dimension_(1)
    
def _init_interpolation(image, trans, fill, target_space):
    image = bt.to_device(bt.to_bttensor(image))
    shape_out = image.shape
    if trans is None or trans.n_dim is None:
        if not image.has_batch:
            image = image.unsqueeze({})
        if not image.has_feature:
            image = image.unsqueeze([])
        n_dim = image.n_space_dim # Get the spatial rank.
    else:
        n_dim = trans.n_dim
        if image.n_dim == n_dim:
            if image.has_special:
                Warn(f"'interpolation' trying to transform [{image.n_special_dim}]+{image.n_space_dim}D image (with batch or channel) by {n_dim}D transformation, auto-removing special dimensions.")
                image.init_special()
            image = image.unsqueeze([]).unsqueeze_({})
        elif image.n_dim == n_dim + 1:
            if not image.has_batch:
                if image.has_channel: image = image.unsqueeze({})
                else: image = image.with_batch_dim(0).unsqueeze([])
            elif not image.has_channel:
                image = image.unsqueeze([])
            else:
                Warn(f"'interpolation' trying to transform [{image.n_special_dim}]+{image.n_space_dim}D image (with batch or channel) by {n_dim}D transformation, auto-removing the channel dimensions.")
                image = image.with_channel_dim(None).unsqueeze({})
        elif image.n_dim == n_dim + 2:
            # _channal/batch dimensions used here as they are n_dim when not existed. 
            if image.n_special_dim == 1:
                Warn(f"'interpolation' trying to transform [1]+{image.n_space_dim}D image (with batch or channel) by {n_dim}D transformation, auto-inserting new special dimension.")
            if not image.has_batch: image.batch_dimension = 0 if image.n_feature_dim > 0 else 1
            if not image.has_channel: image.channel_dimension = 0 if image.n_batch_dim > 0 else 1
    avouch(image.has_batch and image.has_channel, "Please use batorch tensor of size " +
            "({n_batch}, [n_channel/n_feature:optional], m_1, m_2, ..., m_r) [r=n_dim] for " + 
            f"data to be spatially interpolated, instead of {image.shape}. ")
    if trans is not None:
        avouch(image.n_batch == 1 or trans.n_batch in (None, image.n_batch, 1), "Please use transformation of a " +
            f"suitable n_batch to transform image with batchsize {image.n_batch}, currently {trans.n_batch}.")

    # Deal with the shape of input `image`
    n_batch = image.n_batch
    if n_batch == 1 and trans is not None and trans.n_batch is not None and trans.n_batch > 1: n_batch = trans.n_batch
    if n_batch == 1 and isinstance(target_space, bt.Tensor) and target_space.has_batch and target_space.n_batch > 1: n_batch = target_space.n_batch
    if image.n_batch == 1: image = image.repeated(n_batch, {})
    n_feature = image.n_channel
    size = bt.channel_tensor(image.space).int()
    if n_batch > 1 and not shape_out.has_batch: shape_out = bt.Size({n_batch}) + shape_out
    
    # Deal with input  `target_space`
    if target_space is None:
        avouch(trans is not None, "Arguments 'trans' and 'target_space' should not both be None for interpolation. ")
        target_space = perform_reshape(image.space, trans.reshape)
        shape_out = shape_out.with_space(target_space)
    if isinstance(target_space, tuple) and len(target_space) == n_dim: pass
    elif isinstance(target_space, bt.torch.Tensor): pass
    else: raise TypeError(f"Wrong target space for interpolation: {target_space}. ")
    if isinstance(target_space, tuple): 
        # Create a grid X with size ([n_dim], size_1, size_2, ..., size_r) [r=n_dim].
        X = bt.image_grid(target_space).float() # + bt.channel_tensor([float(a-b)/2 for a, b in zip(image.space, target_space)])
        # Compute the transformed coordinates. Y: ({n_batch}, [n_dim], size_1, size_2, ..., size_r) [r=n_dim].
        if trans is None: trans = Identity()
        Y = trans(X, domain='image-space')
        if not Y.has_batch: Y = Y.duplicate(n_batch, {})
        if Y.n_batch == 1: Y = Y.repeated(n_batch, {})
        Y = Y.amplify(n_feature, {})
        shape_out = shape_out.with_space(target_space)
    else:
        target_space = bt.to_bttensor(target_space)
        if not target_space.has_batch:
            if target_space.size(0) == n_batch and n_batch != n_dim or len([x for x in target_space.shape if x == n_dim]) >= 2:
                target_space.with_batch_dim(0)
            else: target_space = target_space.unsqueeze({})
        if not target_space.has_channel:
            if target_space.batch_dimension != 0 and target_space.size(0) == n_dim: target_space.with_channel_dim(0)
            elif target_space.batch_dimension != 1 and target_space.size(1) == n_dim: target_space.with_channel_dim(1)
            elif target_space.batch_dimension != target_space.n_dim - 1 and target_space.size(-1) == n_dim: target_space.with_channel_dim(-1)
        avouch(target_space.has_channel and target_space.n_channel == n_dim, "'target_space' for interpolation should have a channel dimension for coordinates. ")
        Y = target_space.repeated(n_batch // target_space.n_batch, {}).amplify(n_feature, {})
        shape_out = shape_out.with_space(target_space.space)
        
    image = image.mergedims([], {})
    n_batch = image.n_batch

    if isinstance(fill, str):
        if fill.lower() == 'nearest': background = None
        elif fill.lower() == 'background':
            bk_value = bt.stack([image[(slice(None),) + tuple(g)] for g in (bt.image_grid([2]*n_dim) * bt.channel_tensor(size-1)).flatten(...).transpose(0,1)], 1).median(1).values
            background = bk_value
        elif fill.lower() == 'zero': background = 0
    else: background = fill
    n_data = Y.flatten(...).size(-1)

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
    """
    Interpolate using backward transformation.
    i.e. Compute the image I s.t. trans(x) = y for x in I and y in input image using interpolation method:
        method = Linear: Bilinear interpolation
        method = Nearest [NO GRADIENT!!!]: Nearest interpolation

    Args:
        image (bt.Tensor): The target (w.r.t. trans) image.
            size: ({n_batch:optional}, [n_channel:optional], m_1, m_2, ..., m_r) [r=n_dim]
        trans (Function, micomputing.Transformation): Transformation function, mapping
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim] to ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        method (str: linear|nearest): The interpolation method. 
        target_space (tuple, bt.Tensor):
            Size (tuple) of a target ROI at the center of image. 
            OR Transformed coordinate space (bt.Tensor) of the output image. 
            size: length(n_dim) or ({n_batch:optional}, [n_dim:optional], size_1, size_2, ..., size_r) [r=n_dim]
        fill (str: nearest|background, int/float(number)): Indicate the way to fill background outside `Surrounding`. 
        derivative (bool): Whether to return the gradient. One can omit it when using torch.autograd.

        Returns (bt.Tensor): The transformed image.
            size: ({n_batch}, [n_channel:optional], m_1, m_2, ..., m_r) [r=n_dim]
            or when `target_space` is defined by tensor. 
            size: ({n_batch}, size_1, size_2, ..., size_r) [r=n_dim]
            or the derivative for the interpolation. (if `derivative = True`)
            size: ({n_batch}, [n_dim], size_1, size_2, ..., size_r) [r=n_dim]

    Examples::
        >>> Image = bt.rand(3, 100, 120, 80)
        >>> AM = bt.rand(4, 4)
        >>> AM[3, :] = bt.one_hot(-1, 4)
        >>> interpolation(Image, Affine(AM), method='Linear')
    """
    # Deal with input `trans` and special dimensions of `image`
    if trans is not None and trans.domain and not trans.backward:
        if hasattr(trans, 'inv'): trans = trans.inv_with_direct()
        else:
            Warn("Forward transformation found in method `interpolation`. Using `interpolation_forward` instead. ")
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
    W = bt.stack((1 - fY, fY), [1]).view({n_batch}, [2, n_dim], -1) # ({n_batch}, [2, n_dim], n_data).

    # Prepare for the output space: n_batch, m_1, ..., m_s
    if derivative: output = bt.zeros({n_batch}, [n_dim], *shape_out.space)
    else: output = bt.zeros(shape_out)
    
    for G in bt.image_grid([2]*n_dim).flatten(...).transpose(0, 1):
        ## New version of interpolation with indexing.
        Ind = (iY + G).flatten(...) # ({n_batch}, [n_dim], n_data).
        condition = ((Ind < 0) + (Ind > (size - 1).unsqueeze_to(Ind))).sum([]) == 0 # Compute border: ({n_batch}, n_data).
        Ind = bt.min(Ind.clamp(min=0), (size - 1).unsqueeze_to(Ind)) # Find nearest border pix. 
        coeff = bt.where(G.view({1}, [n_dim], 1).repeat(n_batch, 1, n_data) == 0, W[:, 0], W[:, 1]) # ({n_batch}, [n_dim], n_data)
        batch_dim = bt.arange(n_batch).view({n_batch}, [1]).duplicate(n_data, -1) # ({n_batch}, [1], n_data)
        Ind_with_batch = bt.cat(batch_dim, Ind, []) # ({n_batch}, [1 + n_dim], n_data)
        values = image[Ind_with_batch.split(1, [])].squeeze([]) # ({n_batch}, n_data)
        if background is not None: values = bt.where(condition, values, background)
        if not derivative:
            output += (coeff.prod([]) * values).view(shape_out)
        else:
            coeff_tmpmat = coeff.duplicate(n_dim, [1]) # ({n_batch}, [n_dim, n_dim], n_data)
            coeff_tmpmat[:, bt.arange(n_dim), bt.arange(n_dim)] = 1
            dcoeff = coeff_tmpmat.prod(1) * (G * 2 - 1).float().unsqueeze_to(coeff)
            output += (dcoeff * values.unsqueeze([])).view_as(output)
    bt.torch.cuda.empty_cache()
    if - eps < image.min().item() < 0: output = output.clamp(min=0)
    if 1 < image.max().item() < 1 + eps: output = output.clamp(max=1)
    return output.type(image.type())

@alias("resample_forward")
def interpolation_forward(
        image, 
        trans = None, 
        target_space = None,
        fill = 'zero',
        sigma = 1,
        method = '', 
        derivative = False,
    ):
    """
    Interpolate using forward transformation. 
    i.e. Compute the image I s.t. trans(x) = y for x in input image and y in the output I using partial volume method

    Args:
        image (bt.Tensor): The source (w.r.t. trans) image.
            size: ({n_batch:optional}, [n_channel:optional], m_1, m_2, ..., m_r) [r=n_dim]
        trans (Function, micomputing.Transformation): Transformation function, mapping
            size: ({n_batch:optional}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim] to ({n_batch}, [n_dim], n_1, n_2, ..., n_r) [r=n_dim]
        target_space (tuple): Size (tuple) of a target ROI at the center of image.
        fill (str: nearest|background, int, float): Indicate the way to fill background outside `Surrounding`. 
        sigma (int, float): The size of blurry window. The bigger it is, the more running time is needed and more blurry 
            image is generated. The smaller it is, the function runs faster and the image is sharper, except it may take 
            time to mend the wholes in the image. 
            One can set a smaller number for smooth transformations and a bigger one for folded ones.
        method (N/A; str: linear|nearest): The interpolation method, not yet available for forward transformations. 
        derivative (N/A; bool): Whether to return the gradient. One can omit it when using torch.autograd. Not yet available 
            for forward transformations.

        Returns (bt.Tensor): The transformed image.
            size: ({n_batch}, [n_channel:optional], m_1, m_2, ..., m_r) [r=n_dim]
            or when `target_space` is defined by tensor. 
            size: ({n_batch}, size_1, size_2, ..., size_r) [r=n_dim]
            or the derivative for the interpolation. (if `derivative = True`)
            size: ({n_batch}, [n_dim], size_1, size_2, ..., size_r) [r=n_dim]

    Examples::
        >>> Image = bt.rand(3, 100, 120, 80)
        >>> AM = bt.rand(4, 4)
        >>> AM[3, :] = bt.one_hot(-1, 4)
        >>> interpolation(Image, Affine(AM), method='Linear')
    """
    # Deal with input `trans` and special dimensions of `image`
    if trans is not None and trans.backward:
        if hasattr(trans, 'inv'): trans = trans.inv_with_direct()
        else:
            Warn("Backward transformation found in method `interpolation_forward`. Using `interpolation` instead. ")
            return interpolation(image, trans, method = method, target_space = target_space, fill = fill)
    
    image, trans, background, shape_out, n_dim, n_batch, n_feature, n_data, size, Y = _init_interpolation(image, trans, fill, target_space)
    shape_out.with_space(Y.space)
    
    avouch(Y.space == image.space, "Forward interpolation needs 'target_space' the same as input 'image'.")
    n_window = max(int(3 * sigma), 1)
    iY = bt.floor(Y).long() # Generate the integer part of Y
    indices = []; weights = []; values = []
    for G in bt.image_grid([n_window]*n_dim).flatten(...).transpose(0, 1):
        gY = iY + G
        gY = bt.min(gY, bt.channel_tensor(Y.space).expand_to(gY) - 1).clamp(min=0)
        indices.append(gY.flatten(2)) # ({n_batch}, [n_dim], n_data)
        W = bt.exp(- ((Y - gY) ** 2).sum([]) / 2 / (sigma ** 2)) / (sigma ** n_dim) / (math.sqrt(2 * math.pi) ** n_dim) # ({n_batch}, *n_data)
        weights.append(W.flatten(1)) # ({n_batch}, n_data)
        values.append((W * image).flatten(1)) # ({n_batch}, n_data)
    # ({n_batch}, [1 + n_dim], n_data x n_window ^ n_dim)
    Mindices = bt.cat(bt.arange(n_batch).view({n_batch}, [1]).duplicate(len(indices) * n_data, -1), bt.cat(indices, -1), [])
    # ({n_batch}, n_data x n_window ^ n_dim)
    Mweights = bt.cat(weights, -1)
    # ({n_batch}, n_data x n_window ^ n_dim)
    Mvalues = bt.cat(values, -1)
    # ({1 + n_dim}, n_batch x n_data x n_window ^ n_dim)
    with Mindices.hide_special():
        indices = Mindices.transpose(0, 1).flatten(1).with_batch_dim(0)
    # (n_batch x n_data x n_window ^ n_dim,)
    weights = Mweights.flatten(0)
    # (n_batch x n_data x n_window ^ n_dim,)
    values = Mvalues.flatten(0)

    if Version(bt.torch.__version__) >= Version("2"):
        creator = bt.torch.sparse_coo_tensor
    elif indices.device.type == 'cpu': creator = bt.torch.sparse.FloatTensor
    else: creator = bt.torch.cuda.sparse.FloatTensor

    collected_intensities = creator(indices, values, (n_batch,) + shape_out.space).to_dense()
    collected_intensities = bt.Tensor(collected_intensities, batch_dim=True) # ({n_batch}, *n_data)
    collected_weights = creator(indices, weights, (n_batch,) + shape_out.space).to_dense()
    collected_weights = bt.Tensor(collected_weights, batch_dim=True) # ({n_batch}, *n_data)

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
    output = ret.view(shape_out)
    bt.torch.cuda.empty_cache()
    if - eps < image.min().item() < 0: output = output.clamp(min=0)
    if 1 < image.max().item() < 1 + eps: output = output.clamp(max=1)
    return output.type(image.type())

# ############ Image Transformations ############

# class ImageTransformation(Transformation):
        
#     def __call__(self, X):
#         """
#         X (bt.Tensor): Image to be transformed.
#             size: ({n_batch:optional}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
#         Returns (bt.Tensor): The transformed image.
#             size: ({n_batch}, [n_channel], n_1, n_2, ..., n_r) [r=n_dim]
#         """
#         X = bt.to_bttensor(X)
#         if self.n_dim is None:
#             if not X.has_batch: X = X.unsqueeze([])
#             if not X.has_channel: X = X.standard().unsqueeze({1})
#         else:
#             if X.n_dim == self.n_dim: X = X.remove_special_().unsqueeze([]).unsqueeze({1})
#             elif X.n_dim == self.n_dim + 1:
#                 if X.has_batch: X.channel_dimension = None; X = X.unsqueeze({0 if X.batch_dimension > 0 else 1})
#                 elif X.has_channel: X = X.unsqueeze([])
#                 else: X = X.with_batch_dim(True).unsqueeze({1})
#             elif X.n_dim == self.n_dim + 2:
#                 # _channal/batch dimensions used here as they are n_dim when not existed. 
#                 if not X.has_batch: X.batch_dimension = 0 if X._channel_dimension > 0 else 1
#                 if not X.has_channel: X.channel_dimension = 0 if X._batch_dimension > 0 else 1
#         avouch(X.has_batch and X.has_channel, f"Please use batorch tensor of size \
#             ({{n_batch}}, [n_channel/n_feature:optional], m_1, m_2, ..., m_r) [r=n_dim] for \
#                 {self.__name__}, instead of {X.shape}. ")
#         return X.clone()

class Normalize(Transformation):
    """
    Normalize the intensity of an image.
    
    Args:
        _range = (low, high) (int or float or bt.Tensor): The lowest (and highest) intensity. 
            size: length(2) or ({n_batch:optional}, [n_channel:optional], (2))
        
    Args for __call__:
        X (bt.Tensor): Image to be transformed.
            size: ({n_batch:optional}, [n_channel:optional], n_1, n_2, ..., n_r) [r=n_dim]
        Returns (bt.Tensor): The transformed image.
            size: ({n_batch}, [n_channel], n_1, n_2, ..., n_r) [r=n_dim]
    """

    def __init__(self, *_range):
        if len(_range) == 0: _range = None
        elif len(_range) == 1 and isinstance(_range[0], (list, tuple)): _range = bt.tensor(list(_range[0])).with_func_dim(True)
        elif len(_range) == 1 and isinstance(_range[0], bt.Tensor): _range = _range[0]
        elif len(_range) == 1 and isinstance(_range[0], (int, float)): _range = bt.tensor((0, _range)).with_func_dim(True)
        elif len(_range) == 2 and all(isinstance(r, (int, float)) for r in _range): _range = bt.channel_tensor(_range)
        else: raise TypeError(f"Invalid range for Normalize: {_range}. ")
        if _range is None: pass
        else:
            if not _range.has_func:
                if _range.has_channel and _range.channel_size <= 2: _range.change_special_dim(_range.i_chan_dim, bt.FuncDim)
                elif _range.space in ((1,), (2,)): _range.change_special_dim(_range.space_start, bt.FuncDim)
            if not _range.has_batch: _range.unsqueeze_({0})
            if not _range.has_feature: _range.unsqueeze_([1])
            avouch(_range.has_batch and _range.has_channel and _range.has_func and _range.func_size == 2, f"Please use batorch tensor of size \
                ({{n_batch:optional}}, [n_channel:optional], (2)) for Normalizing parameters, instead of {_range.shape}. ")
        super().__init__(_range)
        self.range = _range
        self._range = None
        self.domain = 'non-spatial'

    def __call_non_spatial__(self, X):
        _range = self.range
        if _range is None:
            if X.n_ele >> 24 == 0: quant = bt.quantile
            else: quant = bt.approx_quantile
            _range = quant(X.flatten(...).float(), bt.to_bttensor([0.025, 0.975]), -1).movedim(0, -1)
            self._range = _range
        return ((X - _range[..., 0]) / (_range[..., 1] - _range[..., 0])).clamp(0., 1.)
    
    def __inv__(self):
        if self.range is not None: _range = self.range
        elif self._range is not None: _range = self._range
        else: return Normalize(None)
        den = _range[..., 1] - _range[..., 0]
        _range = bt.stack(-_range[..., 0], 1-_range[..., 0], [-1]) / den
        return Normalize(_range)
    
class Cropping(Transformation):
    """
    Cropping transformation.
        
    Args for __call__:
        Img (bt.Tensor): Image to be cropped.
            size: ({n_batch:optional}, [n_channel:optional], n_1, n_2, ..., n_r)
        Returns (bt.Tensor): The transformed coordinates. 
            size: ({n_batch}, [n_channel:optional], n_1, n_2, ..., n_r)
    """
    def __init__(self, shape, **kwargs):
        super().__init__(shape=shape, **kwargs)
        self.reshape = [shape]
        self.domain = 'non-spatial'
        
    def __call_non_spatial__(self, X):
        return bt.crop_as(X, self.shape)

# ############# Supporting Functions ############

def Bspline(i, U):
    """
    Cubic B-spline function. 
    Note: As long as i and U have the same size, any shape of tensors would do.
    
    Params:
        i (bt.Tensor): the index of segment function of B-spline.
            The value of each element can be chosen in (-1, 0, 1, 2). 
        U (bt.Tensor): the decimal argument of B-spline function. It should be within range [0, 1).
    """
    i = bt.to_bttensor(i); U = bt.to_bttensor(U)
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
    
    Args:
        i (bt.Tensor): the index of segment function of B-spline.
            The value of each element can be chosen in (-1, 0, 1, 2). 
        U (bt.Tensor): the decimal argument of B-spline function. It should be within range [0, 1).
    """
    i = bt.to_bttensor(i); U = bt.to_bttensor(U)
    return (
        bt.where(i == -1, - 3 * (1 - U) ** 2 / 6,
        bt.where(i == 0, 3 * U ** 2 / 2 - 2 * U,
        bt.where(i == 1, (- 3 * U ** 2 + 2 * U + 1) / 2,
        bt.where(i == 2, 3 * U ** 2 / 6,
        bt.zeros_like(U)))))
    )

def fBspline(c, x):
    c = bt.to_bttensor(c); x = bt.to_bttensor(x)
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
    Args:
        params (bt.Tensor):
            t1, t2, θ, s1, s2, ρ1, ρ2 in size: ({n_batch}, [7])
            t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2 in size: ({n_batch}, [9])

    Returns:
        in size: ({n_batch}, [3, 3])
    """
    params = bt.to_bttensor(params)
    if params.n_dim <= 1 and not params.has_batch: params = params.unsqueeze({})
    if params.n_dim <= 1 and not params.has_channel: params = params.unsqueeze([])
    if params.n_dim == 2 and not params.has_batch: params.batch_dimension = 0
    if params.n_dim == 2 and not params.has_channel: params.channel_dimension = 1
    avouch(params.has_batch, f"Please use batorch tensor of size ({{n_batch}}, [7 or 9]) \
        for Affine parameters, instead of {params.shape}. ")
    n_batch = params.n_batch
    if params.size(1) == 7:
        t1, t2, θ, s1, s2, ρ1, ρ2 = params.split()
        c1 = bt.zeros({n_batch}, [1]); c2 = bt.zeros({n_batch}, [1])
    if params.size(1) == 9:
        t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2 = params.split()
    a = (ρ1 * ρ2 + 1) * s1 * bt.cos(θ) + ρ1 * s2 * bt.sin(θ)
    b = - (ρ1 * ρ2 + 1) * s1 * bt.sin(θ) + ρ1 * s2 * bt.cos(θ)
    c = ρ2 * s1 * bt.cos(θ) + s2 * bt.sin(θ)
    d = - ρ2 * s1 * bt.sin(θ) + s2 * bt.cos(θ)
    return bt.cat(
        bt.cat((a, b, t1 - a * c1 - b * c2 + c1, c, d, t2 - c * c1 - d * c2 + c2), []).view({n_batch}, [2, 3]), 
        bt.one_hot(-1, 3).view([1, 3]).duplicate(n_batch, {}), 1
    )

def Quaterns2Matrix(params):
    """
    Args:
        Quatern (bt.Tensor): qb, qc, qd, px, py, pz in size: ({n_batch}, [6])
    Returns:
        Matrix: ({n_batch}, [4, 4])
    """
    params = bt.to_bttensor(params)
    if params.n_dim <= 1 and not params.has_batch: params = params.unsqueeze({})
    if params.n_dim <= 1 and not params.has_channel: params = params.unsqueeze([])
    if params.n_dim == 2 and not params.has_batch: params.batch_dimension = 0
    if params.n_dim == 2 and not params.has_channel: params.channel_dimension = 1
    avouch(params.n_dim == 2 and params.has_batch and params.has_channel, 
           f"Please use batorch tensor of size ({{n_batch}}, [6]) for Affine parameters, instead of {params.shape}. ")
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
        bt.cat((R11, R12, R13, x, R21, R22, R23, y, R31, R32, R33, z), 1).view({n_batch}, [3, 4]),
        bt.one_hot(-1, 4).view([1, 4]).duplicate(n_batch, {}), 1
    )

def Matrix2Quaterns(params):
    """
    Args:
        Matrix (bt.Tensor): ({n_batch}, [4, 4])
    
    Returns:
        Quatern: qb, qc, qd, px, py, pz in size: ({n_batch}, [6])
    """
    params = bt.to_bttensor(params)
    if params.n_dim <= 2 and not params.has_batch: params = params.unsqueeze({})
    if params.n_dim == 3 and not params.has_batch: params.batch_dimension = 0
    if params.n_dim == 3: params.with_n_feature_dim(2)
    avouch(params.n_dim == 3 and params.has_batch, 
           f"Please use batorch tensor of size ({{n_batch}}, [4, 4]) for Affine matrix, instead of {params.shape}. ")
    n_batch = params.n_batch
    x, y, z = params[..., :3, -1].channel_dim_(1).split(1, 1)
    a2 = (bt.diag(params).sum([]).unsqueeze([]) + 1) / 4
    a = bt.sqrt(a2)
    b2 = a2 - (params[..., 1, 1] + params[..., 2, 2]) / 2
    c2 = a2 - (params[..., 2, 2] + params[..., 0, 0]) / 2
    d2 = a2 - (params[..., 0, 0] + params[..., 1, 1]) / 2
    D = params - params.T
    b = bt.sign(D[..., 2, 1]) * bt.sqrt(b2)
    c = - bt.sign(D[..., 2, 0]) * bt.sqrt(c2)
    d = bt.sign(D[..., 1, 0]) * bt.sqrt(d2)
    return bt.cat(b, c, d, x, y, z, [])
