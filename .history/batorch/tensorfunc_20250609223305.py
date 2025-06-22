
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "Tensor functions for `batorch`.",
    requires = ["numpy", "torch", "pycamia", "pyoverload", "matplotlib", "sympy"]
)

__all__ = """
    MultiVariate
    grad
    
    batorch_wrapper
    conv
    crop_as
    decimal
    divide
    dot
    equals
    gaussian_kernel
    image_grid               imagegrid
    center_gravity           mask_center
    one_hot
    pad
    permute_space
    approx_quantile
    normalize01
    input_shape

    grad_image               Jacobian
    up_scale                 down_scale
    
    upper_matrix             upper_triangular_matrix
    lower_matrix             lower_triangular_matrix
    skew_symmetric           cross_matrix
    skew_symmetric_params    uncross_matrix
    
    norm                     norm2
    Fnorm                    Fnorm2
    frobenius_norm
    meannorm                 meannorm2
    mean_norm                mean_norm2
    
    summary                  display
    byte_size                get_memory_used
    num_model_params
""".split()

import math, re

with __info__:
    import numpy as np
    import torch
    import batorch as bt
    from functools import wraps
    from sympy import symbols, Eq, solve
    from pyoverload import *
    from pycamia import avouch, touch, alias, decorator, Warn
    from pycamia import get_environ_vars, get_snakenames, get_args_expression
    from pycamia import SPrint, Version, argmin, tokenize, to_tuple, ByteSize
    from matplotlib import pyplot as plt
    
class MultiVariate(bt.Tensor):
    
    def __new__(cls, *args, inherited=None):
        if inherited is not None:
            assert len(args) == 1, "MultiVariate only takes inherited for inheritance. "
            self = args[0].as_subclass(cls).special_from(args[0])
            # self = super()._make_subclass(cls, args[0]).special_from(args[0])
            self.inherited = inherited.copy()
            return self
        
        if len(args) == 1 and isinstance(args[0], (list, tuple, generator)):
            args = list(args[0])
        else: args = list(args)
        
        assert len(args) > 0, "MultiVariate should at least be initialized by one variable. "
        
        # Default values for basic properties: if no such value is given, use the defaults. 
        match_props = dict(requires_grad=False, device=bt.device('cpu'), n_batch=None, n_sequence=None)
        # The function that cast any tensor to the united state. 
        def cast(prop, x, to):
            if prop == 'requires_grad': return x.requires_grad_(to)
            elif prop == 'device': return x.to(to)
            elif prop == 'n_batch' and to is not None:
                y = (x.repeated(to, {}) if x.n_batch == 1 else x) if x.has_batch else x.duplicate(to, {})
                if isinstance(x, MultiVariate): y.with_batch_size(to)
                return y                    
            elif prop == 'n_sequence' and to is not None:
                y = (x.repeated(to, '') if x.n_time == 1 else x) if x.has_time == 1 else x.duplicate(to, '')
                if isinstance(x, MultiVariate): y.with_sequence_size(to)
                return y
            return x
        # The united state selected by the first defined property value. 
        match_pivots = {}
        for prop, default in match_props.items():
            for arg in args:
                current = match_pivots.get(prop, default)
                if prop == 'n_batch': new_value = touch(lambda: arg.n_batch)
                elif prop == 'n_sequence': new_value = touch(lambda: arg.sequence[0])
                else: new_value = getattr(arg, prop, default)
                if current == default: match_pivots[prop] = new_value
        
        shapes = []
        tensor_seps = [0]
        for i, arg in enumerate(args):
            casted = False
            if not isinstance(arg, bt.Tensor):
                arg = bt.tensor(arg); casted = True
            for prop, target_value in match_pivots.items():
                if prop == 'n_batch': value = arg.n_batch if arg.has_batch else None
                elif prop == 'n_sequence': value = arg.sequence[0] if arg.has_sequence else None
                else: value = getattr(arg, prop, default)
                if value != target_value:
                    # This property requires casting
                    if value is not None: Warn(f"Variates of different '{prop}' detected in constructing MultiVariate, casting them to the first. ")
                    arg = cast(prop, arg, target_value)
                    casted = True
                    
            if match_pivots['n_batch'] is None and match_pivots['n_sequence'] is None:
                shape = arg.shape
                args[i] = arg.flatten().init_special()
            elif match_pivots['n_sequence'] is None:
                if arg.has_batch: arg = arg.move_dim({}, 0)
                else: arg = arg.unsqueeze({})
                shape = arg.shape.with_batch(match_pivots['n_batch'])
                args[i] = arg.flatten(1).special_from(bt.Size({1}, 1))
            elif match_pivots['n_batch'] is None:
                if arg.has_sequence: arg = arg.move_dim('', 0)
                else: arg = arg.unsqueeze('0')
                shape = arg.shape.with_sequence(match_pivots['n_sequence'])
                args[i] = arg.flatten(1).special_from(bt.Size('1', 1))
            else:
                if not arg.has_batch: arg = arg.unsqueeze({})
                if not arg.has_sequence: arg = arg.unsqueeze('')
                arg = arg.movedim({}, 0).movedim('', 1)
                shape = arg.shape.with_batch(match_pivots['n_batch']).with_sequence(match_pivots['n_sequence'])
                args[i] = arg.flatten(2).special_from(bt.Size({1}, '1', 1))
            
            if isinstance(arg, MultiVariate):
                shapes.extend(arg.inherited['shapes'])
                tensor_seps.extend([s + tensor_seps[-1] for s in arg.inherited['tensor_seps'][1:]])
            else:
                shapes.append(shape)
                tensor_seps.append(tensor_seps[-1] + args[i].size(-1))
        
        self = bt.cat(args, -1).as_subclass(cls).special_from(args[0])
        # self = super()._make_subclass(cls, bt.cat(args, -1)).special_from(args[0])
        self.inherited = dict(shapes=shapes, tensor_seps=tensor_seps)
        return self
    
    def with_batch_size(self, n_batch):
        if n_batch is None: return self
        shapes = []
        for s in self.inherited['shapes']:
            if s.has_batch: start, end = s.batch_range
            else: start, end = 0, 0
            shapes.append(s[:start] + bt.Size({n_batch}) + s[end:])
        self.inherited['shapes'] = shapes
        return self
    
    def with_sequence_size(self, n_sequence):
        if n_sequence is None: return self
        shapes = []
        for s in self.inherited['shapes']:
            if s.has_time: start, end = s.time_range
            else: start, end = s.n_batch_dim, s.n_batch_dim
            shapes.append(s[:start] + bt.Size(f'{n_sequence}') + s[end:])
        self.inherited['shapes'] = shapes
        return self
    
    def __str__(self):
        string = self.as_subclass(bt.Tensor).special_from(self).__str__()
        output = string.replace("Tensor", "MultiVariate")
        if output.endswith(')'): output = output[:-1] + f", tensor_shapes={self.inherited['shapes']})".replace("batorch.Size", "")
        return output
    
    def __repr__(self):
        return self.as_subclass(bt.Tensor).special_from(self).__repr__().replace("Tensor", "MultiVariate").replace(", requires_grad", f", tensor_shapes={self.inherited['shapes']}, requires_grad").replace("batorch.Size", "")
    
    def __raw_repr__(self):
        reprs = []
        for v in self.tuple():
            if v.has_batch: v = v.pick(0, dim={})
            if v.has_seque: v = v.pick(0, dim='')
            reprs.append(v.init_special().__raw_repr__())
        return '{' + ', '.join(reprs) + '}'
    
    def __len__(self):
        return len(self.inherited['shapes'])
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def tuple(self):
        return tuple(self[i] for i in range(len(self)))
    
    def __getrange__(self, index):
        if isinstance(index, int):
            shape = self.inherited['shapes'][index]
            start = self.inherited['tensor_seps'][index]
            end = self.inherited['tensor_seps'][index + 1]
        elif isinstance(index, slice):
            shape = self.inherited['shapes'][index]
            start = self.inherited['tensor_seps'][0 if index.start is None else index.start]
            end = self.inherited['tensor_seps'][-1 if index.stop is None else index.stop]
        else: raise TypeError(f"Invalid index {index} for __getrange__. ")
        return start, end, shape
    
    def __getmask__(self, index, value=1):
        start, end, shape = self.__getrange__(index)
        value = bt.bttensor(value)
        if not shape.has_batch:
            value = value.flatten().init_special()
            return bt.cat(bt.zeros({shape.n_batch}, start), bt.ones({shape.n_batch}, end-start) * value, bt.zeros({shape.n_batch}, self.size(-1) - end))
        if value.has_batch: value = value.movedim({}, 0).flatten(1).special_from(bt.Size({1}, 1))
        else: value = value.duplicate(shape.n_batch, {}).flatten(1).special_from(bt.Size({1}, 1))
        return bt.cat(bt.zeros(start), bt.ones(end-start) * value, bt.zeros(self.size(-1) - end))
    
    def __getitem__(self, index):
        try:
            if isinstance(index, int):
                start, end, shape = self.__getrange__(index)
                return super().__getitem__((..., slice(start, end))).as_subclass(bt.Tensor).view(shape)
            elif isinstance(index, slice):
                start, end, shapes = self.__getrange__(index)
                slice_start = 0 if index.start is None else index.start
                slice_stop = None if index.stop is None else (index.stop + 1)
                starting = self.inherited['tensor_seps'][slice_start]
                inherited = dict(shapes=shapes, 
                    tensor_seps=[x - starting for x in self.inherited['tensor_seps'][slice_start:slice_stop]])
                return MultiVariate(super().__getitem__((..., slice(start, end))), inherited=inherited)
        except RuntimeError:
            raise TypeError(f"Cannot get item {f'{index.start}:{index.stop}' if isinstance(index, slice) else index} from MultiVariate(shapes={self.inherited['shapes']}):"
                            f"getting partial[..., {start}:{end}] of shape {super().__getitem__((..., slice(start, end))).shape} but requires shape(s) {shape}")
        raise TypeError("MultiVariates are regarded as tuples, please get tensor items by mv[i] and then index them as Tensors. ")
        
    def __setitem__(self, index, value):
        """WARNING: This is an inplace operation, please be careful. """
        if isinstance(index, (int, slice)):
            start, end, _ = self.__getrange__(index)
            self[..., start:end] = value
        else: return super().__setitem__(index, value)
        
    @property
    def grad_variate(self): return self[len(self) // 2:]    
    @grad_variate.setter
    def grad_variate(self, value):
        """WARNING: This is an inplace operation, please be careful. """
        self[len(self) // 2:] = value

def grad(outputs, inputs, grad_outputs, retain_graph = None, create_graph = False, only_inputs = True, allow_unused = False, is_grads_batched = False):
    if outputs is None: results = (None,) * len(inputs)
    else: results = torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph = retain_graph, create_graph = create_graph, only_inputs = only_inputs, allow_unused = allow_unused, is_grads_batched = is_grads_batched)
    def convert(x, ref):
        if x is None: return bt.zeros_like(ref)
        if not isinstance(x, torch.Tensor): return x
        if isinstance(ref, bt.Tensor): return x.as_subclass(ref.__class__).special_from(ref)
        return x.as_subclass(ref.__class__)
    if isinstance(inputs, torch.Tensor): results = convert(results, inputs)
    return tuple(convert(r, i) for r, i in zip(results, inputs))

@decorator
def batorch_wrapper(func: callable=None, **options):
    """
    The wrapper for batorch functions, ensuring the inputs of the function are batorch 
        tensors, and the return type will be casted as the input type for non-batorch inputs. 
    """
    if func is None: return lambda f: batorch_wrapper(f, **options)
    i_arg = options.pop('i_arg', None)
    default_aux = options
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        values = get_arg_values(get_func_info(func), *args, **kwargs)
        f_varnames = func.__code__.co_varnames
        new_args = []
        new_kwargs = {}
        input_class = None
        for i, (n, v) in enumerate(zip(f_varnames, values)):
            if func.__annotations__.get(n, None) in (array, bt.Tensor):
                if i_arg is None and input_class is None or i == i_arg: input_class = v.__class__
                if v is None: ...
                elif not isinstance(v, torch.Tensor): v = bt.tensor(v)
                elif not isinstance(v, bt.Tensor): v = v.as_subclass(bt.Tensor).init_special()
            if (func.__code__.co_flags & 0x04) and i == func.__code__.co_argcount + func.__code__.co_kwonlyargcount: new_args.extend(v)
            elif (func.__code__.co_flags & 0x08) and i == func.__code__.co_argcount + func.__code__.co_kwonlyargcount + ((func.__code__.co_flags & 0x04) >> 2): new_kwargs.update(v)
            elif i < len(args): new_args.append(v)
            else: new_kwargs[n] = v
        ret = func(*new_args, **new_kwargs)
        if not isinstance(ret, bt.Tensor): return ret
        ret_obj = None
        if input_class is not None:
            if issubclass(input_class, torch.Tensor) and input_class != bt.Tensor: ret_obj = ret.as_subclass(input_class)
            elif issubclass(input_class, np.ndarray): ret_obj = ret.detach().cpu().numpy()
            elif issubclass(input_class, list): ret_obj = ret.detach().cpu().numpy().tolist()
            elif issubclass(input_class, tuple): ret_obj = tuple(ret.detach().cpu().numpy().tolist())
        if ret_obj is None: ret_obj = ret
        for k, default in default_aux.items():
            aux = getattr(ret, k, default)
            if issubclass(default.__class__, torch.Tensor):
                if input_class != bt.Tensor: aux = (aux if isinstance(aux, torch.Tensor) else torch.tensor(aux)).as_subclass(default.__class__)
                else: aux = aux if isinstance(aux, bt.Tensor) else bt.tensor(aux)
            elif issubclass(default.__class__, np.ndarray): aux = aux.detach().cpu().numpy() if isinstance(aux, torch.Tensor) else np.array(aux)
            elif issubclass(default.__class__, list): aux = aux.detach().cpu().numpy().tolist() if isinstance(aux, torch.Tensor) else list(aux)
            elif issubclass(default.__class__, tuple): aux = tuple(aux.detach().cpu().numpy().tolist()) if isinstance(aux, torch.Tensor) else tuple(aux)
            try: setattr(ret_obj, k, aux)
            except AttributeError: pass
        return ret_obj
    return wrapper_func

# @overload
# def batorch_wrapper(func: callable, i_arg=None, **default_aux):
#     @wraps
#     def wrapper_func(*args, **kwargs):
#         values = get_arg_values(get_func_info(func), *args, **kwargs)
#         f_varnames = func.__code__.co_varnames
#         n_posarg = func.__code__.co_posonlyargcount
#         new_args = []
#         new_kwargs = {}
#         input_class = None
#         for i, (n, v) in enumerate(zip(f_varnames, values)):
#             if func.__annotations__.get(n, None) == bt.Tensor:
#                 if i_arg is None and input_class is None or i == i_arg: input_class = v.__class__
#                 if not isinstance(v, torch.Tensor): v = bt.tensor(v)
#                 else: v = v.as_subclass(bt.Tensor)
#             if i < n_posarg: new_args.append(v)
#             else: new_kwargs[n] = v
#         ret = func(*new_args, **new_kwargs)
#         if input_class is not None:
#             if issubclass(input_class, torch.Tensor): ret_obj = ret.as_subclass(input_class)
#             elif issubclass(input_class, np.ndarray): ret_obj = ret.detach().cpu().numpy()
#             elif issubclass(input_class, list): ret_obj = ret.detach().cpu().numpy().tolist()
#             elif issubclass(input_class, tuple): ret_obj = tuple(ret.detach().cpu().numpy().tolist())
#         for k, default in default_aux:
#             aux = getattr(ret, k, default)
#             if issubclass(default.__class__, torch.Tensor): aux = (aux if isinstance(aux, torch.Tensor) else torch.tensor(aux)).as_subclass(default.__class__)
#             elif issubclass(default.__class__, np.ndarray): aux = aux.detach().cpu().numpy() if isinstance(aux, torch.Tensor) else np.array(aux)
#             elif issubclass(default.__class__, list): aux = aux.detach().cpu().numpy().tolist() if isinstance(aux, torch.Tensor) else list(aux)
#             elif issubclass(default.__class__, tuple): aux = tuple(aux.detach().cpu().numpy().tolist()) if isinstance(aux, torch.Tensor) else tuple(aux)
#             setattr(ret_obj, k, aux)
#         return ret_obj
#     return wrapper_func

# @overload
# def batorch_wrapper(i_arg: int | null =None, **default_aux):
#     return wraps(lambda f: batorch_wrapper(f, i_arg, **default_aux))

@batorch_wrapper
def conv(tensor: bt.Tensor, kernel: bt.Tensor, padding='SAME'):
    """
    Convolution for a batorch tensor: only in space dimensions. 
    
    Args:
        tensor (bt.Tensor): With shape ({n_batch: optional}, [n_channel_in: optional], *n_space).
        kernel (bt.Tensor): With shape ([n_channel_out: optional], [n_channel_in: optional], *n_kernel_size).
        padding (str, int, tuple): Convolution padding: string 'SAME' stands for not changing the size of 'tensor'. 
    
    Returns:
        output (bt.Tensor): With shape ({n_batch: optional}, [n_channel_out: optional], *(n_space - n_kernel_size // 2)).
    """
    input_shape = tensor.shape
    if not tensor.has_batch: tensor = tensor.unsqueeze({})
    if not tensor.has_feature: tensor = tensor.unsqueeze([])
    n_dim = tensor.n_space_dim
    avouch(kernel.n_space_dim == n_dim, f"Kernel for {n_dim}D convolution should be of size ([channel_out, channel_in], n_kernel_1, ..., n_kernel_{n_dim}) with a dimension of {n_dim}. ")
    if kernel.has_feature:
        avouch(kernel.n_feature_dim == 2, f"Kernel for {n_dim}D convolution should have exactly 2 channel dimensions, only size ([channel_out, channel_in], n_kernel_1, ..., n_kernel_{n_dim}) is allowed, not {kernel.shape}.")
        avouch(tensor.has_channel, f"Tensor input for convolution should have channel dimension (one feature dimension): not tensor size {tensor.shape} and kernel size {kernel.shape}. Use kernel without special dimension to perform spatial-dimension-only convolution. ")
        avouch(tensor.n_feature_dim == 1 and tensor.n_channel == kernel.feature[1], f"Channels of convolution should match: tensor size {str(tensor.shape).replace('[', '[>').replace(']', '<]')} and kernel size {str(kernel.shape).replace(f'{kernel.feature[-1]}]', f'>{kernel.feature[-1]}<]')}. ")
        space_conv = False
    else: space_conv = True

    is_same = False
    if isinstance(padding, str) and padding == 'SAME':
        is_same = True
        padding = tuple(x // 2 for x in kernel.space)
    if isinstance(padding, int):
        padding = (padding,) * n_dim
    avouch(isinstance(padding, tuple), f"Unrecognized argument 'padding' for convolution: {padding}. ")
    conv = eval("bt.nn.functional.conv%dd"%n_dim)
    
    if space_conv:
        prev_batch_feature = bt.Size({tensor.n_batch}) + tensor.feature
        tensor = tensor.merge_dims([], {}).unsqueeze([])
        kernel = kernel.unsqueeze(0, 0)
    result = conv(tensor, kernel.type(tensor.dtype), padding=padding)
    if space_conv: result = result.squeeze([]).splitdim({}, prev_batch_feature)
    if not input_shape.has_batch: result.squeeze_({})
    if not input_shape.has_feature: result.squeeze_([])
    if is_same: result = crop_as(result, input_shape.space)
    
    # if kernel.size(0) > 1:
    #     if input_shape.has_channel:
    #         input_shape = input_shape.with_dim_size(input_shape.channel_dim, kernel.size(0))
    
    # if is_same: result = crop_as(result, tensor).view(*input_shape)
    # else: result = result.view(input_shape.with_space(tuple(t - k + 1 + 2 * p for t, k, p in zip(tensor.space, kernel.space, padding))))
    return result

@alias("meannorm2", "mean_norm2", root = False, mean = True)
@alias("meannorm", "mean_norm", mean = True)
@alias("Fnorm2", "norm2", root = False)
@alias("Fnorm", "frobenius_norm")
@batorch_wrapper
def norm(tensor: bt.Tensor, p = 2, root = True, mean = False, dim=null):
    """
    The norm of a tensor in (the first available condition):
    (1) all feature dimensions (if n_feature_dim >= 1);
    (2) all space dimensions (if n_space_dim >= 1); 
    (3) all sequence dimensions (if n_sequence_dim >= 1). 
    Use 'dim=None' to compute for all dimensions. 

    Args:
        tensor (bt.Tensor): The tensor to compute norm, commonly in shape ({n_batch}, [n_channel], n_1, ..., n_{n_dim})
        p (int, optional): The order of norm. Defaults to 2.
            p = 1 for L1-norm/ LASSO;
            p = 2 for L2-norm or Frobenius;
            p = math.inf / float('inf') for L∞-norm / maximum. 
        root (bool, optional): Whether to calculate the root. Defaults to True for common norms.
        mean (bool, optional): Whether to use mean across the dimensions. Defaults to False for common norms.
        dim (_type_, optional): The dimension to perform norm. Defaults to null, which auto selects the dimensions.
            One need to manually set 'dim=None' to perform norm in all dimensions. 
    """
    if dim is None: dim = bt.exist_dim(tensor, dim)
    else: dim = bt.linalg_dim[1:](tensor, None if dim is null else dim)
    tensor = tensor.abs()
    if p == float('inf'): return tensor.max(*dim)
    if mean: return (tensor ** p).mean(*dim) ** (1 / p) if root else (tensor ** p).mean(*dim)
    else: return (tensor ** p).sum(*dim) ** (1 / p) if root else (tensor ** p).sum(*dim)

@batorch_wrapper
def decimal(tensor: bt.Tensor):
    """
    The decimal part of tensor: {x} = x-[x]. 
    """
    return tensor - bt.floor(tensor)

@batorch_wrapper
def divide(a: bt.Tensor, b: bt.Tensor, limit=1, tol=1e-6):
    """
    def divide(a: bt.Tensor, b: bt.Tensor, limit=1, tol=1e-6)
    Element-wise division `a / b`, except it equals to `limit` if the divider `b` is smaller than `tol` (which is regarded as the division of zero)
    """
    a_s, b_s = a.shape ^ b.shape
    a = a.view(a_s)
    b = b.view(b_s)
    shape = bt.Size(max(x, y) for x, y in zip(a_s, b_s)).special_from(a_s)
    return bt.where(b.abs() < tol, limit * bt.ones(shape), a / bt.where(b.abs() < tol, tol * bt.ones(shape), b))

@batorch_wrapper
def equals(x: bt.Tensor, y: bt.Tensor, rtol=1e-4, atol=1e-6):
    """
    Determine whether the corresponding elements in two tensors are equal to each other. 
    The region of equivalence is a union of a diamond area at the origin point (⟐) 
        and a cross area in shape 'x' (with borders not passing the origin: '⪥').
    The cross area includes points near lines y=x and y=-x with an angle dilation of approximately arctan(tol). 
    The diamond area has a side length of √2*tol. 
    The reason why a diamond area is added at the origin instead of square (⊡) or circular region (⊙) is that,
        for inputs of difference storage accuracy (e.g. int & float), higher tolerance should be placed for errors 
        in the lower accuracy (e.g. float) tensor than in the higher accuracy (e.g. int) tensor, causing peaks in tolerance map. 
    """
    scale = bt.max(bt.abs(x), bt.abs(y))
    return (bt.abs(x - y) < rtol * scale + atol) | (scale < atol)

def gaussian_kernel(n_dim = 2, kernel_size = 3, sigma = 0, normalize = True):
    r"""
    Create a bt.Tensor Gaussian kernel of size (kernel_size, ..., kernel_size). 
                                                \---{n_dim} kernel_size's---/

    Args:
        n_dim (int, optional): The dimension of the kernel. Defaults to 2.
        kernel_size (int, optional): The side length of the kernel. Defaults to 3.
        sigma (int or float, optional): The standard deviation sigma of Gaussian distribution, 
                                            in unit 'pixel'. Defaults to 0.
        normalize (bool, optional): Whether the sum of kernel should be 1. Defaults to True.
    """
    if not isinstance(kernel_size, bt.torch.Tensor):
        if not isinstance(kernel_size, (tuple, list)): kernel_size = (kernel_size,)
        kernel_size = list(kernel_size)
        if len(kernel_size) == 1: kernel_size *= n_dim
        kernel_size = bt.Tensor(kernel_size)
    elif not isinstance(kernel_size, bt.Tensor): kernel_size = kernel_size.as_subclass(bt.Tensor).init_special()
    kernel_size.n_feature_dim = 1
    radius = (kernel_size - 1) / 2

    if sigma == 0: sigma = radius * 0.6
    elif not isinstance(sigma, bt.torch.Tensor):
        if not isinstance(sigma, (tuple, list)): sigma = (sigma,)
        sigma = list(sigma)
        if len(sigma) == 1: sigma *= n_dim
        sigma = bt.Tensor(sigma)
    elif not isinstance(sigma, bt.Tensor): sigma = sigma.as_subclass(bt.Tensor).init_special()
    sigma.n_feature_dim = 1

    grid = bt.image_grid(*kernel_size.tolist()).float()
    kernel = bt.exp(- (((grid - radius) / sigma) ** 2).sum(0) / 2)
    return (kernel / kernel.sum()) if normalize else kernel

@batorch_wrapper
def dot(g1: bt.Tensor, g2: bt.Tensor, dim=null):
    """
    The dot product of two tensors in (the first available condition):
    (1) all feature dimensions (if n_feature_dim >= 1);
    (2) all space dimensions (if n_space_dim >= 1); 
    (3) all sequence dimensions (if n_sequence_dim >= 1). 
    Use 'dim=None' to compute for all dimensions. 
    """
    if dim is None: dim = bt.exist_dim(g1, dim)
    else: dim = bt.linalg_dim[1:](g1, None if dim is null else dim)
    avouch(g1.shape == g2.shape, "Please make sure the dot product recieve two tensor of a same shape. Use 'bt.expand_to' to adjust if necessary. ")
    avouch(g1.n_feature_dim == g2.n_feature_dim, "Please make sure the inputs of 'dot' have a same feature dimension. ")
    return (g1 * g2).sum(*dim)

@batorch_wrapper
def Jacobian(X, Y, dt=1, pad=False):
    """
    The Jacobian matrix; Note that it is a transpose of grad_image if X is standard grid as it follows the orientation of Jacobian.
    
    Args:
        X (bt.Tensor): in shape ({n_batch}, [n_dim], n_1, ..., n_{n_dim})
        Y (bt.Tensor): in shape ({n_batch}, [n_feature], n_1, ..., n_{n_dim})
        
    Returns:
        Jacobian(output, pad = True): in shape ({n_batch}, [n_feature, n_dim], n_1, ..., n_{n_dim})
        Jacobian(output, pad = False): in shape ({n_batch}, [n_feature, n_dim], n_1 - dt, ..., n_{n_dim} - dt)
    """
    dX = grad_image(X, dx=dt, pad=pad) # ({n_batch}, [n_dim, n_dim], n_1, ..., n_{n_dim})
    dY = grad_image(Y, dx=dt, pad=pad) # ({n_batch}, [n_dim, n_feature], n_1, ..., n_{n_dim})
    # dy_k/dx_i = sum_j [(dy_k/dt_j) / (dx_i/dt_j)] => Jac[{b}, [k, i], ...] = sum_j Jac_j[{b}, [j, k, i], ...] = dY[{b}, [j, k, *], ...] / dX[{b}, [j, *, i], ...]
    return bt.divide(dY.unsqueeze([3]), dX.unsqueeze([2]), 0.0, tol=1e-4).sum([0])

@batorch_wrapper
def grad_image(array: bt.Tensor, dx=1, pad=False):
    '''
        Gradient image / tensor of array (dx is the displacement for finite difference). 
        If pad is not True, the image will trim. The sizes should be: 
        array: ({n_batch}, [n_feature], n_1, ..., n_{n_dim})
        output (pad = True): ({n_batch}, [n_dim, n_feature], n_1, ..., n_{n_dim})
        output (pad = False): ({n_batch}, [n_dim, n_feature], n_1 - dx, ..., n_{n_dim} - dx)
        OR:
        array: ({n_batch}, n_1, ..., n_{n_dim})
        output (pad = True): ({n_batch}, [n_dim], n_1, ..., n_{n_dim})
        output (pad = False): ({n_batch}, [n_dim], n_1 - dx, ..., n_{n_dim} - dx)
    '''
    if not array.has_batch: array = array.unsqueeze({})
    output = []
    size = array.space
    if not pad: size = tuple(x - dx for x in size)
    for d in range(*array.space_range):
        b = (slice(None, None),) * d + (slice(dx, None),) + (slice(None, None),) * (array.ndim - d - 1)
        a = (slice(None, None),) * d + (slice(None, -dx),) + (slice(None, None),) * (array.ndim - d - 1)
        output.append(bt.crop_as((array[b] - array[a]) / dx, size))
    return bt.stack(output, [int(array.has_batch)])

@overload
@batorch_wrapper(roi=tuple())
def crop_as(x: array, y: tuple, center: tuple, fill: number=0) -> array:
    """
    crop a Tensor `x` as the shape given by `y`. 

    Args:
        x (bt.Tensor): The data to crop (or pad if the target shape is larger), 
            Shape: ({n_batch}, [*n_feature], *n_space). 
            Note that only the space dimensions of the tensor are cropped/padded by default.
        y (bt.Tensor or tuple): The target shape in tuple or another tensor to provide the target shape.
        center (tuple, optional): The center of the target box. Defaults to the center of x's shape. 
            Note: Do calculate the center w.r.t. input `x` if one is expanding the tensor, 
                as `x`'s center coordinates in y-space is different from `y`'s center 
                coordinates in x-space (which is correct).
        fill (number, optional): The number to fill for paddings. Defaults to 0.
    """
    size_x = x.shape
    size_y = bt.Size(y)

    if isinstance(size_y, bt.Size) and size_x.n_space_dim == size_y.n_space_dim:
        size_y = size_y.space
    exp_size_x, exp_size_y = size_x ^ bt.Size(-1 if s == 1 else s for s in size_y).special_from(size_y)
    avouch(len(exp_size_x) == len(size_x), f"Cannot match crop size {size_y} with tensor size {size_x} in 'crop_as'. ")
    exp_size_y = bt.Size(-s if s in (1, -1) else s for s in exp_size_y).special_from(exp_size_y)
    size_x, size_y = exp_size_x, exp_size_y

    center = bt.Size(center)
    if len(center) == len(size_x): center = center.special_from(size_x)
    elif len(center) == size_x.n_space_dim: center = (center ^ size_x)[0]
    elif len(x for x in center if x >= 0) == len(x for x in size_y if x >= 0):
        center = bt.Size(a if b >= 0 else -1 for a, b in zip(center, size_y)).special_from(size_x)
    avouch(len(center) == len(size_x), TypeError("Mismatch dimensions for the center in 'crop_as', please use -1 if the dimension that is centered or doesn't need cropping. "))
    center = bt.Size(-1 if sx == sy or sy < 0 else (sx / 2 if c < 0 else c) for c, sx, sy in zip(center, size_x, size_y)).special_from(center)
    size_y = bt.Size(sx if sy < 0 else sy for sx, sy in zip(size_x, size_y)).special_from(size_y)

    z = fill * bt.ones(*size_y).to(x.device).type_as(x)
    """
    We then calculate the intersected boxes in source 'x' and target 'z', 
    to manage the occasions when the two regions are not containing each other (like: ⧉). 
    """
    def intersect(u, v): return max(u[0], v[0]), min(u[1], v[1])
    z_box = [(0, lx) if m < 0 or ly < 0 else intersect((0, ly), (- round(float(m - float(ly) / 2)), - round(float(m - float(ly) / 2)) + lx)) for m, lx, ly in zip(center, size_x, size_y)]
    x_box = [(0, lx) if m < 0 or ly < 0 else intersect((0, lx), (+ round(float(m - float(ly) / 2)), + round(float(m - float(ly) / 2)) + ly)) for m, lx, ly in zip(center, size_x, size_y)]
    # if the two boxes are seperated
    if any([r[0] >= r[1] for r in z_box]) or any([r[0] >= r[1] for r in x_box]): z.roi = tuple(); return z
    region_z = tuple(slice(u, v) for u, v in z_box)
    region_x = tuple(slice(u, v) for u, v in x_box)
    z[region_z] = x[region_x]
    z.roi = region_x
    z.special_from(size_x)
    return z

@overload
def crop_as(x: array, y: array, center: tuple, fill: number=0) -> array:
    return crop_as(x, y.shape, center, fill)

@overload
def crop_as(x: array, y: union(tuple, array), fill: number=0) -> array:
    center = tuple(-1 for m in x.shape)
    return crop_as(x, y, center, fill)

@overload
def crop_as(x: array, *y: int) -> array:
    center = tuple(-1 for m in x.shape)
    return crop_as(x, y, center)

@batorch_wrapper
def pad(x: bt.Tensor, p = 1, fill = 0):
    """
    Pad the tensor by `p` pixels with values `fill`. 
    p can be a tuple of len `n_dim` to pad differently for each dimension. 
    """
    p = to_tuple(p)
    if len(p) == 1: p *= x.n_space_dim
    return crop_as(x, tuple(s + 2 * q for s, q in zip(x.space, p)), fill = fill)

def add_special(size, special, fill=1):
    s = special
    if len(s) == 0: pass
    elif len(s) == 1: size = size[:s[0]] + (fill,) + size[s[0]:]
    else: size = size[:s[0]] + (fill,) + size[s[0]:s[1]-1] + (fill,) + size[s[1]-1:]
    return size

@batorch_wrapper
def up_scale(image: bt.Tensor, *scaling:int):
    """
    up_scale(tensor, s_1, ..., s_{n_dim}): equivalent to tensor.amplify(s_i, i)
    """
    if len(scaling) == 0:
        scaling = (1,)
    elif len(scaling) == 1 and iterable(scaling[0]):
        scaling = scaling[0]
    if len(scaling) == 1:
        if isinstance(scaling[0], int):
            scaling *= image.n_space_dim
            scaling = add_special(scaling, image.special_dims, 1)
        else: raise TypeError("Unknown scaling type for 'up_scale'. ")
    elif len(scaling) < image.n_dim and len(scaling) == image.n_space_dim:
        scaling = add_special(scaling, image.special_dims, 1)
    for i, s in enumerate(scaling):
        image = (
            image
            .transpose(i, -1)
            .unsqueeze(-1)
            .repeat((1,) * image.ndim + (int(s),))
            .flatten(-2)
            .transpose(i, -1)
        )
    return image

@batorch_wrapper
def down_scale(image: bt.Tensor, *scaling:int):
    """
    down_scale(tensor, s_1, ..., s_{n_dim}): equivalent to tensor[::s_i, ..., ::s_{n_dim}]
    """
    if len(scaling) == 0:
        scaling = (1,)
    elif len(scaling) == 1 and iterable(scaling[0]):
        scaling = scaling[0]
    if len(scaling) == 1:
        if isinstance(scaling[0], int):
            scaling *= image.n_space_dim
            scaling = add_special(scaling, image.special_dims, 1)
        else: raise TypeError("Unknown scaling type for 'down_scale'. ")
    elif len(scaling) < image.n_dim and len(scaling) == image.n_space_dim:
        scaling = add_special(scaling, image.special_dims, 1)
    return image[tuple(slice(None, None, s) for s in scaling)]

@alias("imagegrid")
def image_grid(*shape, device=bt.default_device()):
    """
    Create the indices for a space represented by tuple, or the space-dimension space of a tensor. 
    Note that the indices are distributed as indexing = 'ij' in the representation of keyword argument for torch.meshgrid.
    
    Outputs:
        (1) For input of space shape: (n_1, n_2, ..., n_{n_dim}), 
            output a Tensor in shape ([n_dim], n_1, n_2, ..., n_{n_dim}).
        (2) For input of tensor shape: ({n_batch}, n_1, ..., n_{n_dim}, 'n_sequence'), 
            output a Tensor in shape ([n_dim], n_1, ..., n_{n_dim}). 
        (3) ** WARNING: Please be careful with this usage as the feature dimension of the output is irrelevant to the input. **
            For input of tensor shape with feature: ({n_batch}, [n_feature], n_1, ..., n_{n_dim}, 'n_sequence'), 
            output a Tensor in shape ([n_dim], n_1, ..., n_{n_dim}). 
    """
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
    if len(shape) == 1 and hasattr(shape[0], 'space'): shape = shape[0].space
    if len(shape) == 1 and hasattr(shape[0], 'shape'): shape = shape[0].shape
    shape = tuple(bt.Size(*shape).space)
    kwargs = {'indexing': 'ij'} if bt.__torchversion__ >= Version('1.10') else {}
    if any(x < 0 for x in shape): raise TypeError(f"Negative size in shape {shape} for image grid.")
    if len(shape) == 0: return bt.zeros([0])
    return bt.stack(bt.meshgrid(*[bt.arange(x, device=device) for x in shape], **kwargs), [])

@alias("mask_center")
@batorch_wrapper
def center_gravity(mask: bt.Tensor):
    """
    Compute the center of gravity for a mask.
    """
    # avouch(issubclass(mask.dtype, dtype(bool)), TypeError("Only bool mask is allowed in function 'bt.mask_center'. "))
    coords = bt.image_grid(mask).change_special_dim([], bt.Size(0).with_func_dim(True))
    return (coords * mask).float().sum(...) / mask.sum(...)

# def linear(input, weight, bias):
#     result = input @ weight.T
#     if bias is not None:
#         if bias.dim() == 2:
#             return result + bias
#         return result + bias.unsqueeze(0)
#     return result

@batorch_wrapper
def one_hot(k: bt.Tensor, num_classes=None):
    """
    one_hot(k, n) -> bt.Tensor
    Create a one-hot vector as a `bt.Tensor` with length `n` with all elements 0 except the `k`-th element being 1. 
    e.g. k == -1 gives tensor([0, 0, ..., 0, 1])
    """
    avouch(not k.is_floating_point(), "'one_hot' requires the index argument of integers. ")
    if num_classes is None: num_classes = k.max().item() + 1
    l = bt.zeros(k.shape + bt.Size([num_classes])).int()
    l[*image_grid(*k.shape), k] = 1
    return l

@batorch_wrapper
def permute_space(data: bt.Tensor, *dims):
    """
    permute the space section in data. len(dims) should be data.n_space_dim. 
    """
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
    avouch(len(dims) == data.n_space_dim, TypeError(f"'permute_space' only takes 'dims' of length data.n_space_dim={data.n_space_dim}, currently {dims}."))
    dimensions = list(range(data.space_start)) + [d + data.space_start for d in dims] + list(range(data.space_stop, data.n_dim))
    return data.permute(*dimensions)

@batorch_wrapper
def approx_quantile(x, q, dim=...):
    dim = bt.exist_dim(x, dim)
    q = bt.as_tensor(q)
    valid_dims = bt.tensor([x.shape[d] for d in dim]).cumprod(0) >> 24 < 1
    with x.hide_special():
        n_mean_dim = 0
        x = x.move_dim(dim, 0)
        remain_shape = x.shape[len(dim):]
        if all(valid_dims): x = x.flatten(0, len(dim) - 1)
        elif all(~valid_dims):
            n_ele = x.size(0)
            n_replicate = (n_ele >> 24) + 1
            n_sample = n_ele // n_replicate
            x = x[:n_replicate * n_sample].reshape(n_sample, n_replicate, *tuple(x.shape)[1:])
            n_mean_dim = len(dim)
        else:
            n_valid_dim = valid_dims.argmin().item()
            x = x.flatten(0, n_valid_dim-1)
            n_mean_dim = len(dim) - n_valid_dim
    x = x.float().quantile(q, dim=0)
    if n_mean_dim > 0: x = x.mean(*range(q.n_dim, q.n_dim + n_mean_dim))
    return x.view(*q.shape, *remain_shape)

@batorch_wrapper
def normalize01(data: bt.Tensor, func='linear'):
    """
    Normalize the data to range [0, 1]. Outliers are automatically excluded. 
        The result is continuous, even at the two ends.

    Args:
        data (bt.Tensor): The input data of size ({n_batch}, [n_feature], n_1, ..., n_r) [r=n_dim]
        func (str): The basic mapping function. Defaults to 'linear'.
            'linear': x => (x - x_min) / (x_max - x_min)
            'sigmoid': x => 1 / (1 + 19 ^ (2 * x - x_max - x_min) / (x_max - x_min))
    """
    if data.has_feature:
        norm = (data ** 2).sum([], keepdim=True).clamp(bt.eps).sqrt()
        m = norm.quantile(0.05, ..., keepdim=True)
        M = norm.quantile(0.95, ..., keepdim=True)
        new_norm = normalize01(norm.squeeze([]), func=func).view_as(norm)
        non_zero_norm = bt.where(norm == 0, bt.ones_like(norm), norm)
        return data * bt.where(new_norm == 0, bt.ones_like(new_norm), new_norm / non_zero_norm)
    else:
        m = data.quantile(0.05, ..., keepdim=True)
        M = data.quantile(0.95, ..., keepdim=True)
        clamped = bt.min(bt.max(data, m), M)
        if M.max().item() < 1.1 and m.min().item() > -0.1: return clamped.clamp(0, 1)
        if func.lower() == 'linear':
            return (clamped - m) / (M - m).clamp(bt.eps)
        elif func.lower() == 'sigmoid':
            return 1 / (bt.exp(- math.log(19) * (2 * clamped - M - m) / (M - m).clamp(bt.eps)) + 1)
        else: raise TypeError("Only 'linear' and 'sigmoid' functions are available for argument 'func' in 'normalize01'. ")

class input_shape:
    
    def __init__(self, **kwargs):
        self.vars = get_environ_vars()
        self.default_values = kwargs
        self.exp_val_pair = []
        self.opt_exp_val_pair = []
        self.index = 0
        
    def get_variables(self, expr):
        variables = []
        while True:
            try: eval(expr); break
            except NameError as e:
                name = e.__str__().split("'")[1]
                variables.append(name)
                exec(f"{name} = 2")
        return variables
    
    def declose(self, expr):
        for closure in ('{}', '[]', '""', "''", '()'):
            if expr.startswith(closure[0]) and expr.endswith(closure[1]):
                return expr.strip(closure)
        return expr
    
    def split(self, shape_expr, sep=','):
        return [x.strip() for x in tokenize(shape_expr, sep=sep) if x.strip()]
    
    def record_expr_value(self, expr, value):
        if isinstance(expr, str): expr = [expr]
        if isinstance(value, int): value = [value]
        for e, v in zip(expr, value):
            if v > 1: self.exp_val_pair.append((e, v))
            else: self.opt_exp_val_pair.append((e, v))
    
    def substitute(self, expr, values):
        for k, v in values.items():
            while True:
                res = re.search(rf'([^A-Za-z0-9]|^)({k})([^A-Za-z0-9]|$)', expr)
                if res is None: break
                i, j = res.span(2)
                expr = expr[:i] + str(v) + expr[j:]
        return expr
    
    def set(self, **kwargs):
        """
        set shape information of tensors for a function.
        
        Args:
            kwargs: data_name = shape pairs. 
            data_name (str): the name of local variable. 
            shape (str): a string representing the shape of data. The expression is in the shape format. 
                The unknown number of dimensions can be specified with '...' and variable can be specified with words. 
        
        Examples::
            >>> data = bt.zeros({4}, [3, 3], 24, 25, 36)
            >>> data2 = bt.zeros({4}, [3])
            >>> input_shape(n_dim=2).set(
            ...     data = "({n_batch}, [n_dim+1, n_dim+1], n_1, ..., n_r) [r=n_dim]", 
            ...     data2 = "({n_batch}, [(n_dim+1) * n_dim / 2])"
            ... )
            ...
            >>> print(n_batch, n_dim, r)
            4 2 3
        """
        self.exp_val_pair = []
        self.opt_exp_val_pair = []
        variable_names = []
        const_eqs = []
        n_omits = {}
        for data_name, shape_expr in kwargs.items():
            data = self.vars[data_name]
            
            # get constraint eqs
            shape_expr, *const_eq_strs = self.split(shape_expr, sep='')
            is_optional = False
            for ceq in const_eq_strs:
                ceq.strip('[]')
                for x in ceq.split(';'):
                    x = x.strip().strip('[]')
                    if x.lower() == 'optional': is_optional = True; continue
                    left, right = x.split('=')
                    variable_names.extend(self.get_variables(left))
                    variable_names.extend(self.get_variables(right))
                    const_eqs.append(left + '-' + right)
            avouch(isinstance(data, bt.Tensor) or data is None and is_optional,
                   TypeError(f"'{data_name}' needs to be a batorch tensor of shape {shape_expr}, instead of type {data.__class__.__name__}"))
            if data is None: continue
                    
            # get variable names
            variable_names.extend(self.get_variables(shape_expr))
            
            # seperate shape terms
            shape_terms = self.split(shape_expr.strip('()'))
            element_terms = []
            ref_shape = bt.Size()
            opt_shape = bt.Size()
            for term in shape_terms:
                eles = self.split(self.declose(term))
                optional_terms = [x for x in eles if x.endswith(':optional') or x=='n_batch']
                if len(optional_terms) > 0: opt_shape += bt.Size(eval(term[0] + '-1' + term[-1]))
                eles = [x[:-9] if x.endswith(':optional') else x for x in eles]
                element_terms.extend(eles)
                if self.declose(term) == term:
                    if term == '...':
                        n_ele = data.n_space_dim - len([t for t in shape_terms if self.declose(t) == t]) + 1
                    else: n_ele = 1
                    ref_shape += bt.Size(-1) * n_ele
                else:
                    n_ele = len(eles)
                    if '...' in eles: n_ele = getattr(data, 'n_%s_dim'%{'{}':'batch','[]':'feature','""':'sequence',"''":'sequence','()':'func'}[term[0] + term[-1]])
                    ref_shape += bt.Size(eval(term[0] + ', '.join(['-1'] * n_ele) + term[-1]))
            data_shape = data.shape
            if data_shape.has_feature: ref_shape = ref_shape.with_feature(data.feature)
            elif opt_shape.has_feature: ref_shape = ref_shape.with_feature(tuple())
            if data_shape.has_sequence: ref_shape = ref_shape.with_sequence(data.sequence)
            elif opt_shape.has_sequence: ref_shape = ref_shape.with_sequence(tuple())
            if data_shape.has_space: ref_shape = ref_shape.with_space(data.space)
            elif opt_shape.has_space: ref_shape = ref_shape.with_space(tuple())
            data_shape = data.unsqueeze_to(ref_shape).shape
            error = TypeError(f"'{data_name}' with shape {data_shape} does not follow shape {shape_expr}")
            if '...' in element_terms:
                i = element_terms.index('...')
                avouch(0 < i < len(element_terms) - 1, TypeError(f"Ellipsis should be between two elements, not ({shape_expr})"))
                prev = element_terms[i - 1]
                post = element_terms[i + 1]
                _, prev_index = prev.split('_', 1)
                _, post_index = post.split('_', 1)
                prev_index = self.declose(prev_index)
                post_index = self.declose(post_index)
                variable_names.extend(self.get_variables(prev_index))
                variable_names.extend(self.get_variables(post_index))
                
                self.record_expr_value(element_terms[:i], data_shape)
                n_omit = len(data_shape) - len(element_terms) + 3
                avouch(n_omit > 1, error)
                self.record_expr_value(post_index + '-' + prev_index + '+1', n_omit)
                self.record_expr_value(element_terms[i + 1:], data_shape[i + n_omit - 2:])
                n_omits[data_name] = n_omit
            else:
                avouch(len(element_terms) == len(data_shape), error)
                self.record_expr_value(element_terms, data_shape)
                
        if len(variable_names) == 0:
            if any(eval(expr) != value for expr, value in self.exp_val_pair): raise error
            
        else:
        
            # solve the collected equations
            variable_names = list(set(variable_names))
            variables = symbols(' '.join(variable_names))
            exec(', '.join(variable_names) + ' = variables')
            equations = []
            for expr, value in self.exp_val_pair:
                equations.append(Eq(eval(expr), value))
            for ceq in const_eqs:
                equations.append(Eq(eval(ceq), 0))
            variable_values = solve(equations, variables, dict=True)
            exec('del ' + ', '.join(variable_names))
            variable_values = [{k.name: int(v) for k, v in vs.items()} for vs in variable_values if all(x.is_Integer and x > 0 for x in vs.values())]
            
            if len(variable_values) > 1:
                names = [n for n in variable_values[0] if len(set([vs[n] for vs in variable_values])) > 1]
                raise TypeError(f"{names[0] if len(names) == 1 else names} has multiple possible values {list(set([vs[n] for vs in variable_values]))} for {shape_expr} = {data_shape}")
            elif len(variable_values) == 0: raise error
            variable_values = variable_values[0]
            
            # try to solve the optional equations (due to 1 values)
            variable_names = [n for n in variable_names if n not in variable_values]
            if len(variable_names) > 0:
                new_exp_val_pair = []
                for expr, value in self.opt_exp_val_pair:
                    expr = self.substitute(expr, variable_values)
                    if self.get_variables(expr):
                        new_exp_val_pair.append((expr, value))
                locals().update(variable_values)
                variable_names = [self.substitute(n, variable_values) for n in variable_names]
                variables = symbols(' '.join(variable_names))
                exec(', '.join(variable_names) + ' = variables')
                equations = []
                for expr, value in new_exp_val_pair:
                    equations.append(Eq(eval(expr), value))
                new_variable_values = solve(equations, variables, dict=True)
                exec('del ' + ', '.join(variable_names))
                if len(new_variable_values) == 1: variable_values.update({k.name: int(v) for k, v in new_variable_values[0].items() if v.is_Integer and v > 0})
            
            for k, v in self.default_values:
                if k not in variable_values: variable_values[k] = v
            for k, v in variable_values.items(): self.vars.globals[k] = v
            locals().update(variable_values)
        
        for data_name, shape_expr in kwargs.items():
            shape_expr, *_ = self.split(shape_expr, sep='')
            shape_expr = shape_expr.replace(':optional', '')
            data = self.vars[data_name]
            if data is None: continue
            if '...' in shape_expr:
                terms = [x.strip() for x in shape_expr.strip('()').split(',')]
                i = terms.index('...')
                if i > 1: left_size = bt.Size(*eval(','.join(terms[:i-1])))
                else: left_size = bt.Size()
                if i < len(terms) - 2: right_size = bt.Size(*eval(','.join(terms[i+2:])))
                else: right_size = bt.Size()
                size = left_size + (-1,) * n_omits[data_name] + right_size
            else: size = bt.Size(*eval(shape_expr))
            self.vars[data_name] = data.expand_to(size)
            
    def usage(self, index=None, /, **kwargs):
        self.index += 1
        if index is None: index = self.index
        try:
            self.set(**kwargs)
            self.vars.globals['usage'] = index
        except TypeError: ...
        return self
    
def as_shape(self, shape_expr: str):
    env_vars = get_environ_vars()
    loc_vars = env_vars.locals
    shape_list = [x.strip() for x in shape_expr.strip("()").split(',')]
    ibatch, ichannel = None, None
    old_shape = self.shape
    dim_pos = [0] * len(shape_list)
    pointer = 0
    for i, x in enumerate(shape_list):
        if x == '...': break
        if x.startswith('[') and x.endswith(']'): dim_pos[i] = old_shape.batch_dimension; x = x.strip('[]')
        if x.startswith('{') and x.endswith('}'): dim_pos[i] = old_shape.channel_dimension; x = x.strip('{}')
        if x.endswith(":optional"): x = x[:-len(":optional")]
        x = get_snakename(x)
        if x in loc_vars: x = loc_vars[x]

@alias("upper_triangular_matrix")
@batorch_wrapper
def upper_matrix(elements: bt.Tensor):
    '''
    elements: ({n_batch}, [n_dim * (n_dim + 1) / 2])
    output: ({n_batch}, [n_dim, n_dim])
    Compute the matrix with upper triangular elements of `elements`:
        For nxn matrix space (n_dim = n), the input is 𝜔:({n_batch}, [m = n(n+1)/2])
            The output is in shape ({n_batch}, [n, n]) of matrices:
                [[𝜔ₘ 𝜔ₘ₋₁ ... 𝜔ₘ₋ₙ₊₂  𝜔ₘ₋ₙ₊₁ ]
                 [0  𝜔ₘ₋ₙ ... 𝜔ₘ₋₂ₙ₊₄ 𝜔ₘ₋₂ₙ₊₃]
                 [︙  ︙   ⋱    ︙       ︙   ]
                 [0   0   ...   𝜔₃      𝜔₂   ]
                 [0   0   ...   0       𝜔₁   ]]
        ...
    '''
    avouch(elements.n_feature_dim == 1, TypeError("elements for 'upper_matrix' should be a tensor of shape ({n_batch}, [n_dim * (n_dim + 1) / 2])"))
    if not elements.has_batch: elements = elements.unsqueeze({})
    n_batch = elements.n_batch
    n_dim = int(math.sqrt(2 * elements.n_channel))
    elements = elements.flip([])
    output = bt.zeros({n_batch}, [n_dim, n_dim], dtype=elements.dtype, device=elements.device)
    row_len = bt.arange(n_dim, 0, -1)
    for i in range(n_dim): output[:, i, i:] = elements[:, row_len[:i].sum():row_len[:(i+1)].sum()]
    return output

@alias("lower_triangular_matrix")
@batorch_wrapper
def lower_matrix(elements: bt.Tensor):
    '''
    elements: ({n_batch}, [n_dim * (n_dim + 1) / 2])
    output: ({n_batch}, [n_dim, n_dim])
    Compute the matrix with lower triangular elements of `elements`:
        For nxn matrix space (n_dim = n), the input is 𝜔:({n_batch}, [m = n(n+1)/2])
            The output is in shape ({n_batch}, [n, n]) of matrices:
                [[ 𝜔ₘ       0     ...    0   0 ]
                 [𝜔ₘ₋₁    𝜔ₘ₋ₙ    ...    0   0 ]
                 [ ︙       ︙      ⋱   ︙   ︙ ]
                 [𝜔ₘ₋ₙ₊₂  𝜔ₘ₋₂ₙ₊₄  ...   𝜔₃  0  ]
                 [𝜔ₘ₋ₙ₊₁  𝜔ₘ₋₂ₙ₊₃  ...   𝜔₂  𝜔₁ ]]
        ...
    '''
    avouch(elements.n_feature_dim == 1, TypeError("elements for 'upper_matrix' should be a tensor of shape ({n_batch}, [n_dim * (n_dim + 1) / 2])"))
    if not elements.has_batch: elements = elements.unsqueeze({})
    n_batch = elements.n_batch
    n_dim = int(math.sqrt(2 * elements.n_channel))
    elements = elements.flip([])
    output = bt.zeros({n_batch}, [n_dim, n_dim], dtype=elements.dtype, device=elements.device)
    row_len = bt.arange(n_dim, 0, -1)
    for i in range(n_dim): output[:, i, i:] = elements[:, row_len[:i].sum():row_len[:(i+1)].sum()]
    return output.T

@alias("skew_symmetric")
@batorch_wrapper
def cross_matrix(axis: bt.Tensor):
    '''
    axis: ({n_batch}, [n_dim * (n_dim - 1) / 2])
    output: ({n_batch}, [n_dim, n_dim])
    Compute the skew symmetric matrix:
        For 2x2 matrix space (n_dim = 2), the input is 𝜗:({n_batch}, [1])
            The output is in shape ({n_batch}, [2, 2]) of matrices:
                [[0  𝜗]
                 [-𝜗 0]]
        For 3x3 matrix space (n_dim = 3), the input is 𝜔:({n_batch}, [3])
            The output is in shape ({n_batch}, [3, 3]) of matrices:
                [[ 0   𝜔₃ -𝜔₂]
                 [-𝜔₃  0   𝜔₁]
                 [ 𝜔₂ -𝜔₁  0 ]]
        For 4x4 matrix space (n_dim = 4), the input is 𝜔:({n_batch}, [6])
            The output is in shape ({n_batch}, [4, 4]) of matrices:
                [[ 0   𝜔₆ -𝜔₅  𝜔₄]
                 [-𝜔₆  0   𝜔₃ -𝜔₂]
                 [ 𝜔₅ -𝜔₃  0   𝜔₁]
                 [-𝜔₄  𝜔₂ -𝜔₁  0 ]]
        ...
    '''
    avouch(axis.n_feature_dim == 1, TypeError("axis for 'cross_matrix' should be a tensor of shape ({n_batch}, [n_dim * (n_dim - 1) / 2])"))
    if not axis.has_batch: axis = axis.unsqueeze({})
    n_batch = axis.n_batch
    n_dim = int(math.sqrt(2 * axis.n_channel)) + 1
    axis = axis.flip([])
    upper = bt.zeros({n_batch}, [n_dim, n_dim], dtype=axis.dtype, device=axis.device)
    row_len = bt.arange(n_dim - 1, 0, -1)
    row_sign = bt.where(row_len % 2 == 0, bt.ones_like(row_len), -bt.ones_like(row_len)).as_feature_tensor()
    for i in range(n_dim):
        upper[:, i, i+1:] = axis[:, row_len[:i].sum():row_len[:i+1].sum()] * row_sign[:(n_dim - i - 1)]
    return upper - upper.T

@alias("skew_symmetric_params")
@batorch_wrapper
def uncross_matrix(cross_matrix: bt.Tensor):
    '''
    cross_matrix: ({n_batch}, [n_dim, n_dim])
    output: ({n_batch}, [n_dim * (n_dim - 1) / 2])
    The inverse operation of cross_matrix. 
    '''
    avouch(cross_matrix.n_feature_dim == 2 or not cross_matrix.has_feature and cross_matrix.n_space_dim == 2, 
           TypeError("cross_matrix for 'cross_matrix' should be a tensor of shape ({n_batch}, [n_dim, n_dim])"))
    if not cross_matrix.has_batch: cross_matrix = cross_matrix.unsqueeze({})
    if not cross_matrix.has_feature:
        cross_matrix.sz_feature_dim_(cross_matrix.n_space_dim if cross_matrix.sz_sequence_dim < 0 else -cross_matrix.n_space_dim)
    avouch(cross_matrix.feature[0] == cross_matrix.feature[1], 
           TypeError("cross_matrix for 'cross_matrix' should be a tensor of shape ({n_batch}, [n_dim, n_dim])"))
    n_batch = cross_matrix.n_batch
    n_dim = cross_matrix.feature[0]
    axis = bt.zeros({n_batch}, [n_dim * (n_dim - 1) // 2], dtype=cross_matrix.dtype, device=cross_matrix.device)
    row_len = bt.arange(n_dim - 1, 0, -1)
    row_sign = bt.where(row_len % 2 == 0, bt.ones_like(row_len), -bt.ones_like(row_len)).as_feature_tensor()
    for i in range(n_dim):
        axis[:, row_len[:i].sum():row_len[:i+1].sum()] = cross_matrix[:, i, i+1:] * row_sign[:(n_dim - i - 1)]
    axis = axis.flip([])
    return axis

class summary:
    def __init__(self, *Xs, show_thumb = False, thumb_size = None, plot = False):
        self.smys = []
        self.show_as_plot = plot
        if plot: self.canvases = []
        if not hasattr(self, 'var_names'):
            args_exp = get_args_expression("summary")
            tokens = tokenize(args_exp, sep=', ')
            self.var_names = [vn.strip() for vn in tokens if len(tokenize(vn, sep='=')) == 1]
        avouch(len(self.var_names) <= len(Xs), f"Too many variable names recognized: {self.var_names}")
        starts_with_star = [i for i, x in enumerate(self.var_names) if x.startswith('*')]
        n_starts_with_star = 0
        if len(starts_with_star) > 1:
            avouch(None, f"Too much expandable variable in {self.var_names}: more than one. ")
        elif len(starts_with_star) == 0:
            avouch(len(self.var_names) == len(Xs), f"No expandable variable in {self.var_names} but detected more input arguments. ")
        else:
            starts_with_star = starts_with_star[0]
            n_starts_with_star = len(Xs) - len(self.var_names) + 1
        
        if thumb_size is None: thumb_size = 4
        else: show_thumb = True
        self.show_thumb = show_thumb
        self.thumb_size = thumb_size
        
        for i, x in enumerate(Xs):
            if n_starts_with_star > 0:
                if 0 <= i - starts_with_star < n_starts_with_star: j = i - starts_with_star; i = starts_with_star
                elif i >= starts_with_star + n_starts_with_star: i = i - n_starts_with_star
            var_name = self.var_names[i] + (f'[{j}]' if i == starts_with_star else '')
            class_name = x.__class__.__name__
            g = None
            if not isinstance(x, bt.torch.Tensor):
                l = touch(lambda: len(x), default=0)
                if l <= 10:
                    self.smys.append(f">>{var_name}<< " + str(x) + '\n')
                    continue
                else: x = bt.tensor(x)
            elif not isinstance(x, bt.Tensor):
                if x.is_leaf and x.grad is not None: g = bt.tensor(x.grad.clone().detach())
                x = bt.tensor(x)
            special_dim_str = f"[{x.n_special_dim}]+" if x.n_special_dim > 0 else ''
            self.out = SPrint(f">>Summary of {var_name}<<\n")
            def size_str(Size=None, Dimension=None, Class=None):
                return f"{f'[{Size.n_special_dim}]+' if Size.n_special_dim > 0 else ''}{Size.n_space_dim}D {Class.__name__} of size {self.__raw_str__(Size)}"
            self.__add_prop__('Size', value2str_func = size_str, Size=x.shape, Dimension=x.n_dim, Class=x.__class__)
            self.__add_prop__('Device', x.device, lambda d: 'cpu' if d.type == 'cpu' else f"{d.type}:{d.index}")
            self.__add_prop__('Dtype', x.dtype, self.__raw_str__)
            self.__add_prop__('Requires_Gradient', x.requires_grad)
            if x.requires_grad: self.__add_prop__('Gradient_Func', x.grad_fn)

            self.show_grad = True
            self.__display_values__(x)
            if self.show_grad and g is not None:
                self.__add_prop__('Gradient_Size', g.shape, self.__raw_str__)
                self.__display_values__(x, is_grad=True)
            self.smys.append(self.out.text)
            if plot:
                while x.n_dim > 2 and not (x.n_dim == 3 and x.size(-1) == 3):
                    if x.has_special:
                        d_sample = x.i_special_dims[0]
                    else: d_sample = argmin(list(x.shape))[0]
                    x = x.pick(x.size(d_sample) // 2, d_sample)
                self.canvases.append(x.detach())
                
    def __add_prop__(self, name, value=null, value2str_func=None, prompt=None, display_only=False, **assign_values):
        if prompt is not None: prompt = f" ({prompt})"
        else: prompt = ''
        if value is not null:
            if value2str_func is not None: value_str = value2str_func(value)
            else: value_str = str(value)
            if not display_only: setattr(self, name, value)
        else:
            avouch(value2str_func is not None, TypeError("'value2str_func' should be specified when multiple values are given for 'summary.__add_prop__'. "))
            value_str = value2str_func(**assign_values)
            if not display_only:
                for n, v in assign_values.items(): setattr(self, n, v)
        tag = f"【{name}{prompt}】"
        n_align = math.ceil((len(tag) + 2) / 25)
        self.out(tag + ' ' * (n_align * 25 - len(tag) - 2) + value_str)
             
    def __raw_str__(self, x):
        sx = str(x)
        if 'Size(' in sx: return sx.split('Size')[-1]
        return tokenize(tokenize(sx.split('(', 1)[-1], sep=[',', ')'])[0], sep='.')[-1]

    def __approx_quantile__(self, x):
        valid_dims = bt.tensor(x.shape).cumprod(0) >> 24 < 1
        if all(valid_dims): x = x.flatten()
        elif all(~valid_dims):
            n_replicate = (x.n_ele >> 24) + 1
            n_sample = x.n_ele // n_replicate
            x = x[:n_replicate * n_sample].reshape(n_sample, n_replicate)
        else: x = x.flatten(0, valid_dims.argmin().item()-1)
        x = x.float().quantile(bt.arange(6).float() / 5, dim=0)
        if x.n_dim > 1: x = x.mean(*range(1, x.n_dim))
        return x
    
    def __display_values__(self, x, is_grad=False):
        nan_mask = bt.isnan(x)
        num_nans = nan_mask.sum()
        inf_mask = bt.isinf(x)
        num_infs = inf_mask.sum()
        nan_inf_mask = nan_mask | inf_mask
        valid_nums = x[~nan_inf_mask]
        if num_nans + num_infs == 0: show_error = False; show_valid = True
        elif (num_nans + num_infs) / x.n_ele < 0.5: show_error = True; show_valid = True
        else: show_error = True; show_valid = False; self.show_grad = False
        if show_valid:
            values_str = lambda vs: f"(min:{vs[0]}, med:{vs[1]}, max:{vs[2]})"
            if x.is_complex():
                vre_name, vim_name, name = 'Values_Re', 'Values_Im', 'Values_Re/Im'
                if is_grad: vre_name, vim_name, name = ['Gradient_' + x for x in (vre_name, vim_name, name)]
                vre_val = (valid_nums.real.min(), valid_nums.real.median(), valid_nums.real.max())
                vim_val = (valid_nums.imag.min(), valid_nums.imag.median(), valid_nums.imag.max())
                self.__add_prop__(name, value2str_func=lambda **k: f"Re{values_str(k[vre_name])}, Im{values_str(k[vim_name])}", **{vre_name: vre_val, vim_name: vim_val})
            else:
                name = 'Gradient_Values' if is_grad else 'Values'
                val = (valid_nums.min(), valid_nums.median(), valid_nums.max())
                self.__add_prop__(name, val, values_str)
        if show_error:
            if num_nans > 0:
                nan_name = f"{num_nans} NaN"
                if num_nans > 1: nan_name += 's'
                has_nan_maps = nan_mask.sum(...) > 0
                num_nan_maps = has_nan_maps.sum()
                nan_mask = nan_mask[has_nan_maps, ...][0]
                nan_center = bt.mask_center(nan_mask)
                if is_grad:
                    self.Gradient_NaN = self.Gradient_NaNs = num_nans
                else:
                    self.NaN = self.NaNs = num_nans
                    self.NaN_maps = num_nan_maps
                    self.NaN_center = tuple(nan_center)
                self.__add_prop__(nan_name, value2str_func=lambda **_: f"{num_nan_maps}/{has_nan_maps.n_ele} maps with nan, e.g. center: {self.__raw_str__(nan_center)}", display_only=True)
            if num_infs > 0:
                inf_name = f"{num_infs} Inf"
                if num_infs > 1: inf_name += 's'
                has_inf_maps = inf_mask.sum(...) > 0
                num_inf_maps = has_inf_maps.sum()
                inf_mask = inf_mask[has_inf_maps, ...][0]
                inf_center = bt.mask_center(inf_mask)
                if is_grad:
                    self.Gradient_Inf = self.Gradient_Infs = num_infs
                else:
                    self.Inf = self.Infs = num_infs
                    self.Inf_maps = num_inf_maps
                    self.Inf_center = tuple(inf_center)
                self.__add_prop__(inf_name, value2str_func=lambda **_: f"{num_inf_maps}/{has_inf_maps.n_ele} maps with inf, e.g. center: {self.__raw_str__(inf_center)}", display_only=True)
        if show_valid:
            approx_q = x.n_ele >> 24 > 0
            self.__add_prop__('Gradient_Quantiles' if is_grad else 'Quantiles', self.__approx_quantile__(valid_nums), 
                              lambda q: ('\n >> ' if approx_q else '') + f"([0%]{q[0]}, [20%]{q[1]}, [40%]{q[2]}, [60%]{q[3]}, [80%]{q[4]}, [100%]{q[5]})", 
                              prompt='approx. due to too many elements' if approx_q else None)
            values = valid_nums.unique()
            def range_str(values):
                if values.n_ele <= 10:
                    if values.n_ele > 6: return f"\n >> [{', '.join(str(x) for x in values)}]"
                    else:
                        num_pairs = [(v, (x == v).sum().item()) for v in values]
                        return f"\n >> [{', '.join(f'{v}({n})' for v, n in num_pairs)}]"
                else: return f"\n >> [{', '.join(str(x) for x in values[:5])}, ..., {', '.join(str(x) for x in values[-5:])}]"
            self.__add_prop__('Gradient_Range' if is_grad else 'Range', values, value2str_func=range_str, 
                              prompt=f"{values.n_ele} unique element" + ('s' if values.n_ele > 1 else ''))
        if self.show_thumb:
            gaps = bt.floor((bt.channel_tensor(list(x.space)) - 1) / (self.thumb_size - 1)).clamp(min=1).long()
            down_sample = x[(slice(None),) * x.space_start + (image_grid(*(self.thumb_size if s > self.thumb_size else max(s // g, 1) for s, g in zip(x.space, gaps))) * gaps).split(1, dim=[], squeezedim=True)]
            for i in down_sample.i_special_dims[::-1]: down_sample = down_sample.pick(0, dim=i)
            self.__add_prop__('Gradient_Thumb' if is_grad else 'Thumb', down_sample, 
                              value2str_func=lambda x: '\n >> ' + bt.Tensor.__shift_repr__(x.__raw_repr__(), 4), 
                              prompt=f"downsampled to {self.thumb_size}{f' x {self.thumb_size}'*(x.n_space_dim-1)}")
    
    def show(self):
        if not self.show_as_plot:
            print(self)
            return
        plt.figure(figsize=(15, 7))
        
        n_plots = len(self.canvases)
        self.subplot_rows = math.floor(math.sqrt(n_plots))
        self.subplot_cols = math.ceil(n_plots / self.subplot_rows)

        old_minus = plt.rcParams['axes.unicode_minus']
        old_font = plt.rcParams['font.sans-serif']
        font = 'Heiti TC'
        if font not in plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'].insert(0, font)
        plt.rcParams['axes.unicode_minus'] = False
        for i, x in enumerate(self.canvases):
            ax = plt.subplot(self.subplot_rows, self.subplot_cols, i+1)
            old_size = plt.rcParams['font.size']
            plt.rcParams['font.size'] = 6
            if x.n_dim >= 2:
                img = plt.imshow(x, cmap=plt.cm.gray)
                plt.colorbar(mappable=img, ax=ax, orientation='vertical')
                plt.text(-0.5, -0.5, self.smys[i])
            elif x.n_dim == 1:
                plt.plot(bt.arange(len(x)), x)
                plt.text(0, plt.ylim()[1], self.smys[i])
            else: raise TypeError("Failed to display 0-dimensional data in plot.")
            plt.rcParams['font.size'] = old_size
        plt.rcParams['font.sans-serif'] = old_font
        plt.rcParams['axes.unicode_minus'] = old_minus 
        plt.show()

    @alias("__repr__")
    def __str__(self): return ''.join(self.smys)

class display(summary):
    def __init__(self, *args, **kwargs):
        kwargs['plot'] = True
        args_exp = get_args_expression("display")
        tokens = tokenize(args_exp, sep=', ')
        self.var_names = [vn.strip() for vn in tokens if len(tokenize(vn, sep='=')) == 1]
        return super().__init__(*args, **kwargs)
    
def byte_size(tensor):
    return ByteSize(tensor.numel() * dtype(tensor.dtype).bits // 8)

def get_memory_used(*tensors):
    return sum([(t.byte_size() if hasattr(t, 'byte_size') else byte_size(t)) if isinstance(t, torch.Tensor) else ByteSize(1) for t in tensors])
    
def num_model_params(model):
    s = ByteSize(0)
    for param in model.parameters():
        s += byte_size(param)
    return s
