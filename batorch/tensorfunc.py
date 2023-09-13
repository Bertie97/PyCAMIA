
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "Tensor functions for `batorch`.",
    requires = ["torch", "pycamia", "pyoverload", "matplotlib"]
)

__all__ = """
    conv
    crop_as
    decimal
    divide
    dot
    down_scale
    equals
    gaussian_kernel
    image_grid
    matpow
    matprod
    one_hot
    pad
    permute_space
    up_scale

    norm norm2
    Fnorm Fnorm2
    frobenius_norm
    meannorm meannorm2
    mean_norm mean_norm2

    Jacobian
    grad_image

    skew_symmetric
    cross_matrix
    uncross_matrix

    summary
    display
""".split()

import math, re

with __info__:
    import batorch as bt
    from pyoverload import *
    from pycamia import restore_type_wrapper, avouch, touch, alias
    from pycamia import get_environ_vars, get_snakenames, get_args_expression
    from pycamia import SPrint, Version, argmin, tokenize
    from matplotlib import pyplot as plt
    
def batorch_tensor(x):
    if not isinstance(x, bt.torch.Tensor): return bt.tensor(x)
    elif type(x) != bt.Tensor: return x.as_subclass(bt.Tensor)
    return x

def conv(tensor, kernel, padding='SAME'):
    """
    Convolution for a batorch tensor: only in space dimensions. 
    
    Arguments:
    ----------
        tensor [bt.Tensor]: With shape ([n_batch], {n_channel}, *n_space).
        kernel [bt.Tensor]: With shape ([n_channel_out], {n_channel}, *n_kernel_size).
        padding [str, int, tuple]: Convolution padding: string 'SAME' stands for not changing the size of 'tensor'. 
    
    Output:
    ----------
        output [bt.Tensor]: With shape ([n_batch], {n_channel_out}, *(n_space - n_kernel_size // 2)).
    """
    old_tensor_shape = tensor.shape
    tensor = tensor.clone()
    kernel = kernel.clone()
    if tensor.n_space > 4 and not tensor.has_batch and not tensor.has_channel:
        tensor = tensor.with_batchdim(0).with_channeldim(1)
    if tensor.n_space > 3:
        if not tensor.has_batch: tensor = tensor.with_batchdim(1 if tensor.channel_dim == 0 else 0)
        if not tensor.has_channel: tensor = tensor.with_channeldim(1 if tensor.batch_dim == 0 else 0)
    n_dim = tensor.n_space
    if not tensor.has_batch: tensor.unsqueeze_([])
    if not tensor.has_channel: tensor.unsqueeze_({})
    if kernel.n_dim < n_dim + 2 and not kernel.has_batch: kernel.unsqueeze_([])
    if kernel.n_dim < n_dim + 2 and not kernel.has_channel: kernel.unsqueeze_({})
    avouch(kernel.n_dim == n_dim + 2, f"Kernel for {n_dim}D convolution should be of size ([channel_out], {{channel_in}}, n_1, ..., n_{{n_dim}}) with a dimension of {n_dim}")
    if not kernel.has_batch: kernel.with_batchdim(0)
    if not kernel.has_channel: kernel.with_channeldim(1 if kernel.batch_dim == 0 else 0)
    avouch(tensor.n_channel == kernel.n_channel, f"Channels of convolution should match: tensor size {tensor.shape} and kernel size {kernel.shape}. ")

    is_same = False
    if isinstance(padding, str) and padding == 'SAME':
        is_same = True
        padding = tuple(x // 2 for x in kernel.space)
    if isinstance(padding, int):
        padding = (padding,) * n_dim
    avouch(isinstance(padding, tuple), f"Unrecognized argument 'padding' for convolution: {padding}. ")
    conv = eval("bt.nn.functional.conv%dd"%n_dim)
    result = conv(tensor, kernel, padding=padding).squeeze(1)
    if kernel.n_batch > 1:
        if not old_tensor_shape.has_channel:
            old_tensor_shape = bt.Size({1}) + old_tensor_shape
        old_tensor_shape = old_tensor_shape.with_dimsize(kernel.n_batch, {})
    if is_same: result = crop_as(result, tensor).view(*old_tensor_shape)
    else: result = result.view(old_tensor_shape.reset_space(tuple(t - k + 1 + 2 * p for t, k, p in zip(tensor.space, kernel.space, padding))))
    return result

@alias("meannorm2", "mean_norm2", root = False, mean = True)
@alias("meannorm", "mean_norm", mean = True)
@alias("Fnorm2", "norm2", root = False)
@alias("Fnorm", "frobenius_norm")
def norm(tensor, p = 2, root = True, mean = False):
    tensor = batorch_tensor(tensor)
    tensor = tensor.abs()
    if p == float('inf'):
        if tensor.has_channel: return tensor.max({}).values
        else: 
            x = tensor
            for i in range(tensor.ndim - 1, -1, -1):
                if i != x.batch_dimension: x = x.max(i).values
            return x
    if tensor.has_channel:
        if mean: return (tensor ** p).mean({}) ** (1 / p) if root else (tensor ** p).mean({})
        else: return (tensor ** p).sum({}) ** (1 / p) if root else (tensor ** p).sum({})
    else:
        if mean: return (tensor ** p).mean() ** (1 / p) if root else (tensor ** p).mean()
        else: return (tensor ** p).sum() ** (1 / p) if root else (tensor ** p).sum()

def decimal(tensor):
    return tensor - bt.floor(tensor)

def divide(a, b, limit=1, tol=1e-6):
    a = batorch_tensor(a)
    b = batorch_tensor(b)
    a_s, b_s = a.shape ^ b.shape
    a = a.view(a_s)
    b = b.view(b_s)
    shape = bt.Size(max(x, y) for x, y in zip(a_s, b_s))
    return bt.where(b.abs() < tol, bt.where(a.abs() < tol, bt.zeros(shape), limit * bt.ones(shape)), a / bt.where(b.abs() < tol, tol * bt.ones(shape), b))

def equals(x, y):
    x = batorch_tensor(x)
    y = batorch_tensor(y)
    return ((bt.abs(x - y) / (bt.abs(x) + bt.abs(y) + 1e-4)) < 1e-4) | (bt.abs(x) + bt.abs(y) < 1e-4)

def matpow(A, k):
    """return a matrix power of A^k."""
    if Version(bt.torch.__version__) >= '1.10': L, V = bt.linalg.eig(A)
    else:
        K, P = bt.eig(A, eigenvectors=True)
        L = bt.complex(K[:, 0], K[:, 1])
        Vr = bt.where((K[:, 1] < 0).reshape((1, -1)), bt.cat((P[:, :1], P[:, :-1]), 1), P)
        Vi = (K[:, 1] > 0).reshape((1, -1)) * bt.cat((P[:, 1:], P[:, -1:]), 1) - (K[:, 1] < 0).reshape((1, -1)) * P
        V = bt.complex(Vr, Vi)
    R = V @ bt.diag(L ** k) @ bt.inv(V)
    if (bt.is_complex(V) or bt.is_complex(L)) and not bt.is_complex(A): R = bt.real(R)
    return R.as_type(A)

@restore_type_wrapper
def matprod(A, B, dims):
    avouch(isinstance(dims, (list, tuple)) and len(dims) == 2)
    d1, d2 = dims
    avouch(d1 < d2, "'matprod' needs dimension d1 < d1. ")
    avouch(-A.ndim <= d1 < A.ndim and -A.ndim <= d2 < A.ndim)
    avouch(-B.ndim <= d1 < B.ndim and -B.ndim <= d2 < B.ndim)
    avouch(A.shape[:d1] == B.shape[:d1] and A.shape[d1+1:d2] == B.shape[d1+1:d2] and A.shape[d2+1:] == B.shape[d2+1:])
    avouch(A.size(d2) == B.size(d1))
    other_dims = list(range(d1)) + list(range(d1 + 1, d2)) + list(range(d2 + 1, A.ndim))
    other_sizes = A.shape[:d1] + A.shape[d1+1:d2] + A.shape[d2+1:]
    C = A.mergedims(other_dims, target=[]) @ B.mergedims(other_dims, target=[])
    return C.splitdim([], other_sizes.python_repr).movedim(-2, d1).movedim(-1, d2)

def add_special(size, special, fill=1):
    s = special
    if len(s) == 0: pass
    elif len(s) == 1: size = size[:s[0]] + (fill,) + size[s[0]:]
    else: size = size[:s[0]] + (fill,) + size[s[0]:s[1]-1] + (fill,) + size[s[1]-1:]
    return size

def gaussian_kernel(n_dims = 2, kernel_size = 3, sigma = 0, normalize = True):
    radius = (kernel_size - 1) / 2
    if sigma == 0: sigma = radius * 0.6
    grid = bt.image_grid(*(kernel_size,) * n_dims).float()
    kernel = bt.exp(- ((grid - radius) ** 2).sum(0) / (2 * sigma ** 2))
    return (kernel / kernel.sum()) if normalize else kernel

def dot(g1, g2):
    avouch(g1.shape == g2.shape, "Please make sure the dot product recieve two tensor of a same shape. Use 'X.expand_to' to adjust if necessary. ")
    avouch(g1.has_channel and g2.has_channel and g1.channel_dim == g2.channel_dim, "Please make sure the inputs of 'dot' have a same channel dimension. ")
    return (g1 * g2).sum(g1.channel_dimension)

@restore_type_wrapper
def Jacobian(X, Y, dt=1, pad=False):
    """
        The Jacobian matrix; Note that it is a transpose of grad_image if X is standard grid as it follows the orientation of Jacobian.
        X: ([n_batch], {n_dim}, n@1, ..., n@n_dim)
        Y: ([n_batch], {n_feature}, n@1, ..., n@n_dim)
        Jacobian(output, pad = True): ([n_batch], n_feature, {n_dim}, n@1, ..., n@n_dim)
        Jacobian(output, pad = False): ([n_batch], n_feature, {n_dim}, n@1 - dt, ..., n@n_dim - dt)
    """
    dX = grad_image(X, dx=dt, pad=pad).with_channeldim(None) # ([n_batch], n_dim, n_dim, n@1, ..., n@n_dim)
    dY = grad_image(Y, dx=dt, pad=pad).with_channeldim(None) # ([n_batch], n_dim, n_feature, n@1, ..., n@n_dim)
    # dyk/dxi = sum_j [(dyk/dtj) / (dxi/dtj)]: Jac[[b], j, k, i, ...] = dY[[b], j, k, *, ...] - dX[[b], j, *, i, ...]
    # bt.summary(dX).show()
    # bt.summary(dY).show()
    return bt.divide(dY.unsqueeze(3), dX.unsqueeze(2), 0.0).sum(1).with_channeldim(2)

@restore_type_wrapper
def grad_image(array, dx=1, pad=False):
    '''
        Gradient image / tensor of array (dx is the displacement for finite difference). 
        If pad is not True, the image will trim. The sizes should be: 
        array: ([n_batch], {n_feature}, n@1, ..., n@n_dim)
        output (pad = True): ([n_batch], n_dim, {n_feature}, n@1, ..., n@n_dim)
        output (pad = False): ([n_batch], n_dim, {n_feature}, n@1 - dx, ..., n@n_dim - dx)
        OR:
        array: ([n_batch], n@1, ..., n@n_dim)
        output (pad = True): ([n_batch], {n_dim}, n@1, ..., n@n_dim)
        output (pad = False): ([n_batch], {n_dim}, n@1 - dx, ..., n@n_dim - dx)
    '''
    if not isinstance(array, bt.Tensor):
        array = batorch_tensor(array).with_batchdim(0).unsqueeze({})
    grad_dim = int(array.has_batch)
    if not array.has_channel: grad_dim = {grad_dim}
    output = []
    size = array.space
    if not pad: size = tuple(x - dx for x in size)
    for d in range(array.ndim):
        if d in array.special: continue
        b = (slice(None, None),) * d + (slice(dx, None),) + (slice(None, None),) * (array.ndim - d - 1)
        a = (slice(None, None),) * d + (slice(None, -dx),) + (slice(None, None),) * (array.ndim - d - 1)
        output.append(bt.crop_as((array[b] - array[a]) / dx, size))
    return bt.stack(output, grad_dim)

@overload
@restore_type_wrapper("roi")
def crop_as(x: array, y: tuple, center: tuple, fill: number=0) -> array:
    x = batorch_tensor(x)
    size_x = x.shape
    size_y = y

    if isinstance(size_y, bt.Size) and size_x.nspace == size_y.nspace:
        size_y = tuple(size_y.space)
    size_y = tuple(size_y)
    if len(size_y) == len(size_x): pass
    elif len(size_y) == size_x.nspace: size_y = add_special(size_y, size_x.special, -1)
    else: raise TypeError("Mismatch dimensions in 'crop_as', please use -1 if the dimension doesn't need to be cropped. ")
    assert len(size_y) == len(size_x)
    size_y = tuple(a if b == -1 else b for a, b in zip(size_x, size_y))

    if len(center) == len(size_x): pass
    elif len(center) == size_x.nspace: center = add_special(center, size_x.special, -1)
    elif len(x for x in center if x >= 0) == len(x for x in size_y if x >= 0):
        center = tuple(a if b >= 0 else -1 for a, b in zip(center, size_y))
    else: raise TypeError("Mismatch dimensions for the center in 'crop_as', please use -1 if the dimension that is centered or doesn't need cropping. ")
    assert len(center) == len(size_x)
    center = tuple(a / 2 if b == -1 else b for a, b in zip(size_x, center))

    z = fill * bt.ones(*size_y).type_as(x)
    def intersect(u, v):
        return max(u[0], v[0]), min(u[1], v[1])
    z_box = [intersect((0, ly), (- round(float(m - float(ly) / 2)), - round(float(m - float(ly) / 2)) + lx)) for m, lx, ly in zip(center, size_x, size_y)]
    x_box = [intersect((0, lx), (+ round(float(m - float(ly) / 2)), + round(float(m - float(ly) / 2)) + ly)) for m, lx, ly in zip(center, size_x, size_y)]
    # if the two boxes are seperated
    if any([r[0] >= r[1] for r in z_box]) or any([r[0] >= r[1] for r in x_box]): z.roi = None; return z
    region_z = tuple(slice(u, v) for u, v in z_box)
    region_x = tuple(slice(u, v) for u, v in x_box)
    z[region_z] = x[region_x]
    z.roi = region_x
    z.special_from_(x)
    return z

@overload
def crop_as(x: array, y: array, center: tuple, fill: number=0) -> array:
    return crop_as(x, y.shape, center, fill)

@overload
def crop_as(x: array, y: union(tuple, array), fill: number=0) -> array:
    center = tuple(m/2 for m in x.shape)
    return crop_as(x, y, center, fill)

@overload
def crop_as(x: array, *y: int) -> array:
    center = tuple(m/2 for m in x.shape)
    return crop_as(x, y, center)

@restore_type_wrapper
def pad(x, p = 1, fill = 0):
    x = batorch_tensor(x)
    p = to_tuple(p)
    if len(p) == 1: p *= x.n_space
    return crop_as(x, tuple(s + 2 * q for s, q in zip(x.space, p)), fill = fill)

@restore_type_wrapper
def up_scale(image, *scaling:int):
    image = batorch_tensor(image)
    if len(scaling) == 0:
        scaling = (1,)
    elif len(scaling) == 1 and iterable(scaling[0]):
        scaling = scaling[0]
    if len(scaling) == 1:
        if isinstance(scaling[0], int):
            scaling *= image.nspace
            scaling = add_special(scaling, image.special, 1)
        else: raise TypeError("Unknown scaling type for 'up_scale'. ")
    elif len(scaling) < image.ndim and len(scaling) == image.nspace:
        scaling = add_special(scaling, image.special, 1)
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

@restore_type_wrapper
def down_scale(image, *scaling:int):
    image = batorch_tensor(image)
    if len(scaling) == 0:
        scaling = (1,)
    elif len(scaling) == 1 and iterable(scaling[0]):
        scaling = scaling[0]
    if len(scaling) == 1:
        if isinstance(scaling[0], int):
            scaling *= image.nspace
            scaling = add_special(scaling, image.special, 1)
        else: raise TypeError("Unknown scaling type for 'down_scale'. ")
    elif len(scaling) < image.ndim and len(scaling) == image.nspace:
        scaling = add_special(scaling, image.special, 1)
    return image[tuple(slice(None, None, s) for s in scaling)]

def image_grid(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
    if len(shape) == 1 and hasattr(shape[0], 'space'): shape = shape[0].space
    if len(shape) == 1 and hasattr(shape[0], 'shape'): shape = shape[0].shape
    kwargs = {'indexing': 'ij'} if bt.__torchversion__ >= Version('1.10') else {}
    if any(x < 0 for x in shape): raise TypeError(f"Negative size in shape {shape} for image grid.")
    return bt.stack(bt.meshgrid(*[bt.arange(x) for x in shape], **kwargs), {})

def linear(input, weight, bias):
    result = input @ weight.T
    if bias is not None:
        if bias.dim() == 2:
            return result + bias
        return result + bias.unsqueeze(0)
    return result

def one_hot(k, n):
    """
    Create an one-hot vector as a zero `bt.Tensor` with length `n` and the `k`-th element 1. 
    e.g. k == -1 gives tensor([0, 0, ..., 0, 1])
    """
    l = [0] * n; l[k] = 1
    return bt.to_device(batorch_tensor(l))

def permute_space(data, *dims):
    """
    permute the space section in data. len(dims) should be data.n_space. 
    """
    data = batorch_tensor(data)
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
    a, b = data._special; n_dim = data.ndim
    if a < n_dim: dims = [x + 1 if x >= a else x for x in dims]
    if b < n_dim: dims = [x + 1 if x >= b else x for x in dims]
    if a < n_dim: dims = dims[:a] + [a] + dims[a:]
    if b < n_dim: dims = dims[:b] + [b] + dims[b:]
    return data.permute(*dims)

# class Input:
    
#     def __init__(self, **kwargs):
#         """
#         Reshape the tensors to the shape_exprs.

#         Params:
#             shape_expr (str): a string containing a shape expression. The expression is in shape format, 
#                 except unknown number of dimensions can be specified with '...' and variable can be specified with words.
                
#         Examples:
#         ----------
#         >>> a, b = bt.Input(n_dim=3)\
#         ...          .set_shape(bt.zeros(3, 4, 5), "([n_batch], {n_dim}, n@1, ..., n@n_dim)")\
#         ...          .set_shape(bt.zeros([2], 34), "([n_batch], m_1)")\
#         ...          .init()
#         ...
#         >>> n_batch, n_dim
#         (2, 3)
#         >>> a
#         Tensor([[[[[...]]]]], shape=batorch.Size([2], {3}, 3, 4, 5))
#         >>> b
#         Tensor([[...]], shape=batorch.Size([2], 34))
#         >>> m_1, n_1, n_n_dim
#         34, 3, 5
#         """
#         self.env_vars = get_environ_vars()
#         self.vars = self.env_vars.locals
#         self.vars.update(kwargs)
#         self.env_vars.update(kwargs)
#         self.data = []
#         self.shape_exprs = []

#     def set_shape(self, data, shape_expr: str):
#         self.data.append(data)
#         self.shape_exprs.append(shape_expr)
#         return self

#     def init(self):
#         print(sum([get_snakenames(x) for x in self.shape_exprs], []))
#         # for data, shape_list in zip(self.data, self.shape_exprs):
#         #     old_shape = data.shape
#         #     dim_pos = [0] * len(shape_list)
#         #     pointer = 0
#         #     for i, x in enumerate(shape_list):
#         #         if x == '...': break
#         #         if x.startswith('[') and x.endswith(']'): dim_pos[i] = old_shape.batch_dimension; x = x.strip('[]')
#         #         if x.startswith('{') and x.endswith('}'): dim_pos[i] = old_shape.channel_dimension; x = x.strip('{}')
#         #         if x.endswith(":optional"): x = x[:-len(":optional")]
#         #         dim_pos[i] = pointer; pointer += 1
#         #         if pointer in old_shape._special: pointer += 1
#         #         if pointer in old_shape._special: pointer += 1 # two lines needed
#         #         if is_dig
#         #         x = get_snakename(x)
#         #         old_value = self.vars.get(x)
#         #         new_value = old_shape[dim_pos[i]]
            
#         # return tuple(data_list)
#         return tuple(self.data)
        
# def as_shape(self, shape_expr: str):
#     env_vars = get_environ_vars()
#     loc_vars = env_vars.locals
#     shape_list = [x.strip() for x in shape_expr.strip("()").split(',')]
#     ibatch, ichannel = None, None
#     old_shape = self.shape
#     dim_pos = [0] * len(shape_list)
#     pointer = 0
#     for i, x in enumerate(shape_list):
#         if x == '...': break
#         if x.startswith('[') and x.endswith(']'): dim_pos[i] = old_shape.batch_dimension; x = x.strip('[]')
#         if x.startswith('{') and x.endswith('}'): dim_pos[i] = old_shape.channel_dimension; x = x.strip('{}')
#         if x.endswith(":optional"): x = x[:-len(":optional")]
#         x = get_snakename(x)
#         if x in loc_vars: x = loc_vars[x]

@alias("skew_symmetric")
def cross_matrix(axis):
    '''
    axis: ([n_batch], {n_dim * (n_dim - 1) / 2})
    output: ([n_batch], n_dim, n_dim)
    '''
    axis = batorch_tensor(axis)
    n_batch = axis.nbatch
    n_dim = int(math.sqrt(2 * axis.size(1))) + 1
    output = bt.zeros([n_batch], n_dim, n_dim)
    if n_dim == 2:
        output[:, 0, 1] = axis[:, 0]
        output[:, 1, 0] = - axis[:, 0]
    elif n_dim == 3:
        output[:, 1, 2] = axis[:, 0]
        output[:, 2, 0] = axis[:, 1]
        output[:, 0, 1] = axis[:, 2]
        output[:, 2, 1] = - axis[:, 0]
        output[:, 0, 2] = - axis[:, 1]
        output[:, 1, 0] = - axis[:, 2]
    return output
    
def uncross_matrix(cross_matrix):
    '''
    cross_matrix: ([n_batch], n_dim, n_dim)
    output: ([n_batch], {n_dim * (n_dim - 1) / 2})
    '''
    cross_matrix = batorch_tensor(cross_matrix)
    n_batch = cross_matrix.nbatch
    n_dim = cross_matrix.size(1)
    axis = bt.zeros([n_batch], {n_dim * (n_dim - 1) // 2})
    if n_dim == 2:
        axis[:, 0] = cross_matrix[:, 0, 1]
    elif n_dim == 3:
        axis[:, 0] = cross_matrix[:, 1, 2]
        axis[:, 1] = cross_matrix[:, 2, 0]
        axis[:, 2] = cross_matrix[:, 0, 1]
    return axis

class summary:
    def __init__(self, *Xs, show_thumb = False, show_core = False, plot = False):
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
        for i, x in enumerate(Xs):
            if n_starts_with_star > 0:
                if 0 <= i - starts_with_star < n_starts_with_star: j = i - starts_with_star; i = starts_with_star
                elif i >= starts_with_star + n_starts_with_star: i = i - n_starts_with_star
            var_name = self.var_names[i] + (f'[{j}]' if i == starts_with_star else '')
            class_name = x.__class__.__name__
            g = None
            if not isinstance(x, bt.torch.Tensor):
                l = touch(lambda: len(x), 0)
                if l <= 10:
                    self.smys.append(f">>{var_name}<< " + str(x) + '\n')
                    continue
                else: x = batorch_tensor(x)
            elif not isinstance(x, bt.Tensor):
                if x.is_leaf and x.grad is not None: g = batorch_tensor(x.grad.clone().detach())
                x = batorch_tensor(x)
            batch_dim = f"[{x.n_special}]+" if x.n_special > 0 else ''
            xshape = str(x.shape).split('Size')[-1]
            out = SPrint(f">>Summary of {var_name}<<\n")
            out(f"【Size】{batch_dim}{x.n_space}D {class_name} of size {xshape}")
            out(f"【Device】{x.device}")
            out(f"【Type】{x.type()}")
            out(f"【Using gradient】{x.requires_grad}")
            if x.requires_grad: out(f"【Gradient func】{x.grad_fn}")
            nan_mask = bt.isnan(x)
            num_nans = nan_mask.sum().sum()
            if num_nans > 0:
                out(f"【Number of NaNs】{num_nans.long().item()}")
                out(f"【Center of NaNs】{bt.Size([round(x, 1) for x in (bt.image_grid(*x.shape) * nan_mask.remove_special_()).float().mean().tolist()]).special_from_(x)}")
                grad_smy = None
            else:
                out(f"【MinMax】[{round(x.min().item(), 4)}, {round(x.max().item(), 4)}];")
                if g is None and (not x.is_leaf or x.grad is None): grad_smy = None
                else:
                    if g is None: g = x.grad.clone().detach()
                    gshape = str(g.shape).split('Size')[-1]
                    grad_smy = f"【Gradient】{gshape} tensor of values [{round(g.min().item(), 4)}, {round(g.max().item(), 4)}]\n"
                    if g.n_space >= 2:
                        if g.has_batch: g = g.pick(0, [])
                        if g.has_channel: g = g.pick(0, {})
                    if g.n_dim == 1: a = min(4, g.size(0)); core_gradient = bt.crop_as(g, (a,)); label = "%ds"%a
                    elif g.n_dim == 2: a = min(4, g.size(0)); b = min(4, g.size(1)); core_gradient = bt.crop_as(g, (a, b)); label = "%d x %d" % (a, b)
                    else: a = min(4, g.size(-2)); b = min(4, g.size(-1)); core_gradient = bt.crop_as(g.flatten(0, g.n_dim - 3)[0], (a, b)); label = "%d x %d" % (a, b)
                    if show_core: grad_smy += f"【Core {label}-window for gradient】\n{core_gradient}\n".replace('Tensor', ' '*6).split(']'*(len(label)//2))[0] + ']'*(len(label)//2)
                values = x.unique().tolist()
                if x.nele >> 24 <= 0: out("【Quintiles】", [round(u.item(), 4) for u in x.float().quantile(bt.arange(6).float()/5)])
                else: out("【Quintiles (of first 2^24 elements)】", str(x.flatten(0, x.ndim-1).float()[:16777216].quantile(bt.arange(6).float()/5)).replace('Tensor(', '').split(']')[0] + ']')
                if len(values) <= 10:
                    out(f"【Range ({len(values)} unique element{'s' if len(values) > 1 else ''})】")
                    if len(values) > 6:
                        out(f">> [{', '.join(str(round(x, 4)) for x in values)}]")
                    else:
                        num_pairs = [(round(v, 4), (x == v).sum().sum().item()) for v in values]
                        out(f">> [{', '.join(f'{v}({n})' for v, n in num_pairs)}]")
                else: out(f"【Range ({len(values)} unique elements)】\n  [{', '.join(str(round(x, 4)) for x in values[:5])}, ..., {', '.join(str(round(x, 4)) for x in values[-5:])}]")
            if x.n_space >= 2:
                if x.has_batch: x = x.pick(0, [])
                if x.has_channel: x = x.pick(0, {})
            label = "4 x 4"
            if x.n_dim == 1: a = min(4, x.size(0)); core = bt.crop_as(x, (a,)); label = "%ds"%a
            elif x.n_dim == 2: a = min(4, x.size(0)); b = min(4, x.size(1)); core = bt.crop_as(x, (a, b)); label = "%d x %d" % (a, b)
            else: a = min(4, x.size(-2)); b = min(4, x.size(-1)); core = bt.crop_as(x.flatten(0, x.n_dim - 3)[0], (a, b)); label = "%d x %d" % (a, b)
            if show_core: out(f"【Core 4-window (the central {label})】\n{core}".replace('Tensor', ' '*6).split(']'*(len(label)//2))[0] + ']'*(len(label)//2))
            if show_thumb:
                gaps = bt.floor((bt.channel_tensor(list(x.space)) - 1) / 3).clamp(1).long()
                down_sample = x[(image_grid(*(4 if s > 4 else s for s in x.space)) * gaps).split(1, {}, squeeze=True)]
                out(f"【Thumb 4-window (down sampled to 4{' x 4'*(x.n_space-1)})】\n{down_sample}".replace('Tensor', ' '*6).split(']'*x.n_space)[0] + ']'*x.n_space)
            if grad_smy is not None: out(grad_smy)
            self.smys.append(out.text)
            if plot:
                while x.n_dim > 2:
                    i_min = argmin(list(x.shape))[0]
                    x = x.pick(x.size(i_min) // 2, i_min)
                self.canvases.append(x.detach())
    def show(self):
        if not self.show_as_plot:
            print(self)
            return
        plt.figure(figsize=(15, 7))
        
        n_plots = len(self.canvases)
        self.subplot_rows = math.floor(math.sqrt(n_plots))
        self.subplot_cols = math.ceil(n_plots / self.subplot_rows)

        for i, x in enumerate(self.canvases):
            ax = plt.subplot(self.subplot_rows, self.subplot_cols, i+1)
            old_size = plt.rcParams['font.size']
            plt.rcParams['font.size'] = 8
            if x.n_dim == 2:
                img = plt.imshow(x, cmap=plt.cm.gray)
                plt.colorbar(mappable=img, ax=ax, orientation='vertical')
                plt.text(-0.5, -0.5, self.smys[i])
            elif x.n_dim == 1:
                plt.plot(bt.arange(len(x)), x)
                plt.text(0, plt.ylim()[1], self.smys[i])
            else: raise TypeError("Failed to display 0-dimensional data in plot.")
            plt.rcParams['font.size'] = old_size
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
