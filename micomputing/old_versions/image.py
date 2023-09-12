
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File to deal with image data, including transformation, cropping, etc.",
    help = "Use `from micomputing import *`.",
    requires = ""
).check()

__all__ = """
    
""".split()

with __info__:
    import os, sys
    import numpy as np
    import batorch as bt
    from math import *
    from datetime import datetime
    from pycamia import to_tuple, to_list, restore_type_wrapper

if __name__ == "__main__":
    from zyctools import plot as plt

IMG_SIZE = 10#128
TRANSC = IMG_SIZE / 5

def setTRANSC(num):
    global TRANSC
    TRANSC = num
    
def sizeof(x):
    return getattr(x, 'shape', x)

def orderedValue(d):
    return [d[k] for k in sorted(d.keys())]

def meshgrid(*lists):
    return bt.stack(bt.meshgrid([bt.tensor(l) for l in lists])).flatten(1).T.numpy()

def grid(*args):
    return meshgrid(*[range(x) for x in args]).astype(np.long)

def imagegrid(*args):
    if len(args) == 1: args = args[0]
    args = sizeof(args)
    args = to_tuple(args)
    return bt.stack(bt.meshgrid(tuple(bt.arange(i) for i in args)))

def info(tensor, name = '', format = '%.4f'):
    tensor = bt.tensor(tensor)
    output = SPrint()
    output(name, '-', tensor.type(), repr(list(sizeof(tensor))), ':', sep='')
    output("range: (", format%tensor.min().item(), ', ', format%tensor.max().item(), ')', sep='')
    if len(set(to_list(tensor.flatten()))) <= 100:
        output("values: {", end='')
        output(*[format%x for x in set(to_list(tensor.flatten()))], sep=', ', end='')
        output('}')
    else:
        output("values:", 'over 100 values')
    return output.text

def pmkdir(path):
    cumpath = ''
    for p in path.split(os.sep):
        if os.extsep in p and p == path.split(os.sep)[-1]: break
        if not cumpath: cumpath = p
        else: cumpath = os.path.join(cumpath, p)
        if not os.path.exists(cumpath): os.mkdir(cumpath)
    return path

def rlistdir(folder, tofolder=False, relative=False, filter=lambda x: True, ext=''):
    file_list = []
    for f in os.listdir(folder):
        if f == '.DS_Store': continue
        path = os.path.join(folder, f)
        if os.path.isdir(path): file_list.extend(rlistdir(path, tofolder))
        if os.path.isfile(path) and not tofolder: file_list.append(path)
    if tofolder and not file_list: file_list.append(folder)
    return [os.path.relpath(f, folder) if relative else f for f in file_list if filter(f) and f.endswith(ext)]

class SPrint:
    """
    Print to a string.

    example:
    ----------
    >>> output = SPrint("!>> ")
    >>> output("Use it", "like", 'the function', "'print'.", sep=' ')
    !>> Use it like the function 'print'.
    >>> output("A return is added automatically each time", end=".")
    !>> Use it like the function 'print'.
    A return is added automatically each time.
    >>> output.text
    !>> Use it like the function 'print'.
    A return is added automatically each time.
    """

    def __init__(self, init_text=''):
        self.text = init_text

    def __call__(self, *parts, sep=' ', end='\n'):
        if not parts: end = ''
        self.text += sep.join([str(x) for x in parts if str(x)]) + end
        return self.text

    def __str__(self): return self.text

imported = True
try:
    import logging
except ImportError:
    imported = False
if imported:
    class LogPrint:

        def __init__(self, file_name='logging.log', init_text=''):
            # logging.basicConfig(filename=file_name, level=logging.DEBUG, format="%(asctime)s | %(message)s")
            open(file_name, 'w').close()
            self.text = init_text
            self.fp = file_name

        def __call__(self, *parts, sep=' ', end='\n'):
            if not parts: end = ''
            def string(x):
                if prod(getattr(x, 'shape', (0,))): 
                    if isinstance(x, bt.Tensor): x = x.item()
                    else: x = float(x)
                if isinstance(x, float): x = '%.4e'%x
                return str(x) 
            log = sep.join([string(x) for x in parts])
            self.text += log
            print(datetime.now(), '|', log, end=end)
            # logging.info(log)
            with open(self.fp, 'a') as fp:
                fp.write(str(datetime.now()) + ' | ' + log + str(end))
            return self.text

    def myprint(*args, **kwargs):
        string = kwargs.get('sep', ' ').join([str(x) for x in args])
        print(string, end=kwargs.get('end', '\n'))
        logging.info(string)
else:
    def LogPrint(*args, **kwargs): raise ImportError("Cannot import package 'logging'. ")
    def myprint(*args, **kwargs): raise ImportError("Cannot import package 'logging'. ")

#: deprecated
# def myimshow(subplotpos, image):
#     if not isinstance(image, bt.Tensor): image = bt.tensor(image)
#     if isinstance(subplotpos, tuple) and subplotpos[-1] == 1 or \
#         isinstance(subplotpos, int) and subplotpos % 10 == 1: plt.clf()
#     params = bt.tensor([[0, image.size(1) / translation_coefficient, -pi / 2, 1, 1, 0, 0]]).float().to(device)
#     trans = lambda X: AffineTransformation2D(X, params)
#     max_len = max(image.size())
#     cropped = crop(image.unsqueeze(0), (max_len, max_len))
#     rotated = LinearInterpolation(cropped, trans)
#     image = crop(rotated, (image.size(1), image.size(0))).squeeze(0)
#     if isinstance(subplotpos, tuple): plt.subplot(*subplotpos)
#     else: plt.subplot(subplotpos)
#     plt.imshow(image.cpu().data, cmap=plt.cm.gray)
#     if isinstance(subplotpos, tuple) and subplotpos[2] == subplotpos[0] * subplotpos[1] or \
#         isinstance(subplotpos, int) and subplotpos % 10 == (subplotpos // 100) * ((subplotpos // 10) % 10): plt.show()

# def get_local_variable(func, var, inputs):
#     import inspect
#     code = inspect.getsource(func)
#     header = '\n'.join([l for l in open(inspect.getsourcefile(func))
#                         .read().split('\n') if 'import' in l])
#     beg = code.find('return')
#     if beg < 0: code += '\n    return ' + var
#     else:
#         end = code.find('\n', beg)
#         if end < 0: end = len(code)
#         code = code[:beg + 6] + ' ' + var + code[end:]
#     exec(header + '\n' + code)
#     return eval(func.__name__)(*inputs)

@restore_type_wrapper
def crop_as(x, y, center = 'center', n_keepdim = 0, box_only = False, fill = 0):
    if isinstance(x, np.ndarray): size_x = x.shape
    elif isinstance(x, bt.Tensor): size_x = x.size()
    else: raise TypeError("Unknown type for x in crop_as. ")
    if isinstance(y, np.ndarray): size_y = y.shape[n_keepdim:]
    elif isinstance(y, bt.Tensor): size_y = y.size()[n_keepdim:]
    elif isinstance(to_tuple(y), tuple): size_y = to_tuple(y)
    else: raise TypeError("Unknown type for y in crop_as. ")
    size_y = size_x[:n_keepdim] + size_y
    size_y = tuple(a if b == -1 else b for a, b in zip(size_x, size_y))
    if all([a == b for a, b in zip(size_x, size_y)]) and center == 'center': return x
    if isinstance(center, str) and center == 'center': center = tuple(m / 2 for m in size_x)
    elif not isinstance(center, tuple): center = to_tuple(center)
    center = tuple(m / 2 for m in size_x)[:len(size_x) - len(center)] + center
    assert len(size_x) == len(size_y) == len(center)
    if isinstance(x, np.ndarray): z = fill * np.ones(size_y).astype(x.dtype)
    else: z = fill * bt.ones(*size_y).type_as(x)
    intersect = lambda a, b: (max(a[0], b[0]), min(a[1], b[1]))
    z_box = [intersect((0, ly), (- round(float(m - float(ly) / 2)), - round(float(m - float(ly) / 2)) + lx)) for m, lx, ly in zip(center, size_x, size_y)]
    x_box = [intersect((0, lx), (+ round(float(m - float(ly) / 2)), + round(float(m - float(ly) / 2)) + ly)) for m, lx, ly in zip(center, size_x, size_y)]
    if any([r[0] >= r[1] for r in z_box]) or any([r[0] >= r[1] for r in x_box]): return z
    if box_only: return x_box
    region_z = bt.meshgrid([bt.arange(*r) for r in z_box])
    region_x = bt.meshgrid([bt.arange(*r) for r in x_box])
    z[region_z] = x[region_x]
    return z
    
def pad(x, p = 1, n_keepdim = 0, fill = 0):
    p = to_tuple(p)
    if len(p) == 1: p *= len(x.shape) - n_keepdim
    return crop_as(x, tuple(s + 2 * q for s, q in zip(sizeof(x)[n_keepdim:], p)), n_keepdim = n_keepdim, fill = fill)

@restore_type_wrapper
def blur(data, kernel_size = 3, with_batch = False):
    data = bt.tensor(data)
    n_dim = data.n_space
    kernel = bt.tensor(gaussian_kernel(n_dim, kernel_size=kernel_size))
    conv = eval("bt.nn.functional.conv%dd"%n_dim)
    if not with_batch: data = bt.unsqueeze(data, 0)
    result = conv(
        bt.unsqueeze(data.float(), 1),
        bt.unsqueeze(kernel, [0, 1]),
        padding=kernel_size // 2
    ).squeeze(1)
    result = crop_as(result, sizeof(data))
    return result
    # kernel = gaussian_kernel(dimof(data) - int(with_batch), kernel_size=kernel_size)
    # if with_batch: kernel = multiply(kernel, nbatch(data), 0)
    # return conv(data, kernel, with_batch = with_batch)

#
#def gaussian_2d_kernel(kernel_size = 3, sigma = 0):
#    kernel = np.zeros([kernel_size, kernel_size])
#    center = kernel_size // 2
#    if sigma == 0: sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
#
#    s = 2 * (sigma ** 2)
#    sum_val = 0
#    for i in range(kernel_size):
#        for j in range(kernel_size):
#            x = i - center
#            y = j - center
#            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
#            sum_val += kernel[i, j]
#            #/(np.pi * s)
#    sum_val = 1 / sum_val
#    return kernel * sum_val

def gaussian_kernel(n_dims = 2, kernel_size = 3, sigma = 0, normalize = True):
    radius = (kernel_size - 1) / 2
    if sigma == 0: sigma = radius * 0.6
    a, b = map(int, bt.torch.__version__.split('+')[0].split('.')[:2])
    kwargs = {'indexing': 'ij'} if (a, b) >= (1, 10) else {}
    grid = np.meshgrid(*(np.arange(kernel_size),) * n_dims, **kwargs)
    kernel = np.exp(- (sum([(g - radius) ** 2 for g in grid])) / (2 * sigma ** 2))
    return (kernel / np.sum(kernel)) if normalize else kernel

def gather_nd(array, indices):
    '''
    array: ([n_batch], n_1, ..., n_r)
    indices: ([n_batch], {r}, *)
    output: ([n_batch], *)
    '''
    array = bt.tensor(array).standard()
    if not array.has_batch: return array[indices.split(1)]
    n_batch, n_dim, *size = sizeof(indices)
    indices = bt.cat(bt.arange(n_batch).as_type(indices), indices, 1)
    return array[bt.t(indices).split(1)]

def put_nd(array, indices, value, with_batch = True):
    '''
    array: (n_batch, n_1, ..., n_r)
    indices: (n_batch, r, *)
    value: (n_batch, *)
    '''
    if not with_batch: array[indices.split(1)] = value; return
    n_batch, n_dim, *size = sizeof(indices)
    indices = bt.cat(bt.arange(n_batch).as_type(indices), indices, 1)
    array[bt.t(indices).split(1)] = value

#print(gather_nd(bt.tensor([[1,2,3],[2,2,3],[3,3,1]]), bt.tensor([2,1,1,0])))

def Bspline(i, U):
    i = bt.tensor(i); U = bt.tensor(U)
    return (
        bt.where(i == -1, (1 - U) ** 3 / 6,
        bt.where(i == 0, U ** 3 / 2 - U * U + 2 / 3,
        bt.where(i == 1, (- 3 * U ** 3 + 3 * U * U + 3 * U + 1) / 6,
        bt.where(i == 2, U ** 3 / 6,
        bt.zeros_like(U)))))
    )

def dBspline(i, U):
    i = bt.tensor(i); U = bt.tensor(U)
    return (
        bt.where(i == -1, - 3 * (1 - U) ** 2 / 6,
        bt.where(i == 0, 3 * U ** 2 / 2 - 2 * U,
        bt.where(i == 1, (- 3 * U ** 2 + 2 * U + 1) / 2,
        bt.where(i == 2, 3 * U ** 2 / 6,
        bt.zeros_like(U)))))
    )

def fBspline(c, x):
    c = bt.tensor(c); x = bt.tensor(x)
    d = x - c
    return (
        bt.where((-2 <= d) * (d < -1), d ** 3 + 6 * d ** 2 + 12 * d + 8,
        bt.where((-1 <= d) * (d < 0), - 3 * d ** 3 - 6 * d ** 2 + 4,
        bt.where((0 <= d) * (d < 1), 3 * d ** 3 - 6 * d ** 2 + 4,
        bt.where((1 <= d) * (d < 2), - d ** 3 + 6 * d ** 2 - 12 * d + 8,
        bt.zeros_like(d))))) / 6
    )

@restore_type_wrapper
def interpolation(
        image, 
        Transformation, 
        mode = 'Linear', 
        target_space = None,
        Surrounding = None, 
        D = False, 
        bk_type = 'zero',
        toSize = None # Deprecated
    ):
    '''
    Compute the image I s.t. Transformation(x) = y for x in I and y in image using interpolation method:
        mode = Linear: Bilinear interpolation
        mode = Nearest [NO GRADIENT!!!]: Nearest interpolation

    `image`: `bt.Tensor`
        The target image.
        size: `n_batch x m_1 x m_2 x ... x m_{n_dim}`
    `Transformation`: `Function`
        A `(n_dim, n_1, n_2, ..., n_{n_dim})` `->` `(n_batch, n_dim, n_1, n_2, ..., n_{n_dim})` transformation function.
    `target_space`: `Size(tuple) / Coordinate Space(bt.Tensor)`
        The size of the output: a tuple of length `n_dim`
        Or the coordinate space for the output
        size: `n_batch x n_dim x size_1 x size_2 x ... x size_{n_dim}`
    `Surrounding`: `bt.Tensor`
        An extended image containing information out of bound. By default, this image has a same center as the image.
        size: `n_batch x M_1 x M_2 x ... x M_{n_dim}`

    `output`: `bt.Tensor`
        The transformed image.
        size: `n_batch x size_1 x size_2 x ... x size_{n_dim}`
        or
        The derivative for the interpolation. (if `D = True`)
        size: `n_batch x n_dim x size_1 x size_2 x ... x size_{n_dim}`

    ----------
    Example:
    ```python
    >>> Image = bt.rand(3, 100, 120, 80)
    >>> AM = bt.rand(1, 4, 4)
    >>> AM[0, 3, :] = bt.Tensor([0, 0, 0, 1])
    >>> interpolation(Image, Affine(AM), mode='Linear')
    ```
    '''
    image = bt.tensor(image)
    if target_space is None:
        if toSize is not None: target_space = toSize
        else: target_space = image.size()[1:]
    if Surrounding is None: Surrounding = image
    n_dim = image.ndim - 1 # Get the spatial rank.
    n_batch = image.batch_dimension
    m = Ltensor(image.size()[1:])
    M = Ltensor(Surrounding.size()[1:])
    if len(m) == len(M) == n_dim: pass
    elif isinstance(target_space, tuple) and len(target_space) == n_dim: pass
    elif not isinstance(target_space, tuple) and dimof(target_space) == n_dim + 2: pass
    else: raise TypeError("Wrong input shape for interpolation. ")
    if isinstance(target_space, tuple): 
        # Create a grid X with size n_dim x size_1 x size_2 x ... x size_{n_dim}.
        X = imagegrid(target_space)
        # Compute the transformed coordinates. Y: n_batch x n_dim x size_1 x size_2 x ... x size_{n_dim}.
        if Transformation is None: Y = multiply(X, n_batch)
        else: Y = Transformation(X)
    else: Y = target_space

    if mode.upper() == 'FFD':
        if D: raise TypeError("No derivatives for FFD interpolations are available so far. Please write it by yourself. ")
        # TODO: FFD

    iY = bt.floor(Y).long() # Generate the integer part of Y
    jY = iY + 1 # and the integer part plus one.
    if mode.lower() == 'linear': fY = Y - iY.float() # The decimal part of Y.
    elif mode.lower() == 'nearest': fY = bt.floor(Y - iY.float() + 0.5) # The decimal part of Y.
    else: raise TypeError("Unrecognized argument 'mode'. ")
    bY = bt.stack((iY, jY), 1).view(n_batch, 2, n_dim, -1) # n_batch x 2 x s x (m_1 x m_2 x ... x m_s).
    W = bt.stack((1 - fY, fY), 1).view(n_batch, 2, n_dim, -1) # n_batch x 2 x s x (m_1 x m_2 x ... x m_s).
    n_points = bY.size(-1)

    # Prepare for the output space: n_batch x m_1 x ... x m_s
    if D: output = Fzeros(n_batch, n_dim, *sizeof(Y)[2:])
    else: output = Fzeros(n_batch, *sizeof(Y)[2:])
    for G in Lstack(bt.meshgrid((bt.arange(2),) * n_dim)).view(n_dim, -1).transpose(0, 1):
        # Get the indices for the term: bY[:, G[D], D, :], size=(n_batch, s, m_1 x m_2 x ... x m_s)
        Ind = bY.gather(1, G.view(1, 1, n_dim, 1).expand(n_batch, 1, n_dim, n_points)).squeeze(1)
        # Change into the coordinate space for Surrounding.
        Ind += (M - m).view(1, -1, 1).to(device) // 2
        # Clamp the indices in the correct range.
        ceiling = Lto(M.view(n_dim, 1) - Lones(n_dim, 1)).expand(Ind.size())
        # Compute the border condition
        condition = bt.sum((Ind < 0) + (Ind > ceiling), 1)
        Ind = bt.min(bt.clamp(Ind, min=0), ceiling)
        # Convert the indices to 1 dimensional. Dot: (n_batch, m_1 x m_2 x ... x m_s)
        Dot = Ind[:, 0]
        for r in range(1, n_dim): Dot *= M[r]; Dot += Ind[:, r]
        # Get the image values IV: (n_batch, m_1 x m_2 x ... x m_s)
        if bk_type.lower() == 'nearest':
            IV = Fflatten(Surrounding, 1).gather(1, Dot)
        else:
            if bk_type.lower() == 'background':
                background = Fmin(Surrounding) * Fones_like(Dot)
            elif bk_type.lower() == 'zero':
                background = Fzeros_like(Dot)
            IV = bt.where(condition >= 1, background, Surrounding.flatten(1).to(device).gather(1, Dot).float())
        # Weights for each point: [product of W[:, G[D], D, x] for D in range(n_dim)] for point x.
        # Wg: (n_batch, n_dim, m_1 x m_2 x ... x m_s)
        Wg = W.gather(1, G.view(1, 1, n_dim, 1).expand(n_batch, 1, n_dim, n_points)).squeeze(1).to(device)
        if not D: output += (Wg.prod(1) * IV).view_as(output).to(device)
        else:
            dWg = bt.zeros_like(Wg)
            for dim in range(n_dim):
                itm = bt.cat((bt.arange(dim), bt.arange(dim + 1, n_dim))).to(device)
                dWg[:, dim] = Wg[:, itm].prod(1) * (G[dim] * 2 - 1).float()
            output += (dWg * IV.unsqueeze(1)).view_as(output).to(device)
    return output
    
class Transformation:
    
    def __init__(self, params, with_batch=False):
        self.params = params
        self.with_batch = with_batch
        
    def __call__(self, x): pass
    
    def __enter__(self): self._old_with_batch = self.with_batch; self.with_batch = True; return
    
    def __exit__(self, *_): self.with_batch = self._old_with_batch
    
    def __getitem__(self, i):
        from copy import copy
        clone = copy(self)
        try:
            int(tolist(i))
            clone.params = unsqueeze(self.params.clone()[i])
        except: clone.params = self.params.clone()[i]
        clone.with_batch = self.with_batch
        return clone

    def __str__(self):
        return str(type(self)) + " transformation with params: " + str(type(self.params)) + str(sizeof(self.params))

    def __matmul__(x, y):
        return CompoundTransformation(x, y)

    def detach(self): self.params = self.params.detach(); return self
    
    def with_batch_(self, b): self.with_batch = b; return self
    
    def repeat(self, k): return lambda x: repeat(self(x), k)

    def nbatch(self):
        try: return nbatch(self.params)
        except: return self.n_batch
    
    def toDDF(self, *shape):
        if len(shape) == 1: shape = shape[0]
        shape = to_tuple(shape)
        grid = imagegrid(*shape)
        bgrid = grid.unsqueeze(0)
        return self(bgrid if self.with_batch else grid) - bgrid

    def to(self, type = "world", target_affine = bt.eye(4), source_affine = bt.eye(4)):
        if target_affine.ndim <= 2: target_affine = unsqueeze(target_affine)
        if source_affine.ndim <= 2: source_affine = unsqueeze(source_affine)
        target_affine = toFtensor(target_affine)
        source_affine = toFtensor(source_affine)
        if type.lower() == "world": taffine = inv(target_affine); saffine = source_affine
        if type.lower() == "image": taffine = target_affine; saffine = inv(source_affine)
        if type.lower() not in ("world", "image"): raise TypeError("Invalid type for method 'to'.")
        return CompoundTransformation(Affine(taffine, with_batch = self.with_batch), self, Affine(saffine))

class CompoundTransformation(Transformation):

    def __init__(self, *trans_list, with_batch = False):
        self.with_batch = with_batch
        self.trans_list = trans_list

    def __call__(self, x):
        if not self.with_batch: y = unsqueeze(x)
        else: y = x
        for f in self.trans_list:
            with f: y = f(y)
        return y

#class inverse(Transformation):
#    
#    def __init__(self, trans, verbose = True):
#        super().__init__(0)
#        self.trans = trans
#        self.verbose = verbose
#        self.min_gap = 1e-5
#        
#    def __call__(self, grid):
#        trans = self.trans
#        verbose = self.verbose
#        min_gap = self.min_gap
#        n_batch = nbatch(trans.params)
#        disp = trans.toDDF(*grid.shape[1 + int(self.with_batch):]).data
#        if self.with_batch: X = (bt.tensor(grid.clone()).data - disp).requires_grad_(True)
#        else: X = (multiply(bt.tensor(grid.clone()).data, n_batch) - disp).requires_grad_(True)
#        optimizer = bt.optim.Adam([X], lr=1e-1)
#        prev_loss = 0
#        for i in range(1000):
#            Y = DDF(disp, with_batch = True)(X)
#            loss = bt.mean((Y - grid) ** 2)
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#            if verbose: print('inverse transformation: iteration %d, loss %f'%(i + 1, loss.item()))
#            if abs(loss.item() - prev_loss) < min_gap or loss.item() < 1e-3: break
#            prev_loss = loss.item()
#        return X.data

class inverse(Transformation):
    
    def __init__(self, trans, verbose = True):
        super().__init__(0)
        self.trans = trans
        self.verbose = verbose
        self.min_gap = 1e-5
        self.itrans = None
        
    def __call__(self, grid):
        trans = self.trans
        verbose = self.verbose
        min_gap = self.min_gap
        n_batch = nbatch(trans.params)
        size = grid.shape[1 + int(self.with_batch):]
        if self.itrans is None:
            disp = trans.toDDF(*size).data
            idisp = (-disp).requires_grad_(True)
            optimizer = bt.optim.Adam([idisp], lr=1e-1)
            sgrid = multiply(imagegrid(*size), n_batch, 0)
            prev_loss = 0
            for i in range(1000):
                with trans: Y = trans(DDF(idisp, with_batch = True)(sgrid))
                loss = bt.mean((Y - sgrid) ** 2) + bending(idisp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose: print('inverse transformation: iteration %d, loss %f'%(i + 1, loss.item()))
                if abs(loss.item() - prev_loss) < min_gap or loss.item() < 1e-3: break
                prev_loss = loss.item()
            self.itrans = DDF(idisp.data)
        if not self.with_batch: grid = unsqueeze(grid)
        with self.itrans: return self.itrans(grid)

def transform(Source, Transformation, **kwargs):
    '''
    Compute the image Transformation(Source) using nearest interpolation method.
    Source: bt.Tensor
        The source image.
        size: n_batch x m_1 x m_2 x ... x m_{n_dim}
    Transformation: Function
        A (n_dim, n_1, n_2, ..., n_{n_dim}) -> (n_batch, n_dim, n_1, n_2, ..., n_{n_dim}) transformation function.
    '''
    return interpolation(Source, inverse(Transformation), **kwargs)

def rescaleImage(image, scaling=1, with_batch=False, mode="Linear"):
    scaling = to_tuple(scaling)
    is_numpy = isinstance(image, np.ndarray)
    image = bt.tensor(image)
    if len(scaling) == 1: scaling *= image.n_space
    if not with_batch: image = bt.unsqueeze(image)
    affineMatrix = bt.diag(1 / bt.tensor(list(scaling) + [1]))
    scalingTransformation = Affine(affineMatrix.multiply(image.nbatch))
    target_size = tuple(np.ceil(a * b) for a, b in zip(scaling, sizeof(image)[1:]))
    result = interpolation(image, scalingTransformation, target_space = target_size, mode = mode).squeeze()
    return result.detach().cpu().numpy() if is_numpy else result

def rescale_to(image, target, with_batch=False, mode="Linear"):
    if isarray(target): target = sizeof(target)
    return crop_as(rescaleImage(image, tuple(float(x-1) / float(y-1) for x, y in zip(target, sizeof(image)))[with_batch:], with_batch, mode), target, n_keepdim = int(with_batch))

##DEPRECATED
#
#def identity(image, with_batch=False, **kwargs):
#    is_numpy = isinstance(image, np.ndarray)
#    image = bt.tensor(image)
#    if not with_batch: image = unsqueeze(image)
#    affineMatrix = bt.eye(dimof(image) + 1)
#    identityTransformation = Affine(unsqueeze(affineMatrix))
#    result = interpolation(image, identityTransformation, **kwargs).squeeze()
#    return result.detach().cpu().numpy() if is_numpy else result

def repeatEnlarge(image, scaling=1):
    is_numpy = isinstance(image, np.ndarray)
    image = bt.tensor(image)
    scaling = to_tuple(scaling)
    if len(scaling) == 1: scaling *= dimof(image)
    for i, s in enumerate(scaling):
        image = (
            image
            .transpose(i, -1)
            .unsqueeze(-1)
            .repeat((1,) * dimof(image) + (int(s),))
            .flatten(-2)
            .transpose(i, -1)
        )
    return image.detach().cpu().numpy() if is_numpy else image

def Affine2Matrix(params):
    n_batch = params.size(0)
    if params.size(1) == 7:
        t1, t2, θ, s1, s2, ρ1, ρ2 = bt.split(params, 1, 1)
        c1 = bt.zeros(n_batch, 1); c2 = bt.zeros(n_batch, 1)
    if params.size(1) == 9:
        t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2 = bt.split(params, 1, 1)
    a = (ρ1 * ρ2 + 1) * s1 * bt.cos(θ) + ρ1 * s2 * bt.sin(θ)
    b = - (ρ1 * ρ2 + 1) * s1 * bt.sin(θ) + ρ1 * s2 * bt.cos(θ)
    c = ρ2 * s1 * bt.cos(θ) + s2 * bt.sin(θ)
    d = - ρ2 * s1 * bt.sin(θ) + s2 * bt.cos(θ)
    return bt.cat((a, b, t1 - a * c1 - b * c2 + c1, c, d, t2 - c * c1 - d * c2 + c2), 1)

def tomm(x, spacing, translation_coefficient):
    return x * spacing * translation_coefficient
    
#: deprecated
#def toDDF(X, params):
#    """
#    inputs:
#        X: [n_dim, n_1, ..., n_{n_dim}]
#            target grid
#        params: Tensor
#            [n_batch, 7]
#            [n_batch, n_dim + 1, n_dim + 1]
#            [n_batch, n_dim, k_1, ..., k_{n_dim}]
#            [n_batch, n_dim, n_1, ..., n_{n_dim}]
#            parameters for transformation.
#    outputs:
#        [n_batch, n_dim, n_1, ..., n_{n_dim}]
#    """
#    l = params.dim()
#    sparams = sizeof(params)[1:]
#    sI = sizeof(X)
#    if l == 2: trans = ParametricAffine2D(params)
#    elif l == 3: trans = Affine(params.unsqueeze(1))
#    elif tuple(sparams) == tuple(sI): trans = DDF(params)
#    else: trans = FFD(params)
#    return trans(X) - unsqueeze(X)

class ParametricAffine2D(Transformation):
    '''
    Affine transformation with respect to transformation parameters.

    X: bt.Tensor
        Coordinates to be transformed.
        size: (2, n_1, n_2, ..., n_r) if not with_batch
            (n_batch, 2, n_1, n_2, ..., n_r) if with_batch
    params: bt.Tensor
        n_batch pairs of parameters with 7 or 9 for each transformation.
        size: (n_batch, 7) or (n_batch, 9)
        t1, t2, θ, s1, s2, ρ1, ρ2 or t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2

    output: bt.Tensor
        The transformed coordinates.
        size: (n_batch, 2, n_1, n_2, ..., n_r)
    '''
    def __init__(self, params, translation_coefficient=1, with_batch=False, inv=False):
        self.params = params
        self.translation_coefficient = translation_coefficient
        self.with_batch = with_batch
        self.inv = inv

    
    def __call__(self, X):
        params = self.params
        translation_coefficient = self.translation_coefficient
        with_batch = self.with_batch
        inv = self.inv
        if not translation_coefficient:
            translation_coefficient = max(X.size()[1:]) / 5
        n_batch = nbatch(params)
        if params.size(1) == 7:
            t1, t2, θ, s1, s2, ρ1, ρ2 = bt.split(params, 1, 1)
            c1 = bt.zeros(n_batch, 1); c2 = bt.zeros(n_batch, 1)
        if params.size(1) == 9:
            t1, t2, c1, c2, θ, s1, s2, ρ1, ρ2 = bt.split(params, 1, 1)
        for x in (t1, t2, c1, c2): x *= translation_coefficient
        a = (ρ1 * ρ2 + 1) * s1 * bt.cos(θ) + ρ1 * s2 * bt.sin(θ)
        b = - (ρ1 * ρ2 + 1) * s1 * bt.sin(θ) + ρ1 * s2 * bt.cos(θ)
        c = ρ2 * s1 * bt.cos(θ) + s2 * bt.sin(θ)
        d = - ρ2 * s1 * bt.sin(θ) + s2 * bt.cos(θ)
        if not with_batch: X = X.unsqueeze(0).expand(n_batch, *X.size())
        center = bt.cat((c1, c2), 1).to(device)
        while len(center.size()) < len(X.size()):
            center = center.unsqueeze(-1)
        cX = X - center
        if inv:
            det = a * d - b * c
            a, d = d, a
            b, c = -b, -c
            a = a / det; b = b / det; c = c / det; d = d / det
            p = bt.cat((a, b), 1).unsqueeze(1) @ (cX.flatten(2) - t1.unsqueeze(-1))
            q = bt.cat((c, d), 1).unsqueeze(1) @ (cX.flatten(2) - t2.unsqueeze(-1))
            return bt.cat((p, q), 1).view_as(X) + center
        else:
            p = bt.cat((a, b), 1).unsqueeze(1) @ cX.flatten(2) + t1.unsqueeze(-1)
            q = bt.cat((c, d), 1).unsqueeze(1) @ cX.flatten(2) + t2.unsqueeze(-1)
            return bt.cat((p, q), 1).view_as(X) + center

class Affine(Transformation):
    '''
    Affine transformation with respect to transformation matrix.

    `X`: `bt.Tensor`
        Coordinates to be transformed.
        size: `(n_dims, n_1, n_2, ..., n_r)`
    `params`: `bt.Tensor`
        `n_batch` pairs of parameters with `7` for each transformation.
        size: `(n_batch, n_dims + 1, n_dims + 1)`

    `output`: `bt.Tensor`
        The transformed coordinates.
        size: `(n_batch, n_dims, n_1, n_2, ..., n_r)`
    '''
    def __init__(self, params, translation_coefficient=None, with_batch=False):
        self.params = params
        self.translation_coefficient = translation_coefficient
        self.with_batch = with_batch

    
    def __call__(self, X):
        params = self.params
        translation_coefficient = self.translation_coefficient
        with_batch = self.with_batch
        if not translation_coefficient:
            translation_coefficient = 1 # max(X.size()[1:]) / 5
        n_batch = nbatch(params)
        n_dim = params.size(-1) - 1
        if not with_batch: X = multiply(X, n_batch)
        A = params[:, :n_dim, :n_dim]
        b = params[:, :n_dim, n_dim] * translation_coefficient
        Y = (A @ X.flatten(2) + b.view(n_batch, n_dim, 1)).view_as(X)
        return Y

class Identity(Transformation):
    '''
    Identity transformation.

    `X`: `bt.Tensor`
        Coordinates to be transformed.
        size: `(n_dims, n_1, n_2, ..., n_r)`

    `output`: `bt.Tensor`
        The transformed coordinates.
        size: `(n_batch, n_dims, n_1, n_2, ..., n_r)`
    '''
    def __init__(self, n_batch, with_batch=False):
        super().__init__(n_batch, with_batch)
        self.n_batch = n_batch

    
    def __call__(self, X):
        return multiply(X, self.params)

class DDF(Transformation):
    '''
    Dense displacement field transformation.

    X: bt.Tensor
        Coordinates to be transformed.
        size: (n_dim, m_1, m_2, ..., m_r) if not with_batch
            (n_batch, n_dim, m_1, m_2, ..., m_r) if with_batch
    params: bt.Tensor
        n_batch pairs of parameters for each transformation.
        size: (n_batch, n_dim, n_1, n_2, ..., n_{n_dims})

    output: bt.Tensor
        The transformed coordinates.
        size: (n_batch, n_dim, m_1, m_2, ..., m_r)
    '''
    def __init__(self, params, shape = None, with_batch=None):
        if isinstance(params, Transformation):
            self.params = params.toDDF(shape)
            if with_batch is None: self.with_batch = params.with_batch
            else: self.with_batch = with_batch
        else:
            self.params = params
            if with_batch is None: self.with_batch = False
            else: self.with_batch = with_batch
        
    
    def __invert__(self):
#        n_batch, n_dim, *size = sizeof(self.params)
#        assert len(size) == n_dim
#        with self: result_mesh = self(multiply(imagegrid(*size), n_batch))
#        # The closest grid point with all coordinates of its image smaller or equal than the position. 
#        flooring_grid = 1 / bt.zeros_like(result_mesh)
#        dim_ind = expand_to(bt.tensor(range(n_dim)), (n_batch, 1, n_dim, *size), 2)
#        cod_ind = multiply(Lceil(result_mesh), n_dim, 2)
#        cod_ind = clip(cod_ind, 0, expand_to(bt.tensor(size), cod_ind, 1) - 1)
#        put_nd(flooring_grid, bt.cat((dim_ind, cod_ind), 1), multiply(imagegrid(*size), n_batch))
#        unvisited = bt.isinf(flooring_grid)
#        for batch in range(n_batch):
#            unvisited_coord = imagegrid(*size)[unvisited[batch]].view(n_dim, -1).long()
#            for x in unvisited_coord.T:
#                anchor = flooring_grid[(batch, slice(None)) + tuple(i - 1 for i in x)]
#                possible_grid = unsqueeze(anchor, [-1] * n_dim) + imagegrid(*((3,) * n_dim))
#                possible_grid = clip(possible_grid, 0, expand_to(bt.tensor(size), possible_grid))
#                with self: possible_result = self(unsqueeze(possible_grid))
#                possible_result.squeeze(0) - unsqueeze(x, [-1] * n_dim)
#            
#        from zyctools import plot as plt
#        plt.subplot(121); plt.imshow(flooring_grid[0, 0]); plt.scatter(*unvisited_coord.split(1)[::-1])
#        plt.subplot(122); plt.imshow(flooring_grid[0, 1])
#        plt.show()
        n_dim = dimof(self.params) - 2
#        model = os.path.join(os.path.dirname(__file__), "model/DDF_inverse_%dD.model"%(n_dim))
#        if not os.path.exists(model): raise FileNotFoundError("No model {} is found. ".format(model))
        from zyctools.nn import U_Net
        unet = U_Net(
            depth = 2,
            dimension = n_dim,
            in_channels = n_dim,
            out_channels = n_dim,
            initializer = "normal(0, 0.01)",
            conv_block = "residual",
            block_channels = 8,
            with_softmax = False
        ).to(device)
#        unet.load_state_dict(bt.load(model, map_location=device))
        optimizer = Optim(bt.optim.Adam, unet.parameters(), lambda i: 10 ** (-i))
        unet.train()
        disp = self.params.detach()
        grid = imagegrid(*sizeof(disp)[2:]).unsqueeze(0)
        level = 2
        prev_value = 0
        while True:
            idisp = unet(disp)
#            print(info(idisp))
            trans = DDF(disp, with_batch = True)
            itrans = DDF(idisp, with_batch = True)
            rgrid = itrans(trans(grid))
            sgrid = trans(itrans(grid))
            loss = ((sgrid - grid) ** 2).mean()
            value = loss.item()
#            print(value, end = '\n\x1b[A')
            if value < 1e-2: break
            if abs(value - prev_value) < 1e-4: level += 1
            if level >= 5: break
            prev_value = value
#            level = max(1 - int(log10(value)), 2)
            opt = optimizer[level]
            opt.zero_grad()
            loss.backward(retain_graph = True)
            opt.step()
#        bt.save(unet.state_dict(), model)
        unet.eval()
        return DDF(unet(disp), self.with_batch)

    
    def __call__(self, X):
        params = self.params
        with_batch = self.with_batch
        n_batch = nbatch(params)
        n_dim = params.size(1)
        if not with_batch: X = X.unsqueeze(0).expand(n_batch, *X.size())
        if dimof(X) <= 2: X = X.unsqueeze(-1)
        if (X - unsqueeze(imagegrid(*sizeof(params)[2:]))).abs().max() < 1e-6: return X + params
        else:
            Y = bt.zeros_like(X)
            for b in range(n_batch):
                if (X[b] - imagegrid(*sizeof(params)[2:])).abs().max() < 1e-6: Y[b] = X[b] + params[b]
                else: Y[b] = X[b] + interpolation(params[b], None, target_space = multiply(X[b], n_dim)).view_as(X[b])
            return Y
        # interpolation(params.flatten(0, 1), None, target_space = repeatEnlarge(X, (n_dim,) + (1,) * (dimof(X) - 1))).view_as(X)
# [Deprecated!]
#        X = X.transpose(0, 1).flatten(2).long()
#        indices = bt.cat((bt.arange(n_batch).to(device).view(1, n_batch, 1).expand(1, n_batch, Y.size(2)), Y))
#        Y = X + bt.stack([gather_nd(params[:, i], indices) for i in range(n_dims)]).transpose(0, 1).view_as(X)
#        return Y

#params = bt.rand(5, 2, 4, 5)
#print(params)
#print(DDFTransformation(bt.tensor([1, 2.1]).float(), params))

class FFD(Transformation):
    '''
    FFD transformation.

    X: bt.Tensor
        Coordinates to be transformed.
        size: (n_dims, n_1, n_2, ..., n_r) if not with_batch
            or (n_batch, n_dims, n_1, n_2, ..., n_r) if with_batch
    parameters: bt.Tensor
        Parameters for the transformation.
        The coordinates for the control points.
        size: For a m_1 x m_2 x ... x m_{n_dims} grid of Δcontrol points,
        it appears to be of size (n_batch, n_dims, m_1, m_2, ..., m_{n_dims}).

    output: bt.Tensor
        The transformed coordinates.
        size: (n_batch, n_dims, n_1, n_2, ..., n_r)
    '''
    def __init__(self, params, spacing=1, FFD_spacing=1, FFD_origin=0, with_batch=False):
        spacing = to_tuple(spacing)
        FFD_spacing = to_tuple(FFD_spacing)
        FFD_origin = to_tuple(FFD_origin)
        if len(spacing) == 1: spacing *= len(FFD_spacing)
        if len(FFD_spacing) == 1: FFD_spacing *= len(spacing)
        n_dims = params.size(1)
        assert dimof(params) == n_dims + 2
        if len(spacing) == 1: spacing *= n_dims; FFD_spacing *= n_dims
        assert len(spacing) == len(FFD_spacing) == n_dims
        if len(FFD_origin) == 1: FFD_origin *= n_dims
        assert len(FFD_origin) == n_dims
        self.n_dims = n_dims
        self.params = params
        self.spacing = spacing
        self.FFD_spacing = FFD_spacing
        self.FFD_origin = FFD_origin
        self.with_batch = with_batch

    
    def __call__(self, X):
        n_dims = self.n_dims
        params = self.params
        spacing = self.spacing
        FFD_spacing = self.FFD_spacing
        with_batch = self.with_batch
        n_batch = nbatch(params)
        # Coord: size = (n_batch or 1, s, n_1 * ... * n_r)
        Coord = X.view(1 if not with_batch else n_batch, n_dims, -1) # Simplify X.
        if not with_batch:
            assert nbatch(Coord) == 1
            Coord = Coord.repeat(n_batch, 1, 1)
        Coord -= expand_to(bt.tensor(self.FFD_origin), Coord, 1)
        tool = None
        if n_dims == 3: tool = FFD.FFD3D
        elif n_dims == 2: tool = FFD.FFD2D
        if tool:
            result = tool(Coord, params, spacing, FFD_spacing, with_batch)
            result = result.view((n_batch,) + (X.size()[1:] if with_batch else X.size()))
            result += expand_to(bt.tensor(self.FFD_origin), result, 1)
            return result

    @staticmethod
    
    def term3D(a, b, c, ux, uy, uz, ix, iy, iz, params):
        # i(u)x(y, z): (n_batch, 1, n_data)
        mx, my, mz = params.size()[2:]
        x, y, z = ix + a, iy + b, iz + c
        # params[b * Fones_like(x[b]), :, x[b], y[b], z[b]]
        # phi[b, d, k] = params[b, d, (x[b][k] * my + y[b][k]) * mz + z[b][k]]
        ind = repeat((x * my + y) * mz + z, axis = 1, repeats = 3)
        ind = Lclamp(ind, 0, mx * my * mz - 1)
        phi = params.flatten(2).gather(2, ind)
        return (
            Fto(0 <= x) * Fto(x < mx) * Bspline(a * Fones_like(ux), ux) * 
            Fto(0 <= y) * Fto(y < my) * Bspline(b * Fones_like(uy), uy) * 
            Fto(0 <= z) * Fto(z < mz) * Bspline(c * Fones_like(uz), uz) * phi
        )

    @staticmethod
    
    def FFD3D(X, params, spacing, FFD_spacing, with_batch=False):
        n_batch = nbatch(params) # params: (n_batch, n_dims=3, mx, my, mz)
        n_data = X.size(-1) # X: (n_batch, n_dims=3, n_data)
        FFDX = X * unsqueeze(Ftensor(spacing) / Ftensor(FFD_spacing), [0, -1])
        iX = Ffloor(FFDX); uX = FFDX - iX

        args = uX.split(1, 1) + iX.split(1, 1) + (params,)
        delta = sum([
            FFD.term3D(-1, -1, -1, *args), FFD.term3D(-1, -1, 0, *args),
            FFD.term3D(-1, -1, 1, *args),  FFD.term3D(-1, -1, 2, *args),
            FFD.term3D(-1, 0, -1, *args),  FFD.term3D(-1, 0, 0, *args),
            FFD.term3D(-1, 0, 1, *args),   FFD.term3D(-1, 0, 2, *args),
            FFD.term3D(-1, 1, -1, *args),  FFD.term3D(-1, 1, 0, *args),
            FFD.term3D(-1, 1, 1, *args),   FFD.term3D(-1, 1, 2, *args),
            FFD.term3D(-1, 2, -1, *args),  FFD.term3D(-1, 2, 0, *args),
            FFD.term3D(-1, 2, 1, *args),   FFD.term3D(-1, 2, 2, *args),
            FFD.term3D(0, -1, -1, *args),  FFD.term3D(0, -1, 0, *args),
            FFD.term3D(0, -1, 1, *args),   FFD.term3D(0, -1, 2, *args),
            FFD.term3D(0, 0, -1, *args),   FFD.term3D(0, 0, 0, *args),
            FFD.term3D(0, 0, 1, *args),    FFD.term3D(0, 0, 2, *args),
            FFD.term3D(0, 1, -1, *args),   FFD.term3D(0, 1, 0, *args),
            FFD.term3D(0, 1, 1, *args),    FFD.term3D(0, 1, 2, *args),
            FFD.term3D(0, 2, -1, *args),   FFD.term3D(0, 2, 0, *args),
            FFD.term3D(0, 2, 1, *args),    FFD.term3D(0, 2, 2, *args),
            FFD.term3D(1, -1, -1, *args),  FFD.term3D(1, -1, 0, *args),
            FFD.term3D(1, -1, 1, *args),   FFD.term3D(1, -1, 2, *args),
            FFD.term3D(1, 0, -1, *args),   FFD.term3D(1, 0, 0, *args),
            FFD.term3D(1, 0, 1, *args),    FFD.term3D(1, 0, 2, *args),
            FFD.term3D(1, 1, -1, *args),   FFD.term3D(1, 1, 0, *args),
            FFD.term3D(1, 1, 1, *args),    FFD.term3D(1, 1, 2, *args),
            FFD.term3D(1, 2, -1, *args),   FFD.term3D(1, 2, 0, *args),
            FFD.term3D(1, 2, 1, *args),    FFD.term3D(1, 2, 2, *args),
            FFD.term3D(2, -1, -1, *args),  FFD.term3D(2, -1, 0, *args),
            FFD.term3D(2, -1, 1, *args),   FFD.term3D(2, -1, 2, *args),
            FFD.term3D(2, 0, -1, *args),   FFD.term3D(2, 0, 0, *args),
            FFD.term3D(2, 0, 1, *args),    FFD.term3D(2, 0, 2, *args),
            FFD.term3D(2, 1, -1, *args),   FFD.term3D(2, 1, 0, *args),
            FFD.term3D(2, 1, 1, *args),    FFD.term3D(2, 1, 2, *args),
            FFD.term3D(2, 2, -1, *args),   FFD.term3D(2, 2, 0, *args),
            FFD.term3D(2, 2, 1, *args),    FFD.term3D(2, 2, 2, *args),
        ])
        return delta / unsqueeze_to(Ftensor(spacing), 3, 1) + X

    @staticmethod
    
    def term2D(a, b, ux, uy, ix, iy, params):
        # i(u)x(y, z): (n_batch, 1, n_data)
        mx, my = params.size()[2:]
        x, y = ix + a, iy + b
        # params[b * Fones_like(x[b]), :, x[b], y[b], z[b]]
        # phi[b, d, k] = params[b, d, (x[b][k] * my + y[b][k]) * mz + z[b][k]]
        ind = repeat(x * my + y, axis = 1, repeats = 2)
        ind = Lclamp(ind, 0, mx * my - 1)
        phi = params.flatten(2).gather(2, ind)
        return (
            Fto(0 <= x) * Fto(x < mx) * Bspline(a * Fones_like(ux), ux) * 
            Fto(0 <= y) * Fto(y < my) * Bspline(b * Fones_like(uy), uy) * phi
        )

    @staticmethod
    
    def FFD2D(X, params, spacing, FFD_spacing, with_batch=False):
        n_batch = nbatch(params) # params: (n_batch, n_dims=2, mx, my)
        n_data = X.size(-1) # X: (n_batch, n_dims=2, n_data)
        FFDX = X * unsqueeze(Ftensor(spacing) / Ftensor(FFD_spacing), [0, -1])
        iX = Ffloor(FFDX); uX = FFDX - iX

        args = uX.split(1, 1) + iX.split(1, 1) + (params,)
        delta = sum([
            FFD.term2D(-1, -1, *args), FFD.term2D(-1, 0, *args),
            FFD.term2D(-1, 1, *args),  FFD.term2D(-1, 2, *args),
            FFD.term2D(0, -1, *args),  FFD.term2D(0, 0, *args),
            FFD.term2D(0, 1, *args),   FFD.term2D(0, 2, *args),
            FFD.term2D(1, -1, *args),  FFD.term2D(1, 0, *args),
            FFD.term2D(1, 1, *args),   FFD.term2D(1, 2, *args),
            FFD.term2D(2, -1, *args),  FFD.term2D(2, 0, *args),
            FFD.term2D(2, 1, *args),   FFD.term2D(2, 2, *args),
        ])
        return delta / unsqueeze_to(Ftensor(spacing), 3, 1) + X

# [Deprecated!]
# 
# def FFDTransformation(
#         X, 
#         params, 
#         image_spacing=(1, 1), 
#         FFD_spacing=(1, 1),
#         image_size=(IMG_SIZE, IMG_SIZE), 
#         with_batch=False
#     ):
#     n_dim = params.size(1) # Get spatial rank s and name it as n_dim.
#     n_batch = params.size(0) # Get batch size.
#     # Coord: size = (n_batch or 1, s, n_1 * ... * n_r)
#     Coord = X.view(1 if not with_batch else n_batch, n_dim, -1) # Simplify X.
#     if not with_batch: Coord = Coord.repeat(n_batch, 1, 1)
#     n_data = Coord.size(-1)
#     # Normalize X in the domain (m_1, m_2, ..., m_s).
#     M = Ltensor(params.size()[2:])
#     Coord *= (Ltensor(image_spacing) / Ltensor(FFD_spacing)).unsqueeze(0).unsqueeze(-1)
#     # Coord = Coord * (M.view(n_dim, 1).float() - bt.ones(n_dim, 1).to(device)) / bt.tensor(image_size).float().to(device).view(n_dim, 1)
#     # Compute the integer and decimal part of the coordinates
#     iCoord = bt.floor(Coord)
#     dCoord = Coord - iCoord
#     # Compute the weights. W: 4 x n_batch x s x (n_1 * ... * n_r)
#     i = T(bt.arange(-1, 3)).view(4, 1, 1, 1).expand(4, n_batch, n_dim, n_data)
#     W = Bspline(i, dCoord.unsqueeze(0).repeat(4, 1, 1, 1))
#     "Compute FFD Transformation"
#     output = bt.zeros(n_batch, n_dim, n_data).to(device)
#     # Loop in the space {-1, 0, 1, 2} ^ s
#     for G in bt.stack(bt.meshgrid((bt.arange(-1, 3),) * n_dim)).view(n_dim, -1).transpose(0, 1).long().to(device):
#         # Weights for each point: [product of W[G[D], t, D, x] for D in range(n_dim)] for point x and batch t.
#         # Wg: n_batch x 1 x (n_1 * ... * n_r)
#         Wg = W.gather(0, G.view(1, 1, n_dim, 1) + bt.ones(1, *W.size()[1:]).long().to(device))[0].prod(1, keepdim=True)
#         # Compute the indices of related control points. Ind: n_batch x n_dim x (n_1 * ... * n_r)
#         Ind = bt.clamp(iCoord.long() + G.view(1, n_dim, 1), min=0)
#         Ind = bt.min(Ind, (M.view(1, n_dim, 1) - bt.ones(1, n_dim, 1).long().to(device)))
#         # Convert the indices to 1 dimensional. Dot: n_batch x (n_1 * ... * n_r)
#         Dot = Ind[:, 0]
#         for r in range(1, n_dim): Dot *= M[r]; Dot += Ind[:, r]
#         # Obtain the coordinates of the control points. CPoints: n_batch x s x (n_1 * ... * n_r)
#         CPoints = params.flatten(2).gather(2, Dot.unsqueeze(1).repeat(1, n_dim, 1)).float()
#         # Add the weighted control coordinates to the output coordinates.
#         output += Wg * CPoints.float()
#     # Denormalize the outputs.
#     output += X.view(1, n_dim, n_data)
#     if with_batch: output = output.view_as(X)
#     else: output = output.view(n_batch, *X.size())
#     return output

# 
# def LocallyAffineTransformation2D(X, params, image_size=(IMG_SIZE, IMG_SIZE), translation_coefficient=TRANSC):
#     '''
#     Locally Affine transformation with respect to transformation parameters.

#     params: bt.Tensor
#         n_batch pairs of parameters with 7 for each transformation.
#         size: (n_batch, n_grid, n_grid, 7)
#     X: bt.Tensor
#         Coordinates to be transformed.
#         size: (2, n_1, n_2, ..., n_r)

#     output: bt.Tensor
#         The transformed coordinates.
#         size: (n_batch, 2, n_1, n_2, ..., n_r)
#     '''
#     # Flatten X and params.
#     Xflatten = X.view(2, -1)
#     n_batch, n_grid, _, _ = params.size()
#     Pflatten = params.view(n_batch, n_grid * n_grid, 7)
#     # Compute the local affine in use for each x. place.size() = (2, N)
#     place = bt.clamp(bt.floor(Xflatten * n_grid / bt.tensor(image_size).float().to(device).view(2, -1)), 0, n_grid - 1).long()
#     # Compute the index of this local affine in the flattened parameters. size=(N,)
#     place = place[0] * n_grid + place[1]
#     # Compute the parameters for each point x:
#     # Params_x[batch][i][k] = Pflatten[batch][place[i]][k]; size = (n_batch, N, 7)
#     params_x = Pflatten.gather(1, place.view(1, -1, 1).expand(n_batch, Xflatten.size(1), 7))
#     # split the parameters, each of the following parameter is of size (n_batch, N, 1)
#     t1, t2, θ, s1, s2, ρ1, ρ2 = bt.split(params_x, 1, -1)
#     a = (ρ1 * ρ2 + 1) * s1 * bt.cos(θ) + ρ1 * s2 * bt.sin(θ)
#     b = - (ρ1 * ρ2 + 1) * s1 * bt.sin(θ) + ρ1 * s2 * bt.cos(θ)
#     c = ρ2 * s1 * bt.cos(θ) + s2 * bt.sin(θ)
#     d = - ρ2 * s1 * bt.sin(θ) + s2 * bt.cos(θ)
#     # calculate the two components of the transformed coordinates. p.size() = (n_batch, N, 1)
#     p = bt.sum(bt.cat((a, b), -1) * Xflatten.unsqueeze(0).transpose(1, 2), -1, keepdim=True) + translation_coefficient * t1
#     q = bt.sum(bt.cat((c, d), -1) * Xflatten.unsqueeze(0).transpose(1, 2), -1, keepdim=True) + translation_coefficient * t2
#     return bt.cat((p, q), -1).transpose(1, 2).view(n_batch, *X.size())

class PolyAffine(Transformation):
    '''
    Poly affine transformation with respect to transformation matrices.

    `X`: `bt.Tensor`
        Coordinates to be transformed.
        size: `(n_dims, n_1, n_2, ..., n_r)`
    `params`: `bt.Tensor`
        An affine matrix for each region. 
        size: `(n_batch, n_region, n_dims + 1, n_dims + 1)`
    `masks`: `bt.Tensor`
        A 0-1 mask for each region. 
        size: `(n_batch, n_region, n_1, n_2, ..., n_r)

    `output`: `bt.Tensor`
        The transformed coordinates.
        size: `(n_batch, n_dims, n_1, n_2, ..., n_r)`
    '''
    def __init__(self, params, masks, translation_coefficient=None, with_batch=False):
        try:
            import SimpleITK as sitk
        except: raise AssertionError("Please install package SimpleITK before using Poly Affine. ")
        # preprocess masks
        masks = tonumpy(masks).astype(np.int)
        n_batch = masks.shape[0]
        n_region = masks.shape[1]
        size = masks.shape[2:]
        _dis_map = bt.zeros(*masks.shape)
        for i in range(n_batch):
            for j in range(n_region):
                mask_image = sitk.GetImageFromArray(masks[i, j], isVector = False)
                dis_map = sitk.GetArrayViewFromImage(sitk.SignedMaurerDistanceMap(mask_image, squaredDistance = False, useImageSpacing = False))
                # from matplotlib import pyplot as plt
                # plt.imshow(dis_map * (dis_map < 0).astype(np.float)); plt.show()
        #         _dis_map[i, j] = Ftensor(-dis_map * (dis_map > 0).astype(np.float))
        # max_dis_map = _dis_map.max(1).values
        # for j in range(n_region):
        #     lower = 2 * _dis_map[:, j][max_dis_map == _dis_map[:, j]].view(n_batch, -1).min(1).values
        #     _dis_map[:, j] = bt.min(bt.max(_dis_map[:, j], lower), bt.tensor(0.)) / lower
        # exp_dis_map = bt.exp(10 * _dis_map)
        # weights = exp_dis_map / exp_dis_map.sum(1, keepdim = True)
                _dis_map[i, j] = Ftensor(dis_map * (dis_map > 0).astype(np.float))
        k = 2
        invpowk_dis_map = 1 / (_dis_map ** k + 1e-5)
        sum_dis_map = invpowk_dis_map.sum(1, keepdim = True)
        weights = invpowk_dis_map / sum_dis_map
        # from matplotlib import pyplot as plt
        # plt.subplot(121); plt.imshow(weights[0, 0])
        # plt.subplot(122); plt.imshow(weights[0, 1])
        # plt.show()
        self.params = params
        self.weights = weights
        self.translation_coefficient = translation_coefficient
        self.with_batch = with_batch
        self._inv = False

    
    def __call__(self, X):
        params = self.params
        weights = self.weights
        translation_coefficient = self.translation_coefficient
        with_batch = self.with_batch
        if not translation_coefficient:
            translation_coefficient = 1 # max(X.size()[1:]) / 5
        n_batch = nbatch(params)
        n_dim = params.size(-1) - 1
        n_region = params.size(1)
        if not with_batch: X = multiply(X, n_batch)
        Xs = multiply(X, n_region, 1)
        A = params[..., :n_dim, :n_dim]
        b = params[..., :n_dim, n_dim:] * translation_coefficient
        Y = (A @ Xs.flatten(3) + b).view_as(Xs)
        D = (Y * weights.unsqueeze(2)).sum(1) - X
        if self._inv: D = -D
        trans = DDF(D, with_batch = True)
        trans2 = DDF(trans(trans(X)) - X, with_batch = True)
        trans4 = DDF(trans2(trans2(X)) - X, with_batch = True)
        trans8 = DDF(trans4(trans4(X)) - X, with_batch = True)
        trans16 = DDF(trans8(trans8(X)) - X, with_batch = True)
        trans32 = DDF(trans16(trans16(X)) - X, with_batch = True)
        trans64 = DDF(trans32(trans32(X)) - X, with_batch = True)
        return trans64(X)

    def inv(self): self._inv = not self._inv; return self


def SumSquaredDifference(A, B): return ((A - B) ** 2).flatten(1).sum(1)
  
def MeanSquaredErrors(A, B): return ((A - B) ** 2).flatten(1).mean(1)

def LocalLinearDifference(A, B, truncated = True, kernel = "Gaussian", kernel_size = 3, dilation = 1, stride = 1):
    n_dim = dimof(A) - 1
    if isinstance(kernel, str):
        if kernel.lower() == "gaussian": kernel = unsqueeze(toFtensor(gaussian_kernel(n_dims = n_dim, kernel_size = kernel_size)), [0, 0])
        elif kernel.lower() == "mean": kernel = unsqueeze(Fones(*(kernel_size,) * n_dim), [0, 0])
    elif isarray(kernel): kernel_size = sizeof(kernel)[-1]
    blur = lambda a: eval("bt.nn.functional.conv%dd"%n_dim)(unsqueeze(a), kernel, padding = kernel_size // 2).squeeze(0)
    mean = lambda a: eval("bt.nn.functional.conv%dd"%n_dim)(unsqueeze(a), kernel * kernel.numel(), 
        padding = ((kernel_size - 1) * dilation + 1) // 2, dilation = dilation, stride = stride).squeeze(0)
    with bt.no_grad():
        A_ = mean(A)
        B_ = mean(B)
        AA_ = mean(A * A)
        AB_ = mean(A * B)
        BB_ = mean(B * B)
        N1 = A_*BB_-B_*AB_
        N2 = B_*AA_-A_*AB_
        R = AA_*BB_-AB_**2
        if truncated:
            tN1 = bt.where(N1.abs() < N2.abs() / 5, N2.abs() / 2 * bt.randn(N2.size()).sign(), N1)
            tN2 = bt.where(N2.abs() < N1.abs() / 5, N1.abs() / 2 * bt.randn(N1.size()).sign(), N2)
        else: tN1 = N1; tN2 = N2
    rec = lambda x: crop_as(repeatEnlarge(x, (1,) + (stride,) * n_dim), A)
    tN1 = blur(rec(tN1))
    tN2 = blur(rec(tN2))
    R = blur(rec(R))
    divide = lambda a, b, s: bt.where(b.abs() < 1e-4, s * Fzeros_like(a), a / b)
    from zyctools import plot as plt
    plt.subplot(151); plt.sliceshow(A)
    plt.subplot(152); plt.sliceshow(B)
    plt.subplot(153); plt.sliceshow(divide(tN2 * B - R, -tN1, 0.0).clamp(0, 1))
    plt.subplot(154); plt.sliceshow(tN1 ** 2 + tN2 ** 2)
    plt.subplot(155); plt.sliceshow(divide((tN1 * A + tN2 * B - R) ** 2, tN1 ** 2 + tN2 ** 2, 0.0))
    plt.show()
    return divide((tN1 * A + tN2 * B - R) ** 2, tN1 ** 2 + tN2 ** 2, 0.0).flatten(1).mean(1)

def LocalCrossCorrelation(A, B, kernel = "Gaussian", kernel_size = 3):
    n_dim = dimof(A) - 1
    if isinstance(kernel, str):
        if kernel.lower() == "gaussian": kernel = unsqueeze(toFtensor(gaussian_kernel(n_dims = n_dim, kernel_size = kernel_size)), [0, 0])
        elif kernel.lower() == "mean": kernel = unsqueeze(Fones(*(kernel_size,) * n_dim), [0, 0])
    elif isarray(kernel): kernel_size = sizeof(kernel)[-1]
    mean = lambda a: eval("bt.nn.functional.conv%dd"%n_dim)(unsqueeze(a), kernel, padding = kernel_size // 2).squeeze(0)
    divide = lambda a, b, s: bt.where(b.abs() < 1e-6, s * Fones_like(a), a / b.clamp(min=1e-6))
    A_ = mean(A); B_ = mean(B)
    cc = (divide((mean(A * B) - A_ * B_) ** 2, (mean(A * A) - A_ ** 2) * (mean(B * B) - B_ ** 2), 1.0 - (A_ - B_) ** 2) + 1e-6).sqrt()
    return cc.flatten(1).mean(1)

def CrossEntropy(y, label): return -bt.sum(label * bt.log(y.clamp(1e-10, 1.0)), 1).flatten(1).mean(1)

def CrossCorrelation(A, B):
    A = A.flatten(1).float(); B = B.flatten(1).float()
    dA = A - expand_to(A.mean(1), A); dB = B - expand_to(A.mean(1), B)
    return (dA * dB).sum(1)

def NormalizedCrossCorrelation(A, B):
    A = A.flatten(1).float(); B = B.flatten(1).float()
    dA = A - expand_to(A.mean(1), A); dB = B - expand_to(A.mean(1), B)
    return (dA * dB).sum(1) / bt.sqrt(bt.sum(dA ** 2, 1)) / bt.sqrt(bt.sum(dB ** 2, 1))

def Dice(A, B, multi_label = False):
    '''
    if multi_label:
        A: (n_batch, n_label, n_1, n_2, ..., n_k)
        B: (n_batch, n_label, n_1, n_2, ..., n_k)
        return: (n_batch, n_label)
    else:
        A: (n_batch, n_1, n_2, ..., n_k)
        B: (n_batch, n_1, n_2, ..., n_k)
        return: (n_batch,)
    '''
    eps = 1e-8
#    assert bt.sum(A * (1 - A)).abs().item() < eps and bt.sum(B * (1 - B)).abs().item() < eps
    if not multi_label: A = A.unsqueeze(1); B = B.unsqueeze(1)
    A = toFtensor(A).flatten(2); B = toFtensor(B).flatten(2)
    ABsum = A.sum(-1) + B.sum(-1)
    dice = 2 * bt.sum(A * B, -1) / (ABsum + eps)
    return dice

def LabelDice(A, B, class_labels):
    '''
    :param A: (n_batch, n_1, ..., n_k)
    :param B: (n_batch, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    A = bt.tensor(A)
    B = bt.tensor(B)
    n_batch = nbatch(A)
    n_class = len(class_labels)
    A_labels = [1 - bt.clamp(bt.abs(A - i), 0, 1) for i in class_labels]
    B_labels = [1 - bt.clamp(bt.abs(B - i), 0, 1) for i in class_labels]
    A_maps = bt.cat(A_labels)
    B_maps = bt.cat(B_labels)
    dice = Dice(A_maps, B_maps)
    return dice.view(n_class, n_batch).transpose(0, 1)

if __name__ == "zyctools.image":
    from zyctools.miloss import MutualInformation
    from zyctools.miloss import NormalizedMutualInformation
    
    def ImageLoss(trans, I1, I2, ltype = 'MSE', **kwargs):
        if isinstance(trans, Identity) or "MI" in ltype: I2T = I2
        else: I2T = interpolation(I2, trans, toSize = sizeof(I1)[1:])
        if ltype.upper() == "SSD": result = SumSquaredDifference(I1, I2T)
        elif ltype.upper() == "MSE": result = MeanSquaredErrors(I1, I2T)
        elif ltype.upper() == "LLD": result = LocalLinearDifference(I1, I2T, truncated = False)
        elif ltype.upper() == "TLLD": result = LocalLinearDifference(I1, I2T)
        elif ltype.upper() == "LCC": result = - LocalCrossCorrelation(I1, I2T)
        elif ltype.upper() == "CC": result = - CrossCorrelation(I1, I2T)
        elif ltype.upper() == "NCC": result = - NormalizedCrossCorrelation(I1, I2T)
        elif ltype.upper() == "DICE": result = - LabelDice(I1, I2T, sorted(list(set(tonumpy(I1).flatten())))).mean(1)
        elif ltype.upper() == "MI": result = - MutualInformation(trans, I1, I2, nbins=kwargs.get('nbins', 100))
        elif ltype.upper() == "NMI": result = - NormalizedMutualInformation(trans, I1, I2, nbins=kwargs.get('nbins', 100))
        elif ltype.upper() == "NMI-1": result = - NormalizedMutualInformation(trans, I1, I2, nbins=kwargs.get('nbins', 100)) + 1
        else: raise TypeError("Unknown ltype[%s] for function ImageLoss. "%ltype)
        loss = result.mean()
        value = loss.detach().item() * (-1 if ltype.upper() not in ("SSD", "MSE", "LLD", "TLLD") else 1)
        return loss, value
    
@restore_type_wrapper
def bending(weights):
    '''
        Bending energy for weights
        weights: (n_batch, n_feature, n_1, ..., n_{n_dim})
    '''
    weights = bt.tensor(weights)
    n_feature = weights.size(1)
    n_dim = dimof(weights) - 2
    #### delete the axis that has a size smaller than 3
    todel = 0
    axis_record = bt.tensor([])
    for i, l in enumerate(sizeof(weights)[2:]):
        if l >= 3: continue
        weights = weights.unsqueeze(0).transpose(0, 3 + i + todel)
        axis_record += 1
        axis_record = bt.cat((bt.tensor([3 + i + todel]), axis_record))
        todel += 1
    n_dim -= todel
    size_record = sizeof(weights)
    weights = weights.flatten(0, todel)
    weights = weights[tuple(slice(None) if i < 2 or l > 1 else 0 for i, l in enumerate(sizeof(weights)))]
    ###
    kernel = Fzeros(*((3,) * n_dim))
    conv = eval("bt.nn.functional.conv%dd" % n_dim)
    wsize = sizeof(weights)[:2] + tuple(l - 2 for l in sizeof(weights)[2:])
    laplace = bt.tensor([])
    for d in range(n_dim):
        for e in range(d, n_dim):
            tmpkernel = kernel.clone()
            if d == e: tmpkernel[(1,) * d + (slice(None),) + (1,) * (n_dim - d - 1)] = Ftensor([1, -2, 1])
            else:
                tmpkernel[(1,) * d + (slice(None),) + (1,) * (e - d - 1) + (slice(None),) + (1,) * (n_dim - e - 1)] = Ftensor([
                    [ 1/2, 0, -1/2],
                    [   0, 0,    0],
                    [-1/2, 0,  1/2]
                ])
            pdpe = bt.sqrt((conv(weights.flatten(0, 1).unsqueeze(1), unsqueeze(tmpkernel, [0, 1]), padding=0).view(*wsize) ** 2).sum(1, keepdim=True) + 1e-6)
            laplace = bt.cat((laplace, pdpe), 1)
    return laplace.mean()

def grad(array):
    '''
        Gradient image of array
        array: (n_batch, n_feature, n_1, ..., n_{n_dim})
        output: (n_batch, n_dim, n_feature, n_1, ..., n_{n_dim})
    '''
    n_dim = dimof(array) - 2
    output = astype(np.zeros_like(tonumpy(array)), array)
    output = multiply(output, n_dim, 1)
    for d in range(n_dim):
        b = (slice(None, None),) * (d+2) + (slice(4, None),) + (slice(None, None),) * (n_dim-d-1)
        a = (slice(None, None),) * (d+2) + (slice(None, -4),) + (slice(None, None),) * (n_dim-d-1)
        output[:, d] = crop_as((array[b] - array[a]) / 4, array)
    return crop_as(output, tuple(x if i<3 else x - 2 for i, x in enumerate(sizeof(output))))

def MeanLocalRigidity(displacement, mask = None):
    '''
        displacement: (n_batch, n_feature, n_1, ..., n_{n_dim})
        mask: (n_batch, n_1, ..., n_{n_dim})
    '''
    n_batch = nbatch(displacement)
    if mask is None: mask = bt.ones_like(n_batch, *sizeof(displacement)[2:])
    mask = toFtensor(mask)
    
    # G = imagegrid(*mask.shape[1:]).unsqueeze(0)
    # X = (G * mask.unsqueeze(1)).flatten(2)
    # Y = ((G + displacement) * mask.unsqueeze(1)).flatten(2)
    # x = X.sum(2, keepdim=True)
    # y = Y.sum(2, keepdim=True)
    # n = mask.flatten(1).sum(1, keepdim=True).unsqueeze(-1)
    # cA = X @ T(X) - (x @ T(x)) / n
    # A = (Y @ T(X) - (y @ T(x)) / n) @ bt.inverse(cA)
    # return Fnorm2(T(A) @ A - eye(A))
    
    gd = grad(displacement)
    JacOfDelta = join_to(gd.flatten(3).transpose(1, 2), [0, 3], 0)
    JacOfPoints = JacOfDelta + eye(JacOfDelta)
    RigOfPoints = Fnorm2(T(JacOfPoints) @ JacOfPoints - eye(JacOfPoints)).view(n_batch, -1)
    return (RigOfPoints * crop_as(mask, sizeof(gd)[3:], n_keepdim=1).flatten(1)).clamp(0, 1).sum(1) / mask.flatten(1).sum(1)

    # DetOfPoints = bt.det(JacOfPoints + eye(JacOfPoints)).view(n_batch, -1)
    # return (DetOfPoints * crop_as(mask, sizeof(gd)[3:], n_keepdim=1).flatten(1)).sum(1) / mask.flatten(1).sum(1)

#    if n_dim == 1:
#        kernel = Ftensor([1, -2, 1])
#        conv = bt.nn.functional.conv1d
#    elif n_dim == 2:
#        kernel = Ftensor([
#            [1/2, 1, -1/2],
#            [1, -4, 1],
#            [-1/2, 1, 1/2]
#        ])
#        conv = bt.nn.functional.conv2d
#    elif n_dim == 3:
#        kernel = Fzeros(3, 3, 3)
#        kernel[0] = Ftensor([
#            [0, 1/2, 0],
#            [1/2, 1, -1/2],
#            [0, -1/2, 0]
#        ])
#        kernel[1] = Ftensor([
#            [1/2, 1, -1/2],
#            [1, -6, 1],
#            [-1/2, 1, 1/2]
#        ])
#        kernel[2] = Ftensor([
#            [0, -1/2, 0],
#            [-1/2, 1, 1/2],
#            [0, 1/2, 0]
#        ])
#        conv = bt.nn.functional.conv3d
#    wsize = sizeof(weights)[:2] + tuple(l - 2 for l in sizeof(weights)[2:])
#    laps_image = bt.sqrt((conv(weights.flatten(0, 1).unsqueeze(1), unsqueeze(kernel, [0, 1]), padding=0).view(*wsize) ** 2).sum(1, keepdim=True))
#    return bt.abs(laps_image.squeeze(1)).mean(0).sum()
    
def crossMatrix(axis):
    '''
    axis: (n_batch, n_dim * (n_dim - 1) / 2)
    output: (n_batch, n_dim, n_dim)
    '''
    n_batch = nbatch(axis)
    n_dim = int(sqrt(2 * axis.size(1))) + 1
    output = Fzeros(n_batch, n_dim, n_dim)
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
    
def uncross(cross_matrix):
    '''
    axis: (n_batch, n_dim, n_dim)
    output: (n_batch, n_dim * (n_dim - 1) / 2)
    '''
    n_batch = nbatch(cross_matrix)
    n_dim = cross_matrix.size(1)
    axis = Fzeros(n_batch, n_dim * (n_dim - 1) // 2)
    if n_dim == 2:
        axis[:, 0] = cross_matrix[:, 0, 1]
    elif n_dim == 3:
        axis[:, 0] = cross_matrix[:, 1, 2]
        axis[:, 1] = cross_matrix[:, 2, 0]
        axis[:, 2] = cross_matrix[:, 0, 1]
    return axis
    
def expCross_minus_I(axis):
    eps = 1e-6
    theta = bt.sqrt((axis ** 2).sum(1))
    c1 = bt.where(bt.abs(theta) < eps, bt.ones_like(theta) / 2, (1 - bt.cos(theta)) / (theta ** 2))
    c2 = bt.where(bt.abs(theta) < eps, bt.ones_like(theta), bt.sin(theta) / theta)
    wx = crossMatrix(axis)
    return expand_to(c1, wx) * (wx @ wx) + expand_to(c2, wx) * wx
    
def Fnorm2(X):
    '''
    X: (n_batch, m, n)
    return tr(XX^T) / m / n
    '''
    return (X ** 2).flatten(1).mean(-1).clamp(min = 0)
    
def norm_p(X, p = 2):
    '''||X||p'''
    return ((X ** p).flatten(1).sum(-1).clamp(min = 0)) ** (1/p)
    
def Wnorm2(X, w):
    '''
    X: (n_batch, m, n)
    w >= 0, (n_batch, n)
    return tr(XWX^T) / m / n where W = diag(w)
    '''
    Y = X * w.unsqueeze(1)
    return (Y * X).flatten(1).mean(-1).clamp(min = 0)
    
def mask_encode(X, w):
    '''
    X: (n_batch, m, n_1, ..., n_r)
    w >= 0, (n_batch, n_1, ..., n_r)
    return X W^{1/2}(I - (W^{1/2} 1 1^T W^{1/2}) / (1^T W 1))
    '''
    sw = bt.sqrt(w.unsqueeze(1).flatten(2))
    Y = X.flatten(2) * sw
    Z = (Y * sw).sum(-1, keepdim = True) * sw
    return (Y - divide(Z, unsqueeze(w.flatten(1).sum(-1), [-1, -1]), 0.0)).view_as(X)
    
def QXT(X, w):
    '''
    X: (n_batch, m, n)
    w >= 0, (n_batch, n)
    return QX^T where Q = diag(w) - ww^T/∑w = W - (W11^TW)/(1^TW1)
    '''
    Y = w.unsqueeze(-1) * X.transpose(1, 2)
    return Y - w.unsqueeze(-1) @ Y.sum(1, keepdim = True) / unsqueeze(w.sum(-1), [-1, -1])
    
def Qnorm2(X, w):
    '''
    X: (n_batch, m, n)
    w >= 0, (n_batch, n)
    return tr(XQX^T) / m / n where Q = diag(w) - ww^T/∑w
    '''
    Y = X - (X * w.unsqueeze(1)).sum(-1, keepdim = True) / unsqueeze(w.sum(-1), [-1, -1])
    return Wnorm2(Y, w)
# [deprecated]
#    _, n_dim, n_data = X.size()
#    Y = X @ w.unsqueeze(-1)
#    return bt.clamp(Wnorm(X, w) - (Y ** 2).flatten(1).sum(-1) / w.sum(-1) / n_dim / n_data, min = 0)

def center(mask):
    '''
    mask: (n_batch, n_1, n_2, ..., n_{n_dim})
    output: (n_batch, n_dim)
    '''
    return (unsqueeze(imagegrid(*sizeof(mask)[1:])) * unsqueeze(mask, 1)).flatten(2).sum(-1) / mask.flatten(1).sum(-1, keepdim = True)
    
def mask2surface(mask):
    '''
    mask:(n_batch, n_1, ..., n_{n_dim})
    output:(n_batch, n_dim, N)
    '''
    mask = bt.tensor(mask)
    shape = mask.size()
    n_batch = shape[0]
    n_dim = len(shape) - 1
    grid = imagegrid(*shape[1:])
    smask = []
    smesh = []
    for d in range(n_dim):
        indicesA = (slice(None),) * (d + 1) + (slice(1, None),) + (slice(None),) * (n_dim - d - 1)
        indicesB = (slice(None),) * (d + 1) + (slice(None, -1),) + (slice(None),) * (n_dim - d - 1)
        filter = (mask[indicesA] ^ mask[indicesB]) != 0
        points = (grid[indicesA] + grid[indicesB]) / 2
        smask.append(filter.flatten(1))
        smesh.append(points.flatten(1))
    # n_data = sum_i(prod_j(n_j - delta(i, j)))
    surface = bt.cat(smask, 1) # n_batch x n_data
    mesh = bt.cat(smesh, 1) # n_dim x n_data
    
    n_samples = Lzeros(n_batch + 1)
    n_samples[1:] = surface.sum(1).long()
    tshape = (n_dim, n_batch, surface.size(-1))
    pos = expand_to(mesh, tshape, [0, 2])[expand_to(surface, tshape, [1, 2])].view(n_dim, -1)
    return n_samples, pos

def update_surface(surface, n_samples, target):
    keep = n_samples[1:] * target[1:] > 0
    n_samples = bt.cat((n_samples[:1], n_samples[1:][keep]))
    target = bt.cat((target[:1], target[1:][keep]))
    n_deltas = n_samples[1:] - n_samples[:-1]
    cum_ns = target.cumsum(0)
    places, N = cum_ns[:-1], cum_ns[-1]
    cn = Lzeros(N); cn[places] = n_deltas; cn = cn.cumsum(0)
    ts = Lzeros(N); ts[places] = target[:-1]; ts = ts.cumsum(0)
    cs = Lzeros(N); cs[places] = n_samples[:-1]; cs = cs.cumsum(0)
    p = Larange(N)
    return surface[:, (p - ts) % cn + cs]
    
def cycle(data, n_samples):
    n_dim = data.size(0)
    n_samples = bt.cat((n_samples[:1], n_samples[1:][n_samples[1:] > 0]))
    n_deltas = n_samples[1:] - n_samples[:-1]
    cum_ns = n_samples.cumsum(0)
    places, N = cum_ns[:-1], cum_ns[-1]
    cn = Lzeros(N); cn[places] = n_deltas; cn = cn.cumsum(0).view(1, N)
    cs = Lzeros(N); cs[places] = n_samples[:-1]; cs = cs.cumsum(0).view(1, N)
    M, N = n_samples.max().item(), N.item()
    assert data.size(1) == N
    grid = imagegrid(M, N)
    index = (Lsum(grid, 0) - cs) % cn + cs
    cyc = expand_to(data, (n_dim, M, N), [0, 2]).gather(2, expand_to(index, (n_dim, M, N), [1, 2]))
    return cyc, cn, grid[0] < cn

def SurfaceDistance(mask1, mask2, type="ASD"):
    '''
    mask:(n_batch, n_1, ..., n_{n_dim})
    '''
    mask1 = bt.tensor(mask1)
    mask2 = bt.tensor(mask2)
    n_batch = nbatch(mask1)
    assert nbatch(mask2) == n_batch
    shape = [max(a, b) + 2 for a, b in zip(mask1.size(), mask2.size())]
    shape[0] = n_batch
    mask1 = crop_as(mask1, shape)
    mask2 = crop_as(mask2, shape)
    n_dim = len(shape) - 1
    n_samples1, surface1 = mask2surface(mask1)
    n_samples2, surface2 = mask2surface(mask2)
    n_samples = bt.max(n_samples1, n_samples2)
    n_samples = (n_samples1 * n_samples2 != 0).long() * n_samples
    if all(n_samples == 0): return bt.zeros(n_batch)
    surface1 = update_surface(surface1, n_samples1, n_samples)
    surface2 = update_surface(surface2, n_samples2, n_samples)
    
    cyc1, cn1, msk1 = cycle(surface1, n_samples)
    dis1 = ((cyc1 - surface2.unsqueeze(1)) ** 2).sum(0).sqrt()
    dis1 = bt.where(msk1, dis1, expand_to(dis1[0:1, :], dis1))
    cyc2, cn2, msk2 = cycle(surface2, n_samples)
    dis2 = ((surface1.unsqueeze(1) - cyc2) ** 2).sum(0).sqrt()
    dis2 = bt.where(msk2, dis2, expand_to(dis2[0:1, :], dis2))
    dis = bt.stack((dis1.min(0).values, dis2.min(0).values))
    if type == 'ASD': return bt.stack([
        bt.cat((x[0, :n_samples1[i+1]], x[1, :n_samples2[i+1]])).mean() 
        if n_samples1[i+1] + n_samples2[i+1] > 0 else bt.tensor(0.0) 
        for i, x in enumerate(dis.split(tonumpy(n_samples).tolist()[1:], 1))
    ])
    elif type == 'HD': return bt.stack([
        bt.cat((x[0, :n_samples1[i+1]], x[1, :n_samples2[i+1]])).max() 
        if n_samples1[i+1] + n_samples2[i+1] > 0 else bt.tensor(0.0) 
        for i, x in enumerate(dis.split(tonumpy(n_samples).tolist()[1:], 1))
    ])
    
def ASD(mask1, mask2): return SurfaceDistance(mask1, mask2, type = 'ASD')
def HD(mask1, mask2): return SurfaceDistance(mask1, mask2, type = 'HD')

imported = True
try:
    import SimpleITK as sitk
except ImportError:
    imported = False
if imported:
    def dilate(mask, distance = 0, spacing = 1, class_labels = None):
        imagetype = type(mask)
        mask = tonumpy(mask).astype(np.int)
        if not class_labels:
            class_labels = list(set(tonumpy(mask).reshape((-1,))))
            class_labels.sort()
        spacing = to_tuple(spacing)
        if len(spacing) == 1: spacing *= dimof(mask)
        mask_image = sitk.GetImageFromArray(mask, isVector = False)
        mask_image.SetSpacing(spacing)
        canvas = np.zeros((len(class_labels) - 1,) + mask.shape)
        i = 0
        for l in class_labels:
            dis_map = sitk.GetArrayViewFromImage(sitk.SignedMaurerDistanceMap(mask_image == l, insideIsPositive = False, squaredDistance = False, useImageSpacing = True))
            if l > 0:
                canvas[i] = np.where(dis_map <= distance, dis_map, np.inf)
                i += 1
        output = np.zeros_like(mask).astype(np.int)
        label_map = np.argmin(canvas, 0)
        i = 0
        for l in class_labels:
            if l == 0: continue
            output[label_map == i] = l
            i += 1
        output[np.min(canvas, 0) == np.inf] = 0
        if issubclass(imagetype, np.ndarray): return tonumpy(output)
        if issubclass(imagetype, bt.Tensor): return bt.tensor(output)
    
    class SurfaceDistanceImageFilter:
        def __init__(self): self.all_dis = bt.tensor([0])
        def Execute(self, A, B):
            array = lambda x: np.array(sitk.GetArrayViewFromImage(x)).astype(np.float)
            ADisMap = sitk.Abs(sitk.SignedMaurerDistanceMap(A, squaredDistance = False, useImageSpacing = True))
            BDisMap = sitk.Abs(sitk.SignedMaurerDistanceMap(B, squaredDistance = False, useImageSpacing = True))
            Asurface = sitk.LabelContour(A)
            Bsurface = sitk.LabelContour(B)
            
            # for a pixel 'a' in A, compute aBdis = dis(a, B)
            aBDis = array(BDisMap)[array(Asurface) > 0]
            # for a pixel 'b' in B, compute aBdis = dis(b, A)
            bADis = array(ADisMap)[array(Bsurface) > 0]
            self.all_dis = bt.tensor(np.concatenate((aBDis, bADis), 0))
            
        def GetHausdorffDistance(self): return self.all_dis.max()
        def GetMedianSurfaceDistance(self): return self.all_dis.median()
        def GetAverageSurfaceDistance(self): return self.all_dis.mean()
        def GetDivergenceOfSurfaceDistance(self): return self.all_dis.std()

    
    def Metric(A, B, spacing = 1, metric = "HD"):
        '''
        A: (n_batch, n_1, n_2, ..., n_k)
        B: (n_batch, n_1, n_2, ..., n_k)
        return: (n_batch,)
        '''
        A = tonumpy(A) != 0
        B = tonumpy(B) != 0
        spacing = to_tuple(spacing)
        n_dim = dimof(A) - 1
        n_batch = nbatch(A)
        if len(spacing) == 1: spacing *= n_dim
        Overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        SD_filter = SurfaceDistanceImageFilter()
        Overlap_execs = {
            'Dice': lambda x: x.GetDiceCoefficient(),
            'Jaccard': lambda x: x.GetJaccardCoefficient(),
            'Volume': lambda x: x.GetVolumeSimilarity(),
            'Falsenegative': lambda x: x.GetFalseNegativeError(),
            'Falsepositive': lambda x: x.GetFalsePositiveError()
        }
        SD_execs = {
            'HD': lambda x: x.GetHausdorffDistance(),
            'MSD': lambda x: x.GetMedianSurfaceDistance(),
            'ASD': lambda x: x.GetAverageSurfaceDistance(),
            'STDSD': lambda x: x.GetDivergenceOfSurfaceDistance()
        }
        measures = np.zeros((n_batch,))
        for b in range(n_batch):
            ITKA = sitk.GetImageFromArray(A[b].astype(np.int), isVector = False)
            ITKA.SetSpacing(spacing)
            ITKB = sitk.GetImageFromArray(B[b].astype(np.int), isVector = False)
            ITKB.SetSpacing(spacing)
            metric = metric.capitalize()
            if metric in Overlap_execs:
                Overlap_filter.Execute(ITKA, ITKB)
                measures[b] = Overlap_execs[metric](Overlap_filter)
            metric = metric.upper()
            if metric in SD_execs:
                SD_filter.Execute(ITKA, ITKB)
                measures[b] = SD_execs[metric](SD_filter)
        return measures

    
    def LabelMetric(A, B, spacing = 1, metric = "HD", class_labels = None):
        '''
        :param A: (n_batch, n_1, ..., n_k)
        :param B: (n_batch, n_1, ..., n_k)
        :param class_labels: list[n_class]
        :return: (n_batch, n_class)
        '''
        A = tonumpy(A)
        B = tonumpy(B)
        if not class_labels:
            class_labels = list(set(A.astype(np.int).reshape((-1,))))
            class_labels.sort()
        n_batch = nbatch(A)
        n_class = len(class_labels)
        A_labels = [A == i for i in class_labels]
        B_labels = [B == i for i in class_labels]
        A_maps = np.concatenate(A_labels)
        B_maps = np.concatenate(B_labels)
        metric = Metric(A_maps, B_maps, spacing, metric)
        return metric.reshape((n_class, n_batch)).T
    
    template1 = "\ndef m{metric}(*args, **kwargs): return Metric(*args, **kwargs, metric = '{metric}')"
    template2 = "\ndef mLabel{metric}(*args, **kwargs): return LabelMetric(*args, **kwargs, metric = '{metric}')"
    for metric in ('Dice', 'Jaccard', 'Volume', 'FalseNegative', 'FalsePositive', 'HD', 'MSD', 'ASD', 'STDSD'):
        exec(template1.format(metric=metric))
        exec(template2.format(metric=metric))

slicer = lambda n: {i: slice(n-i-1, None if i == 0 else -i) for i in range(n)}

def surface(mask):
    '''
    mask:(n_1, ..., n_{n_dim})
    output:(n_1, ..., n_{n_dim})
    '''
    mask = bt.tensor(mask)
    shape = sizeof(mask)
    shape = tuple(x+2 for x in shape)
    mask = crop_as(mask, shape)
    n_batch = nbatch(mask)
    n_dim = dimof(mask)
    mgrid = imagegrid(*shape)
    smask = []
    smesh = []
    for d in range(n_dim):
        indicesA = (slice(None),) * d + (slice(1, None),) + (slice(None),) * (n_dim - d - 1)
        indicesB = (slice(None),) * d + (slice(None, -1),) + (slice(None),) * (n_dim - d - 1)
        filter = (mask[indicesA] ^ mask[indicesB]) != 0
        points = (mgrid[(slice(None),) + indicesA] + mgrid[(slice(None),) + indicesB]) / 2
        smask.append(filter)
        smesh.append(points * filter.unsqueeze(0))
    centers = 0
    nums = 0
    for d in range(n_dim):
        ranges = [3] * n_dim
        ranges[d] -= 1
        for x in grid(*ranges):
            g = tuple(slicer(2)[t] if i == d else slicer(3)[t] for i, t in enumerate(x))
            centers += smesh[d][(slice(None),) + g]
            nums += smask[d][g]
    res = Fzeros(*nums.shape)
    idx = Lround(bt.where(nums > 0, centers / nums.unsqueeze(0), Ftensor(0)))
    res[idx.flatten(1).split(1, 0)] = 1
    return res
    
def conv(image, kernel = None, padding = 'SAME', with_batch = False):
    is_same = padding == 'SAME'
    if with_batch: n_batch = image.batch_dimension
    else: n_batch = 1
    if kernel is None: kernel = Fones((n_batch,) + (3,) * dimof(image)) / (3 ** dimof(image))
    else:
        kernel = bt.tensor(kernel)
        if with_batch:
            if nbatch(kernel) == n_batch: pass
            elif nbatch(kernel) == 1: kernel = repeat(kernel, n_batch)
            else: raise TypeError("Incorrect kernel batch size. ")
        else: kernel = multiply(kernel, n_batch)
    image = bt.tensor(image)
    if not with_batch: image = multiply(image, n_batch)
    if padding == 'SAME': padding = tuple(x//2 for x in sizeof(kernel)[1:])
    else: padding = to_tuple(padding)
    if len(padding) == 1: padding *= dimof(image) - 1
    if is_same: old_shape = sizeof(image)[1:]
    image = crop_as(image, tuple(x + 2 * y for x, y in zip(sizeof(image)[1:], padding)), n_keepdim = 1) 
    conv = 0
    for x in grid(*sizeof(kernel)[1:]):
        conv += kernel[(slice(None),) + tuple(x)] * image[(slice(None),) + tuple(slicer(sizeof(kernel)[1:][i])[t] for i, t in enumerate(x))]
    if with_batch: return crop_as(conv, old_shape, n_keepdim = 1) if is_same else conv
    return crop_as(conv.squeeze(0), old_shape) if is_same else conv.squeeze(0)
    
def roi(x, threshold = 0.2, mode = 'inner'):
    old_x = bt.tensor(x)
    n_dim = dimof(old_x)
    places = [i for i, l in enumerate(old_x.size()) if l != 1]
    x = old_x.squeeze().float()
    roi_result = tuple()
    for d in range(dimof(x)):
        percents = unsqueeze(x).transpose(0, d+1).flatten(1).mean(1)
        if all(percents < 1e-2): return (slice(None),) * n_dim
        filt = percents > threshold
        if all(~filt): return roi(old_x, threshold / 2, mode)
        valid = filt.long() * Larange(x.size(d))
        valid = valid[valid > 0]
        if mode != 'inner': roi_result += (slice(valid.min().item(), valid.max().item()),)
        else:
            new_filt = Lzeros(len(filt) + 2) != 0
            new_filt[1:-1] = filt
            ranges = (new_filt[1:] ^ new_filt[:-1]).long() * Larange(1, len(new_filt))
            ranges = ranges[ranges > 0] - 1
            ranges = ranges.view(-1, 2)
            range_lens = ranges[:, 1] - ranges[:, 0]
            irange = bt.argmax(range_lens)
            roi_result += (slice(ranges[irange, 0].item(), ranges[irange, 1].item()),)
    res = [slice(None)] * n_dim
    for i, x in enumerate(places): res[x] = roi_result[i]
    return tuple(res)
    
def margined(x):
    out = tuple()
    for rg in x:
        if rg.step is not None: raise TypeError("Unsuitable range for function margined. ")
        s, e = rg.start, rg.stop
        if s is None: s = 0
        if e is not None: out += (slice(s - (e - s) // 5, e + (e - s) // 5),)
        else: out += (slice(s // 2, e),)
    return out
    
    
class rigidity:
    
    def __init__(self, type = "isometry", *args):
        super().__init__()
        self.type = type
        self.R = self.b = None
        
    
    def __call__(self, trans, itrans = None, source_mask = None, target_mask = None, n_label_source_per_batch = None, n_label_target_per_batch = None, **kwargs):
        '''
        trans: the transformation from target (fixed) to source (moving). 
            ([n_batch, ]n_dim, n_1, n_2, ..., n_{n_dim}) => (n_batch, n_dim, n_1, n_2, ..., n_{n_dim})
        source_mask: (n_label_source, n_1, n_2, ..., n_{n_dim})
        target_mask: (n_label_target, n_1, n_2, ..., n_{n_dim})
        '''
        if source_mask is None and target_mask is None: return bt.tensor(0.0)
        if source_mask is not None and target_mask is None: n_label_source, *size = sizeof(source_mask); n_label_target = 0
        elif source_mask is None and target_mask is not None: n_label_target, *size = sizeof(target_mask); n_label_source = 0
        else:
            n_label_source, *size = sizeof(source_mask)
            n_label_target, *size_ = sizeof(target_mask)
            if size != size_: raise TypeError("Wrong size of inputs for rigidity constraint. ")
        n_label = n_label_source + n_label_target
        n_dim = len(size)
        displacement = trans.toDDF(*size)
        n_batch, n_dim_, *size_ = sizeof(displacement)
        if source_mask is not None and n_label_source_per_batch is None: n_label_source_per_batch = [1] * n_batch
        if target_mask is not None and n_label_target_per_batch is None: n_label_target_per_batch = [1] * n_batch
        n_label_source_per_batch = tolist(n_label_source_per_batch)
        n_label_target_per_batch = tolist(n_label_target_per_batch)
        if n_dim == n_dim_ and size == size_: pass
        elif not n_label_source_per_batch or len(n_label_source_per_batch) == n_batch and sum(n_label_source_per_batch) == n_label_source: pass
        elif not n_label_target_per_batch or len(n_label_target_per_batch) == n_batch and sum(n_label_target_per_batch) == n_label_target: pass
        else: raise TypeError("Wrong size of transformation for rigidity constraint. ")
        standard_grid = multiply(imagegrid(*size), n_batch)
#        if not itrans: itrans = inverse(trans, verbose = False)
        if source_mask is not None:
            index = Lzeros(n_label_source)
            idones = bt.cumsum(bt.tensor(n_label_source_per_batch), 0)
            index[idones[idones < n_batch]] = 1
            index = bt.cumsum(index, 0)
            
#            with itrans: warped_grid = itrans(standard_grid).data
            with trans: source_mesh = trans(standard_grid)
            X_source = source_mesh[index]
            Y_source = standard_grid[index]
            W_source = interpolation(source_mask, trans[index], mode='Nearest')
        else:
            X_source = bt.tensor([])
            Y_source = bt.tensor([])
            W_source = bt.tensor([])
        if target_mask is not None:
            index = Lzeros(n_label_target)
            idones = bt.cumsum(bt.tensor(n_label_target_per_batch), 0)
            index[idones[idones < n_batch]] = 1
            index = bt.cumsum(index, 0)
            with trans: source_mesh = trans(standard_grid)
            X_target = standard_grid[index]
            Y_target = source_mesh[index]
            W_target = target_mask
        else:
            X_target = bt.tensor([])
            Y_target = bt.tensor([])
            W_target = bt.tensor([])
#        if source_mask is not None:
#            index = Lzeros(n_label_source)
#            idones = bt.cumsum(bt.tensor(n_label_source_per_batch), 0)
#            index[idones[idones < n_batch]] = 1
#            index = bt.cumsum(index, 0)
#            if not itrans: itrans = ~trans
#            with itrans: warped_grid = itrans(standard_grid).data
#            with trans: source_mesh = trans(warped_grid)
#            X_source = source_mesh[index]
#            Y_source = warped_grid[index]
#            W_source = source_mask
##            if not itrans: itrans = ~trans
##            with itrans: target_mesh = itrans(standard_grid)
##            X_source = standard_grid[index]
##            Y_source = target_mesh[index]
##            W_source = source_mask
#        else:
#            X_source = bt.tensor([])
#            Y_source = bt.tensor([])
#            W_source = bt.tensor([])
#        if target_mask is not None:
#            index = Lzeros(n_label_target)
#            idones = bt.cumsum(bt.tensor(n_label_target_per_batch), 0)
#            index[idones[idones < n_batch]] = 1
#            index = bt.cumsum(index, 0)
#            if not itrans: itrans = ~trans
#            with trans: warped_grid = trans(standard_grid).data
#            with itrans: target_mesh = itrans(warped_grid)
#            X_target = target_mesh[index]
#            Y_target = warped_grid[index]
#            W_target = target_mask
##            with trans: source_mesh = trans(standard_grid)
##            X_target = standard_grid[index]
##            Y_target = source_mesh[index]
##            W_target = target_mask
#        else:
#            X_target = bt.tensor([])
#            Y_target = bt.tensor([])
# #            W_target = bt.tensor([])
        '''
        X: (n_label, n_dim, n_1, n_2, ..., n_{n_dim})
        Y: (n_label, n_dim, n_1, n_2, ..., n_{n_dim})
        W: (n_label, n_1, n_2, ..., n_{n_dim})
        '''
        X = bt.cat((X_source, X_target))
        Y = bt.cat((Y_source, Y_target))
        W = bt.cat((W_source, W_target))
        if self.type != "3term local":
            X_old, Y_old = X, Y
            X = mask_encode(X, W)
            Y = mask_encode(Y, W)
        X_ = bt.cat((X, Fones(n_label, 1, *size)), 1)
        Y_ = bt.cat((Y, Fones(n_label, 1, *size)), 1)
        Del = None; Aux = None; R = None
        if self.type == "non-rigid affinity":
            X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
            R = Y @ T(X) @ inv(X @ T(X))
            Del = Y - R @ X
        elif self.type == "affinity":
            X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
            R = Y @ T(X) @ inv(X @ T(X))
            Del = Y - R @ X
            Aux = Fnorm2(T(R) @ R - eye(R))
        elif self.type == "isometry":
            Xdissq, Ydissq = [], []
            for d in range(n_dim):
                u = (slice(None),) * (2 + d) + (slice(1, None),) + (slice(None),) * (n_dim - d - 1)
                l = (slice(None),) * (2 + d) + (slice(None, -1),) + (slice(None),) * (n_dim - d - 1)
                Xdissq.append(bt.where(X[u] * X[l] == 0, Fzeros_like(X[u]), (X[u] - X[l]) ** 2).sum(1).flatten(1))
                Ydissq.append(bt.where(Y[u] * Y[l] == 0, Fzeros_like(Y[u]), (Y[u] - Y[l]) ** 2).sum(1).flatten(1))
            Xdissq = bt.cat(Xdissq, 1)
            Ydissq = bt.cat(Ydissq, 1)
            Aux = 1e-2 * (Fnorm2(unsqueeze(Xdissq[Xdissq > 0] - 1)) + Fnorm2(unsqueeze(Ydissq[Ydissq > 0] - 1)))
            
            Xdissq, Ydissq = [], []
            for d in grid(*((2,) * n_dim)):
                sl = (slice(None, -1), slice(1, None))
                u = (slice(None), slice(None)) + tuple(sl[i] for i in d)
                l = (slice(None), slice(None)) + tuple(sl[1-i] for i in d)
                Xdissq.append(bt.where(X[u] * X[l] == 0, Fzeros_like(X[u]), (X[u] - X[l]) ** 2).sum(1).flatten(1))
                Ydissq.append(bt.where(Y[u] * Y[l] == 0, Fzeros_like(Y[u]), (Y[u] - Y[l]) ** 2).sum(1).flatten(1))
            Xdissq = bt.cat(Xdissq, 1)
            Ydissq = bt.cat(Ydissq, 1)
            Aux += 1e-3 * (Fnorm2(unsqueeze(Xdissq[Xdissq > 0] - n_dim)) + Fnorm2(unsqueeze(Ydissq[Ydissq > 0] - n_dim)))
        elif self.type == "rotation":
            X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
            A = Y @ T(X) @ inv(X @ T(X))
            omega = uncross(A - T(A))
            norm = norm2(omega)
            omega = omega / norm
            theta = bt.asin(norm / 2)
            wx = crossMatrix(omega)
            R = (1 - bt.cos(theta)) * (wx @ wx) + bt.sin(theta) * wx + eye(wx)
            Del = Y - R @ X
        elif self.type == "global":
            if n_dim == 2:
                X = bt.cat((X_.flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
                Y = bt.cat((Y_.flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
                W = bt.cat((W.flatten(1), multiply(bt.tensor([1.0]), n_label)), 1)
                X_old = bt.cat((bt.cat((X_old, Fones(n_label, 1, *size)), 1).flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
                Y_old = bt.cat((bt.cat((Y_old, Fones(n_label, 1, *size)), 1).flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
            elif n_dim == 3:
                X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
            A = X @ T(Y) + Y @ T(X)
            A = A - unsqueeze(trace(A), [-1, -1]) * eye(A)
            b = uncross(Y @ T(X)) - uncross(X @ T(Y))
            thetas = []
            vectors = []
            for t in range(nbatch(A)):
                if Fnorm2(A)[t] < 1e-4: thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
                L, P = bt.eig(A[t], eigenvectors = True)
                l = L[:, 0]
                c = squeeze(T(P) @ unsqueeze(b[t], -1), -1)
                f = ~ equal(c ** 2, 0)
                if sum(f) >=1:
                    coeff2 = divide(l.prod(), l, 0.0).sum() - (c ** 2).sum()
                    coeff1 = ((c ** 2) * (l.sum() - l)).sum() - l.prod()
                    coeff0 = - ((c ** 2) * divide(l.prod(), l, 0.0)).sum()
                    p = np.poly1d([1, - l.sum().item(), coeff2.item(), coeff1.item(), coeff0.item()])
                else: thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
                proots = bt.tensor(np.real(p.roots[np.abs(np.imag(p.roots)) < 1e-4])).to(device)
                if proots.numel() == 0: thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
                mu = unsqueeze(proots, [-1, -1]) * eye(unsqueeze(A[t].to(bt.float64)))
                v = - inv(unsqueeze(A[t].to(bt.float64)) - mu) @ unsqueeze(b[t].to(bt.float64), [0, -1])
                v = v.float()
                vTb = (T(v) @ unsqueeze(b[t], -1)).squeeze()
                if not any(equal(proots, vTb)): thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
                v = v[equal(proots, vTb)]
                theta = 2 * bt.atan(norm2(v))
                loss = unsqueeze(1 + bt.cos(theta), [-1, -1]) * ((T(v) @ unsqueeze(A[t]) @ v) / 2 + T(v) @ unsqueeze(b[t], -1))
                tmax = bt.argmax(loss)
                thetas.append(theta[tmax])
                vectors.append(v[tmax].squeeze(-1))
            th = bt.stack(thetas)
            vec = bt.stack(vectors)
            vx = crossMatrix(vec)
            R = unsqueeze(1 + bt.cos(th), [-1, -1]) * (vx @ vx + vx) + eye(vx)
            Del = Y - R.detach() @ X
        elif self.type == "3term local":
            S = lambda i, j, slc: (slice(None), i) + (slice(None),) * j + (slc,) + (slice(None),) * (n_dim - j - 1)
            if not hasattr(self, 'gap'): self.gap = 2
            dsize = tuple(x - self.gap for x in size)
            dX = Fstack([crop_as(X[S(i, i, slice(self.gap, None))] - X[S(i, i, slice(None, -self.gap))], dsize, n_keepdim=1) for i in range(n_dim)], 1)
            dY = Fstack([Fstack([crop_as(Y[S(i, j, slice(self.gap, None))] - Y[S(i, j, slice(None, -self.gap))], dsize, n_keepdim=1) for i in range(n_dim)], 1) for j in range(n_dim)], 2)
            dM = unsqueeze(crop_as(W, dsize, n_keepdim=1), [1, 2]).float()
            JacOfPoints = dY / (dX.unsqueeze(1) + (1 - dM)) * dM
            ddsize = tuple(x - 2 * self.gap for x in size)
            ddY = Fstack([Fstack([crop_as(JacOfPoints[(slice(None),) + S(i, j, slice(self.gap, None))] - JacOfPoints[(slice(None),) + S(i, j, slice(None, -self.gap))], ddsize, n_keepdim=2) for i in range(n_dim)], 2) for j in range(n_dim)], 3)
            ddM = unsqueeze(crop_as(W, ddsize, n_keepdim=1), [1, 2, 3]).float()
            HesOfPoints = ddY / (unsqueeze(crop_as(dX, ddsize, n_keepdim=2), [1, 2]) + (1 - ddM)) * ddM
            Aux = ((HesOfPoints ** 2 * ddM).flatten(4).sum(-1) / ddM.flatten(4).sum(-1)).flatten(1).mean(1)
            JacCollect = JacOfPoints.flatten(3).unsqueeze(1).transpose(1, -1).squeeze(-1).flatten(0, 1)
            JacDeltaOfPoints = (T(JacCollect) @ JacCollect - eye(JacCollect)).view(n_label, -1, n_dim, n_dim).unsqueeze(-1).transpose(1, -1).squeeze(1).view_as(JacOfPoints)
            Aux += ((JacDeltaOfPoints ** 2 * dM).flatten(3).sum(-1) / dM.flatten(3).sum(-1)).flatten(1).mean(1)
            Aux += ((bt.det(JacCollect) - 1).view(n_label, *dsize) ** 2 * squeeze(dM, [1, 1])).flatten(1).sum(-1) / dM.flatten(1).sum(-1)
        elif self.type == "local":
            X = pad(X_, n_keepdim = 2)
            Y = pad(Y_, n_keepdim = 2)
            X = bt.stack([X[(slice(None), slice(None)) + tuple(slicer(3)[t] for t in g)] for g in grid(*((3,) * n_dim))], 2)
            Y = bt.stack([Y[(slice(None), slice(None)) + tuple(slicer(3)[t] for t in g)] for g in grid(*((3,) * n_dim))], 2)
            # X, Y: n_batch x (n_dim + 1) x (3 ^ n_dim) x n_1 x ... x n_{n_dim}
            Nx = join_to(X.flatten(3), [0, 3], 0)
            Ny = join_to(Y.flatten(3), [0, 3], 0)
            val = bt.det(Nx @ T(Nx)) > 1e-2
            Nx = Nx[val]
            Ny = Ny[val]
            # Nx, Ny: (n_batch x n_data) x (n_dim + 1) x (3 ^ n_dim)
            NR = Ny @ T(Nx) @ inv(Nx @ T(Nx))
            # NR: (n_batch x n_data) x (n_dim + 1) x (n_dim + 1)
            dis = Fnorm2(T(NR) @ NR - eye(NR)).view(n_label, -1)
            Aux = 1e-2 * dis.mean()
        else: raise TypeError("Unknown type ({}) for rigidity constraint. ".format(self.type))
        cons = 0
        if Del is not None: cons += divide(Fnorm2(Del) * Del.size(-1), W.flatten(1).sum(1), 0.0)
#        if Del is not None:
#            if len(kwargs['Lambda']) == 0:
#                Lambda = bt.randn(*Del.size()).requires_grad_(True); kwargs['Lambda'].append(Lambda)
#            else: Lambda = kwargs['Lambda'][0]
#            cons += bt.abs(Lambda) * bt.abs(Del)
        if Aux is not None: cons += Aux
        if R is not None:
            self.R = R
            if self.type == "3term local": X_old, Y_old = X, Y
            X_old, Y_old = X_old.view_as(X), Y_old.view_as(Y)
            self.b = ((Y_old - R@X_old) * W.unsqueeze(1)).sum(-1) / W.sum(-1, keepdim=True)
        return cons.mean() #(cons * W.flatten(1).sum(1)).mean()

#         '''
#         X: (n_label, n_dim, n_1, n_2, ..., n_{n_dim})
#         Y: (n_label, n_dim, n_1, n_2, ..., n_{n_dim})
#         W: (n_label, n_1, n_2, ..., n_{n_dim})
#         '''
#         X = bt.cat((X_source, X_target))
#         Y = bt.cat((Y_source, Y_target))
#         W = bt.cat((W_source, W_target))
#         X_org = bt.cat((X, Fones(n_label, 1, *size)), 1).flatten(2)
#         Y_org = bt.cat((Y, Fones(n_label, 1, *size)), 1).flatten(2)
#         X = mask_encode(X, W)
#         Y = mask_encode(Y, W)
#         X_ = bt.cat((X, Fones(n_label, 1, *size)), 1)
#         Y_ = bt.cat((Y, Fones(n_label, 1, *size)), 1)
#         Del = None; Aux = None
#         if self.type == "non-rigid affinity":
#             X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
#             R = Y @ T(X) @ inv(X @ T(X))
#             Del = Y - R @ X
#         elif self.type == "affinity":
#             X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
#             R = Y @ T(X) @ inv(X @ T(X))
#             Del = Y - R @ X
#             Aux = Fnorm2(T(R) @ R - eye(R))
#         elif self.type == "isometry":
#             Xdissq, Ydissq = [], []
#             for d in range(n_dim):
#                 u = (slice(None),) * (2 + d) + (slice(1, None),) + (slice(None),) * (n_dim - d - 1)
#                 l = (slice(None),) * (2 + d) + (slice(None, -1),) + (slice(None),) * (n_dim - d - 1)
#                 Xdissq.append(bt.where(X[u] * X[l] == 0, Fzeros_like(X[u]), (X[u] - X[l]) ** 2).sum(1).flatten(1))
#                 Ydissq.append(bt.where(Y[u] * Y[l] == 0, Fzeros_like(Y[u]), (Y[u] - Y[l]) ** 2).sum(1).flatten(1))
#             Xdissq = bt.cat(Xdissq, 1)
#             Ydissq = bt.cat(Ydissq, 1)
#             Aux = 1e-2 * (Fnorm2(unsqueeze(Xdissq[Xdissq > 0] - 1)) + Fnorm2(unsqueeze(Ydissq[Ydissq > 0] - 1)))
            
#             Xdissq, Ydissq = [], []
#             for d in grid(*((2,) * n_dim)):
#                 sl = (slice(None, -1), slice(1, None))
#                 u = (slice(None), slice(None)) + tuple(sl[i] for i in d)
#                 l = (slice(None), slice(None)) + tuple(sl[1-i] for i in d)
#                 Xdissq.append(bt.where(X[u] * X[l] == 0, Fzeros_like(X[u]), (X[u] - X[l]) ** 2).sum(1).flatten(1))
#                 Ydissq.append(bt.where(Y[u] * Y[l] == 0, Fzeros_like(Y[u]), (Y[u] - Y[l]) ** 2).sum(1).flatten(1))
#             Xdissq = bt.cat(Xdissq, 1)
#             Ydissq = bt.cat(Ydissq, 1)
#             Aux += 1e-3 * (Fnorm2(unsqueeze(Xdissq[Xdissq > 0] - n_dim)) + Fnorm2(unsqueeze(Ydissq[Ydissq > 0] - n_dim)))
#         elif self.type == "rotation":
#             X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
#             A = Y @ T(X) @ inv(X @ T(X))
#             omega = uncross(A - T(A))
#             norm = norm2(omega)
#             omega = omega / norm
#             theta = bt.asin(norm / 2)
#             wx = crossMatrix(omega)
#             R = (1 - bt.cos(theta)) * (wx @ wx) + bt.sin(theta) * wx + eye(wx)
#             Del = Y - R @ X
#         elif self.type == "global":
#             if n_dim == 2:
#                 X_org = bt.cat((X_org.flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
#                 Y_org = bt.cat((Y_org.flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
#                 X = bt.cat((X_.flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
#                 Y = bt.cat((Y_.flatten(2), multiply(unsqueeze(toFtensor([0, 0, 2]), -1), n_label)), 2)
#                 W = bt.cat((W.flatten(1), multiply(bt.tensor([1.0]), n_label)), 1)
#             elif n_dim == 3:
#                 X = X.flatten(2); Y = Y.flatten(2); W = W.flatten(1)
#             A = X @ T(Y) + Y @ T(X)
#             A = A - unsqueeze(trace(A), [-1, -1]) * eye(A)
#             b = uncross(Y @ T(X)) - uncross(X @ T(Y))
#             thetas = []
#             vectors = []
#             for t in range(nbatch(A)):
#                 if Fnorm2(A)[t] < 1e-4: thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
#                 L, P = bt.eig(A[t], eigenvectors = True)
#                 l = L[:, 0]
#                 c = squeeze(T(P) @ unsqueeze(b[t], -1), -1)
#                 f = ~ equal(c ** 2, 0)
#                 if sum(f) >=1:
#                     coeff2 = divide(l.prod(), l, 0.0).sum() - (c ** 2).sum()
#                     coeff1 = ((c ** 2) * (l.sum() - l)).sum() - l.prod()
#                     coeff0 = - ((c ** 2) * divide(l.prod(), l, 0.0)).sum()
#                     p = np.poly1d([1, - l.sum().item(), coeff2.item(), coeff1.item(), coeff0.item()])
#                 else: thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
#                 proots = bt.tensor(np.real(p.roots[np.abs(np.imag(p.roots)) < 1e-4])).to(device)
#                 if proots.numel() == 0: thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
#                 mu = unsqueeze(proots, [-1, -1]) * eye(unsqueeze(A[t].to(bt.float64)))
#                 v = - inv(unsqueeze(A[t].to(bt.float64)) - mu) @ unsqueeze(b[t].to(bt.float64), [0, -1])
#                 v = v.float()
#                 vTb = (T(v) @ unsqueeze(b[t], -1)).squeeze()
#                 if not any(equal(proots, vTb)): thetas.append(bt.tensor(0.0)); vectors.append(toFtensor([0, 0, 0])); continue
#                 v = v[equal(proots, vTb)]
#                 theta = 2 * bt.atan(norm2(v))
#                 loss = unsqueeze(1 + bt.cos(theta), [-1, -1]) * ((T(v) @ unsqueeze(A[t]) @ v) / 2 + T(v) @ unsqueeze(b[t], -1))
#                 tmax = bt.argmax(loss)
#                 thetas.append(theta[tmax])
#                 vectors.append(v[tmax].squeeze(-1))
#             th = bt.stack(thetas)
#             vec = bt.stack(vectors)
#             vx = crossMatrix(vec)
#             R = unsqueeze(1 + bt.cos(th), [-1, -1]) * (vx @ vx + vx) + eye(vx)
#             Del = Y - R @ X
#         elif self.type == "3term local":
#             S = lambda i, j, slc: (slice(None), i) + (slice(None),) * j + (slc,) + (slice(None),) * (n_dim - j - 1)
#             dsize = tuple(x - 2 for x in size)
#             dX = Fstack([crop_as(X[S(i, i, slice(2, None))] - X[S(i, i, slice(None, -2))], dsize, n_keepdim=1) for i in range(n_dim)], 1)
#             dY = Fstack([Fstack([crop_as(Y[S(i, j, slice(2, None))] - Y[S(i, j, slice(None, -2))], dsize, n_keepdim=1) for i in range(n_dim)], 1) for j in range(n_dim)], 2)
#             dM = unsqueeze(crop_as(W, dsize, n_keepdim=1), [1, 2]).float()
#             JacOfPoints = dY / (dX.unsqueeze(1) + (1 - dM)) * dM
#             ddsize = tuple(x - 4 for x in size)
#             ddY = Fstack([Fstack([crop_as(JacOfPoints[(slice(None),) + S(i, j, slice(2, None))] - JacOfPoints[(slice(None),) + S(i, j, slice(None, -2))], ddsize, n_keepdim=2) for i in range(n_dim)], 2) for j in range(n_dim)], 3)
#             ddM = unsqueeze(crop_as(W, ddsize, n_keepdim=1), [1, 2, 3]).float()
#             HesOfPoints = ddY / (unsqueeze(crop_as(dX, ddsize, n_keepdim=2), [1, 2]) + (1 - ddM)) * ddM
#             Aux = ((HesOfPoints ** 2 * ddM).flatten(4).sum(-1) / ddM.flatten(4).sum(-1)).flatten(1).mean(1)
#             JacCollect = JacOfPoints.flatten(3).unsqueeze(1).transpose(1, -1).squeeze(-1).flatten(0, 1)
#             JacDeltaOfPoints = (T(JacCollect) @ JacCollect - eye(JacCollect)).view(n_label, -1, n_dim, n_dim).unsqueeze(-1).transpose(1, -1).squeeze(1).view_as(JacOfPoints)
#             Aux += ((JacDeltaOfPoints ** 2 * dM).flatten(3).sum(-1) / dM.flatten(3).sum(-1)).flatten(1).mean(1)
#             Aux += ((bt.det(JacCollect) - 1).view(n_label, *dsize) ** 2 * squeeze(dM, [1, 1])).flatten(1).sum(-1) / dM.flatten(1).sum(-1)
#         elif self.type == "local":
#             X = pad(X_, n_keepdim = 2)
#             Y = pad(Y_, n_keepdim = 2)
#             X = bt.stack([X[(slice(None), slice(None)) + tuple(slicer(3)[t] for t in g)] for g in grid(*((3,) * n_dim))], 2)
#             Y = bt.stack([Y[(slice(None), slice(None)) + tuple(slicer(3)[t] for t in g)] for g in grid(*((3,) * n_dim))], 2)
#             # X, Y: n_batch x (n_dim + 1) x (3 ^ n_dim) x n_1 x ... x n_{n_dim}
#             Nx = join_to(X.flatten(3), [0, 3], 0)
#             Ny = join_to(Y.flatten(3), [0, 3], 0)
#             val = bt.det(Nx @ T(Nx)) > 1e-2
#             Nx = Nx[val]
#             Ny = Ny[val]
#             # Nx, Ny: (n_batch x n_data) x (n_dim + 1) x (3 ^ n_dim)
#             NR = Ny @ T(Nx) @ inv(Nx @ T(Nx))
#             # NR: (n_batch x n_data) x (n_dim + 1) x (n_dim + 1)
#             dis = Fnorm2(T(NR) @ NR - eye(NR)).view(n_label, -1)
#             Aux = 1e-2 * dis.mean()
#         else: raise TypeError("Unknown type ({}) for rigidity constraint. ".format(self.type))
#         cons = 0
#         if Del is not None: cons += divide(Fnorm2(Del) * Del.size(-1), W.flatten(1).sum(1), 0.0)
# #        if Del is not None:
# #            if len(kwargs['Lambda']) == 0:
# #                Lambda = bt.randn(*Del.size()).requires_grad_(True); kwargs['Lambda'].append(Lambda)
# #            else: Lambda = kwargs['Lambda'][0]
# #            cons += bt.abs(Lambda) * bt.abs(Del)
#         if Aux is not None: cons += Aux
#         if self.type in ('global', 'rotation', 'affinity', 'non-rigid affinity'):
#             self.R = R
#             if self.type == "3term local": X_org, Y_org = X, Y
#             self.b = ((Y_org - R @ X_org) * W.unsqueeze(1)).sum(-1) / W.sum(1, keepdim=True)
#         return cons.mean() #(cons * W.flatten(1).sum(1)).mean()
