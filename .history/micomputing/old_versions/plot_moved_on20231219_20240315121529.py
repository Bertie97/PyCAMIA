
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to visualize features of 3D volume.",
    help = "Use `from micomputing import plot as plt`.",
    requires = "matplotlib"
).check()

__all__ = """
    set_font
    plot
    subplots
    background
    imshow
    imsshow
    volshow
    maskshow
    bordershow
    fieldshow
    gridshow
""".split()

import math
from .trans import DenseDisplacementField, Transformation
from .stdio import IMG
from .funcs import reorient

with __info__:
    import batorch as bt
    from matplotlib import pyplot as plt
    from matplotlib import colors as mc
    from matplotlib.pyplot import *
    from pyoverload import *
    from pycamia import prod, to_tuple, argmax, avouch, touch
    __all__.extend([x for x in dir(plt) if not x.startswith('_') and x not in __all__])
    plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300

canvas = None
colors = ['crimson', 'forestgreen', 'royalblue', 'red', 'green', 'blue', 'gold', 'purple', 'gray', 'pink', 'darkgreen', 'dodgerblue']
assert all(c in mc.CSS4_COLORS for c in colors)
subplot_rows = 1
subplot_cols = 1
subplot_id = 1
default_orient = None

TRS_list = ["trans", "trs", "transversal", "transverse", "ax", "axial", 
            "axis", "crs", "cross", "hor", "horizontal", "slice"]
COR_list = ["cor", "coro", "coronal", "coron"]
SAG_list = ["sag", "sagit", "sagittal", "vert", "vertical"]
name_to_orient = {k: 'AR' for k in TRS_list}
name_to_orient.update({k: 'SR' for k in COR_list})
name_to_orient.update({k: 'SA' for k in SAG_list})

def set_font(font: str):
    plt.rcParams['font.sans-serif'] = font
    
def set_orient(ref):
    if isinstance(ref, IMG): ref = ref.orientation
    global default_orient
    default_orient = ref

def to_image(data: array, nslice: (int, null)=None, dim: int=-1, as_orient=None, to_orient='AR'):
    if as_orient is None: as_orient = default_orient
    if to_orient is None: to_orient = 'AR' # default orient for output
    as_orient = name_to_orient.get(as_orient.lower(), as_orient.upper()) if as_orient else None
    to_orient = name_to_orient.get(to_orient.lower(), to_orient.upper()) if to_orient else None
    axis_upper = {'L':'R', 'I':'S', 'P':'A'}
    to_orilist = [axis_upper.get(o, o) for o in to_orient]
    if as_orient and len(as_orient) == 3 and len(to_orilist) < 3:
        to_orilist += [x for x in "ARS" if x not in to_orilist]
        to_orient += to_orilist[-1]
    if isinstance(data, IMG) and as_orient is None:
        data = data.reorient(to_orient)
        if data.ndim == 3 and len(to_orilist) < 3:
            to_orilist += [x for x in "ARS" if x not in to_orilist]
            to_orient += to_orilist[-1]
        if isinstance(dim, int): # LPI => to_orient
            from_orilist = [axis_upper.get(o, o) for o in data.orientation]
            remapper = {k:to_orilist.index(from_orilist[k]) for k in range(-3, 3)}
            dim = remapper[dim]
    sp = getattr(data, 'spacing', None)
    data = bt.tensor(data).squeeze()
    if sp is not None:
        for d, s in enumerate(sp):
            data = data.amplify(round(s), d)
    if as_orient is not None: data = reorient(data, from_orient=as_orient, to_orient=to_orient)
    if isinstance(dim, str) and len(dim) == 2: dim = argmax([o in dim for o in to_orient])
    if data.ndim <= 1: raise TypeError("Please don't use 'plot.imshow' to demonstrate an array or a scalar. ")
    if data.nspace > 3: raise TypeError(f"'plot.imshow' takes 2 or 3D-data as input (currently {data.shape}), please reduce the dimension manually or specify special dimensions to reduce. ")
    if data.nspace == 3:
        if data.has_batch: data = data.pick(0, [])
        if data.has_channel: 
            if dim < 0: dim += data.nspace
            dim += 1 if data.channel_dim <= dim else 0
            if nslice is None: nslice = data.size(dim) // 2
            data = data.pick(nslice, dim) \
                       .sample(number=min(data.channel_size, 3), random=False, dim={}) \
                       .movedim(data.channel_dimension, -1)
        else:
            if data.space[-1] <= 3: pass
            elif data.space[0] <= 3: data = data.movedim(0, 2)
            if nslice is None: nslice = data.size(dim) // 2
            data = data.pick(nslice, dim)
    elif data.nspace == 2:
        if data.has_batch: data = data.pick(0, [])
        if data.has_channel:
            data = data.sample(number=min(data.channel_size, 3), random=False, dim={}) \
                       .movedim(data.channel_dimension, -1)
    elif data.ndim == 3: data = data.pick(0, []).remove_special_()
    elif data.ndim == 2: data = data.remove_special_()
    else: TypeError(f"Cannot extrat image from data of size {data.shape}. ")
    if data.ndim == 3 and data.size(2) < 3: data = bt.cat(data, bt.zeros(data.shape[:-1] + (3 - data.size(2),)), 2)
    if data.ndim == 2 and data.max() == data.min(): data = data.multiply(3, -1)
    return data.float()

def to_RGB(*color):
    if len(color) == 0: return (1.,) * 3
    elif len(color) == 1:
        c = color[0]
        if isinstance(c, float) and 0 <= c <= 1: return (c,) * 3
        elif isinstance(c, (int, float)) and 0 <= c <= 255: return (c / 255,) * 3
        elif isinstance(c, tuple): return to_RGB(*c)
        elif isinstance(c, str):
            if not c.startswith('#'):
                c = mc.BASE_COLORS.get(c.lower(), mc.CSS4_COLORS.get(c.lower(), c))
                if isinstance(c, tuple): return to_RGB(*c)
            if not c.startswith('#'): raise TypeError(f"Unrecognized color {c}.")
            return mc.hex2color(c)
        else: raise TypeError("Unaccepted color type. ")
    elif len(color) == 3:
        if all(isinstance(c, float) and 0 <= c <= 1 for c in color): return color
        elif all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in color): return tuple(c / 255 for c in color)
        else: raise TypeError("Unaccepted color type. ")
    else: raise TypeError("Unaccepted color type. ")

def subplots(n=1, m=None):
    if m is None:
        r = math.floor(math.sqrt(n))
        c = math.ceil(n / r)
        n, m = r, c
    global subplot_rows
    global subplot_cols
    global subplot_id
    subplot_rows = n
    subplot_cols = m
    subplot_id = 1

def clf():
    plt.clf()
    subplots()
    
def create_subplot():
    global subplot_rows
    global subplot_cols
    global subplot_id
    if subplot_id > subplot_rows * subplot_cols:
        raise TypeError(f"Cannot create subplot {subplot_id} in grid {subplot_rows} x {subplot_cols}, " + 
                        "please use `micomputing.plot.subplots(rows, cols)` to specify the number of images. ")
    plt.subplot(subplot_rows, subplot_cols, subplot_id); subplot_id += 1

def plot(*data: (array, null), **kwargs):
    """
    An alias of matplotlib.pyplot.plot, compatible with batorch. 
    
    Args: data (bt.Tensor): [n_batch] for multiple lines, while {n_channel} for x and y.
    """
    create_subplot()
    return line(*data, **kwargs)

def line(*data: (array, null), **kwargs):
    """
    Args: data (bt.Tensor): [n_batch] for multiple lines, while {n_channel} for x and y.
    """
    if len(data) == 1: 
        y = bt.tensor(data[0])
        avouch(y.n_space_dim == 1)
        if y.n_channel == 2: x, y = [u.squeeze() for u in y.split(1, {})]
        else:
            length = y.space[0]
            x = bt.arange(0, length).expand_to(y)
    elif len(data) == 2:
        x, y = data
    else: raise TypeError("Too much arguments for micomputing.plot/line. ")
    if isinstance(x, bt.torch.Tensor):
        if x.device.type == 'cuda': x = x.cpu()
        x = x.detach().cpu().numpy()
    if isinstance(y, bt.torch.Tensor):
        if y.device.type == 'cuda': y = y.cpu()
        y = y.detach().cpu().numpy()
    return plt.plot(x.T, y.T, **kwargs)

@typehint
def imshow(data: (array, null)=None, nslice: (int, null)=None, dim: int=-1, **kwargs):
    """
    An automatic image display function for all kinds of tensors. 
    The first image in batched images will be selected to be showed. 
    For medical images:
    Displacements with channel dimension identified will be displayed as RGB colored maps.
    If there are no dimension <=3, gray scaled images will be showed. 
    Transverse medical image with the right hand side of the subject shown on the left
        and anterior shown at the bottom will be selected for 3D volumes.
    `nslice` and `dim` are used for 3D volumes only, meaning to show the `nslice` slice of dimension `dim`. 
    """
    global canvas
    if kwargs.pop('create_subplot', data is not None): create_subplot()
    if data is not None:
        canvas = to_image(data, nslice, dim, kwargs.pop('as_orient', None), kwargs.pop('to_orient', 'ARS'))
    if canvas is None or isinstance(canvas, tuple):
        raise TypeError("Please input data in 'imshow' or 'background' to show. ")
    if isinstance(canvas, bt.torch.Tensor):
        if canvas.device.type == 'cuda': canvas_show = canvas.cpu()
        canvas_show = canvas.detach().cpu().numpy()
    else: canvas_show = canvas
    plt.axis('off')
    kwargs['cmap'] = kwargs.get('cmap', 'gray')
    if canvas_show.ndim >= 3:
        Max = canvas_show.max()
        if canvas_show.dtype.kind == 'i' and Max > 255: canvas_show = canvas_show * 255 // canvas_show.max(); plt.title(f"x {Max / 255:.04f}")
        elif Max > 1: canvas_show = canvas_show / canvas_show.max(); plt.title(f"x {Max:.04f}")
    return plt.imshow(canvas_show, **kwargs)

@typehint
def imsshow(*data: (array, null), nslice: (int, null)=None, dim: int=-1, show=False, **kwargs):
    """
    Display a series of images by 'imshow'. Use argument 'show=True' to directly call plt.show(). 
    The first image in batched images will be selected to be showed. 
    For medical images:
    Displacements with channel dimension identified will be displayed as RGB colored maps.
    If there are no dimension <=3, gray scaled images will be showed. 
    Transverse medical image with the right hand side of the subject shown on the left
        and anterior shown at the bottom will be selected for 3D volumes.
    `nslice` and `dim` are used for 3D volumes only, meaning to show the `nslice` slice of dimension `dim`. 
    """
    return [imshow(d, nslice, dim, **kwargs) for d in data] + ([plt.show()] if show else [])

@typehint
def volshow(data: (array, null), show=False, **kwargs):
    subplots(2, 2)
    return [imshow(data, None, (d + 2) % 3, **kwargs) for d in range(len(data.shape))]

def background(*color):
    """
    Set a background color by RGB or a gray scale, conflict with imshow. 
    """
    global canvas
    canvas = to_RGB(*color)
    create_subplot()

@typehint
def maskshow(*masks, on=None, alpha=0.5, nslice=None, dim=-1, stretch=False, **kwargs):
    global canvas
    orient_as = kwargs.pop('as_orient', None)
    orient_to = kwargs.pop('to_orient', None)
    if on is not None:
        if isinstance(on, (int, tuple)): background(*on)
        elif isinstance(on, array): canvas = to_image(on, nslice, dim, orient_as, orient_to)
        elif isinstance(on, list): canvas = to_image(bt.tensor(on), nslice, dim, orient_as, orient_to)
        else: raise TypeError("Unrecognized argument 'on' for 'maskshow'. ")
    elif canvas is None:
        canvas = (1.,) * 3
    if len(masks) == 1 and not isinstance(masks[0], array):
        raise TypeError("Non-array mask for function 'maskshow', unzip the tuple if necessary. ")
    if len(masks) == 0: return imshow(create_subplot = on is not None)
    alpha = to_tuple(alpha, len(masks))
    new_masks = []
    new_alpha = []
    for m, a in zip(masks, alpha):
        img = to_image(m, nslice, dim, orient_as, orient_to)
        if img.ndim == 3:
            new_masks.extend(x.squeeze(-1) for x in img.split(1, dim=dim))
            new_alpha.extend([a] * img.size(dim))
        else:
            new_masks.append(img)
            new_alpha.append(a)
    color_mask_map = [(to_RGB(c), m, a) for c, m, a in zip(colors*(len(new_masks) // len(colors) + 1), new_masks, new_alpha)]
    for c, m in kwargs.items():
        if touch(lambda: to_RGB(c)) is not None: 
            color_mask_map.append((to_RGB(c), m, alpha[0]))
            kwargs.pop(c)

    if not stretch:
        shapes = [m.ishape for _, m, _ in color_mask_map]
        target_shape = shapes[0]
        if len(set(shapes)) > 1 or not isinstance(canvas, tuple) and target_shape != canvas.shape:
            raise TypeError(f"Please use masks (size: {target_shape}) of the same size as the background image (size: {canvas.shape}), "
                            "or use 'stretch=True' in 'maskshow' to automatically adjust the image sizes. ")
    else:
        def adjust(m, to):
            ms = tuple(m.shape)
            scaling = tuple((a // b, b // a) for a, b in zip(to, ms))
            return m.down_scale([max(v, 1) for u, v in scaling]).up_scale([max(u, 1) for u, v in scaling]).crop_as(to)
        shapes = [m.ishape for _, m, _ in color_mask_map]
        if not isinstance(canvas, tuple): shapes.append(canvas.shape[:2])
        areas = [u * v for u, v in shapes]
        target_shape = shapes[areas.index(max(areas))]
        color_mask_map = [(c, adjust(m, to=target_shape), a) for c, m, a in color_mask_map]
        canvas = adjust(canvas, to=target_shape)

    target_shape = bt.Size(*target_shape, {3})
    if isinstance(canvas, tuple): canvas = bt.tensor(list(canvas)).expand_to(target_shape)
    elif canvas.ndim == 2: canvas = canvas.expand_to(target_shape)
    coeff = prod(1 - a * m for _, m, a in color_mask_map) # color_mask_map: (color, mask, alpha)
    canvas *= coeff
    for i, (c, m, a) in enumerate(color_mask_map):
        coeff = prod(a * m if j == i else 1 - a * m for j, (_, m, a) in enumerate(color_mask_map))
        canvas += coeff.unsqueeze(-1) * m.unsqueeze(-1) * bt.tensor(list(c)).unsqueeze(0, 1)
    if isinstance(canvas, bt.torch.Tensor):
        if canvas.device.type == 'cuda': canvas_show = canvas.cpu()
        canvas_show = canvas.detach().cpu().numpy()
    else: canvas_show = canvas
    plt.axis('off')
    create_subplot()
    kwargs['cmap'] = kwargs.get('cmap', 'gray')
    if canvas_show.ndim >= 3:
        Max = canvas_show.max()
        if canvas_show.dtype.kind == 'i' and Max > 255: canvas_show = canvas_show * 255 // canvas_show.max(); plt.title(f"x {Max / 255:.04f}")
        elif Max > 1: canvas_show = canvas_show / canvas_show.max(); plt.title(f"x {Max:.04f}")
    return plt.imshow(canvas_show, **kwargs)

def smooth(curve):
    """
    curve: 2xn
    """
    middle = (curve[:, :-2] + curve[:, 1:-1] + curve[:, 2:]) / 3
    if all(curve[:, 0] == curve[:, -1]):
        head = (curve[:, :1] + curve[:, 1:2] + curve[:, -2:-1]) / 3
        return bt.cat(head, middle, head, dim=1)
    return bt.cat(curve[:, :1], middle, curve[:, -1:], dim=1)

def sharpen(curve, old_curve):
    """
    sharpen towards old_curve
    curve, old_curve: 2xn
    """
    a = bt.cat(curve[:, -2:-1] - curve[:, -1:], curve[:, :-1] - curve[:, 1:], dim=1)
    b = bt.cat(curve[:, 1:] - curve[:, :-1], curve[:, 1:2] - curve[:, :1], dim=1)
    costheta = (a*b).sum(0) / bt.sqrt((a*a).sum(0)) / bt.sqrt((b*b).sum(0))
    new_curve = curve.clone()
    new_curve[:, costheta > -0.9] = old_curve[:, costheta > -0.9]
    return new_curve

def constraint(new_curve, old_curve, constraint_curve):
    """
    constraint new_curve towards constraint_curve
    curve, old_curve: 2xn
    """
    dis_sqs = bt.sum((new_curve - constraint_curve) ** 2, 0)
    percentile = 2 #min(np.sort(dis_sqs)[98 * len(dis_sqs) // 100], 1)
    return bt.where(dis_sqs <= percentile, new_curve, old_curve)

def border(mask, min_length = 10):
    grid = bt.image_grid(*mask.shape)
    mask = mask > 0.5
    idx = mask[1:, :] ^ mask[:-1, :]
    idx = idx.expand_to(2, -1, mask.size(1))
    locs1 = (grid[:, 1:, :] + grid[:, :-1, :])[idx] / 2
    idx = mask[:, 1:] ^ mask[:, :-1]
    idx = idx.expand_to(2, mask.size(0), -1)
    locs2 = (grid[:, :, 1:] + grid[:, :, :-1])[idx] / 2
    locs = bt.cat(locs1.reshape(2, -1), locs2.reshape(2, -1), dim=1)
    if locs.size == 0: return []
    curves = []
    unvisited = bt.ones(locs.shape[-1])
    while True:
        if not any(unvisited): break
        first = bt.argmax(unvisited).item()
        cloc = locs[:, first:first + 1]
        unvisited[first] = 0
        curve = cloc
        while True:
            dissq = bt.sum((locs - cloc) ** 2, 0)
            inloc = bt.argmax(bt.where((unvisited > 0) & (dissq > 0), 1/dissq.clamp(min=0.1), bt.tensor(0).float()))
            if dissq[inloc] > 2: break
            cloc = locs[:, inloc:inloc + 1]
            curve = bt.cat(curve, cloc, dim=1)
            unvisited[inloc] = 0
            if not any(unvisited): break
        sloc = locs[:, first:first + 1]
        if bt.sum((cloc - sloc) ** 2) <= 2:
            curve = bt.cat(curve, sloc, dim=1)
        if curve.shape[1] <= min_length: continue
        scurve = curve
        for _ in range(100): scurve = constraint(smooth(scurve), scurve, curve)
        ccurve = scurve
        for _ in range(100):
            scurve = constraint(sharpen(scurve, curve), scurve, ccurve)
            scurve = constraint(smooth(scurve), scurve, curve)
        curves.append(scurve)
    return curves

def bordershow(*masks, on=None, mask_alpha=0., nslice=None, dim=-1, stretch=False, min_length = 10, **kwargs):
    global canvas
    orient_as = kwargs.pop('as_orient', None)
    orient_to = kwargs.pop('to_orient', None)
    if on is not None:
        if isinstance(on, (int, tuple)): background(*on)
        elif isinstance(on, array): canvas = to_image(on, nslice, dim, orient_as, orient_to)
        elif isinstance(on, list): canvas = to_image(bt.tensor(on), nslice, dim, orient_as, orient_to)
        else: raise TypeError("Unrecognized argument 'on' for 'maskshow'. ")
    elif canvas is None:
        canvas = (1.,) * 3
    if len(masks) == 0: return
    new_masks = []
    for m in masks:
        img = to_image(m, nslice, dim, orient_as, orient_to)
        if img.ndim == 3:
            new_masks.extend(x.squeeze(-1) for x in img.split(1, dim=dim))
        else:
            new_masks.append(img)
    color_mask_map = [(to_RGB(c), m) for c, m in zip(colors*(len(new_masks) // len(colors) + 1), new_masks)]
    for c, m in kwargs.items():
        if touch(lambda: to_RGB(c)) is not None: 
            color_mask_map.append((to_RGB(c), m))
            kwargs.pop(c)

    new_masks = [m for _, m in color_mask_map]
    shapes = [m.ishape for _, m in color_mask_map]
    if not stretch:
        target_shape = shapes[0]
        if len(set(shapes)) > 1 or not isinstance(canvas, tuple) and target_shape != canvas.shape[:2]:
            raise TypeError("Please use masks of the same size as the background image, "
                            "or use 'stretch=True' in 'maskshow' to automatically adjust the image sizes. ")
    else:
        def adjust(m, to):
            ms = tuple(m.shape)
            scaling = tuple((a // b, b // a) for a, b in zip(to, ms))
            return m.down_scale([max(v, 1) for u, v in scaling]).up_scale([max(u, 1) for u, v in scaling]).crop_as(to)
        if not isinstance(canvas, tuple): shapes.append(canvas.shape[:2])
        areas = [u * v for u, v in shapes]
        target_shape = shapes[areas.index(max(areas))]
        color_mask_map = [(c, adjust(m, to=target_shape)) for c, m in color_mask_map]
        canvas = adjust(canvas, to=target_shape)
    if isinstance(canvas, tuple):
        canvas = bt.channel_tensor(canvas).expand_to(target_shape + bt.Size({3},)).with_channeldim(None)

    plots = []
    if on is not None: plots.append(imshow(create_subplot=True))
    if mask_alpha > 0: plots.append(maskshow(*new_masks, alpha=mask_alpha))
    else: plots.append(imshow())
    for color, mask in color_mask_map:
        curves = border(mask, min_length)
        for c in curves:
            if isinstance(c, bt.torch.Tensor):
                if c.device.type == 'cuda': c = c.cpu()
                c = c.detach().cpu().numpy()
            plots.append(plt.plot(c[1], c[0], color = color, **kwargs))
    return plots

def fieldshow(disp, on=None, nslice=None, dim=-1, stretch=False, **kwargs):
    """
    fieldshow shows displacements with RGB channels to form a facke colored map 
        indicating different directions by different color and magnitude by brightness.
    """
    global canvas
    orient_as = kwargs.pop('as_orient', None)
    orient_to = kwargs.pop('to_orient', None)
    if on is not None:
        if isinstance(on, (int, tuple)): background(*on)
        elif isinstance(on, array): canvas = to_image(on, nslice, dim, orient_as, orient_to)
        elif isinstance(on, list): canvas = to_image(bt.tensor(on), nslice, dim, orient_as, orient_to)
        else: raise TypeError("Unrecognized argument 'on' for 'maskshow'. ")
    elif canvas is None:
        canvas = (1.,) * 3
	
    if isinstance(disp, Transformation):
        if canvas is None or isinstance(canvas, tuple) or not hasattr(on, 'ndim') or on.ndim != disp.n_dim:
            raise TypeError("Please input data in 'imshow' or 'background' before trying to display grid. ")
        disp = disp.toDDF(on.space).pick(0, [])
    elif canvas is None or isinstance(canvas, tuple): canvas = bt.zeros(disp.space)
    disp = disp.abs()
    if orient_as is not None: disp = reorient(disp, from_orient=orient_as, to_orient=orient_to)
    disp = to_image(disp, nslice, dim)
    if disp.size(2) > 3: disp = disp[..., :3]
    return imshow(disp, create_subplot = on is not None, **kwargs)

def gridshow(disp, on=None, gap=None, nslice=None, dim=-1, stretch=False, **kwargs):
    """
    gridshow shows grid for the displacements. 
    """
    global canvas
    orient_as = kwargs.pop('as_orient', None)
    orient_to = kwargs.pop('to_orient', None)
    if on is not None:
        if isinstance(on, (int, tuple)): background(*on)
        elif isinstance(on, array): canvas = to_image(on, nslice, dim, orient_as, orient_to)
        elif isinstance(on, list): canvas = to_image(bt.tensor(on), nslice, dim, orient_as, orient_to)
        else: raise TypeError("Unrecognized argument 'on' for 'maskshow'. ")
    elif canvas is None:
        canvas = (1.,) * 3
	
    if isinstance(disp, Transformation):
        if canvas is None or isinstance(canvas, tuple) or not hasattr(on, 'ndim') or on.n_space_dim != disp.n_dim:
            raise TypeError("Please input data for displacement in 'imshow' or 'background' before trying to display grid: need size of the canvas. ")
        disp = disp.toDDF(bt.tensor(on).space)
    elif canvas is None or isinstance(canvas, tuple): canvas = bt.zeros(disp.space)
    if disp.has_batch: disp = disp.sample(random=False, dim=[])
    if orient_as is not None: disp = reorient(disp, from_orient=orient_as, to_orient=orient_to)
    Y = DenseDisplacementField(disp)(bt.image_grid(disp.space)).squeeze([])
    Y = to_image(Y, nslice, dim).movedim(-1, 0)
    if Y.size(0) > 2:
        if dim < 0: dim += Y.size(0)
        Y = bt.cat(Y[:dim], Y[dim+1:], {})

    plots = [imshow(canvas, create_subplot = on is not None)]
    plt.ylim((Y.space[0], 0))
    plt.xlim((0, Y.space[1]))
    if gap is None: gap = Y.space[0] // 8, Y.space[1] // 8
    gap = to_tuple(gap)
    if len(gap) == 1: gap = gap * 2
    if 'color' not in kwargs: kwargs['color'] = 'gold'
    if isinstance(Y, bt.torch.Tensor):
        if Y.device.type == 'cuda': Y = Y.cpu()
        Y_data = Y.detach().cpu().numpy()
    for row in list(bt.arange(0, Y.space[0], gap[0])) + [Y.space[0] - 1]:
        plots.append(plt.plot(Y_data[1, row, :], Y_data[0, row, :], **kwargs))
    for col in list(bt.arange(0, Y.space[1], gap[0])) + [Y.space[1] - 1]:
        plots.append(plt.plot(Y_data[1, :, col], Y_data[0, :, col], **kwargs))
    return plots
