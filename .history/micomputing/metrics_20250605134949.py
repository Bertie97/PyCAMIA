
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File containing commonly used similarity measures in medical image analysis. ",
    help = "Use `metric['ABBR.'](A, B)` to compute the similarity.",
    requires = "SimpleITK"
).check()

__all__ = """
    metric
    ITKMetric
    ITKLabelMetric
    MutualInformation
    NormalizedMutualInformation
    KLDivergence
    CorrelationOfLocalEstimation
    NormalizedVectorInformation
    Cos2Theta
    SumSquaredDifference
    MeanSquaredErrors
    PeakSignalToNoiseRatio
    CrossEntropy
    CrossCorrelation
    NormalizedCrossCorrelation
    StructuralSimilarity
    Dice DiceScore DiceScoreCoefficient
    LabelDice LabelDiceScore LabelDiceScoreCoefficient
    ITKDiceScore
    ITKJaccardCoefficient
    ITKVolumeSimilarity
    ITKFalsePositive
    ITKFalseNegative
    ITKHausdorffDistance
    ITKMedianSurfaceDistance
    ITKAverageSurfaceDistance
    ITKDivergenceOfSurfaceDistance
    ITKLabelDiceScore
    ITKLabelJaccardCoefficient
    ITKLabelVolumeSimilarity
    ITKLabelFalsePositive
    ITKLabelFalseNegative
    ITKLabelHausdorffDistance
    ITKLabelMedianSurfaceDistance
    ITKLabelAverageSurfaceDistance
    ITKLabelDivergenceOfSurfaceDistance
    LocalNonOrthogonality
    RigidProjectionError
    
    joint_hist
""".split()

with __info__:
    import torch
    import batorch as bt
    import numpy as np
    import SimpleITK as sitk
    from pycamia import to_tuple, avouch, Version, alias

######### Section 1: Information Based ########
eps = 1e-6

@bt.batorch_wrapper
def Bspline(i: bt.Tensor, U: bt.Tensor):
    """
    Cubic B-spline function. 
    Args: Segment i with argument U; i and U are of the same size. 
    """
    return (
        bt.where(i == -1, (1 - U) ** 3 / 6,
        bt.where(i == 0, U ** 3 / 2 - U * U + 2 / 3,
        bt.where(i == 1, (- 3 * U ** 3 + 3 * U * U + 3 * U + 1) / 6,
        bt.where(i == 2, U ** 3 / 6,
        bt.zeros_like(U)))))
    )

@bt.batorch_wrapper
def dBspline(i: bt.Tensor, U: bt.Tensor):
    """
    Derivative of cubic B-spline function. 
    Args: Segment i with argument U; i and U are of the same size. 
    """
    return (
        bt.where(i == -1, - 3 * (1 - U) ** 2 / 6,
        bt.where(i == 0, 3 * U ** 2 / 2 - 2 * U,
        bt.where(i == 1, (- 3 * U ** 2 + 2 * U + 1) / 2,
        bt.where(i == 2, 3 * U ** 2 / 6,
        bt.zeros_like(U)))))
    )

@bt.batorch_wrapper
def multi_Bspline(i: bt.Tensor, U: bt.Tensor):
    """
    Multi-variate cubic B-spline function. 
    Args: Segment i with argument U; i and U are of the same size:
        ({n_batch}, [n_image], n_1, ..., n_r) [r=n_dim] 
    """
    return Bspline(i, U).prod([])

@bt.batorch_wrapper
def multi_dBspline(i: bt.Tensor, U: bt.Tensor, j):
    """
    Derivative of j-th term of multi-variate cubic B-spline function. 
    Args: Segment i with argument U; i and U are of the same size:
        ({n_batch}, [n_image], n_1, ..., n_r) [r=n_dim] 
    """
    return Bspline(i, U)[:, :j].prod([]) * dBspline(i, U)[:, j] * Bspline(i, U)[:, j+1:].prod([])

class JointHistogram(bt.autograd.Function):
    
    @staticmethod
    @bt.batorch_wrapper
    def forward(ctx, I1: bt.Tensor, I2: bt.Tensor, nbin=100, mask=None):
        """batorch version of JointHistogram.
        """
        if mask is None: mask = bt.ones_like(I1)
        with bt.no_grad():
            n_dim = I1.n_space_dim
            n_bin = nbin
            data_pair = bt.stack(I1.flatten(1), I2.flatten(1), dim=[]) # ({n_batch}, [n_image=2], n_data)
            n_batch, n_image, n_data = tuple(data_pair.shape)
            device = data_pair.device
            indices = []; values = []
            l_window = 4
            window = (bt.image_grid([l_window] * n_image) - 1).flatten(...).transpose(0, 1).to(device) # (l_window ^ n_image, [n_image])
            for shift in window: # ([n_image],)
                hist_pos = data_pair * n_bin # ({n_batch}, [n_image], n_data)
                index = bt.clamp(bt.floor(hist_pos).long() + shift, 0, n_bin - 1)
                batch_idx = bt.arange(n_batch, device=device).view({n_batch}, [1], 1).repeat(1, 1, n_data)
                index = bt.cat(batch_idx, index, []) # ({n_batch}, [n_image + 1], n_data)
                value = multi_Bspline(shift.expand_to(data_pair), bt.decimal(hist_pos)) # ({n_batch}, n_data)
                value *= mask.flatten(...) # MASK HERE: ({n_batch}, n_data)
                indices.append(index)
                values.append(value)
            # ({n_batch}, [1 + n_image], n_data x l_window ^ n_image)
            Mindices = bt.cat(indices, -1)
            # ({n_batch}, n_data x l_window ^ n_image)
            Mvalues = bt.cat(values, -1)
            # ([1 + n_image], n_batch x n_data x l_window ^ n_image)
            indices = Mindices.merge_dims({}, -1)
            # (n_batch x n_data x l_window ^ n_image,)
            values = Mvalues.flatten(0)
            if Version(bt.torch.__version__) >= Version("2"):
                creator = lambda *x: bt.torch.sparse_coo_tensor(*x, device=indices.device)
            elif indices.device.type == 'cpu': creator = bt.torch.sparse.FloatTensor
            else: creator = bt.torch.cuda.sparse.FloatTensor
            collected = creator(indices, values, (n_batch,) + (n_bin,) * n_image).to_dense()
            collected = bt.Tensor(collected, batch_dim=True) / mask.sum(...) # ({n_batch}, n_bin, ..., n_bin)

            # JH = collected / n_data
            JH = collected
            ctx.save_for_backward(bt.tensor(list(I1.shape)), data_pair, JH, mask)
        return collected

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: ({n_batch}, n_bin, ..., n_bin)
        grad_output = grad_output.as_subclass(bt.Tensor).init_special().with_batch_dim(True)
        Ishape, data_pair, _, mask = ctx.saved_tensors
        device = grad_output.device
        n_bin = grad_output.size(-1)
        Ishape = tuple(Ishape.int().tolist())
        with bt.no_grad():
            # data_pair: ({n_batch}, [n_image=2], n_data)
            n_batch, n_image, n_data = data_pair.shape
            dEdI1 = bt.zeros({n_batch}, n_data).to(device) # ({n_batch}, n_1, ..., n_r) [r=n_dim]
            dEdI2 = bt.zeros({n_batch}, n_data).to(device)
            if Version(bt.__version__) >= '1.9': window = bt.stack(bt.meshgrid(*([bt.arange(-1, 3)] * n_image), indexing='ij'), 0).flatten(1).transpose(0, 1).to(device) # (4 ^ n_image, n_image)
            else: window = bt.stack(bt.meshgrid(*([bt.arange(-1, 3)] * n_image), indexing='ij'), 0).flatten(1).transpose(0, 1).to(device) # (4 ^ n_image, n_image)
            for shift in window:
                shift = shift.view({1}, [2], 1) # ({1}, [n_image], 1) -> (1, n_image, 1)
                hist_pos = data_pair * n_bin # ({n_batch}, [n_image], n_data) -> (n_batch, n_image, n_data)
                index = bt.clamp(bt.floor(hist_pos).long() + shift.long(), 0, n_bin - 1)
                decimal = hist_pos - bt.floor(hist_pos)
                value = grad_output[(bt.arange({n_batch}).long().duplicate(n_data, -1).to(device),) + tuple(x.squeeze(1) for x in index.split(1, 1))] # ({n_batch}, n_data) -> (n_batch, n_data)
                dEdI1 += value * multi_dBspline(shift, decimal, 0) # ({n_batch}, n_data) -> (n_batch, n_data)
                dEdI2 += value * multi_dBspline(shift, decimal, 1)
            # dEdI1 = bt.tensor(dEdI1.data).view(Ishape) * n_bin / n_data
            # dEdI2 = bt.tensor(dEdI2.data).view(Ishape) * n_bin / n_data
            n_dim = len(Ishape) - 1
            dEdI1 = dEdI1.data.view(Ishape) * mask * n_bin / mask.flatten(1).sum(1).view((n_batch,) + (1,) * n_dim)
            dEdI2 = dEdI2.data.view(Ishape) * mask * n_bin / mask.flatten(1).sum(1).view((n_batch,) + (1,) * n_dim)
        return dEdI1, dEdI2, None, None
        # Deprecated: using gather to pick elements is not necessary. 
        #         grad_y = grad_output[(slice(None),) + index.split(1, 1)].squeeze(2)
        #         value = grad_y.gather(0, bt.arange(nbatch).long().unsqueeze(0).unsqueeze(-1).repeat(1, 1, ndata)).view(ctx.Ishape)
        #         dPdI1 += value * multi_dBspline(shift, bt.decimal(data_pair * nbin), 0).view(ctx.Ishape)
        #         dPdI2 += value * multi_dBspline(shift, bt.decimal(data_pair * nbin), 1).view(ctx.Ishape)
        # return dPdI1, dPdI2, None

# class JointHistogram(bt.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, I1, I2, nbin=100, mask=None):
#         """
#         Estimate the joint histogram between I1 & I2. 
        
#         Args:
#             I1 (bt.Tensor): (n_batch, n@1, n@2, ..., n@n_dim)
#             I2 (bt.Tensor): (n_batch, n@1, n@2, ..., n@n_dim)
#             nbin (int): the size of histogram. Note that it should be set to smaller numbers if 
#                 the number of intensity is low. Otherwise, gradients would be too small.

#         Returns:
#             histogram (bt.Tensor): (n_batch, n_bin, ..., n_bin)
#         """
#         if isinstance(I1, bt.Tensor): I1 = I1.as_subclass(bt.Tensor)
#         if isinstance(I2, bt.Tensor): I2 = I2.as_subclass(bt.Tensor)
#         if isinstance(mask, bt.Tensor): mask = mask.as_subclass(bt.Tensor)
#         device = I1.device
#         if mask is None: mask = bt.ones_like(I1).to(device)
#         with bt.no_grad():
#             if hasattr(ctx, 'JH'): del ctx.JH
#             n_bin = bt.tensor(nbin).to(device)
#             n_dim = I1.ndim - 1
#             # mask_pos = bt.stack(bt.meshgrid(*[bt.arange(x) for x in mask.shape[1:]]), 0).unsqueeze(0).to(device) * mask.unsqueeze(1) # (n_batch, n_dim, n@1, ..., n@n_dim)
#             # level = bt.tensor(bt.cat((bt.tensor([1]), bt.cumprod(bt.tensor(mask.shape[:0:-1]), 0)[:-1]), 0).tolist()[::-1]).to(device)
#             # mask_select = bt.unique((mask_pos * level.view((1, n_dim) + (1,) * n_dim)).sum(1)).long()
#             # data_pair = bt.stack((I1.flatten(1), I2.flatten(1)), 1)[..., mask_select] # ({n_batch}, {n_hist=2}, n_data) -> (n_batch, n_hist, n_data)
#             data_pair = bt.stack((I1.flatten(1), I2.flatten(1)), 1) # ({n_batch}, {n_hist=2}, n_data) -> (n_batch, n_hist, n_data)
#             n_batch, n_hist, n_data = data_pair.shape
#             indices = []; values = []
#             if Version(bt.__version__) >= '1.9': window = bt.stack(bt.meshgrid(*([bt.arange(-1, 3)] * n_hist), indexing='ij'), 0).flatten(1).transpose(0, 1).to(device) # (4 ^ n_hist, n_hist)
#             else: window = bt.stack(bt.meshgrid(*([bt.arange(-1, 3)] * n_hist)), 0).flatten(1).transpose(0, 1).to(device) # (4 ^ n_hist, n_hist)
#             for shift in window: # (n_hist,)
#                 hist_pos = data_pair * n_bin # ({n_batch}, {n_hist}, n_data) -> (n_batch, n_hist, n_data)
#                 decimal = hist_pos - bt.floor(hist_pos)
#                 index = bt.clamp(bt.floor(hist_pos).long() + shift.view(1, -1, 1).long(), 0, n_bin - 1)
#                 batch_idx = bt.arange(n_batch).unsqueeze(-1).unsqueeze(-1).expand(n_batch, 1, n_data).to(device)
#                 index = bt.cat((batch_idx, index), 1) # ({n_batch}, {n_hist + 1}, n_data) -> (n_batch, n_hist + 1, n_data)
#                 value = Bspline(shift.view(1, -1, 1), decimal).prod(1) # ({n_batch}, n_data)
#                 # value *= mask.flatten(1) # MASK HERE: ({n_batch}, n_data)
#                 indices.append(index)
#                 values.append(value)
#             # ({n_batch}, {1 + n_hist}, n_data x 4 ^ n_hist)
#             Mindices = bt.cat(indices, -1)
#             # ({n_batch}, n_data x 4 ^ n_hist)
#             Mvalues = bt.cat(values, -1)
#             # ({1 + n_hist}, n_batch x n_data x 4 ^ n_hist)
#             indices = Mindices.transpose(0, 1).flatten(1)
#             # (n_batch x n_data x 4 ^ n_hist,)
#             values = Mvalues.flatten(0)
#             if indices.device == bt.device('cpu'): creator = bt.sparse.FloatTensor
#             else: creator = bt.cuda.sparse.FloatTensor
#             collected = creator(indices, values, (n_batch,) + (n_bin,) * n_hist).to_dense()
#             # collected = bt.Tensor(collected, batch_dim=True) # ({n_batch}, n_bin, ..., n_bin)

#             ctx.nbin = n_bin
#             ctx.Ishape = I1.shape
#             ctx.data_pair = data_pair
#             # ctx.JH = collected / mask.flatten(1).sum(1).unsqueeze(-1).unsqueeze(-1)
#             ctx.JH = collected / n_data
#             ctx.mask = mask
#         return ctx.JH

#     @staticmethod
#     def backward(ctx, grad_output):
#         # grad_output: (n_batch, n_bin, ..., n_bin)
#         device = grad_output.device
#         with bt.no_grad():
#             n_bin = ctx.nbin # (scalar)
#             data_pair = ctx.data_pair # ({n_batch}, {n_hist=2}, n_data) -> (n_batch, n_hist=2, n_data)
#             n_batch, n_hist, n_data = data_pair.shape
#             dEdI1 = bt.zeros(n_batch, n_data).to(device) # ({n_batch}, n@1, ..., n@n_dim) -> (n_batch, n@1, ..., n@n_dim)
#             dEdI2 = bt.zeros(n_batch, n_data).to(device)
#             if Version(bt.__version__) >= '1.9': window = bt.stack(bt.meshgrid(*([bt.arange(-1, 3)] * n_hist), indexing='ij'), 0).flatten(1).transpose(0, 1).to(device) # (4 ^ n_hist, n_hist)
#             else: window = bt.stack(bt.meshgrid(*([bt.arange(-1, 3)] * n_hist)), 0).flatten(1).transpose(0, 1).to(device) # (4 ^ n_hist, n_hist)
#             for shift in window:
#                 shift = shift.view(1, 2, 1) # ([1], {n_hist}, 1) -> (1, n_hist, 1)
#                 hist_pos = data_pair * n_bin # ({n_batch}, {n_hist}, n_data) -> (n_batch, n_hist, n_data)
#                 index = bt.clamp(bt.floor(hist_pos).long() + shift.long(), 0, n_bin - 1)
#                 decimal = hist_pos - bt.floor(hist_pos)
#                 value = grad_output[(bt.arange(n_batch).long().unsqueeze(-1).expand(n_batch, n_data).to(device),) + tuple(x.squeeze(1) for x in index.split(1, 1))] # ({n_batch}, n_data) -> (n_batch, n_data)
#                 dEdI1 += value * multi_dBspline(shift, decimal, 0) # ({n_batch}, n_data) -> (n_batch, n_data)
#                 dEdI2 += value * multi_dBspline(shift, decimal, 1)
#             n_dim = len(ctx.Ishape) - 1
#             # dEdI1 = dEdI1.data.view(ctx.Ishape) * ctx.mask * n_bin / ctx.mask.flatten(1).sum(1).view((n_batch,) + (1,) * n_dim)
#             # dEdI2 = dEdI2.data.view(ctx.Ishape) * ctx.mask * n_bin / ctx.mask.flatten(1).sum(1).view((n_batch,) + (1,) * n_dim)
#             dEdI1 = dEdI1.data.view(ctx.Ishape) * n_bin / n_data
#             dEdI2 = dEdI2.data.view(ctx.Ishape) * n_bin / n_data
#             # n_data = 1
#             # for x in ctx.Ishape[1:]: n_data *= x
#             # dEdI1_whole = bt.zeros(n_batch, n_data).to(device)
#             # dEdI1_whole[..., ctx.mask_select] = dEdI1
#             # dEdI1 = dEdI1_whole.data.view(ctx.Ishape) * n_bin / n_data
#             # dEdI2_whole = bt.zeros(n_batch, n_data).to(device)
#             # dEdI2_whole[..., ctx.mask_select] = dEdI2
#             # dEdI2 = dEdI2_whole.data.view(ctx.Ishape) * n_bin / n_data
#         return dEdI1, dEdI2, None, None

def joint_hist(A, B, nbin=100, mask=None):
    if not A.has_batch: A = A.unsqueeze({})
    if not B.has_batch: B = B.unsqueeze({})
    return JointHistogram.apply(A, B, nbin, mask)

def MutualInformation(A, B, nbin=100, mask=None):
    func = 'MutualInformation'
    n_batch, i_batch, i_channel = None, None, None
    if isinstance(A, bt.Tensor):
        if A.has_channel and A.has_batch: i_channel = A.channel_dim; A = A.mergedims([], {})
        if A.has_channel and not A.has_batch: i_channel = A.channel_dim; A.with_channel_dim(None).with_batch_dim(i_channel)
        if A.has_batch: n_batch = A.n_batch; i_batch = A.batch_dim; A = A.movedim({}, 0)
        # A = A.as_subclass(bt.Tensor)
    if isinstance(B, bt.Tensor):
        if B.has_channel and B.has_batch: B = B.mergedims([], {})
        if B.has_channel and not B.has_batch: B.with_channel_dim(None).with_batch_dim(i_channel)
        if B.has_batch: B = B.movedim({}, 0)
        # B = B.as_subclass(bt.Tensor)
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape: {A.shape} and {B.shape}")
    avouch(A.max() <= 1 and A.min() >= 0 and B.max() <= 1 and B.min() >= 0, f"Please make sure inputs of '{func}' are normalized images with intensity in [0, 1]. Currently of ranges [{A.min().item()}, {A.max().item()}] and [{B.min().item()}, {B.max().item()}].")

    Pab = JointHistogram.apply(A, B, nbin, mask)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - bt.sum(Pab * bt.log2(bt.where(Pab < eps, bt.ones_like(Pab), Pab)), [1, 2])
    Hx = - bt.sum(Pa * bt.log2(bt.where(Pa < eps, bt.ones_like(Pa), Pa)), 1)
    Hy = - bt.sum(Pb * bt.log2(bt.where(Pb < eps, bt.ones_like(Pb), Pb)), 1)
    MI = (Hx + Hy - Hxy).as_subclass(bt.Tensor).init_special()
    if i_batch is not None: MI.with_batch_dim(0).movedim_(0, i_batch)
    if i_channel is not None and i_channel == i_batch: MI.with_batch_dim(None).with_channel_dim(i_channel)
    if i_channel is not None and i_batch is not None: MI = MI.splitdim({}, {n_batch}, []).movedim([], i_channel)
    return MI

def NormalizedMutualInformation(A, B, nbin=100, mask=None):
    func = 'NormalizedMutualInformation'
    n_batch, i_batch, i_channel = None, None, None
    if isinstance(A, bt.Tensor):
        if A.has_channel and A.has_batch: i_channel = A.channel_dim; A = A.mergedims([], {})
        if A.has_channel and not A.has_batch: i_channel = A.channel_dim; A.with_channel_dim(None).with_batch_dim(i_channel)
        if A.has_batch: n_batch = A.n_batch; i_batch = A.batch_dim; A = A.movedim({}, 0)
    if isinstance(B, bt.Tensor):
        if B.has_channel and B.has_batch: B = B.mergedims([], {})
        if B.has_channel and not B.has_batch: B.with_channel_dim(None).with_batch_dim(i_channel)
        if B.has_batch: B = B.movedim({}, 0)
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape: {A.shape} and {B.shape}")
    avouch(A.max() <= 1 + eps and A.min() >= -eps and B.max() <= 1 + eps and B.min() >= -eps, f"Please make sure inputs of '{func}' are normalized images with intensity in [0, 1]. Currently of ranges [{A.min().item()}, {A.max().item()}] and [{B.min().item()}, {B.max().item()}].")

    Pab = JointHistogram.apply(A, B, nbin, mask)
    # Pab = JointHistogram.apply(A, B, nbin, mask)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - bt.sum(Pab * bt.log2(bt.where(Pab < eps, bt.ones_like(Pab), Pab)), [1, 2])
    Hx = - bt.sum(Pa * bt.log2(bt.where(Pa < eps, bt.ones_like(Pa), Pa)), 1)
    Hy = - bt.sum(Pb * bt.log2(bt.where(Pb < eps, bt.ones_like(Pb), Pb)), 1)
    NMI = bt.divide(Hx + Hy, Hxy, 0.0)
    if i_batch is not None: NMI.movedim_({}, i_batch)
    if i_channel is not None and i_channel == i_batch: NMI.with_batch_dim(None).with_channel_dim(i_channel)
    if i_channel is not None and i_batch is not None: NMI = NMI.splitdim({}, {n_batch}, []).movedim([], i_channel)
    return NMI

def KLDivergence(A, B, nbin=100):
    func = 'KLDivergence'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")
    avouch(A.max() <= 1 and A.min() >= 0 and B.max() <= 1 and B.min() >= 0, f"Please make sure inputs of '{func}' are normalized images with intensity in [0, 1]. Currently of ranges [{A.min().item()}, {A.max().item()}] and [{B.min().item()}, {B.max().item()}].")

    Pab = JointHistogram_bt.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    return (Pa * bt.log2(bt.where(Pb < eps, bt.ones_like(Pa), Pa / Pb.clamp(min=eps)).clamp(min=eps))).sum(1)

###############################################

######## Section 2: Cross Correlation #########

@bt.batorch_wrapper
def local_matrix(A: bt.Tensor, B: bt.Tensor, s=0, kernel="Gaussian", kernel_size=3):
    if isinstance(kernel, str):
        if kernel.lower() == "gaussian": kernel = bt.gaussian_kernel(n_dim = A.n_space_dim, kernel_size = kernel_size).unsqueeze(0, 0)
        elif kernel.lower() == "mean": kernel = bt.ones(*(kernel_size,) * A.n_space_dim).unsqueeze(0, 0) / (kernel_size ** A.n_space_dim)
    elif hasattr(kernel, 'shape'): kernel_size = kernel.size(-1)

    def mean(a):
        op = eval("bt.nn.functional.conv%dd"%A.n_space_dim)
        if a.has_batch and a.has_channel: x = a
        elif not a.has_channel: x = a.unsqueeze([])
        elif not a.has_batch: x = a.unsqueeze({})
        else: x = a.unsqueeze({}, [])
        return op(x, kernel, padding = kernel_size // 2).squeeze(*((1,) if a.has_batch else (0, 0)))

    if s > 0:
        GA = bt.grad_image(A, pad=True)
        GB = bt.grad_image(B, pad=True)
        point_estim = bt.stack(bt.dot(GA, GA), bt.dot(GA, GB), bt.dot(GB, GB), dim=bt.FuncDim)
    else: point_estim = 0

    MA = mean(A)
    MB = mean(B)
    local_estim = bt.stack(mean(A * A) - MA ** 2, mean(A * B) - MA * MB, mean(B * B) - MB ** 2, dim=bt.FuncDim)

    print(point_estim.shape, local_estim.shape)
    return s * point_estim + local_estim

def CorrelationOfLocalEstimation(A, B, s=0, kernel="Gaussian", kernel_size=3):
    func = 'CorrelationOfLocalEstimation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    S11, S12, S22 = local_matrix(A, B, s=s, kernel=kernel, kernel_size=kernel_size).split()
    return (bt.divide(S12 ** 2, S11 * S22, tol=eps).squeeze(1) + eps).sqrt().mean(...)

###############################################

########## Section 3: Local Gradient ##########

def NormalizedVectorInformation(A, B, mask=None):
    func = 'NormalizedVectorInformation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    GA = bt.grad_image(A, pad=True)
    GB = bt.grad_image(B, pad=True)
    if mask is None: mask = bt.ones_like(A)
    return (bt.divide(bt.dot(GA, GB) ** 2, bt.dot(GA, GA) * bt.dot(GB, GB), tol=eps) * mask).mean(...)

def Cos2Theta(A, B, mask=None):
    func = 'Cos2Theta'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    GA = bt.grad_image(A, pad=True)
    GB = bt.grad_image(B, pad=True)
    if mask is None: mask = bt.ones_like(A)
    return (bt.divide(bt.dot(GA, GB) ** 2, bt.dot(GA, GA) * bt.dot(GB, GB), tol=eps) * mask)

###############################################

####### Section 4: Intensity Difference #######

@alias("SumOfSquaredDifference")
def SumSquaredDifference(A, B, mask=None):
    func = 'SumSquaredDifference'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    return ((A - B) ** 2 * mask).sum(...)

@alias("MeanOfSquaredErrors")
def MeanSquaredErrors(A, B, mask=None):
    func = 'MeanSquaredErrors'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    return ((A - B) ** 2 * mask).mean(...)

def PeakSignalToNoiseRatio(A, B, mask=None):
    func = 'PeakSignalToNoiseRatio'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    return 10 * bt.log10(bt.max(((A * mask).max(), (B * mask).max())) ** 2 / ((A - B) ** 2 * mask).mean(...))

###############################################

##### Section 5: Distribution Similarity ######

def CrossEntropy(y, label):
    func = 'CrossEntropy'
    avouch(isinstance(y, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(label, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(y.has_batch and label.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(y.has_channel and label.has_channel, f"Please make sure inputs of '{func}' have channel dimensions to calculate entropy along." +
           "Use X.channel_dim = 0 to identify (or X.unsqueeze([]) if no existed channel, though this should not be commonly seen).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    ce = - label * bt.log(y.clamp(1e-10, 1.0))
    return ce.sum(ce.channel_dimension).mean(...)

def CrossCorrelation(A, B, mask=None):
    func = 'CrossCorrelation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    dA = A - (A * mask).sum(...) / mask.sum(...); dB = B - (B * mask).sum(...) / mask.sum(...)
    return (dA * dB * mask).sum(...)

def NormalizedCrossCorrelation(A, B, mask=None):
    func = 'NormalizedCrossCorrelation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    dA = A - (A * mask).sum(...) / mask.sum(...); dB = B - (B * mask).sum(...) / mask.sum(...)
    return (dA * dB * mask).sum(...) / (dA ** 2 * mask).sum(...).sqrt() / (dB ** 2 * mask).sum(...).sqrt()

def StructuralSimilarity(A, B, k1=0.01, k2=0.03, mask=None):
    func = 'StructuralSimilarity'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    A_mean = (A * mask).sum(...) / mask.sum(...)
    B_mean = (B * mask).sum(...) / mask.sum(...)
    dA = A - A_mean; dB = B - B_mean
    varA = (dA ** 2 * mask).mean(...)
    varB = (dB ** 2 * mask).mean(...)
    covAB = (dA * dB * mask).mean(...)
    L = bt.max(((A * mask).max(), (B * mask).max()))
    c1, c2 = k1 * L, k2 * L
    num = (2 * A_mean * B_mean + c1 ** 2) * (2 * covAB + c2 ** 2)
    den = (A_mean ** 2 + B_mean ** 2 + c1 ** 2) * (varA + varB + c2 ** 2)
    return num / den

###############################################

########## Section 6: Region Overlap ##########

@alias('Dice', 'DiceScore')
def DiceScoreCoefficient(A, B, mask=None):
    '''
    The Dice score between A and B where A and B are 0-1 masks. 
    The sizes are as follows: 
    A: ({n_batch}, [n_label: optional], n@1, n@2, ..., n@n_dim)
    B: ({n_batch}, [n_label: optional], n@1, n@2, ..., n@n_dim)
    return: ({n_batch}, [n_label: optional])
    '''
    func = 'Dice'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if mask is None: mask = bt.ones_like(A)
    ABsum = (A * mask).sum(...) + (B * mask).sum(...)
    return 2 * (A * B * mask).sum(...) / (ABsum + eps)

@alias('LabelDice', 'LabelDiceScore')
def LabelDiceScoreCoefficient(A, B, class_labels=None, mask=None):
    '''
    The Dice score between A and B where A and B are integer label maps. 
    
    Params:
        A [bt.Tensor]: label map 1 with size ({n_batch}, n@1, ..., n@n_dim).
        B [bt.Tensor]: label map 2 with size ({n_batch}, n@1, ..., n@n_dim).
        class_labels [list or NoneType]: integers representing different labels, a list of length `n_class`. 
            If it is not given, it will be automatically detected by collecting all sorted labels (except the minimum as background) in A and B. 
            It is time consuming, especially if A and B are accidentally float images. Please be careful when using this default. 
        
    output [bt.Tensor]: the Dice scores for each label. 
        size: ({n_batch}, {n_class})
    '''
    func = 'LabelDice'
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if not class_labels: class_labels = sorted(list(set(A.unique().tolist() + B.unique().tolist())))[1:]
    A_labels = [1 - bt.clamp(bt.abs(A - i), 0, 1) for i in class_labels]
    B_labels = [1 - bt.clamp(bt.abs(B - i), 0, 1) for i in class_labels]
    A_maps = bt.stack(A_labels, [1])
    B_maps = bt.stack(B_labels, [1])
    return Dice(A_maps, B_maps, mask=mask)

###############################################

######### Section 7: Surface distance #########
class SurfaceDistanceImageFilter:
    def __init__(self): self.all_dis = bt.tensor([0])
    def Execute(self, A, B):
        array = lambda x: np.array(sitk.GetArrayViewFromImage(x)).astype(np.float32)
        ADisMap = sitk.Abs(sitk.SignedMaurerDistanceMap(A, squaredDistance = False, useImageSpacing = True))
        BDisMap = sitk.Abs(sitk.SignedMaurerDistanceMap(B, squaredDistance = False, useImageSpacing = True))
        Asurface = sitk.LabelContour(A)
        Bsurface = sitk.LabelContour(B)
        
        # for a pixel 'a' in A, compute aBdis = dis(a, B)
        aBDis = array(BDisMap)[array(Asurface) > 0]
        # for a pixel 'b' in B, compute aBdis = dis(b, A)
        bADis = array(ADisMap)[array(Bsurface) > 0]
        if aBDis.size == 0 or bADis.size == 0:
            raise TypeError("Cannot detect surfaces. ")
        self.all_dis = bt.tensor(np.concatenate((aBDis, bADis), 0))
        
    def GetHausdorffDistance(self): return self.all_dis.max()
    def GetMedianSurfaceDistance(self): return self.all_dis.median()
    def GetAverageSurfaceDistance(self): return self.all_dis.mean()
    def GetDivergenceOfSurfaceDistance(self): return self.all_dis.std()

def ITKMetric(A, B, spacing = 1, metric = "HD"):
    '''
    The metrics between A and B where A and B are 0-1 masks. 
    The sizes are as follows: 
    A: ({n_batch}, [n_label: optional], n@1, n@2, ..., n@n_dim)
    B: ({n_batch}, [n_label: optional], n@1, n@2, ..., n@n_dim)
    return: ({n_batch}, [n_label: optional])
    '''
    func = 'Metric ' + metric
    avouch(isinstance(A, bt.Tensor), f"Please use 'bt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'bt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    n_batch = A.n_batch
    has_channel = False
    if A.has_channel and B.has_channel:
        has_channel = True
        n_channel = A.n_channel
        A = A.mergedims([], {})
        B = B.mergedims([], {})
    n_maps = A.n_batch
    filter_A = A.sum(...) != 0
    filter_B = B.sum(...) != 0
    filter_both = filter_A & filter_B
    A = A[filter_both]
    B = B[filter_both]
    A = A.numpy() != 0
    B = B.numpy() != 0
    spacing = to_tuple(spacing)
    n_dim = A.ndim
    n_data = A.shape[0]
    if len(spacing) == 1: spacing *= n_dim
    Overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    SD_filter = SurfaceDistanceImageFilter()
    Overlap_execs = {
        'Dice': lambda x: x.GetDiceCoefficient(),
        'Jaccard': lambda x: x.GetJaccardCoefficient(),
        'Volume': lambda x: x.GetVolumeSimilarity(),
        'Falsepositive': lambda x: x.GetFalsePositiveError(),
        'Falsenegative': lambda x: x.GetFalseNegativeError()
    }
    SD_execs = {
        'HD': lambda x: x.GetHausdorffDistance(),
        'MSD': lambda x: x.GetMedianSurfaceDistance(),
        'ASD': lambda x: x.GetAverageSurfaceDistance(),
        'STDSD': lambda x: x.GetDivergenceOfSurfaceDistance()
    }
    measures = np.zeros((n_data,))
    for b in range(n_data):
        ITKA = sitk.GetImageFromArray(A[b].astype(np.int32), isVector = False)
        ITKA.SetSpacing(spacing)
        ITKB = sitk.GetImageFromArray(B[b].astype(np.int32), isVector = False)
        ITKB.SetSpacing(spacing)
        metric = metric.capitalize()
        if metric in Overlap_execs:
            Overlap_filter.Execute(ITKA, ITKB)
            measures[b] = Overlap_execs[metric](Overlap_filter)
        metric = metric.upper()
        if metric in SD_execs:
            SD_filter.Execute(ITKA, ITKB)
            measures[b] = SD_execs[metric](SD_filter)
    
    resulting_measures = bt.zeros({n_maps})
    resulting_measures[filter_both] = bt.tensor_like(measures, target=resulting_measures)
    if has_channel:
        return resulting_measures.view({n_batch}, [n_channel])
    return resulting_measures.with_batch_dim(True)

def ITKLabelMetric(A, B, spacing = 1, metric = "HD", class_labels = None):
    '''
    The metrics between A and B where A and B are integer label maps. 
    
    Params:
        A [bt.Tensor]: label map 1 with size ({n_batch}, n@1, ..., n@n_dim).
        B [bt.Tensor]: label map 2 with size ({n_batch}, n@1, ..., n@n_dim).
        class_labels [list or NoneType]: integers representing different labels, a list of length `n_class`. 
            If it is not given, it will be automatically detected by collecting all sorted labels (except the minimum as background) in A and B. 
            It is time consuming, especially if A and B are accidentally float images. Please be careful when using this default. 
        
    output [bt.Tensor]: the metric values for each label. 
        size: ({n_batch}, [n_class])
    '''
    func = 'LabelMetric ' + metric
    avouch(isinstance(A, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' have batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(A.shape == B.shape, f"Please make sure inputs of '{func}' have the same shape.")

    if not class_labels: class_labels = sorted(list(set(A.unique().tolist() + B.unique().tolist())))[1:]
    n_batch = A.n_batch
    n_class = len(class_labels)
    A_labels = [A == i for i in class_labels]
    B_labels = [B == i for i in class_labels]
    A_maps = bt.stack(A_labels, [])
    B_maps = bt.stack(B_labels, [])
    metric = ITKMetric(A_maps, B_maps, spacing, metric)
    return metric.view([n_class], {n_batch}).transpose(0, 1)

def ITKLabelDiceScore(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='Dice')
def ITKLabelJaccardCoefficient(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='Jaccard')
def ITKLabelVolumeSimilarity(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='Volume')
def ITKLabelFalsePositive(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='FalsePositive')
def ITKLabelFalseNegative(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='FalseNegative')
def ITKLabelHausdorffDistance(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='HD')
def ITKLabelMedianSurfaceDistance(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='MSD')
def ITKLabelAverageSurfaceDistance(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='ASD')
def ITKLabelDivergenceOfSurfaceDistance(A, B, spacing = 1, class_labels=None):
    return ITKLabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='STDSD')

def ITKDiceScore(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='Dice')
def ITKJaccardCoefficient(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='Jaccard')
def ITKVolumeSimilarity(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='Volume')
def ITKFalsePositive(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='FalsePositive')
def ITKFalseNegative(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='FalseNegative')
def ITKHausdorffDistance(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='HD')
def ITKMedianSurfaceDistance(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='MSD')
def ITKAverageSurfaceDistance(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='ASD')
def ITKDivergenceOfSurfaceDistance(A, B, spacing = 1):
    return ITKMetric(A, B, spacing=spacing, metric='STDSD')

###############################################

########## Section 6: Region Overlap ##########

def LocalNonOrthogonality(Disp_world_coord, mask = None, target_affine=None):
    """
    Local non-orthogonality metric defined for a displacement Disp_world_coord [?].
        Please use trans.toDDF(shape) to convert a transformation into a displacement field first before using the function. 
        Use .to_world_space to transform it to world space if necessary.
        Please refer to the reference [?] for more information. 
    [?] To be added. 

    Args:
        Disp_world_coord (bt.Tensor): The displacements for calculation. 
            size: ({n_batch}, {n_dim}, n@1, ..., n@n_dim)
        mask (bt.Tensor or NoneType): The mask in which we calculate the metric. It is the whole image by default. 
            size: ({n_batch}, n@1, ..., n@n_dim)
        spacing (int or tuple): The spacing of space

    Returns:
        bt.Tensor: Values of size ({n_batch},)
    """
    from .funcs import dilate
    func = 'LocalNonOrthogonality'
    avouch(isinstance(Disp_world_coord, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(mask is None or isinstance(mask, bt.Tensor), f"Please use 'babt.Tensor' object 'mask' for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(Disp_world_coord.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(Disp_world_coord.has_channel, f"Please make sure input displacement of '{func}' has a channel dimension for the coordinates. Use X.channel_dim = 0 to identify. ")
    # spacing = to_tuple(spacing)
    # if len(spacing) < Disp_world_coord.n_space_dim: spacing *= Disp_world_coord.n_space_dim
    if target_affine is None: target_affine = lambda x: x

    n_batch = Disp_world_coord.n_batch
    if mask is None: mask = bt.ones(n_batch, *Disp_world_coord.space)
    mask = dilate((mask.float() > 0).float(), -1)
    X = target_affine(bt.image_grid(Disp_world_coord).duplicate(n_batch, {}).float()) # ({n_batch}, [n_dim], n@1, n@2, ..., n@n_dim)
    gd = bt.Jacobian(X, X + Disp_world_coord, dt=2) # of size ({n_batch}, n_dim, {n_dim}, n@1-dx, ..., n@n_dim-dx)
    JacOfPoints = gd.flatten(3).with_channel_dim(None).mergedims(3, 0) # ([n_batch x n_data(-dx)], n_dim, n_dim)
    RigOfPoints = bt.Fnorm(JacOfPoints.T @ JacOfPoints - bt.eye_like(JacOfPoints)).view({n_batch}, -1) # ({n_batch}, n_data(-dx))
    MaskedRig = RigOfPoints.abs().view({n_batch}, *tuple(x-2 for x in Disp_world_coord.space)) * mask[(slice(None),) + (slice(1, -1),) * Disp_world_coord.n_channel]
    # print(JacOfPoints[MaskedRig[0].argmax()])
    # from micomputing import plot as plt
    # plt.subplots(3)
    # plt.gridshow(Disp_world_coord[0], on=bt.ones(Disp_world_coord.space), color='gray', as_orient="ILP")
    # plt.imshow(RigOfPoints.abs().view({n_batch}, *tuple(x-2 for x in Disp_world_coord.space))[0], as_orient = "ILP")
    # plt.imshow(MaskedRig[0], as_orient = "ILP")
    # plt.show()
    return MaskedRig.sum(...) / mask.sum(...)
    # return (RigOfPoints * bt.crop_as(mask, gd.shape[3:]).flatten(1)).sum(...) / mask.sum(...)
    
def RigidProjectionError(Disp_world_coord, mask = None, target_affine=None):
    '''
        Rigid projection error measures the rigidity of a matrix by evaluating the distance between it and its projection on rigid matrices. 
            This is defined for a displacement Disp_world_coord [?]. Please use trans.toDDF(shape) to convert a transformation into a displacement 
            field first before using the function. Use .to_world_space to transform it to world space if necessary.
            Please refer to the reference [?] for more information. 
        [?] To be added. 

        Disp_world_coord [bt.Tensor]: The displacements for calculation. 
            size: ({n_batch}, [n_dim], n@1, ..., n@n_dim)
        mask [bt.Tensor or NoneType]: The mask in which we calculate the metric. It is the whole image by default. 
            size: ({n_batch}, [n_region:optional], n@1, ..., n@n_dim)
        output [bt.Tensor]: Values of size ({n_batch},)
    '''
    func = 'LocalNonOrthogonality'
    avouch(isinstance(Disp_world_coord, bt.Tensor), f"Please use 'babt.Tensor' objects for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(mask is None or isinstance(mask, bt.Tensor), f"Please use 'babt.Tensor' object 'mask' for '{func}' in 'micomputing'. Use X.as_subclass(bt.Tensor) to create one. ")
    avouch(Disp_world_coord.has_batch and mask.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze({{}}) if no existed batch).")
    avouch(Disp_world_coord.has_channel, f"Please make sure input displacement of '{func}' has a channel dimension for the coordinates. Use X.channel_dim = 0 to identify. ")
    
    def K(r):
        '''r: ({n_batch}, n_dim); K(r): ({n_batch}, n_dim, n_dim)'''
        return bt.cross_matrix(r)
    def Q(r):
        '''r: ({n_batch}, n_dim + 1); Q(r): ({n_batch}, n_dim + 1, n_dim + 1)'''
        r13, r4 = r.split([r.size(1) - 1, 1], 1)
        return bt.cat(bt.cat(r4 * bt.eye(r13) + K(r13), -r13.unsqueeze(1), 1), r.unsqueeze(-1), -1)
    def W(r):
        '''r: ({n_batch}, n_dim + 1); W(r): ({n_batch}, n_dim + 1, n_dim + 1)'''
        r13, r4 = r.split([r.size(1) - 1, 1], 1)
        return bt.cat(bt.cat(r4 * bt.eye(r13) - K(r13), -r13.unsqueeze(1), 1), r.unsqueeze(-1), -1)
    def R(r):
        '''r: ({n_batch}, n_dim + 1); R(r): ({n_batch}, n_dim, n_dim)'''
        n_dim = r.size(1) - 1
        return (W(r).T @ Q(r))[..., :n_dim, :n_dim]
    def t(r, s):
        '''r: ({n_batch}, n_dim + 1); s: ({n_batch}, n_dim + 1); t(r, s): ({n_batch}, n_dim)'''
        n_batch = r.n_batch
        n_dim = r.size(1) - 1
        return 2 * (W(r).T @ s)[..., :n_dim]
    def T(r, s):
        '''r: ({n_batch}, n_dim + 1); s: ({n_batch}, n_dim + 1); T(r, s): ({n_batch}, n_dim + 1, n_dim + 1)'''
        n_batch = r.n_batch
        n_dim = r.size(1) - 1
        return bt.cat(bt.cat(R(r), bt.zeros({n_batch}, 1, n_dim), 1), bt.cat(t(r, s).unsqueeze(-1), bt.ones({n_batch}, 1, 1), 1), -1)
    def max_eigvec(A):
        '''A: ({n_batch}, n_dim + 1, n_dim + 1); max_eigvec(A): ({n_batch}, n_dim + 1)'''
        n_batch = A.n_batch
        n_dim = A.size(1) - 1
        max_vec = bt.zeros({n_batch}, n_dim + 1)
        for b in range(n_batch):
            if Version(bt.bt.__version__) >= '1.10':
                l, v = bt.linalg.eig(A[b])
                max_vec[b] = bt.real(v)[:, bt.real(l).argmax()]
            else:
                l, v = bt.eig(A[b], eigenvectors=True)
                max_vec[b] = v[:, l[:, 0].argmax()]
        return max_vec
    def max_eigvec_iter(A):
        '''A: ({n_batch}, n_dim + 1, n_dim + 1); max_eigvec(A): ({n_batch}, n_dim + 1)'''
        n_batch = A.n_batch
        n_dim = A.size(1) - 1
        x = bt.ones({n_batch}, n_dim + 1)
        for _ in range(40): x += 2e-3 * ((x**2).sum(...) * (A @ x) - (x * (A @ x)).sum(...) * x) / (x**2).sum(...) ** 2
        return x / bt.norm(x) # eigenvalue: ((x * (A @ x)).sum(...) / (x**2).sum(...))
    
    n_batch = Disp_world_coord.n_batch
    if mask is None: mask = bt.ones({n_batch}, {1}, *Disp_world_coord.shape[2:])
    if mask.max() > 1:
        labels = sorted(mask.unique().tolist())[1:]
        mask = bt.stack([mask == l for l in labels], {})
    if not mask.has_channel:
        mask = mask.unsqueeze({})
    avouch(mask.has_channel, f"Please make sure mask of '{func}' has a channel dimension for the coordinates. Use X.channel_dim = 0 to identify. ")
    # spacing = to_tuple(spacing)
    # if len(spacing) < Disp_world_coord.n_space_dim: spacing *= Disp_world_coord.n_space_dim
    
    n_region = mask.n_channel
    X = bt.image_grid(Disp_world_coord).duplicate(n_batch, {}).float() # ({n_batch}, {n_dim}, n@1, n@2, ..., n@n_dim)
    if target_affine: X = target_affine(X)
    Y = X + Disp_world_coord
    if Disp_world_coord.n_channel == 2:
        X = bt.cat(X, bt.zeros({n_batch}, {1}), 1)
        Y = bt.cat(Y, bt.zeros({n_batch}, {1}), 1)
    n_dim = X.n_channel
    maxes = X.flatten(2).max(2).values.max({}).values # ({n_batch},)
    X = X * mask.unsqueeze({2})
    Y = Y * mask.unsqueeze({2}) # ({n_batch}, n_region, {n_dim}, n@1, n@2, ..., n@n_dim)
    p_mod = (X / maxes).flatten(3).mergedims(1, 3, 0).with_channel_dim(None)
    p_obs = (Y / maxes).flatten(3).mergedims(1, 3, 0).with_channel_dim(None) # ([n_batch x n_region x n_data], {n_dim})
    matW = W(bt.cat(p_mod, bt.zeros([p_mod.n_batch], 1), 1) / 2)
    matQ = Q(bt.cat(p_obs, bt.zeros([p_obs.n_batch], 1), 1) / 2) # ([n_batch x n_region x n_data], n_dim + 1, n_dim + 1)
    C1 = - 2 * matQ.T @ matW
    C1 = (C1.view({n_batch}, {n_region}, -1, n_dim + 1, n_dim + 1) * mask.flatten(2).unsqueeze(-1, -1)).sum(2) # ({n_batch}, {n_region}, n_dim + 1, n_dim + 1)
    if n_dim == 2: C1 -= 2 * bt.diag([-1, -1, 1, 1]).duplicate(n_batch, {}).duplicate(n_region, [])
    l = mask.sum(...).clamp(1)
    C2 = l * bt.eye(C1)
    C3 = 2 * ((matW - matQ).view({n_batch}, {n_region}, -1, n_dim + 1, n_dim + 1) * mask.flatten(2).unsqueeze(-1, -1)).sum(2)
    A = (C3.T @ C3 / (2 * l) - C1 - C1.T) / 2 # A = (C3.T @ (C2 + C2.T).inv() @ C3 - C1 - C1.T) / 2; it can be simplified by the substitution of C2
    r = max_eigvec(A.mergedims([], {}))
    s = - C3.mergedims([], {}) @ r / (2 * l.mergedims([], {})) # s = - (C2 + C2.T).inv() @ C3 @ r; it can be simplified by the substitution of C2
    matT = T(r, s).splitdim([], {n_batch}, {n_region})
    hatY = ((matT @ bt.cat(X.with_channel_dim(2) / maxes, bt.ones({n_batch}, n_region, {1}), {}).flatten(3).with_channel_dim(1))[:, :, :-1] * maxes).view({n_batch}, n_region, {n_dim}, *Disp_world_coord.shape[2:])
    return ((bt.Fnorm(Y - hatY).with_channel_dim(1) * mask).sum(...) / l).mean({})

###############################################

# Metric abbreviations
class Metric:
    @alias("__getitem__")
    def __call__(self, key):
        """
            List
            ----------
            MI = MutualInformation,
            NMI = NormalizedMutualInformation,
            KL = KLDivergence,
            CLE = CorrelationOfLocalEstimation,
            NVI = NormalizedVectorInformation,
            SSD = SumSquaredDifference,
            MSE = MeanSquaredErrors,
            PSNR = PeakSignalToNoiseRatio,
            CE = CrossEntropy,
            CC = CrossCorrelation,
            NCC = NormalizedCrossCorrelation,
            SSIM = StructuralSimilarity,
            DSC = LabelDiceScore,
            JCD = ITKLabelJaccardCoefficient,
            VS = ITKLabelVolumeSimilarity,
            FP = ITKLabelFalsePositive,
            FN = ITKLabelFalseNegative,
            HD = ITKLabelHausdorffDistance,
            MdSD = ITKLabelMedianSurfaceDistance,
            ASD = ITKLabelAverageSurfaceDistance,
            MSD = ITKLabelAverageSurfaceDistance,
            divSD = ITKLabelDivergenceOfSurfaceDistance,
            stdSD = ITKLabelDivergenceOfSurfaceDistance,
            LNO = LocalNonOrthogonality,
            RPE = RigidProjectionError
        """
        return dict(
            MI = MutualInformation,
            NMI = NormalizedMutualInformation,
            KL = KLDivergence,
            CLE = CorrelationOfLocalEstimation,
            NVI = NormalizedVectorInformation,
            SSD = SumSquaredDifference,
            MSE = MeanSquaredErrors,
            PSNR = PeakSignalToNoiseRatio,
            CE = CrossEntropy,
            CC = CrossCorrelation,
            NCC = NormalizedCrossCorrelation,
            SSIM = StructuralSimilarity,
            DSC = LabelDiceScore,
            JCD = ITKLabelJaccardCoefficient,
            VS = ITKLabelVolumeSimilarity,
            FP = ITKLabelFalsePositive,
            FN = ITKLabelFalseNegative,
            HD = ITKLabelHausdorffDistance,
            MdSD = ITKLabelMedianSurfaceDistance,
            ASD = ITKLabelAverageSurfaceDistance,
            MSD = ITKLabelAverageSurfaceDistance,
            divSD = ITKLabelDivergenceOfSurfaceDistance,
            stdSD = ITKLabelDivergenceOfSurfaceDistance,
            LNO = LocalNonOrthogonality,
            RPE = RigidProjectionError
       )[key]
metric = Metric()
