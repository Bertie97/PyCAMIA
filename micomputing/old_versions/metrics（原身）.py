
from pycamia import info_manager
from pycamia.exception import avouch

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
    Dice
    LabelDice
    LabelDiceScore
    LabelJaccardCoefficient
    LabelVolumeSimilarity
    LabelFalsePositive
    LabelFalseNegative
    LabelHausdorffDistance
    LabelMedianSurfaceDistance
    LabelAverageSurfaceDistance
    LabelDivergenceOfSurfaceDistance
    LocalNonOrthogonality
    RigidProjectionError
""".split()

with __info__:
    import torch
    import batorch as bt
    import numpy as np
    import SimpleITK as sitk
    from pycamia import to_tuple

######### Section 1: Information Based ########
eps = 1e-6

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

def dBspline_WRT_I1(i, U):
    '''
    The derivative of Bspline function with respect to I1.
    i, U: indices and decimal parts. size: (n_batch, n_hist, n_data)
    '''
    return dBspline(i[:, 0], U[:, 0]) * Bspline(i[:, 1], U[:, 1])

def dBspline_WRT_I2(i, U):
    '''
    The derivative of Bspline function with respect to I2.
    i, U: indices and decimal parts. size: (n_batch, n_hist, n_data)
    '''
    return Bspline(i[:, 0], U[:, 0]) * dBspline(i[:, 1], U[:, 1])

class JointHistogram(bt.autograd.Function):

    @staticmethod
    def forward(ctx, I1, I2, nbin=100):
        with bt.no_grad():
            if hasattr(ctx, 'JH'): del ctx.JH
            nbin = bt.tensor(nbin)
            data_pair = bt.stack(I1.flatten(1), I2.flatten(1), dim={1})
            nbatch, nhist, ndata = data_pair.ishape
            indices = []; values = []
            ctx.window = (bt.image_grid(4, 4) - 1).flatten(1).transpose(0, 1)
            for shift in ctx.window:
                shift = bt.channel_tensor(shift)
                # [nbatch] x {nhist} x ndata
                hist_pos = data_pair * nbin
                index = bt.clamp(bt.floor(hist_pos).long() + shift, 0, nbin - 1)
                batch_idx = bt.arange(nbatch).expand_to([nbatch], {1}, ndata)
                index = bt.cat(batch_idx, index, 1)
                value = Bspline(shift.expand_to(data_pair), bt.decimal(hist_pos)).prod(1)
                indices.append(index)
                values.append(value)
            # n_batch x (1 + n_hist) x (n_data x 4 ** n_hist)
            Mindices = bt.cat(indices, -1)
            # n_batch x (n_data x 4 ** n_hist)
            Mvalues = bt.cat(values, -1)
            # (1 + n_hist) x (n_batch x n_data x 4 ** n_hist)
            indices = Mindices.transpose(0, 1).flatten(1)
            # (n_batch x n_data x 4 ** n_hist)
            values = Mvalues.flatten(0)
            if indices.device == torch.device('cpu'): creator = torch.sparse.FloatTensor
            else: creator = torch.cuda.sparse.FloatTensor
            collected = creator(indices, values, (nbatch, nbin, nbin)).to_dense()
            collected = bt.Tensor(collected, batch_dim=0)

            ctx.nbin = nbin
            ctx.Ishape = I1.shape
            ctx.data_pair = data_pair
            ctx.JH = collected / ndata
        return ctx.JH

    @staticmethod
    def backward(ctx, grad_output):
        with bt.no_grad():
            nbin = ctx.nbin
            data_pair = ctx.data_pair
            nbatch, nhist, ndata = data_pair.ishape
            dPdI1 = bt.zeros(ctx.Ishape)
            dPdI2 = bt.zeros(ctx.Ishape)
            for shift in ctx.window:
                # [nbatch] x {nhist} x ndata
                shift = shift.view(1, 2, 1)
                hist_pos = data_pair * nbin
                index = torch.clamp(torch.floor(hist_pos).long() + shift, 0, nbin - 1)
                grad_y = grad_output[(slice(None),) + index.split(1, 1)].squeeze(2)
                value = grad_y.gather(0, bt.arange(nbatch).long().unsqueeze(0).unsqueeze(-1).repeat(1, 1, ndata)).view(ctx.Ishape)
                dPdI1 += value * dBspline_WRT_I1(shift, bt.decimal(data_pair * nbin)).view(ctx.Ishape)
                dPdI2 += value * dBspline_WRT_I2(shift, bt.decimal(data_pair * nbin)).view(ctx.Ishape)
        return dPdI1, dPdI2, None

def MutualInformation(A, B, nbin=100):
    func = 'MutualInformation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - bt.sum(Pab * bt.log2(bt.where(Pab < eps, bt.ones_like(Pab), Pab)), [1, 2])
    Hx = - bt.sum(Pa * bt.log2(bt.where(Pa < eps, bt.ones_like(Pa), Pa)), 1)
    Hy = - bt.sum(Pb * bt.log2(bt.where(Pb < eps, bt.ones_like(Pb), Pb)), 1)
    return Hx + Hy - Hxy

def NormalizedMutualInformation(A, B, nbin=100):
    func = 'NormalizedMutualInformation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - bt.sum(Pab * bt.log2(bt.where(Pab < eps, bt.ones_like(Pab), Pab)), [1, 2])
    Hx = - bt.sum(Pa * bt.log2(bt.where(Pa < eps, bt.ones_like(Pa), Pa)), 1)
    Hy = - bt.sum(Pb * bt.log2(bt.where(Pb < eps, bt.ones_like(Pb), Pb)), 1)
    return (Hx + Hy) / Hxy

def KLDivergence(A, B, nbin=100):
    func = 'KLDivergence'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    return (Pa * bt.log2(bt.where(Pb < eps, bt.ones_like(Pa), Pa / Pb.clamp(min=eps)).clamp(min=eps))).sum(1)

###############################################

######## Section 2: Cross Correlation #########

def local_matrix(A, B, s=0, kernel="Gaussian", kernel_size=3):
    if isinstance(kernel, str):
        if kernel.lower() == "gaussian": kernel = bt.gaussian_kernel(n_dims = A.nspace, kernel_size = kernel_size).unsqueeze(0, 0)
        elif kernel.lower() == "mean": kernel = bt.ones(*(kernel_size,) * A.nspace).unsqueeze(0, 0) / (kernel_size ** A.nspace)
    elif hasattr(kernel, 'shape'): kernel_size = kernel.size(-1)

    def mean(a):
        op = eval("bt.nn.functional.conv%dd"%A.nspace)
        if a.has_batch: x = a.unsqueeze({1})
        else: x = a.unsqueeze([0], {1})
        return op(x, kernel, padding = kernel_size // 2).squeeze(*((1,) if a.has_batch else (0, 0)))

    if s > 0:
        GA = bt.grad(A)
        GB = bt.grad(B)
        point_estim = bt.stack(bt.dot(GA, GA), bt.dot(GA, GB), bt.dot(GB, GB), dim={int(A.has_batch)})
    else: point_estim = 0

    MA = mean(A)
    MB = mean(B)
    local_estim = bt.stack(mean(A * A) - MA ** 2, mean(A * B) - MA * MB, mean(B * B) - MB ** 2, dim={int(A.has_batch)})

    return s * point_estim + local_estim

def CorrelationOfLocalEstimation(A, B, s=0, kernel="Gaussian", kernel_size=3):
    func = 'CorrelationOfLocalEstimation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    S11, S12, S22 = local_matrix(A, B, s=s, kernel=kernel, kernel_size=kernel_size).split()
    return (bt.divide(S12 ** 2, S11 * S22, tol=eps).squeeze(1) + eps).sqrt().mean()

###############################################

########## Section 3: Local Gradient ##########

def NormalizedVectorInformation(A, B):
    func = 'NormalizedVectorInformation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    GA = bt.grad(A)
    GB = bt.grad(B)
    return bt.divide(bt.dot(GA, GB) ** 2, bt.dot(GA, GA) * bt.dot(GB, GB), tol=eps).mean()

def Cos2Theta(A, B):
    func = 'Cos2Theta'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    GA = bt.grad(A)
    GB = bt.grad(B)
    return bt.divide(bt.dot(GA, GB) ** 2, bt.dot(GA, GA) * bt.dot(GB, GB), tol=eps)

###############################################

####### Section 4: Intensity Difference #######

def SumSquaredDifference(A, B):
    func = 'SumSquaredDifference'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    return ((A - B) ** 2).sum()

def MeanSquaredErrors(A, B):
    func = 'MeanSquaredErrors'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    return ((A - B) ** 2).mean()

def PeakSignalToNoiseRatio(A, B):
    func = 'PeakSignalToNoiseRatio'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    return 10 * bt.log10(bt.max((A.max(), B.max())) ** 2 / ((A - B) ** 2).mean())

###############################################

##### Section 5: Distribution Similarity ######

def CrossEntropy(y, label):
    func = 'CrossEntropy'
    avouch(isinstance(y, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(label, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(y.has_batch and label.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")
    avouch(y.has_channel and label.has_channel, f"Please make sure inputs of '{func}' has channel dimensions to calculate entropy along." +
           "Use X.channel_dim = 0 to identify (or X.unsqueeze({{}}) if no existed channel, though this should not be commonly seen).")

    ce = - label * bt.log(y.clamp(1e-10, 1.0))
    return ce.sum(ce.channel_dimension).mean()

def CrossCorrelation(A, B):
    func = 'CrossCorrelation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    dA = A - A.mean(); dB = B - B.mean()
    return (dA * dB).sum()

def NormalizedCrossCorrelation(A, B):
    func = 'NormalizedCrossCorrelation'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    dA = A - A.mean(); dB = B - B.mean()
    return (dA * dB).sum() / (dA ** 2).sum().sqrt() / (dB ** 2).sum().sqrt()

def StructuralSimilarity(A, B, k1=0.01, k2=0.03):
    func = 'StructuralSimilarity'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    varA = ((A - A.mean()) ** 2).mean()
    varB = ((B - B.mean()) ** 2).mean()
    covAB = ((A - A.mean()) * (B - B.mean())).mean()
    L = bt.max((A.max(), B.max()))
    c1, c2 = k1 * L, k2 * L
    num = (2 * A.mean() * B.mean() + c1 ** 2) * (2 * covAB + c2 ** 2)
    den = (A.mean() ** 2 + B.mean() ** 2 + c1 ** 2) * (varA + varB + c2 ** 2)
    return num / den

###############################################

########## Section 6: Region Overlap ##########

def Dice(A, B):
    '''
    The Dice score between A and B where A and B are 0-1 masks. 
    The sizes are as follows: 
    A: ([n_batch], {n_label}, n_1, n_2, ..., n_|n_dim|)
    B: ([n_batch], {n_label}, n_1, n_2, ..., n_|n_dim|)
    return: ([n_batch], {n_label})
    OR:
    A: ([n_batch], n_1, n_2, ..., n_k)
    B: ([n_batch], n_1, n_2, ..., n_k)
    return: ([n_batch],)
    '''
    func = 'Dice'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    ABsum = A.sum() + B.sum()
    return 2 * (A * B).sum() / (ABsum + eps)

def LabelDice(A, B, class_labels=None):
    '''
    The Dice score between A and B where A and B are integer label maps. 
    
    Params:
        A [bt.Tensor]: label map 1 with size ([n_batch], n_1, ..., n_|n_dim|).
        B [bt.Tensor]: label map 2 with size ([n_batch], n_1, ..., n_|n_dim|).
        class_labels [list or NoneType]: integers representing different labels, a list of length `n_class`. 
            If it is not given, it will be automatically detected by collecting all sorted labels in A and B. 
            It is time consuming, especially if A and B are accidentally float images. Please be careful when using this default. 
        
    output [bt.Tensor]: the Dice scores for each label. 
        size: ([n_batch], {n_class})
    '''
    func = 'LabelDice'
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    if not class_labels: class_labels = sorted(list(set(A.unique().tolist() + B.unique().tolist())))
    A_labels = [1 - bt.clamp(bt.abs(A - i), 0, 1) for i in class_labels]
    B_labels = [1 - bt.clamp(bt.abs(B - i), 0, 1) for i in class_labels]
    A_maps = bt.stack(A_labels, {1})
    B_maps = bt.stack(B_labels, {1})
    return Dice(A_maps, B_maps)

###############################################

######### Section 7: Surface distance #########
class SurfaceDistanceImageFilter:
    def __init__(self): self.all_dis = bt.tensor([0])
    def Execute(self, A, B):
        array = sitk.GetArrayViewFromImage
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
    The metrics between A and B where A and B are 0-1 masks. 
    The sizes are as follows: 
    A: ([n_batch], n_1, n_2, ..., n_k)
    B: ([n_batch], n_1, n_2, ..., n_k)
    return: ([n_batch],)
    
    Note that function 'Metric' does not accept ([n_batch], {n_label}, n_1, n_2, ..., n_|n_dim|) images as function 'Dice' does due to the inner logic of it. 
        Please manually use X.mergedims({}, []) to merge label dimension into batch dimension to do the computation. 
    '''
    func = 'Metric ' + metric
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    A = np.array(A) != 0
    B = np.array(B) != 0
    spacing = to_tuple(spacing)
    n_dim = A.ndim
    n_batch = A.shape[0]
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
    return bt.tensor(measures).batch_dimension_(0)

def LabelMetric(A, B, spacing = 1, metric = "HD", class_labels = None):
    '''
    The metrics between A and B where A and B are integer label maps. 
    
    Params:
        A [bt.Tensor]: label map 1 with size ([n_batch], n_1, ..., n_|n_dim|).
        B [bt.Tensor]: label map 2 with size ([n_batch], n_1, ..., n_|n_dim|).
        class_labels [list or NoneType]: integers representing different labels, a list of length `n_class`. 
            If it is not given, it will be automatically detected by collecting all sorted labels in A and B. 
            It is time consuming, especially if A and B are accidentally float images. Please be careful when using this default. 
        
    output [bt.Tensor]: the metric values for each label. 
        size: ([n_batch], {n_class})
    '''
    func = 'LabelMetric ' + metric
    avouch(isinstance(A, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(isinstance(B, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(A.has_batch and B.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")

    A = np.array(A)
    B = np.array(B)
    if not class_labels: class_labels = sorted(list(set(np.unique(A.astype(np.int)).tolist() + np.unique(B.astype(np.int)).tolist())))
    n_batch = A.shape[0]
    n_class = len(class_labels)
    A_labels = [A == i for i in class_labels]
    B_labels = [B == i for i in class_labels]
    A_maps = np.concatenate(A_labels)
    B_maps = np.concatenate(B_labels)
    metric = Metric(A_maps, B_maps, spacing, metric)
    return bt.tensor(metric.reshape((n_class, n_batch)).T).batch_dimension_(0).channel_dimension_(1)

template1 = "def m{metric}(*args, **kwargs): return Metric(*args, **kwargs, metric = '{metric}')"
template2 = "def mLabel{metric}(*args, **kwargs): return LabelMetric(*args, **kwargs, metric = '{metric}')"
for metric in ('Dice', 'Jaccard', 'Volume', 'FalsePositive', 'FalseNegative', 'HD', 'MSD', 'ASD', 'STDSD'):
    exec(template1.format(metric=metric))
    exec(template2.format(metric=metric))

def LabelDiceScore(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='Dice')
def LabelJaccardCoefficient(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='Jaccard')
def LabelVolumeSimilarity(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='Volume')
def LabelFalsePositive(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='FalsePositive')
def LabelFalseNegative(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='FalseNegative')
def LabelHausdorffDistance(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='HD')
def LabelMedianSurfaceDistance(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='MSD')
def LabelAverageSurfaceDistance(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='ASD')
def LabelDivergenceOfSurfaceDistance(A, B, spacing = 1, class_labels=None):
    return LabelMetric(A, B, spacing=spacing, class_labels=class_labels, metric='STDSD')

###############################################

########## Section 6: Region Overlap ##########

def LocalNonOrthogonality(D, mask = None):
    '''
        Local non-orthogonality metric defined for a displacement D [?].
            Please use trans.toDDF(shape) to convert a transformation into a displacement field first before using the function. 
            Please refer to the reference [?] for more information. 
        [?] To be added. 

        D [bt.Tensor]: The displacements for calculation. 
            size: ([n_batch], {n_dim}, n_1, ..., n_|n_dim|)
        mask [bt.Tensor or NoneType]: The mask in which we calculate the metric. It is the whole image by default. 
            size: ([n_batch], n_1, ..., n_|n_dim|)
        output [bt.Tensor]: Values of size ([n_batch],)
    '''
    func = 'LabelMetric ' + metric
    avouch(isinstance(D, bt.Tensor), f"Please use 'batorch.Tensor' objects for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(mask is None or isinstance(mask, bt.Tensor), f"Please use 'batorch.Tensor' object 'mask' for '{func}' in 'micomputing'. Use bt.tensor(X) to create one. ")
    avouch(D.has_batch, f"Please make sure inputs of '{func}' has batch dimensions. Use X.batch_dim = 0 to identify (or X.unsqueeze([]) if no existed batch).")
    avouch(D.has_channel, f"Please make sure input displacement of '{func}' has a channel dimension for the coordinates. Use X.channel_dim = 0 to identify. ")

    n_batch = D.n_batch
    if mask is None: mask = bt.ones(n_batch, *D.space)
    mask = mask.float()
    gd = bt.grad(D) # of size ([n_batch], n_dim, {n_dim}, n_1, ..., n_|n_dim|)
    JacOfDelta = gd.flatten(3).mergedims(3, 0).T # ([n_batch x n_data], {n_dim}, n_dim)
    JacOfDelta.channel_dimension = None
    JacOfPoints = JacOfDelta + bt.eye_like(JacOfDelta)
    RigOfPoints = bt.Fnorm2(JacOfPoints.T @ JacOfPoints - bt.eye_like(JacOfPoints)).view([n_batch], -1)
    return (RigOfPoints * bt.crop_as(mask, gd.shape[3:]).flatten(1)).sum() / mask.sum()
    
def RigidProjectionError(D, mask = None):
    '''
        Rigid projection error measures the rigidity of a matrix by evaluating the distance between it and its projection on rigid matrices. 
            This is defined for a displacement D [?]. Please use trans.toDDF(shape) to convert a transformation into a displacement field first before using the function. 
            Please refer to the reference [?] for more information. 
        [?] To be added. 

        D [bt.Tensor]: The displacements for calculation. 
            size: ([n_batch], {n_dim}, n_1, ..., n_|n_dim|)
        mask [bt.Tensor or NoneType]: The mask in which we calculate the metric. It is the whole image by default. 
            size: ([n_batch], n_1, ..., n_|n_dim|)
        output [bt.Tensor]: Values of size ([n_batch],)
    '''
    def K(r):
        '''r: ([n_batch], n_dim); K(r): ([n_batch], n_dim, n_dim)'''
        return bt.cross_matrix(r)
    def Q(r):
        '''r: ([n_batch], n_dim + 1); Q(r): ([n_batch], n_dim + 1, n_dim + 1)'''
        r13, r4 = r.split([r.size(1) - 1, 1], 1)
        return bt.cat(bt.cat(r4 * bt.eye(r13) + K(r13), -r13.unsqueeze(1), 1), r.unsqueeze(-1), -1)
    def W(r):
        '''r: ([n_batch], n_dim + 1); W(r): ([n_batch], n_dim + 1, n_dim + 1)'''
        r13, r4 = r.split([r.size(1) - 1, 1], 1)
        return bt.cat(bt.cat(r4 * bt.eye(r13) - K(r13), -r13.unsqueeze(1), 1), r.unsqueeze(-1), -1)
    def R(r):
        '''r: ([n_batch], n_dim + 1); R(r): ([n_batch], n_dim, n_dim)'''
        n_dim = r.size(1) - 1
        return (W(r).T @ Q(r))[..., :n_dim, :n_dim]
    def T(r, s):
        '''r: ([n_batch], n_dim + 1); s: ([n_batch], n_dim + 1); T(r, s): ([n_batch], n_dim + 1, n_dim + 1)'''
        n_batch = r.n_batch
        n_dim = r.size(1) - 1
        t = 2 * (W(r).T @ s)[..., :n_dim]
        return bt.cat(bt.cat(R(r), bt.zeros([n_batch], 1, n_dim), 1), bt.cat(t.unsqueeze(-1), bt.ones([n_batch], 1, 1), 1), -1)
    def max_eigvec(A):
        '''A: ([n_batch], n_dim + 1, n_dim + 1); max_eigvec(A): ([n_batch], n_dim + 1)'''
        n_batch = A.n_batch
        n_dim = A.size(1) - 1
        x = bt.ones([n_batch], n_dim + 1)
        for _ in range(5): x += 2e-3 * ((x**2).sum() * (A @ x) - (x * (A @ x)).sum() * x) / (x**2).sum() ** 2
        return x / bt.norm(x) # eigenvalue: ((x * (A @ x)).sum() / (x**2).sum())
    
    n_batch = D.n_batch
    n_dim = D.n_channel
    X = bt.image_grid(D).unsqueeze([]).float()
    Y = X + D
    if n_dim == 2:
        X = bt.cat(X, bt.zeros([n_batch], {1}), 1)
        Y = bt.cat(Y, bt.zeros([n_batch], {1}), 1)
    maxes = X.flatten(2).max(2).values.max({}).values
    p_mod = (X / maxes).amplify(n_batch, []).flatten(2).mergedims(2, 0).with_channeldim(None)
    p_obs = (Y / maxes).flatten(2).mergedims(2, 0).with_channeldim(None) # ([n_batch x n_data], {n_dim})
    # p_mod = bt.zeros_like(p_mod)
    # p_obs = p_obs - p_mod
    matW = W(bt.cat(p_mod, bt.zeros([p_mod.n_batch], 1), 1) / 2)
    matQ = Q(bt.cat(p_obs, bt.zeros([p_obs.n_batch], 1), 1) / 2) # ([n_batch x n_data], n_dim + 1, n_dim + 1)
    if mask is None: mask = bt.ones([n_batch], *D.shape[2:])
    C1 = - 2 * matQ.T @ matW
    C1 = (C1.view([n_batch], -1, n_dim + 1, n_dim + 1) * mask.flatten(1).unsqueeze(-1, -1)).sum(1) # ([n_batch], n_dim + 1, n_dim + 1)
    if n_dim == 2: C1 -= 2 * bt.diag([-1, -1, 1, 1]).multiply(n_batch, [])
    C2 = mask.sum() * bt.eye(C1)
    C3 = 2 * ((matW - matQ).view([n_batch], -1, n_dim + 1, n_dim + 1) * mask.flatten(1).unsqueeze(-1, -1)).sum(1)
    A = (C3.T @ C3 / (2 * mask.sum()) - C1 - C1.T) / 2 # A = (C3.T @ (C2 + C2.T).inv() @ C3 - C1 - C1.T) / 2; it can be simplified by the substitution of C2
    r = max_eigvec(A)
    print(A, r)
    s = - C3 @ r / (2 * mask.sum()) # s = - (C2 + C2.T).inv() @ C3 @ r; it can be simplified by the substitution of C2
    matT = T(r, s)
    hatY = (matT @ bt.cat((X / maxes).amplify(n_batch, []), bt.ones([n_batch], {1}), 1).flatten(2).with_channeldim(None))[:, :-1].view_as(Y) * maxes
    print(matT[0], X[0, :, 20, 20, 20], hatY[0, :, 20, 20, 20])
    return bt.meannorm2(bt.Fnorm(Y - hatY))

###############################################

# Metric abbreviations
def metric(key):
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
    JCD = LabelJaccardCoefficient,
    VS = LabelVolumeSimilarity,
    FP = LabelFalsePositive,
    FN = LabelFalseNegative,
    HD = LabelHausdorffDistance,
    MdSD = LabelMedianSurfaceDistance,
    ASD = LabelAverageSurfaceDistance,
    MSD = LabelAverageSurfaceDistance,
    divSD = LabelDivergenceOfSurfaceDistance,
    stdSD = LabelDivergenceOfSurfaceDistance,
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
        JCD = LabelJaccardCoefficient,
        VS = LabelVolumeSimilarity,
        FP = LabelFalsePositive,
        FN = LabelFalseNegative,
        HD = LabelHausdorffDistance,
        MdSD = LabelMedianSurfaceDistance,
        ASD = LabelAverageSurfaceDistance,
        MSD = LabelAverageSurfaceDistance,
        divSD = LabelDivergenceOfSurfaceDistance,
        stdSD = LabelDivergenceOfSurfaceDistance,
        LNO = LocalNonOrthogonality,
        RPE = RigidProjectionError
    )[key]
