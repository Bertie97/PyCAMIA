
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
""".split()

with __info__:
    import torch
    import batorch as bt
    import SimpleITK as sitk

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
    THe derivative of Bspline function with respect to I2.
    i, U: n_batch x n_hist x n_data
    '''
    return dBspline(i[:, 0], U[:, 0]) * Bspline(i[:, 1], U[:, 1])

def dBspline_WRT_I2(i, U):
    '''
    THe derivative of Bspline function with respect to I2.
    i, U: n_batch x n_hist x n_data
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
            if bt.Device == bt.DeviceCPU: creator = torch.sparse.FloatTensor
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
    assert A.has_batch and B.has_batch
    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - bt.sum(Pab * bt.log2(bt.where(Pab < eps, bt.ones_like(Pab), Pab)), [1, 2])
    Hx = - bt.sum(Pa * bt.log2(bt.where(Pa < eps, bt.ones_like(Pa), Pa)), 1)
    Hy = - bt.sum(Pb * bt.log2(bt.where(Pb < eps, bt.ones_like(Pb), Pb)), 1)
    return Hx + Hy - Hxy

def NormalizedMutualInformation(A, B, nbin=100):
    assert A.has_batch and B.has_batch
    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - bt.sum(Pab * bt.log2(bt.where(Pab < eps, bt.ones_like(Pab), Pab)), [1, 2])
    Hx = - bt.sum(Pa * bt.log2(bt.where(Pa < eps, bt.ones_like(Pa), Pa)), 1)
    Hy = - bt.sum(Pb * bt.log2(bt.where(Pb < eps, bt.ones_like(Pb), Pb)), 1)
    return (Hx + Hy) / Hxy

def KLDivergence(A, B, nbin=100):
    assert A.has_batch and B.has_batch
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
        GA = bt.grad_image(A)
        GB = bt.grad_image(B)
        point_estim = bt.stack(bt.dot(GA, GA), bt.dot(GA, GB), bt.dot(GB, GB), dim={int(A.has_batch)})
    else: point_estim = 0

    MA = mean(A)
    MB = mean(B)
    local_estim = bt.stack(mean(A * A) - MA ** 2, mean(A * B) - MA * MB, mean(B * B) - MB ** 2, dim={int(A.has_batch)})

    return s * point_estim + local_estim

def CorrelationOfLocalEstimation(A, B, s=0, kernel="Gaussian", kernel_size=3):
    assert A.has_batch and B.has_batch
    S11, S12, S22 = local_matrix(A, B, s=s, kernel=kernel, kernel_size=kernel_size).split()
    return (bt.divide(S12 ** 2, S11 * S22, tol=eps).squeeze(1) + eps).sqrt().mean()

###############################################

########## Section 3: Local Gradient ##########

def NormalizedVectorInformation(A, B):
    assert A.has_batch and B.has_batch
    GA = bt.grad_image(A)
    GB = bt.grad_image(B)
    return bt.divide(bt.dot(GA, GB) ** 2, bt.dot(GA, GA) * bt.dot(GB, GB), tol=eps).mean()

def Cos2Theta(A, B):
    assert A.has_batch and B.has_batch
    GA = bt.grad_image(A)
    GB = bt.grad_image(B)
    return bt.divide(bt.dot(GA, GB) ** 2, bt.dot(GA, GA) * bt.dot(GB, GB), tol=eps)

###############################################

####### Section 4: Intensity Difference #######

def SumSquaredDifference(A, B):
    assert A.has_batch and B.has_batch
    return ((A - B) ** 2).sum()

def MeanSquaredErrors(A, B):
    assert A.has_batch and B.has_batch
    return ((A - B) ** 2).mean()

def PeakSignalToNoiseRatio(A, B):
    assert A.has_batch and B.has_batch
    return 10 * bt.log10(bt.max((A.max(), B.max())) ** 2 / ((A - B) ** 2).mean())

###############################################

##### Section 5: Distribution Similarity ######

def CrossEntropy(y, label):
    assert y.has_batch and label.has_batch and y.has_channel and label.has_channel
    ce = - label * bt.log(y.clamp(1e-10, 1.0))
    return ce.sum(ce.channel_dimension).mean()

def CrossCorrelation(A, B):
    assert A.has_batch and B.has_batch
    dA = A - A.mean(); dB = B - B.mean()
    return (dA * dB).sum()

def NormalizedCrossCorrelation(A, B):
    assert A.has_batch and B.has_batch
    dA = A - A.mean(); dB = B - B.mean()
    return (dA * dB).sum() / (dA ** 2).sum().sqrt() / (dB ** 2).sum().sqrt()

def StructuralSimilarity(A, B, k1=0.01, k2=0.03):
    assert A.has_batch and B.has_batch
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
    assert A.has_batch and B.has_batch
    ABsum = A.sum() + B.sum()
    return 2 * (A * B).sum() / (ABsum + eps)

def LabelDice(A, B, class_labels=None):
    '''
    :param A: (n_batch, n_1, ..., n_k)
    :param B: (n_batch, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    assert A.has_batch and B.has_batch
    if not class_labels: class_labels = sorted(A.unique().tolist() + B.unique().tolist())
    A_labels = [1 - bt.clamp(bt.abs(A - i), 0, 1) for i in class_labels]
    B_labels = [1 - bt.clamp(bt.abs(B - i), 0, 1) for i in class_labels]
    A_maps = bt.stack(A_labels, {1})
    B_maps = bt.stack(B_labels, {1})
    return Dice(A_maps, B_maps)

###############################################

######### Section 7: Surface distance #########
# not done yet.
class SurfaceDistanceImageFilter:
    def __init__(self): self.all_dis = totensor([0])
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
        self.all_dis = totensor(np.concatenate((aBDis, bADis), 0))
        
    def GetHausdorffDistance(self): return self.all_dis.max()
    def GetMedianSurfaceDistance(self): return self.all_dis.median()
    def GetAverageSurfaceDistance(self): return self.all_dis.mean()
    def GetDivergenceOfSurfaceDistance(self): return self.all_dis.std()

@torchext
def Metric(A, B, spacing = 1, metric = "HD"):
    '''
    A: (n_batch, n_1, n_2, ..., n_k)
    B: (n_batch, n_1, n_2, ..., n_k)
    return: (n_batch,)
    '''
    A = tonumpy(A) != 0
    B = tonumpy(B) != 0
    spacing = totuple(spacing)
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

@torchext
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

template1 = "@torchext\ndef m{metric}(*args, **kwargs): return Metric(*args, **kwargs, metric = '{metric}')"
template2 = "@torchext\ndef mLabel{metric}(*args, **kwargs): return LabelMetric(*args, **kwargs, metric = '{metric}')"
for metric in ('Dice', 'Jaccard', 'Volume', 'FalseNegative', 'FalsePositive', 'HD', 'MSD', 'ASD', 'STDSD'):
    exec(template1.format(metric=metric))
    exec(template2.format(metric=metric))

###############################################

# Metric abbreviations
metric = dict(
    MI = MutualInformation,
    NMI = NormalizedMutualInformation,
    KL = KLDivergence,
    CLE = CorrelationOfLocalEstimation,
    NVI = NormalizedVectorInformation,
    SSD = SumSquaredDifference,
    MSE = MeanSquaredErrors,
    PSNR = PeakSignaltoNoiseRatio,
    CE = CrossEntropy,
    CC = CrossCorrelation,
    NCC = NormalizedCrossCorrelation,
    SSIM = StructuralSimilarity,
    DSC = LabelDice
)
