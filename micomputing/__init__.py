
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'micomputing',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.1.31',
    contact = 'bertiezhou@163.com',
    keywords = ['medical image', 'image registration', 'image similarities'],
    description = "'micomputing' is a package for medical image computing. ",
    requires = ['numpy', 'torch>=1.5.1', 'pycamia', 'pyoverload', 'batorch']
).check()

from .stdio import NII, DCM #*
from .sim import MutualInformation, NormalizedMutualInformation, KLDivergence, CorrelationOfLocalEstimation, NormalizedVectorInformation, Cos2Theta, SumSquaredDifference, MeanSquaredErrors, PeakSignalToNoiseRatio, CrossEntropy, CrossCorrelation, NormalizedCrossCorrelation, StructuralSimilarity, Dice, LabelDice #*














































































































