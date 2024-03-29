
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'micomputing',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.1.46',
    contact = 'bertiezhou@163.com',
    keywords = ['medical image', 'image registration', 'image similarities'],
    description = "'micomputing' is a package for medical image computing. ",
    requires = ['numpy', 'torch>=1.5.1', 'batorch', 'matplotlib', 'pycamia', 'pyoverload', 'nibabel', 'pydicom', 'SimpleITK'],
    update = '2023-07-10 20:53:03',
    package_data = True
).check()
__version__ = '1.1.46'

from . import plot as plt
from .stdio import IMG, dcm2nii, nii2dcm #*
from .data import Info, Subject, ImageObject, Dataset, MedicalDataset #*
from .network import U_Net, CNN, FCN
from .funcs import reorient, rescale, reflect, dilate, blur, bending, distance_map, registration, local_prior, center_of_gravity #*
from .trans import Transformation, SpatialTransformation, ImageTransformation, ComposedTransformation, CompoundTransformation, Identity, Id, Rotation90, Rotation180, Rotation270, Reflect, Reflection, Permutedim, DimPermutation, Rescale, Rescaling, Translate, Translation, Rigid, Rig, Affine, Aff, PolyAffine, logEu, LocallyAffine, LARM, FreeFormDeformation, FFD, DenseDisplacementField, DDF, MultiLayerPerception, MLP, Normalize, resample, interpolation, interpolation_forward, Affine2D2Matrix, Quaterns2Matrix, Matrix2Quaterns #*
from .metrics import metric, ITKMetric, ITKLabelMetric, MutualInformation, NormalizedMutualInformation, KLDivergence, CorrelationOfLocalEstimation, NormalizedVectorInformation, Cos2Theta, SumSquaredDifference, MeanSquaredErrors, PeakSignalToNoiseRatio, CrossEntropy, CrossCorrelation, NormalizedCrossCorrelation, StructuralSimilarity, Dice, DiceScore, DiceScoreCoefficient, LabelDice, LabelDiceScore, LabelDiceScoreCoefficient, ITKDiceScore, ITKJaccardCoefficient, ITKVolumeSimilarity, ITKFalsePositive, ITKFalseNegative, ITKHausdorffDistance, ITKMedianSurfaceDistance, ITKAverageSurfaceDistance, ITKDivergenceOfSurfaceDistance, ITKLabelDiceScore, ITKLabelJaccardCoefficient, ITKLabelVolumeSimilarity, ITKLabelFalsePositive, ITKLabelFalseNegative, ITKLabelHausdorffDistance, ITKLabelMedianSurfaceDistance, ITKLabelAverageSurfaceDistance, ITKLabelDivergenceOfSurfaceDistance, LocalNonOrthogonality, RigidProjectionError #*
