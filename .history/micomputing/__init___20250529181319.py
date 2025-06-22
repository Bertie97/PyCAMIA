
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'micomputing',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.1.49',
    contact = 'bertiezhou@163.com',
    keywords = ['medical image', 'image registration', 'image similarities'],
    description = "'micomputing' is a package for medical image computing. ",
    requires = ['numpy', 'torch>=1.5.1', 'batorch', 'matplotlib', 'pycamia', 'pyoverload', 'nibabel', 'pydicom', 'SimpleITK'],
    update = '2024-12-24 16:23:03',
    package_data = True
).check()
__version__ = '1.1.49'
from .trans import Transformation, ComposedTransformation, CompoundTransformation, interpolation, interpolation_forward, Identity, Id, Rotation90, Rotation180, Rotation270, Reflect, Reflection, Permutedim, DimPermutation, Rescale, Rescaling, rand_Rescaling, Translate, Translation, rand_Translation, CenterAlign, Rigid, Rig, rand_Rigidity, rand_Rig, Affine, Aff, rand_Affinity, rand_Aff, PolyAffine, rand_PolyAffine, logEuclidean, logEu, rand_logEuclidean, rand_logEu, LocallyAffine, LARM, rand_LocallyAffine, rand_LARM, FreeFormDeformation, FFD, rand_FreeFormDeformation, rand_FFD, DenseDisplacementField, DDF, rand_DenseDisplacementField, rand_DDF, VelocityField, VF, rand_VelocityField, rand_VF, Normalize, Cropping #*
from . import plot as plt
from .stdio import IMG, dcm2nii, nii2dcm, affine2orient, affine2spacing, orient2affine, spacing2affine, dim_of_orient_axis #*
from .data import Key, DataObject, BatchObject, Dataset, LossCollector #*
from .network import U_Net, CNN, FCN, RNN, NeuralODE, NeuralODEMarch, CoupledNeuralODEMarch, Convolution_Block, Convolution, Models #*
from .funcs import reorient, rescale, dilate, blur, bending, distance_map, registration, local_prior, center_of_gravity #*
# from .trans import Transformation, WoldCoordsTransformation, ImageCoordsTransformation, IntensityTransformation, ComposedTransformation, CompoundTransformation, Identity, Id, Rotation90, Rotation180, Rotation270, Reflect, Reflection, Permutedim, DimPermutation, Rescale, Rescaling, Translate, Translation, Rigid, Rig, Affine, Aff, PolyAffine, logEu, LocallyAffine, LARM, FreeFormDeformation, FFD, DenseDisplacementField, DDF, MultiLayerPerception, MLP, Normalize, resample, interpolation, interpolation_forward, Affine2D2Matrix, Quaterns2Matrix, Matrix2Quaterns #*
from .metrics import metric, ITKMetric, ITKLabelMetric, MutualInformation, NormalizedMutualInformation, KLDivergence, CorrelationOfLocalEstimation, NormalizedVectorInformation, Cos2Theta, SumSquaredDifference, MeanSquaredErrors, PeakSignalToNoiseRatio, CrossEntropy, CrossCorrelation, NormalizedCrossCorrelation, StructuralSimilarity, Dice, DiceScore, DiceScoreCoefficient, LabelDice, LabelDiceScore, LabelDiceScoreCoefficient, ITKDiceScore, ITKJaccardCoefficient, ITKVolumeSimilarity, ITKFalsePositive, ITKFalseNegative, ITKHausdorffDistance, ITKMedianSurfaceDistance, ITKAverageSurfaceDistance, ITKDivergenceOfSurfaceDistance, ITKLabelDiceScore, ITKLabelJaccardCoefficient, ITKLabelVolumeSimilarity, ITKLabelFalsePositive, ITKLabelFalseNegative, ITKLabelHausdorffDistance, ITKLabelMedianSurfaceDistance, ITKLabelAverageSurfaceDistance, ITKLabelDivergenceOfSurfaceDistance, LocalNonOrthogonality, RigidProjectionError, joint_hist #*
