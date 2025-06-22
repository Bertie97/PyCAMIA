
import ctypes
import numpy as np
import micomputing as mc
from pycamia import Path

dll = ctypes.cdll.LoadLibrary(Path(__file__).parent/'micfunctions.so')
distance_map_func = dll.distance_map
distance_map_func.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS"),
                              np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS"),
                              ctypes.c_int,
                              np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS"),
                              np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS")]

masks = ((np.stack(np.meshgrid(np.arange(100), np.arange(100), indexing='ij'), 0) - np.array((20, 20)).reshape((2, 1, 1))) ** 2).sum(0) < 100
mask_sent = masks.flatten().astype(np.int32)
dismap_get = np.zeros_like(mask_sent).flatten().astype(np.float32)
size_in = np.array((1, 100, 100)).astype(np.int32)
spacing_in = np.array((1, 1)).astype(np.float32)
distance_map_func(mask_sent, dismap_get, 3, size_in, spacing_in)
dismap_get = dismap_get.reshape(masks.shape)

mc.plot.subplots(2)
mc.plot.imshow(masks)
mc.plot.imshow(dismap_get)
mc.plot.show()
