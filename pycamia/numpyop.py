
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2022-01",
    fileinfo = "File of numpy operations. ",
    requires = 'numpy'
)

__all__ = """
    toU
    toI
""".split()

try:
    with __info__:
        import numpy as np
    
    def toU(dt):
        """
        Convert a numpy dtype `dt` to unsigned version. 
        """
        if hasattr(dt, 'dtype'): x = dt; dt = dt.dtype
        dt = np.dtype(str(dt).split('.')[-1])
        if dt.kind == np.dtype(np.int16).kind: rdt = np.dtype('uint%d'%(8*dt.itemsize))
        else: rdt = dt
        if isinstance(x, np.ndarray): return x.astype(rdt)
        
    def toI(dt):
        """
        Convert a numpy dtype `dt` to signed version. 
        """
        if hasattr(dt, 'dtype'): x = dt; dt = dt.dtype
        dt = np.dtype(str(dt).split('.')[-1])
        if dt.kind == np.dtype(np.uint).kind: rdt = np.dtype('int%d'%(8*dt.itemsize))
        else: rdt = dt
        if isinstance(x, np.ndarray): return x.astype(rdt)

except: pass
