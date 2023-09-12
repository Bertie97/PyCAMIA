
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Unit test for overload @typehint.",
    requires = ["pycamia", "pyoverload", "unittest"]
)

import unittest
with __info__:
    from pycamia import avouch
    import numpy as np
    import batorch as bt
    from pyoverload import *
    
class DtypeTests(unittest.TestCase):
    
    def test_get_dtype_details(self):
        def unzip(x): return dtype(x).module, dtype(x).name, dtype(x).bits
        avouch(unzip(int) == (None, 'int', None), f"{unzip(int)!r}")
        avouch(unzip('int') == (None, 'int', None), f"{unzip('int')!r}")
        avouch(unzip(long) == (None, 'int', 64), f"{unzip(long)!r}")
        avouch(unzip(bt.int) == ('torch', 'int', 32), f"{unzip(bt.int)!r}")
        avouch(unzip(np.int32) == ('numpy', 'int', 32), f"{unzip(np.int32)!r}")
        avouch(unzip(bt.uint8) == ('torch', 'uint', 8), f"{unzip(bt.uint8)!r}")
    
    def test_subclass_relations(self):
        avouch(issubclass(int, real))
        avouch(issubclass(float, real))
        avouch(issubclass(long, int))
        avouch(issubclass(double, float))
        avouch(issubclass(dtype(bt.int16), int))
        avouch(issubclass(dtype(bt.int32), int))
        avouch(issubclass(dtype(bt.int64), int))
        avouch(not issubclass(dtype(bt.int16), dtype(bt.int32)))
        avouch(not issubclass(dtype(bt.int16), long))
        avouch(issubclass(dtype(bt.int64), long))
        avouch(issubclass(dtype(bt.float32), float))
        avouch(issubclass(dtype(bt.float64), float))
        avouch(not issubclass(dtype(bt.float32), double))
        avouch(issubclass(dtype(bt.float64), double))

    def test_tensor_shape(self):
        avouch(isinstance(bt.zeros(1)[0], float)) # size of len 0
        avouch(isinstance(bt.zeros(20, 10), float[20][10]))
        avouch(isinstance(np.eye(10), real[...][...]))

if __name__ == "__main__":
    unittest.main()