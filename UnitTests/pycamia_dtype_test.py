
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Unit test for pycamia loop service",
    requires = ["pycamia", "unittest"]
)

import unittest
import torch, numpy
with __info__:
    from pycamia import touch
    from pyoverload import dtype
    from pyoverload import dtypes

class DtypeTests(unittest.TestCase):

    def test_dtypes(self):
        error_func = '*' #lambda e: f"{e.__class__.__name__}: {e}"
        skipped = []
        for test_name, test_array in dict(
            torch_dtype = dtypes.torch_dtype, 
            torch_type = dtypes.torch_type, 
            numpy_type = dtypes.numpy_type
        ).items():
            for n in test_array:
                module = eval(test_name.split('_')[0])
                if not hasattr(module, n): skipped.append(n); continue
                x = getattr(module, n)
                print(
                    f"{test_name:<15s}{n:>20s}",
                    touch(lambda: '-' if dtypes.to_dtype(x).__name__ else 'x', default=error_func), 
                    touch(lambda: '-' if dtypes.to_torch_dtype(x) else 'x', default=error_func), 
                    touch(lambda: '-' if dtypes.to_torch_type(x) else 'x', default=error_func), 
                    touch(lambda: '-' if dtypes.to_numpy_dtype(x) else 'x', default=error_func), 
                    touch(lambda: '-' if dtypes.to_numpy_type(x) else 'x', default=error_func)
                )

        print("Skipped:", skipped)

if __name__ == "__main__":
    unittest.main()