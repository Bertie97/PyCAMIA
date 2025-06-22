
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-07-03",
    fileinfo = "Unit test for static c++ version of dilation",
    requires = ["batorch", "pycamia", "micomputing", "unittest"]
)

import unittest
import os, sys, re
with __info__:
    import batorch as bt
    from micomputing.funcs import distance_map_cpp, dilate_sitk, dilate, dilate_python
    from pycamia import tokenize, scope, Workflow
    
workflow = Workflow()

class InheritTests(unittest.TestCase):
    
    def (self):
    
if __name__ == "__main__":
    unittest.main()