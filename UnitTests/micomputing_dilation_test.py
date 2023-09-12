
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

class DilationTests(unittest.TestCase):
    
    def __init__(self, *args):
        super().__init__(*args)
        self.image1 = ((bt.image_grid(100, 100, 100) - bt.channel_tensor([50, 50, 50])) ** 2).sum({}) < 400
        self.image2 = bt.zeros(100, 100, 100)
        self.image2[20:-20, 20:-20, 20:-20] = 1;
        self.image2 *= ~self.image1;
        self.image = bt.stack(self.image1, self.image2, [])
        
    def test_dismap(self):
        dismap = distance_map_cpp(self.image, spacing=(1, 2, 1))
        with workflow("display dismap"), workflow.use_tag:
            bt.display(self.image[1, ..., 50], dismap[1, ..., 50]).show()
    
    def test_cpp_dilation(self):
        with scope("use cpp"):
            dc = dilate(self.image2, 20)
            dc = dilate(self.image2, 10)
        with workflow("display dilated"), workflow.use_tag:
            bt.display(dc[..., 50]).show()
    
    def test_itk_dilation(self):
        with scope("use itk"):
            di = dilate_sitk(self.image2, 20)
            di = dilate_sitk(self.image2, 10)
        with workflow("display dilated"), workflow.use_tag:
            di.display(dc[..., 50]).show()
            
    @unittest.skip('python dilation')
    def test_python_dilation(self):
        with scope("use python"):
            dp = dilate_python(self.image2, 20)
            dp = dilate_python(self.image2, 10)
        with workflow("display dilated"), workflow.use_tag:
            dp.display(dc[..., 50]).show()

if __name__ == "__main__":
    unittest.main()
