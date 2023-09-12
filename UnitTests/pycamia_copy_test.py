
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-08",
    fileinfo = "Unit test for file copying.",
    requires = ["pycamia", "unittest"]
)

import unittest
import os, sys, re, math
with __info__:
    from pycamia import Path, Workflow, iterate
workflow = Workflow("pycamia copy folder")

class FileCopyTests(unittest.TestCase):
    
    def __init__(self, *args):
        super().__init__(*args)
        # self.source = Path("/Volumes/Photoshop 2022.23.5.2_kkmac.com/ps2022破解补丁.pkg")
        self.source = Path("/Volumes/Photoshop 2022.23.5.2_kkmac.com/products/PHSP/AdobePhotoshop23-Core.zip")
        self.target = Path.curdir

    @unittest.skip("system copy")
    def test_system_command_copy(self):
        source = self.source.replace(' ', '\ ')
        os.system(f"cp {source} {self.target}")

    @unittest.skip("stdio copy")
    def test_stdio_copy(self):
        if self.target.is_dir(): target = self.target / self.source.filename
        with open(self.source, 'rb') as fp_in:
            with open(target, 'wb') as fp_out:
                fp_out.write(fp_in.read())
            
    @unittest.skip("process copy")
    def test_process_copy(self):
        file_size = self.source.size()
        copy_unit = 1 << (10 * 2)
        n_share = math.ceil(file_size / copy_unit)
        n_remain = file_size - n_share * copy_unit
        if self.target.is_dir(): target = self.target / self.source.filename
        with open(target, 'w'): ...
        with open(self.source, 'rb') as fp_in:
            for i in iterate(n_share):
                if i == n_share - 1: n_copy = n_remain
                n_copy = copy_unit
                with open(target, 'ab') as fp_out:
                    fp_out.write(fp_in.read(n_copy))
    
    @unittest.skip("pycamia copy")
    def test_pycamia_copy_same_folder(self):
        (self.target/self.source.filename).copy_to(self.target)

if __name__ == "__main__":
    unittest.main()

