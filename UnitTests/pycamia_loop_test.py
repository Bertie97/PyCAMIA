
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
import os, sys, re
import time

with __info__:
    from pycamia import iterate

class LoopTests(unittest.TestCase):

    def test_iterate_double_loop(self):
        for i in iterate(10):
            for j in iterate(40):
                time.sleep(0.03)
                print(iterate.prefix, f"i = {i}, j = {j}")
            print(iterate.prefix, i)

if __name__ == "__main__":
    unittest.main()