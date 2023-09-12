
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-08-22",
    fileinfo = "Unit test for pycamia scope",
    requires = ["pycamia", "unittest"]
)

import unittest
import os, sys, re

with __info__:
    from pycamia import scope, jump, Jump

class ScopeTests(unittest.TestCase):

    def test_scope_jump_time(self):
        with scope("this") as s:
            print("Part I")
            s.exit()
            print("Part II")
        with scope("this"), Jump(False) as stop:
            print("Part I")
            stop()
            print("Part II")
        with scope("this again", False) as s:
            print("Part I")
            print("Part II")
        print(s.recorded_time)

if __name__ == "__main__":
    unittest.main()