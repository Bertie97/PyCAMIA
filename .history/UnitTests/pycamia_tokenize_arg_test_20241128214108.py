
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

# __info__ = info_manager(
#     project = "PyCAMIA",
#     package = "<main>.unittest",
#     author = "Yuncheng Zhou", 
#     create = "2023-08-21",
#     fileinfo = "Unit test for function `tokenize`, take extracting arguments as an example",
#     requires = ["pycamia", "unittest"]
# )

import unittest
# import os, sys, re
# with __info__:
    # from pycamia import tokenize

class TokenizeTests(unittest.TestCase):
    def test_tokenize_getargexp(self): ...
#         for case in [
#             dict(
#                 inputs=""
#                 line="func1(sum(2, 3, this_to_be(extracted)), other_args); i++"
#                 func_name="sum"
#             )
#         ]
#         # l = 
#         l = "z(sum(2, 3).x)))"
#         l = "show(Tensor.inherit_from(obj, pivot, shape=...))"
#         func_name = 'show'
#         error = TypeError(f"Cannot get args expression. Please avoid using two function '{func_name}'s in a single code line. ")
#         lines_with_func = [x for x in tokenize(l, sep=[';']) if func_name in x]
#         if len(lines_with_func) > 1: raise error
#         line = lines_with_func[0]
#         parts = line.split(func_name)
#         if len(parts) > 2: raise error
#         exp = tokenize(parts[-1], sep=[',', ')'])[0]
#         exp = tokenize(exp, sep='.')[0]
#         print(exp)
#         return False

if __name__ == "__main__":
    unittest.main()
