
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "",
    package = "",
    author = "", 
    create = "",
    fileinfo = "",
    requires = ""
)

# import os, sys, re
# import batorch as bt
# import micomputing as mc
# from pycamia import *
from tensordim import new_dim, exist_dim, del_dim, iter_dim, linalg_dim, Size, FakeSize, func_dim_size, func_dim

def main():
    print(func_dim)

if __name__ == "__main__": main()
