
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Get time took for decorator. ",
    requires = "unittest"
)

import unittest
from functools import wraps
from pycamia import scope
from pyoverload import *

class DecoratorTimeTests(unittest.TestCase):

    def decorator_time(self, dec, n_test = int(1e5)):
        def test_dec(func):
            @wraps(func)
            def func2(*args, **kwargs):
                return func(*args, **kwargs)
            return func2
        @test_dec
        def func(self, pos1, pos2: str="def_val1", / , arg3: (int, float)=3, arg4=[], *args: int, kwarg5: str='extra', **kwargs) -> str:
            return ''.join(str(x) for x in (pos1, pos2, arg3, arg4, args, kwarg5, kwargs))
        with scope("without_decorator") as scope_wo:
            string = ''
            for _ in range(n_test):
                string = func(12, string, kwarg5='best')
        func = dec(func)
        with scope("with_decorator") as scope_w:
            string = ''
            for _ in range(n_test):
                string = func(12, string, kwarg5='best')
        print(max(scope_w.recorded_time - scope_wo.recorded_time, 0.))
        
    def test_overload_time(self):
        self.decorator_time(overload)
        
    def test_typehint_time(self):
        self.decorator_time(typehint)

if __name__ == "__main__":
    unittest.main()
