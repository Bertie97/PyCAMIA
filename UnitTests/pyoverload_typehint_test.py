
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
    from pycamia import scope, Workflow, touch, avouch, crashed, get_declaration
    import torch
    import batorch as bt
    from pyoverload import *
    from pyoverload.old_version_files_deprecated.typehint import Int, Float
    
workflow = Workflow()

class TypingsTests(unittest.TestCase):
    
    def test_type_check(self):
        avouch(isinstance(bt.zeros(3,4), array))
        avouch(bool(True) or bool(False))
        avouch(not (bool(False) or bool(False)))
        avouch(bool(True) | bool(False))
        avouch(isinstance(True, bool), isinstance([True, True, True], bool[3]))
        print(int, int(23.), isinstance(1, int), int('2'), int[4])
        avouch(crashed(lambda: int('2.')))
        avouch(isinstance(torch.zeros(10), iterable[10]))
        avouch(issubclass(torch.int, dtype(int)))
        avouch(not isinstance(torch.zeros(2,3,4), dtype('int')[2, 3, 4]))
        avouch(isinstance(torch.zeros(2,3,4), dtype('float')[2, 3, 4]))
        avouch(isinstance(torch.zeros(2).type(torch.float64)[0], double&scalar))
        avouch(isinstance(None, null))
        avouch(isinstance(torch.zeros(10, 20).type(torch.int), array[torch.Tensor:torch.int][10, 20]))
        avouch(isinstance((torch.int, torch.float), dtype[2]))
        avouch(not isinstance(1, dtype('int')))
        avouch(isinstance([[2,3], [3,4], [4,5]], list[3][list[2]]))
        avouch(isinstance(bt.zeros(2, 3).int(), int[2, ..., 3]))
        avouch(isinstance(lambda: _, callable))
        avouch(issubclass(int, int&scalar))

    def test_typehint(self):
        class A:
            def func(self, a: int, b):
                return a + b
            
            @typehint
            @classmethod
            def func2(cls, a, *b: int, c=2): ...
            
            @staticmethod
            def load(path): ...
        
        print(*[get_virtual_declaration(f) for f in [A.func, A().func, A.func2, A().func2, A.load, A().load]])
        try: A.func2(10, 2, 4., '')
        except Exception as e: print(str(e))

        @typehint
        def func(a: int, b, /, c: callable, d:int =2, *e: int, bad=3, **f) -> str:
            x = 1
            return repr(c(a, len(b)))
        print(get_virtual_declaration(func))

        @typehint
        def fun(*args: [list[...], float]):
            print(args)
            
        fun([], [1], 5.)

    def test_overload(self):
        @overload
        def size(*s: (int, list[0], list[1], set[1], dict[0]), keep_special: bool=False):
            return s

        @overload
        def size(s: tuple[(int, list[0], list[1], set[1], dict[0]),], keep_special: bool=True):
            return s
        
        @overload
        def size(*s: int, batch_dim: (null, int), channel_dim: (null, int), keep_special: bool=True):
            s = list(s)
            if batch_dim is not None: s[batch_dim] = [s[batch_dim]]
            if channel_dim is not None: s[channel_dim] = {s[channel_dim]}
            return s

        def size_direct(*s: int, batch_dim: (null, int), channel_dim: (null, int), keep_special: bool=True):
            s = list(s)
            if batch_dim is not None: s[batch_dim] = [s[batch_dim]]
            if channel_dim is not None: s[channel_dim] = {s[channel_dim]}
            return s

        @overload
        def f(x:float, y:Int, z=3, *k:int, **b:float):
            return x, y, z, k, b

        @overload
        def f(x:float, y:Float, z=3, *, t=5):
            return x, y, z, t

        print(size([4], {5}, 3, 4))
        print(size(4, 5, 3, 4, batch_dim=0, channel_dim=1))
        print(f(1., 1, 4, 5, 7, 9))
        
    with workflow("Test param time"), workflow.use_tag:
        with scope("func_usage1"):
            x = ''
            for _ in range(int(1e5)):
                x = size([4], {5}, 3, 4)
        with scope("func_usage2"):
            x = ''
            for _ in range(int(1e5)):
                x = size(4, 5, 3, 4, batch_dim=0, channel_dim=1)
        with scope("func_direct"):
            x = ''
            for _ in range(int(1e5)):
                x = size_direct(4, 5, 3, 4, batch_dim=0, channel_dim=1)

    def test_method_overload(self):
        class A:
            @overload
            def __int__(self): ...

            @overload
            def __int__(self, y): self.y = y
            
            @overload
            def run(self, x: str): ...
            
            @overload
            def run(self, x: int[2]): ...
            
        print(A().run(''))
    
    def test_rational(self):

        def GCD(a, b):
            while b > 0:
                c = b
                b = a % b
                a = c
            return a
        
        isint = lambda x: abs(x - round(x)) < 1e-4

        class rat:

            @overload
            def __init__(self, numerator: int, denominator: int):
                self.numerator, self.denominator = numerator, denominator
                self.value = self.numerator / self.denominator
                self.cancelation()

            @overload
            def __init__real__(self, numerator: real, denominator: real):
                if isint(numerator) and isint(denominator):
                    self.__init__(int(numerator), int(denominator))
                else: self.__init__float__(numerator / denominator)

            @overload
            def __init__str__(self, string: str):
                try: arg = [float(x) for x in string.split('/')]
                except Exception: raise SyntaxError("Invalid Format")
                self.__init__real__(*arg)

            @overload
            def __init__int__(self, num: int): self.__init__(num, 1)

            @overload
            def __init__float__(self, num: float):
                if isint(num): self.__init__int__(int(num))
                else:
                    self.value = num
                    self.__init__(*rat.nearest(num))
            print({k:v for k, v in locals().items() if '__init__' in k})
            def tuple(self): return self.numerator, self.denominator

            def cancelation(self):
                d = GCD(*self.tuple())
                self.numerator //= d
                self.denominator //= d
                if self.denominator < 0:
                    self.numerator = - self.numerator
                    self.denominator = - self.denominator

            def __add__(self, other):
                return rat(self.numerator * other.denominator +
                        self.denominator * other.numerator,
                        self.denominator * other.denominator)

            def __mul__(self, other):
                return rat(self.numerator * other.numerator,
                        self.denominator * other.denominator)

            def __str__(self):
                if self.denominator == 1: return str(self.numerator)
                return str(self.numerator)+'/'+str(self.denominator)

            @staticmethod
            def nearest(num, maxiter=100):
                def iter(x, d):
                    if isint(x) or d >= maxiter: return int(round(x)), 1
                    niter = iter(1 / (x - int(x)), d+1)
                    return int(x) * niter[0] + niter[1], niter[0]
                if num >= 0: return iter(num, 0)
                num = iter(-num, 0)
                return -num[0], num[1]

        print(rat(126/270) + rat(25, 14))
        print(rat(126, 270) * rat(25, 14))

if __name__ == "__main__":
    unittest.main()

