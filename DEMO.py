## This is the DEMO file for project PyCAMIA.
## Change the arguments of `Workflow` for variable `demo` to test different packages. 
if __name__ != "__main__": exit()
from pycamia import Workflow
demo = Workflow("pycamia")

with demo("pyoverload"), demo.jump:

    from pyoverload import *
    print(isoftype({'a': 2, '3': 43}, Dict[str:int]))
    print(isoftype({'a': 2, '3': 43.}, Dict[str:int]))
    
    @params(Functional, Int, +Int, __return__ = Real[2])
    def test_func(a, b=2, *k):
        return k
    
    print(test_func(lambda x: 1, 3, 4, 5))
    
    isint = lambda x: abs(x - round(x)) < 1e-4
    rint = lambda x: int(round(x))
    def GCD(a, b):
        while b > 0:
            c = b
            b = a % b
            a = c
        return a
    
    class rat:

        @overload
        def __init__(self, numerator, denominator):
            self.numerator, self.denominator = numerator, denominator
            self.value = self.numerator / self.denominator
            self.cancelation()

        @overload
        def __init__(self, numerator: Real, denominator: Real):
            if isint(numerator) and isint(denominator):
                self.__init__numdenint(rint(numerator), rint(denominator))
            else: self.__init__float(numerator / denominator)

        @overload
        def __init__(self, string: str):
            try: arg = [float(x) for x in string.split('/')]
            except Exception: raise SyntaxError("Invalid Format")
            self.__init__numdenfloat(*arg)

        @overload
        def __init__(self, num: int): self.__init__numdenint(num, 1)

        @overload
        def __init__(self, num: float):
            if isint(num): self.__init__(rint(num))
            else:
                self.value = num
                self.__init__(*rat.nearest(num))

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

with demo("pycamia"), demo.jump:
    from pycamia import enclosed_object, info_manager
    code = """
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    version = "1.0.0",
    contact = "bertiezhou@163.com",
    keywords = ["environment", "path", "touch"],
    description = "The main package and a background support of project PyCAMIA. ",
    requires = ["torch>1.5"]
).check()

from .environment import *
from .exception import *
from .functions import *
    """
    pivot = "info_manager("
    info_str = enclosed_object(code, by="()", start=code.index(pivot))
    info = info_manager.parse(info_str).check()
    info.x = "hello"
    print(info)
    
    from pycamia import flat_list
    print(flat_list([0, 2, [1, 4, 2], [1, 3, 4]]))
    
    from pycamia import touch
    a = 1
    print(touch("a"))
    
with demo(""), demo.jump: pass
