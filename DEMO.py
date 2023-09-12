## This is the DEMO file for project PyCAMIA.
## Change the arguments of `Workflow` for variable `demo` to test different packages. 
if __name__ != "__main__": exit()
from pycamia import Workflow
demo = Workflow()

with demo("pyoverload"), demo.use_tag:

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
            else: self.__initrefat(numerator / denominator)

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

with demo("info_manager"), demo.use_tag:
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

with demo("pycamia"), demo.use_tag:
    from pycamia import flatten_list
    print(flatten_list([0, 2, [1, 4, 2], [1, 3, 4]]))
    
    from pycamia import touch
    a = 1
    print(touch("a"))
    
    from pycamia import alias
    @alias("func_a", b=1)
    @alias("func_c", b=2)
    def func_b(x, b):
        print(x+b)
    
    func_a(1)
    func_b(2, 4)
    func_c(7)
    
    class A:
        @alias('x', 'a', 'b')
        @property
        def y(self): return '1'
    
    a = A()
    print(a.x)
    print(a.y)
    print(a.a)
    print(a.b)
    
    def compose(f, g):
        return lambda x: f(g(x))
    print(compose(alias("square")(lambda x: x**2), square)(3))
    
    from pycamia import Path, curdir, pardir, copy
    Path(curdir, ref=pardir).cmd('ls {}; pwd')
    print(Path(curdir).size())

with demo("environ"), demo.use_tag:
    from pycamia import get_args_expression, get_declaration
    def image_grid(x):
        return 2
    def f(x):
        print(get_args_expression())
    def get_declaration(f):
        with open(f.__code__.co_filename) as fp:
            for _ in range(f.__code__.co_firstlineno-1): fp.readline()
            l = fp.readline()
            print(l)
    get_declaration(f)

with demo("batorch"), demo.use_tag:
    # Deprecated: 2023-09
    import batorch as bt
    import numpy as np
    from pyoverload import overload, Array
    from pycamia import restore_type_wrapper, get_declaration

    @overload
    # @restore_type_wrapper
    def image_grid(x: Array):
        return image_grid(x.space)

    @overload
    def image_grid__default__(*shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
        a, b = map(int, bt.torch.__version__.split('+')[0].split('.')[:2])
        kwargs = {'indexing': 'ij'} if (a, b) >= (1, 10) else {}
        return bt.stack(bt.meshgrid(*[bt.arange(x) for x in shape], **kwargs), {})
    print(get_declaration(image_grid))
    # t = bt.Tensor(np.zeros((3, 4, 3)), batch_dim=1)
    # print(t, t.shape)
    # a = bt.zeros(3, [4], 5, {2}, 7)
    # a.c_dim = -1
    # print(bt.zeros([2], {4}, 3, 5).split(2, {}))
    # print(a.standard().shape)
    # a = bt.rand([2], 3, 3)
    # b = bt.rand([2], 3)
    # print(a @ b)
    # print(bt.zeros([3]).unsqueeze(-2))
    
with demo("micomputing"), demo.use_tag:
    from micomputing import metric
    
with demo("batorch2"), demo.use_tag:
    from batorch import tensor2 as bt
    print(bt.zeros({2}, 3))
    print(bt.zeros({2}, 3) + 1, bt.zeros({2}, 3) + 1 == bt.ones({2}, 3))
    print(bt.rand({3}, [4], 2, 1).squeeze(-1))

with demo("iterate"), demo.use_tag: 
    from pycamia import iterate
    for i in iterate(100):
        for _ in range(100000): i+_
        print(i)
        
