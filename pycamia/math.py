
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "Some special math operations."
)

__all__ = """
    GCD
    isint
    rational
""".split()

def GCD(a, b):
    """
    Find the Greatest Common Divisor for `a` and `b`. 
    """
    while b > 0:
        c = b
        b = a % b
        a = c
    return a

isint = lambda x: abs(x - round(x)) < 1e-4

class rational:

    def __init__(self, *arg):
        self.value = None
        if len(arg) > 2: raise SyntaxError("Invalid initialization")
        if len(arg) == 2 and False in [isint(x) for x in arg]:
            arg = [arg[0] / arg[1]]
        if len(arg) == 1:
            if type(arg[0]) == str:
                try: arg = [int(x) for x in arg[0].split('/')]
                except Exception: raise SyntaxError("Invalid Format")
            elif type(arg[0]) == rational: arg = arg[0].tuple()
            else:
                try: arg = float(arg[0])
                except Exception: raise SyntaxError("Invalid initialization")
                self.value = arg; arg = rational.nearest(arg)
        self.numerator, self.denominator = (int(round(x)) for x in arg)
        if self.value is None: self.value = self.numerator / self.denominator
        self.cancelation()

    def tuple(self): return self.numerator, self.denominator

    def cancelation(self):
        d = GCD(*self.tuple())
        self.numerator //= d
        self.denominator //= d
        if self.denominator < 0:
            self.numerator = - self.numerator
            self.denominator = - self.denominator

    @staticmethod
    def nearest(num, maxiter=None):
        def iter(x, d):
            if not maxiter and abs(x - round(x)) < 0.01: return int(round(x)), 1
            elif d >= (100 if maxiter is None else maxiter): return int(x), 1
            niter = iter(1 / (x - int(x)), d+1)
            return int(x) * niter[0] + niter[1], niter[0]
        if num >= 0: return iter(num, 0)
        num = iter(-num, 0)
        return -num[0], num[1]

    @staticmethod
    def floor(arg): return rational(arg.numerator // arg.denominator)
    def floor(self): return rational(self.numerator // self.denominator)
    def __int__(self): return int(self.numerator // self.denominator)
    def __round__(self): return rational(round(self.value))

    def __add__(self, other):
        return rational(self.numerator * other.denominator +
                        self.denominator * other.numerator,
                        self.denominator * other.denominator)

    def __sub__(self, other):
        return rational(self.numerator * other.denominator -
                        self.denominator * other.numerator,
                        self.denominator * other.denominator)

    def __mul__(self, other):
        return rational(self.numerator * other.numerator,
                        self.denominator * other.denominator)

    def __truediv__(self, other):
        return rational(self.numerator * other.denominator,
                        self.denominator * other.numerator)

    def __floordiv__(self, other):
        return rational.floor(self.__truediv__(other))

    def __mod__(self, other): return self - self // other

    def __pow__(self, other):
        if isint(self.numerator ** other) and isint(self.denominator ** other):
            return rational(self.numerator ** other, self.denominator ** other)
        return self.value ** other

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other): return -self.__sub__(other)
    def __rtruediv__(self, other): return ~self.__truediv__(other)
    def __rfloordiv__(self, other): return rational.floor(self.__rtruediv__(other))
    def __rmod__(self, other): return other - other // self
    def __rpow__(self, other):
        other = rational(other)
        num = other.numerator ** self.value
        den = other.denominator ** self.value
        if isint(num) and isint(den) and \
           other.numerator ** self.numerator == num ** self.denominator and \
           other.denominator ** self.numerator == den ** self.denominator:
            return rational(num, den)
        return other ** self.value

    def __matmul__(self, other):
        return rational(*rational.nearest(self.value, maxiter=other))

    def __str__(self):
        if self.denominator == 1: return str(self.numerator)
        return str(self.numerator)+'/'+str(self.denominator)

    __repr__ = __str__

    def __format__(self, format):
        text = self.__str__()
        if format[-2:] == 'df':
            format = format[:-2]
            text = "{d} {f}".format(d=int(self), f=self % 1)
        elif format[-1:] == 'f': text = self.value
        try:
            return ("{txt:" + format + "}").format(txt=text)
        except ValueError as error:
            raise ValueError(str(error).replace("'str'", "'rational'"))

    def __neg__(self): return rational(-self.numerator, self.denominator)

    def __pos__(self): return self

    def __abs__(self): return max(self, -self)

    def __invert__(self): return rational(self.denominator, self.numerator)

    def __lt__(self, other):
        other = rational(other)
        return self.numerator * other.denominator < \
               other.numerator * self.denominator

    def __eq__(self, other):
        other = rational(other)
        return self.numerator * other.denominator == \
              other.numerator * self.denominator

    def __le__(self, other): return self.__lt__(other) or self.__eq__(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __ge__(self, other): return not self.__lt__(other)
    def __gt__(self, other): return not self.__le__(other)
    def __bool__(self): return self.numerator != 0

    def __hash__(self): return self.value

