   
from pyoverload import *

@overload
def func(x: int): print("func1", x)
@overload
def func(x: str): print("func2", x)
@overload
def func(x: int[4]): print("func3", x)
@overload
def func(x: 'numpy.ndarray'): print("func4", x)
@overload
def func(x): print("func5", x)

func()
