
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File of different functions."
)

__all__ = """
    empty_function
    const_function
    identity_function
""".split()

def empty_function(*args, **kwargs):
    """ Just an empty function, one can use it as a placeholder. """
    pass

def const_function(a):
    """Create a function that always returns `a`. """
    def f(*args, **kwargs): return a
    return f

def identity_function(*x):
    """A funtion that returns the input, just as it was. """
    if len(x) == 1: return x[0]
    else: return x
