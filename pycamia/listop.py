
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File of list operations. "
)

__all__ = """
    prod
    argmin
    argmax
    flat_list
    item
    to_list
    to_tuple
""".split()

from pycamia import avouch, touch

def prod(*x):
    """
    Returns the product of elements, just like built-in function `sum`.
    
    Example:
    ----------
    >>> prod([5, 2, 1, 4, 2])
    80
    """
    if len(x) == 1 and isinstance(x[0], list): x = x[0]
    p = 1
    for i in x: 
        if hasattr(i, "__mul__"):
            p *= i
    return p

def argmin(y, x=None):
    """
    Find the indices of minimal element in `y` given domain `x`.
    
    Example:
    ----------
    >>> argmin([0, 2, 1, 4, 2], [1, 3, 4])
    [1, 4]
    """
    if x is None: x = range(len(y))
    if len(x) <= 0: return []
    m = min([y[i] for i in x])
    return [i for i in x if y[i] == m]

def argmax(y, x):
    """
    Find the indices of maximal element in `y` given domain `x`.
    
    Example:
    ----------
    >>> argmin([0, 2, 1, 4, 2], [1, 3, 4])
    [3]
    """
    if x is None: x = range(len(y))
    if len(x) <= 0: return []
    m = max([y[i] for i in x])
    return [i for i in x if y[i] == m]

def flat_list(list_):
    """
    Flat the nested lists `list_`.
    
    Example:
    ----------
    >>> flat_list([0, 2, [1, 4, 2], [1, 3, 4]])
    [0, 2, 1, 4, 2, 1, 3, 4]
    """
    # Deprecated realization of the function, as elements may be strings with characters '[' or ']'.
    # items = str(list_).replace('[', '').replace(']', '').split(',')
    # return list(eval(','.join([x for x in items if x.strip() != ''])))
    flattened = []
    for x in list_:
        if isinstance(x, list):
            flattened.extend(flat_list(x))
        else: flattened.append(x)
    return flattened

def item(list_):
    """
    Assert if the length of the list `list_` is not 1 and return the only element. 
    
    Example:
    ----------
    >>> item([0])
    0
    >>> item([1,2])
    AssertError: ...
    """
    avouch(len(list_) == 1, f"Failure in itemize as the length of the list {repr(list_)} is not 1. ")
    return list_[0]

def to_list(x):
    """
    Try to cast element `x` into a list
    
    Example:
    ----------
    >>> item(0)
    [0]
    >>> to_list((1,2))
    [1,2]
    """
    func_candidates = ['tolist', 'to_list', 'aslist', 'as_list', '__list__']
    for fname in func_candidates:
        if hasattr(x, fname) and callable(getattr(x, fname)): return getattr(x, fname)()
    return touch(lambda: list(x), [x])

def to_tuple(x):
    """
    Try to cast element `x` into a tuple
    
    Example:
    ----------
    >>> item(0)
    (0,)
    >>> to_tuple([1,2])
    (1,2)
    """
    func_candidates = ['totuple', 'to_tuple', 'astuple', 'as_tuple', '__tuple__']
    for fname in func_candidates:
        if hasattr(x, fname) and callable(getattr(x, fname)): return getattr(x, fname)()
    return touch(lambda: tuple(x), (x,))
