
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
    flatlist
""".split()

def prod (*x):
    """
    Returns the product of elements, just like built-in function `sum`.
    
    Example:
    ----------
    >>> prod([5, 2, 1, 4, 2])
    80
    """
    if len(x) == 1 and isinstance(x[0], list): x = x[0]
    p = 1
    for i in x: p *= i
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

def flatlist(list_):
    """
    Flat the nested lists `list_`.
    
    Example:
    ----------
    >>> flatlist([0, 2, [1, 4, 2], [1, 3, 4]])
    [0, 2, 1, 4, 2, 1, 3, 4]
    """
    # Deprecated realization of the function, as elements may be strings with characters '[' or ']'.
    # items = str(list_).replace('[', '').replace(']', '').split(',')
    # return list(eval(','.join([x for x in items if x.strip() != ''])))
    flattened = []
    for x in list_:
        if isinstance(x, list):
            flattened.extend(flatlist(x))
        else: flattened.append(x)
    return flattened
