
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
    cartesian_prod
    argmin
    argmax
    min_argmin
    max_argmax
    kth_biggest
    kth_smallest
    median
    flatten_list
    item
    to_list
    to_tuple
    to_set
    map_ele
    sublist
    arg_tuple
    arg_extract
    count
    unique
""".split()

from .exception import avouch, touch
from typing import Iterable
from random import randint

def prod(*x):
    """
    Returns the product of elements, just like built-in function `sum`.
    
    Example:
    ----------
    >>> prod([5, 2, 1, 4, 2])
    80
    """
    if len(x) == 1: x = x[0]
    p = 1
    for i in x: 
        if hasattr(i, "__mul__") or hasattr(i, "__rmul__"):
            p *= i
    return p

def cartesian_prod(x, *y):
    """
    Returns the cartesian product of lists.
    
    Example:
    ----------
    >>> cartesian_prod([1, 2], [3, 1], [1, 4, 2])
    [[1, 3, 1], [1, 3, 4], [1, 3, 2], [1, 1, 1], [1, 1, 4], [1, 1, 2], [2, 3, 1], [2, 3, 4], [2, 3, 2], [2, 1, 1], [2, 1, 4], [2, 1, 2]]
    """
    if not isinstance(x, Iterable): x = [x]
    if len(y) == 0: return [[i] for i in x]
    res = []
    for i in x:
        res.extend([[i] + s for s in cartesian_prod(*y)])
    return res

def argmin(y, x=None):
    """
    Find the indices of minimal element in `y` given index domain `x`.
    
    Example:
    ----------
    >>> argmin([0, 2, 1, 4, 2], [1, 3, 4])
    [1, 4]
    """
    if x is None: x = range(len(y))
    if len(x) <= 0: return []
    m = min([y[i] for i in x])
    return [i for i in x if y[i] == m]

def argmax(y, x=None):
    """
    Find the indices of maximal element in `y` given index domain `x`.
    
    Example:
    ----------
    >>> argmin([0, 2, 1, 4, 2], [1, 3, 4])
    [3]
    """
    if x is None: x = range(len(y))
    if len(x) <= 0: return []
    m = max([y[i] for i in x])
    return [i for i in x if y[i] == m]

def min_argmin(y, x=None):
    """
    Find the minimal value as well as the indices in `y` given domain `x`.
    
    Example:
    ----------
    >>> min_argmin([0, 2, 1, 4, 2], [1, 3, 4])
    (2, [1, 4])
    """
    if x is None: x = range(len(y))
    if len(x) <= 0: return None, []
    m = min([y[i] for i in x])
    return m, [i for i in x if y[i] == m]

def max_argmax(y, x=None):
    """
    Find the maximal value as well as the indices in `y` given domain `x`.
    
    Example:
    ----------
    >>> argmin([0, 2, 1, 4, 2], [1, 3, 4])
    (4, [3])
    """
    if x is None: x = range(len(y))
    if len(x) <= 0: return None, []
    m = max([y[i] for i in x])
    return m, [i for i in x if y[i] == m]

def kth_biggest(list_, k: int):
    """
    Find the k-th biggest element in the list.
    
    Example:
    ----------
    >>> kth_biggest([0, 2, 1, 4, 2], 3)
    2
    """
    n = len(list_)
    avouch(k <= n, f"Cannot find the {k}-th element in a list of length {n}.")
    if n <= 1: return list_[0]
    p = randint(0, n - 1)
    lp = list_[p]
    gt_p = [x for x in list_ if x > lp]
    lt_p = [x for x in list_ if x < lp]
    n_gt_p = len(gt_p)
    n_lt_p = len(lt_p)
    if k <= n_gt_p: return kth_biggest(gt_p, k)
    elif n_gt_p < k <= n - n_lt_p: return lp
    else: return kth_biggest(lt_p, k - n + n_lt_p)
    
def kth_smallest(list_, k: int):
    n = len(list_)
    return kth_biggest(list_, n - k + 1)
    
def median(list_):
    n = len(list_)
    if n % 2 == 1: return kth_biggest(list_, n // 2)
    return (kth_biggest(list_, n // 2) + kth_biggest(list_, n // 2 + 1)) / 2

def flatten_list(list_):
    """
    Flat the nested lists `list_`.
    
    Example:
    ----------
    >>> flatten_list([0, 2, [1, 4, 2], [1, 3, 4]])
    [0, 2, 1, 4, 2, 1, 3, 4]
    """
    # Deprecated realization of the function, as elements may be strings with characters '[' or ']'.
    # items = str(list_).replace('[', '').replace(']', '').split(',')
    # return list(eval(','.join([x for x in items if x.strip() != ''])))
    flattened = []
    for x in list_:
        if isinstance(x, list):
            flattened.extend(flatten_list(x))
        else: flattened.append(x)
    return flattened

def item(list_):
    """
    Assert if the length of the list/tuple/set `list_` is not 1 and return the only element. 
    
    Example:
    ----------
    >>> item([0])
    0
    >>> item([1,2])
    AssertError: ...
    """
    list_ = to_list(list_)
    avouch(len(list_) == 1, f"Failure in itemize as the length of {repr(list_)} is not 1. ")
    return list_[0]

def to_list(x, l = None):
    """
    Try to cast element `x` into a list
    
    Example:
    ----------
    >>> to_list(1)
    [1]
    >>> to_list(0, 4)
    [0, 0, 0, 0]
    >>> to_list((1,2))
    [1, 2]
    >>> to_list((1,2), 4)
    [1, 2, 1, 2]
    """
    func_candidates = ['tolist', 'to_list', 'aslist', 'as_list', '__list__']
    for fname in func_candidates:
        if hasattr(x, fname) and callable(getattr(x, fname)):
            x = getattr(x, fname)(); break
    if isinstance(x, Iterable) and not isinstance(x, str): x = list(x)
    if not isinstance(x, list): x = [x]
    if l is None: return x
    if l % len(x) == 0: return x * (l // len(x))
    raise TypeError(f"{x} can not be converted into a list of length {l}. ")

def to_tuple(x, l = None):
    """
    Try to cast element `x` into a tuple of length `l`
    
    Example:
    ----------
    >>> to_tuple(1)
    (1,)
    >>> to_tuple(0, 4)
    (0, 0, 0, 0)
    >>> to_tuple([1,2])
    (1, 2)
    >>> to_tuple([1,2], 4)
    (1, 2, 1, 2)
    """
    func_candidates = ['totuple', 'to_tuple', 'astuple', 'as_tuple', '__tuple__']
    for fname in func_candidates:
        if hasattr(x, fname) and callable(getattr(x, fname)):
            x = getattr(x, fname)(); break
    try:
        return tuple(to_list(x, l))
    except TypeError:
        raise TypeError(f"{x} can not be converted into a tuple of length {l}. ")

def to_set(x):
    """
    Try to cast element `x` into a set
    
    Example:
    ----------
    >>> to_set(0)
    {0}
    >>> to_set([1,2])
    {1,2}
    """
    func_candidates = ['toset', 'to_set', 'asset', 'as_set', '__set__']
    for fname in func_candidates:
        if hasattr(x, fname) and callable(getattr(x, fname)): return getattr(x, fname)()
    return touch(lambda: set(x), touch(lambda: set(to_list(x)), {x}))

def map_ele(func, list_, index_ = None):
    """
    In-place! Map elements in `list_` at indices `index_` by function `func`. 
    
    Example:
    ----------
    >>> map_ele(lambda x: x+1, [1,2], 1)
    [1, 3]
    >>> map_ele(to_list, [1,2,3,4], [1,2])
    [1, [2], [3], 4]
    """
    if index_ is None: index_ = range(len(list_))
    if not index_: return list_
    index_ = to_list(index_)
    for i in index_: list_[i] = func(list_[i])
    return list_

def sublist(list_: list, index_):
    """
    Return elements in `list_` at indices `index_`. 
    
    Example:
    ----------
    >>> map_ele([1,2], [1])
    [2]
    >>> map_ele([1,2,3,4], [1,2])
    [2, 3]
    """
    if isinstance(index_, slice): index_ = range(index_.start, index_.stop)
    return [list_[i] for i in index_]

def arg_extract(arg:tuple, arg_type=None):
    """
    For *args, extract the only element if length is 1. 
    Set kwarg arg_type to define the types of objects that can be extracted. 
    By default, arg_type = None, which means any single element will be extracted. 
    
    Example:
    ----------
    >>> def f(*args): print(args)
    ... 
    >>> f([1,2,3,4])
    ([1, 2, 3, 4],)
    >>> def f(*args): print(arg_tuple(args))
    ... 
    >>> f([1,2,3,4])
    [1, 2, 3, 4]
    """
    if len(arg) == 0: return ()
    if len(arg) > 1: arg = tuple(arg); return arg
    if arg_type is None: arg = arg[0]
    elif isinstance(arg_type, (tuple, list)):
        if isinstance(arg[0], type(arg_type)): arg = arg[0]
    elif isinstance(arg[0], arg_type): arg = arg[0]
    return arg

def arg_tuple(arg:tuple, no_list=False):
    """
    Return the raw tuple. 
    
    Example:
    ----------
    >>> def f(*args): print(args)
    ... 
    >>> f([1,2,3,4])
    ([1, 2, 3, 4],)
    >>> def f(*args): print(arg_tuple(args))
    ... 
    >>> f([1,2,3,4])
    (1, 2, 3, 4)
    """
    if len(arg) == 1 and isinstance(arg[0], tuple): arg = arg[0]
    if len(arg) == 1 and isinstance(arg[0], list) and not no_list: arg = arg[0]
    arg = tuple(arg)
    return arg

def count(list_:list, filter):
    """
    Return number of elements in `list_` that satisfies `filter`. 
    
    Example:
    ----------
    >>> count([1,2], lambda x: x > 1)
    1
    """
    count = 0
    for x in list_:
        if filter(x): count += 1
    return count

def unique(list_:list):
    """
    Return a list of unique element in `list_`
    
    Example:
    ----------
    >>> count([1, 1, 3, 2])
    [1, 3, 2]
    """
    ulist = []
    for x in list_:
        if x not in ulist: ulist.append(x)
    return ulist
