
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File of string operations. "
)

__all__ = """
    strlen
    strslice
    findall
    sorted_dict_repr
    enclosed_object
    tokenize
""".split()

def strlen(str_:str, r:[int, float]=2):
    """
    Returen the ASCII string length of `str_`. 
    
    Arguments:
        r: bytes a wide character stands for. 
    
    Example:
    ----------
    >>> print(strlen("12"), len("你好"), strlen("你好"))
    2 2 4
    """
    return int(len(str_) + (r-1) * len([c for c in str_ if ord(u'\u4e00') <= ord(c) <= ord(u'\u9fa5')]))

def findall(str_:str, key:str):
    """
    Returen all the starting indices of string `key` in string `str_`. 
    
    Example:
    ----------
    >>> findall("abcaa", 'a')
    [0, 3, 4]
    """
    p, indices = -1, []
    while True:
        p = str_.find(key, p + 1)
        if p < 0: break
        indices.append(p)
    return indices

def strslice(str_:str, indices:list):
    """
    Split the string `str_` by breaks in list `indices`.
    
    Example:
    ----------
    >>> slice("abcaa", [2,4])
    ["ab", "ca", "a"]
    """
    indices.insert(0, 0); indices.append(len(str_))
    return [str_[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]

def sorted_dict_repr(d:dict, order:[list, tuple]):
    """
    Representer of dictionary `d`, with key order `order`.
    
    Example:
    ----------
    >>> sorted_dict_repr({'a': 1, '0': 3}, ['0', 'a'])
    "{'0': 3, 'a': 1}"
    """
    return '{' + ', '.join([f"{repr(k)}: {repr(d[k])}" for k in order]) + '}'

def enclosed_object(str_, by=["([{", ")]}", "$`'\""], start=0):
    """
    Return the first object enclosed with a whole pair of parenthesis in `str_` after index `start`.
    
    Example:
    ----------
    >>> enclosed_object("function(something inside), something else. ")
    function(something inside)
    """
    if len(by) == 3: left, right, both = by
    elif len(by) == 2: left, right = by; both = ""
    elif len(by) == 1: left = ""; right = ""; both = by[0]
    else: raise TypeError("Invalid argument `by` for function `tokenize`. ")
    depth = {'all': 0}
    for i in range(start, len(str_)):
        s = str_[i]
        if s in right:
            if depth.get(s, 0) == 0: return str_[start:i]
            assert depth[s] > 0 and depth['all'] > 0
            depth[s] -= 1
            depth['all'] -= 1
            if depth[s] == 0 and depth['all'] == 0: return str_[start:i+1]
        elif s in left:
            r = right[left.index(s)]
            depth.setdefault(r, 0)
            depth[r] += 1
            depth['all'] += 1
        elif s in both and str_[i-1] != '\\':
            depth.setdefault(s, 0)
            if depth[s] > 0:
                depth[s] -= 1;
                depth['all'] -= 1
                if depth[s] == 0 and depth['all'] == 0: return str_[start:i+1]
            else: depth[s] += 1; depth['all'] += 1
    raise RuntimeError(f"Cannot find enclosed object from string {repr(str_)}.")

def tokenize(str_:str, sep=[' ', '\n'], by=["([{", ")]}", "$`'\""], start=0, strip='', keep_empty=True) -> list:
    """
    Split the string `str_` by elements in `sep`, but keep enclosed objects not split.
    
    Example:
    ----------
    >>> tokenize("function(something inside), something else. ")
    ["function(something inside),", "something", "else.", ""]
    """
    if isinstance(sep, str): sep = [sep]
    if len(by) == 3: left, right, both = by
    elif len(by) == 2: left, right = by; both = ""
    elif len(by) == 1: left = ""; right = ""; both = by[0]
    else: raise TypeError("Invalid argument `by` for function `tokenize`. ")
    depth = {'all': 0}
    tokens = []
    p = start
    for i in range(start, len(str_)):
        s = str_[i]
        both_done = False
        if s in right:
            if depth.get(s, 0) == 0: break
            assert depth[s] > 0 and depth['all'] > 0
            depth[s] -= 1
            depth['all'] -= 1
        elif s in both and str_[i-1] != '\\':
            depth.setdefault(s, 0)
            if depth[s] > 0:
                depth[s] -= 1
                depth['all'] -= 1
                both_done = True
        if depth['all'] == 0:
            for x in sep:
                if str_[i:i + len(x)] == x:
                    t = str_[p:i].strip(strip)
                    if keep_empty or t != '': 
                        tokens.append(t)
                    p = i + len(x)
        if s in left:
            r = right[left.index(s)]
            depth.setdefault(r, 0)
            depth[r] += 1
            depth['all'] += 1
        elif both_done: pass
        elif s in both and str_[i-1] != '\\':
            depth.setdefault(s, 0)
            if depth[s] == 0:
                depth[s] += 1
                depth['all'] += 1
    t = str_[p:].strip(strip)
    if keep_empty or t != '': 
        tokens.append(t)
    return tokens
    # if isinstance(keys, str): keys = [keys]
    # tokensin, tokensout = "([{<$\"|`'", "'`|\"$>}])"
    # d = {'()':0, '[]':0, '{}':0, '<>':0, '$$':0, '""':0, '||':0, '``':0, "''":0}
    # pair = {x:y for x, y in flatlist([[(c, x) for c in x] for x in d.keys()])}
    # depth = [0] * len(str_)
    # for i in range(len(str_)):
    #     if i > 0 and str_[i] == '\\': depth[i] = sum(d.values()); continue
    #     if str_[i] in tokensout and d[pair[str_[i]]] > 0: d[pair[str_[i]]] -= 1
    #     elif str_[i] in tokensin: d[pair[str_[i]]] += 1
    #     depth[i] = sum(d.values())
    # strs = [str_]
    # for key in keys:
    #     strs = flatlist([[sl[len(key):] if sl.startswith(key) else sl for sl in
    #                       slice(s, argmin(findall(s, key), depth))] for s in strs])
    # if len(keys) > 1: return [s for s in strs if s]
    # return strs
