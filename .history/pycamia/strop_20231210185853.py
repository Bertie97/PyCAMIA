
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File of string operations. "
)

__all__ = """
    is_digits
    is_alphas
    is_snakename
    get_digits
    get_alphas
    get_snakename
    get_snakenames
    str_len
    str_slice
    find_all
    enclosed_object
    tokenize
    token_replace
    columns
""".split()

import re, math

def is_digits(str_:str):
    """
    Return whether `str_` characters are all digits. 
    
    Examples::
        >>> print(is_digits("123"), is_digits("12 3"))
        True False
    """
    return all([48 <= ord(x) <= 57 for x in str(str_)])

def is_alphas(str_:str):
    """
    Return whether `str_` characters are all alphabets. 
    
    Examples::
        >>> print(is_alphas("abc"), is_alphas("1bc"))
        True False
    """
    return all([65 <= ord(x) <= 90 or 97 <= ord(x) <= 122 for x in str(str_)])

def is_snakename(str_:str):
    """
    Return whether `str_` is a snake name (i.e. for variable names or file names), whose characters are all alphabets/digits/underlines. 
    
    Examples::
        >>> print(is_snakename("abc2_x"), is_snakename("1bc"), is_snakename("x_ y"))
        True False False
    """
    str_ = str(str_)
    return (65 <= ord(str_[0]) <= 90 or 97 <= ord(str_[0]) <= 122) and all([48 <= ord(x) <= 57 or 65 <= ord(x) <= 90 or ord(x) == 95 or 97 <= ord(x) <= 122 for x in str_])

def get_digits(str_:str):
    """
    Return all digits in `str_`. 
    
    Examples::
        >>> print(repr(get_digits("12 3")), repr(get_digits("12a3")))
        '123' '123'
    """
    return ''.join(x for x in str(str_) if 48 <= ord(x) <= 57)

def get_alphas(str_:str):
    """
    Return all alphabets in `str_`. 
    
    Examples::
        >>> print(repr(get_alphas("12 3")), repr(get_alphas("12a3")))
        '' 'a'
    """
    return ''.join(x for x in str(str_) if 65 <= ord(x) <= 90 or 97 <= ord(x) <= 122)

def get_snakename(str_:str):
    """
    Return all alphabets/digits/underlines characters in `str_` so that it is a snake name (i.e. for variable names or file names). 
    
    Examples::
        >>> print(repr(get_snakename("abc2_x")), repr(get_snakename("1bc")), repr(get_snakename("x_ y:")))
        'abc2_x' 'bc' 'x_y'
    """
    past_prefix = False
    result = ""
    for x in str(str_):
        if (65 <= ord(x) <= 90 or 97 <= ord(x) <= 122):
            result += x
            past_prefix = True
        elif (48 <= ord(x) <= 57 or ord(x) == 95) and past_prefix:
            result += x
        else:
            continue
    return result

def get_snakenames(str_:str):
    """
    Return all alphabets/digits/underlines (snakename) words in `str_`. 
    
    Examples::
        >>> print(repr(get_snakename("abc2_x")), repr(get_snakename("1bc")), repr(get_snakename("x_ y:")))
        ['abc2_x'] ['bc'] ['x', 'y']
    """
    rec = False
    ret = []
    for c in str(str_):
        if not rec and (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122): rec = True; ret.append(c); continue
        if rec and (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122 or 48 <= ord(c) <= 57 or ord(c) == 95): ret[-1] += c; continue
        if rec: rec = False
    return ret
    

def str_len(str_:str, r:int=2):
    """
    Returen the ASCII string length of `str_`. 
    
    Args:
        r: bytes a wide character stands for. 
    
    Examples::
        >>> print(str_len("12"), len("你好"), str_len("你好"))
        2 2 4
    """
    length = 0
    for c in str(str_):
        if ord(u'\u4e00') <= ord(c) <= ord(u'\u9fa5'): length += r
        elif c == '\t': length += 4
        else: length += 1
    return length

def find_all(str_:str, key:str):
    """
    Returen all the starting indices of string `key` in string `str_`. 
    
    Examples::
        >>> find_all("abcaa", 'a')
        [0, 3, 4]
    """
    p, indices = -1, []
    while True:
        p = str(str_).find(key, p + 1)
        if p < 0: break
        indices.append(p)
    return indices

def str_slice(str_:str, indices:list):
    """
    Split the string `str_` by breaks in list `indices`.
    
    Examples::
        >>> str_slice("abcaa", [2,4])
        ["ab", "ca", "a"]
    """
    indices.insert(0, 0); indices.append(len(str(str_)))
    return [str(str_)[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]

def enclosed_object(str_, by="()[]{}$$``''\"\"", start=0):
    """
    Return the first object enclosed with a whole pair of parenthesis in `str_` after index `start`.
    
    Argument by[str]: A string with brackets such as "()[]''", pairs by pairs. 
        Same characters on both sides like quotes are also included. 
    Note: One may also use the old way - a list of 'left char', 'right char' and 'chars for both sides',
        e.g. ["([{", ")]}", "$`'\""], but this is deprecated. 
    
    Examples::
        >>> enclosed_object("function(something inside), something else. ")
        function(something inside)
    """
    str_ = str(str_)
    if isinstance(by, list):
        if len(by) == 3: left, right, both = by
        elif len(by) == 2: left, right = by; both = ""
        elif len(by) == 1: left = ""; right = ""; both = by[0]
        else: raise TypeError(f"Invalid argument `by` of length {len(by)} for function `enclosed_object`, should be 1-3. ")
    elif isinstance(by, str):
        left = ""; right = ""; both = ""
        for i in range(0, len(by), 2):
            if by[i] == by[i+1]: both += by[i]
            else: left += by[i]; right += by[i+1]
        by = [left, right, both]
    else: raise TypeError(f"Invalid argument `by` of type {type(by)} for function `enclosed_object`. ")
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

def tokenize(str_:str, sep=[' ', '\n'], by="()[]{}$$``''\"\"", start=0, strip='', keep_empty=True):
    """
    Split the string `str_` by elements in `sep`, but keep enclosed objects not split.
    
    Args:
        sep (list of str's): all strings that seperate the tokens.
        by (str): A string with brackets such as "()[]''", pairs by pairs. 
            Same characters on both sides like quotes are also included. 
        Note: One may also use the old way - a list of 'left char', 'right char' and 'chars for both sides',
            e.g. ["([{", ")]}", "$`'\""], but this is deprecated. 

    Examples::
        >>> tokenize("function(something inside), something else. ")
        ["function(something inside),", "something", "else.", ""]
    """
    str_ = str(str_)
    if strip == True: strip = None
    if strip == False: strip = ''
    if isinstance(sep, str): sep = [sep]
    if isinstance(by, list):
        if len(by) == 3: left, right, both = by
        elif len(by) == 2: left, right = by; both = ""
        elif len(by) == 1: left = ""; right = ""; both = by[0]
        else: raise TypeError(f"Invalid argument `by` of length {len(by)} for function `tokenize`, should be 1-3. ")
    elif isinstance(by, str):
        left = ""; right = ""; both = ""
        for i in range(0, len(by), 2):
            if by[i] == by[i+1]: both += by[i]
            else: left += by[i]; right += by[i+1]
        by = [left, right, both]
    else: raise TypeError(f"Invalid argument `by` of type {type(by)} for function `tokenize`. ")
    depth = {'all': 0}
    class Tokens(list):
        def __new__(cls, *args, **kwargs):
            self = super().__new__(cls, *args, **kwargs)
            self.indices = []
            return self
        def get(self, k, default=None):
            if not default: return self[self.index(k) + 1]
            else:
                try: return self[self.index(k) + 1]
                except ValueError: return default
        def get_all(self, k, default=None):
            if not default: return self[self.index(k) + 1]
            else:
                try: return self[self.index(k) + 1]
                except ValueError: return default
    tokens = Tokens()
    tokens.starting_indices = []
    p = start
    for i in range(start, len(str_)):
        if depth['all'] == 0:
            for x in sep:
                if str_[i:i + len(x)] == x:
                    t = str_[p:i].strip(strip)
                    if keep_empty or t != '': 
                        tokens.append(t)
                        tokens.indices.append((p, i))
                    p = i + len(x)
        elif depth['all'] == 1:
            for x in sep:
                if len(x) != 1 or x not in right: continue
                if left[right.index(x)] in sep and depth[x] == 1 and str_[i] == x:
                    t = str_[p:i].strip(strip)
                    if keep_empty or t != '': 
                        tokens.append(t)
                        tokens.indices.append((p, i))
                    p = i + 1
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
        tokens.indices.append((p, len(str_)))
    return tokens

def token_replace(str_:str, expr:str, replacement:str, sep=[' ', '\n'], by="()[]{}$$``''\"\"", start=0, strip='', keep_empty=True):
    if not (isinstance(expr, str) or callable(expr)):
        raise TypeError("'token_replace' only accept str or function for the keyword expression. ")
    tokens = tokenize(str_, sep=sep, by=by, start=start, strip=strip, keep_empty=keep_empty)
    output = []
    p = 0
    for i, t in enumerate(tokens):
        if isinstance(expr, str) and t != expr: continue
        if callable(expr) and not expr(t): continue
        output.append(str_[p:tokens.indices[i][0]])
        output.append(replacement)
        p = tokens.indices[i][1]
    output.append(str_[p:])
    return ''.join(output)

def columns(*strs, sep=' | ', line_width = 100):
    strs = list(strs)
    n = len(strs)
    col_width = (line_width - (n-1) * len(sep)) // n
    strs_lines = [str(s).split('\n') for s in strs]
    output = ""
    for L in range(max([len(str_lines) for str_lines in strs_lines])):
        lines = []
        for str_lines in strs_lines:
            str_line = str_lines[L] if L < len(str_lines) else ''
            column = []
            if len(str_line) > col_width:
                for p in range(math.ceil(len(str_line) / col_width)):
                    column.append(str_line[p*col_width:(p+1)*col_width])
            else: column.append(str_line)
            lines.append(column)
        for i in range(max(len(c) for c in lines)):
            output += sep.join((c[i] + ' ' * (col_width - len(c[i]))) if i < len(c) else ' ' * col_width for c in lines) + '\n'
    return output

