
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2023-12",
    fileinfo = "File of python code operations, such as execute blocks or python code manipulation. ",
)

__all__ = """
    sorted_dict_repr
    dict_parse
    set_indent_len
    get_indent_len
    tab_to_indent
    no_indent
    with_indent
    get_num_indent
    python_lines
    python_line
    execblock
    add_lineno
""".split()

import re, io

indent_len = 4

def sorted_dict_repr(d:dict, order:list):
    """
    Representer of dictionary `d`, with key order `order`.
    
    Examples::
        >>> sorted_dict_repr({'a': 1, '0': 3}, ['0', 'a'])
        "{'0': 3, 'a': 1}"
    """
    return '{' + ', '.join([f"{repr(k)}: {repr(d[k])}" for k in order]) + '}'

def dict_parse(str_):
    return dict([(x.strip().split('=')[0].strip(), eval(x.strip().split('=')[1].strip())) for x in str(str_).split(',')])

def set_indent_len(int_):
    global indent_len
    indent_len = int_

def get_indent_len():
    global indent_len
    return indent_len

def tab_to_indent(str_):
    global indent_len
    return '\n'.join(' ' * ((len(l) - len(l.lstrip('\t'))) * indent_len) + l.lstrip('\t') for l in str_.split('\n'))

def no_indent(str_):
    min_indent = None
    lines = str(str_).split('\n')
    for l in lines:
        indent = re.search(r'^([ \t]*)[^ \t]', l.rstrip())
        if indent is None: continue
        n_indent = len(indent.group(1))
        if min_indent is None or n_indent < min_indent: min_indent = n_indent
    return '\n'.join([l[min_indent:] for l in lines])

def with_indent(str_, indent):
    global indent_len
    if isinstance(indent, int): indent = ' ' * (indent * indent_len)
    else: indent = str(indent)
    return '\n'.join([indent + l for l in no_indent(str_).split('\n')])

def get_num_indent(line: str):
    global indent_len
    num_indent = 0
    while line.startswith(' ' * indent_len) or line.startswith('\t'):
        if line.startswith('\t'): line = line[1:]
        else: line = line[indent_len:]
        num_indent += 1
    return num_indent

def python_lines(file, n_line=None):
    qts = '\'"'
    nesters = r"()[]{}"
    find_left = {r:l for l, r in zip(nesters[::2], nesters[1::2])}
    find_right = {l:r for l, r in zip(nesters[::2], nesters[1::2])}
    depth = {}
    lines = []
    cur_line_block = []
    while True:
        if isinstance(file, str): 
            line, *rest = file.split('\n', 1)
            if len(rest) == 0 and not line.strip(): break
            file = '\n'.join(rest)
        elif isinstance(file, io.TextIOWrapper):
            line = file.readline()
            if not line: break
            line = line.rstrip('\n')
        else: raise TypeError(f"'python_lines' only takes a file stream or a string of text.")
        skip_char = False
        for i, c in enumerate(line):
            if skip_char: skip_char = False; continue
            in_str = depth.get('"', 0) + depth.get("'", 0) + depth.get('"""', 0) + depth.get("'''", 0) > 0
            if c == '#' and not in_str: break
            if c == '\\' and not in_str:
                if depth.get(c, 0) > 0: raise SyntaxError("unexpected character after line continuation character")
                depth[c] = 1
            elif c == '\\': skip_char = True; continue
            elif c in qts:
                if depth.get(qts[1-qts.index(c)], 0) + depth.get(qts[1-qts.index(c)]*3, 0) == 0:
                    if line[i:i+3] == c*3: depth[c*3] = 1 - depth.get(c*3, 0)
                    else: depth[c] = 1 - depth.get(c, 0)
            elif c in find_right:
                depth[c] = depth.get(c, 0) + 1
            elif c in find_left:
                c_left = find_left[c]
                if depth.get(c_left, 0) <= 0: raise SyntaxError(f"unmatched {c!r}")
                depth[c_left] -= 1
        cur_line_block.append(line)
        if sum(depth.values()) == 0:
            lines.append('\n'.join(cur_line_block))
            cur_line_block = []
            if n_line is not None and len(lines) >= n_line: break
        elif depth.get('\\', 0) > 0: depth['\\'] = 0
    if sum(depth.values()) > 0:
        c = [k for k, n in depth.items() if n > 0][0]
        raise SyntaxError(f"{c!r} was never closed")
    if cur_line_block: lines.append('\n'.join(cur_line_block))
    return lines

def python_line(file): return python_lines(file, n_line=1)[0]

def execblock(code):
    """
    Execute `code` with indents eliminated. 
    
    Note: Assigning local variables in functions would fail just as built-in 
        method `exec`. Use `locals()[var_name]` instead to fetch the result. 

    Examples::
        >>> class A:
        ...     def run(self, x): return x ** 2
        ...     exec('''
        ...     def additional_method(self, x):
        ...         return self.run(x)
        ...          ''')
        ...
        Traceback (most recent call last):
            [...omitted...]
        IndentationError: unexpected indent
        >>> class A:
        ...     def run(self, x): return x ** 2
        ...     execblock('''
        ...     def additional_method(self, x):
        ...         return self.run(x) + self.run(x+1)
        ...          ''')
        ...
        >>> A().additional_method(3)
        25
    """
    from .environment import get_environ_vars
    code = no_indent(code)
    vars = get_environ_vars()
    loc_vars = vars.locals
    try: exec(code, vars.globals, loc_vars)
    except Exception as e:
        raise NameError(f"Error ({e}) in block execution (make sure the variable is defined in either the most global or the very local scopes): \n{code}")
    vars.update(loc_vars)
    
def add_lineno(code):
    str_i = lambda i: f"[{i:2d}]" if i < 100 else (f"{i:3d}]" if i < 1000 else f"{i:4d}")
    return '\n'.join(f"{str_i(i+1)}: {l}" for i, l in enumerate(code.split('\n')))
