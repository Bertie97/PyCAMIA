
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to handle the outputs.",
    help = "Use `sprint = SPrint()` to start printing to a string, etc. "
)

__all__ = """
    no_print
    str_print
    SPrint
""".split()

import sys

class _strIO:
    def __init__(self): self._str_ = ''
    def write(self, s): self._str_ += s
    def __str__(self): return self._str_
    def split(self, c=None): return self._str_.split(c)

class NPrint:
    """
    Suppress the outputs, designed for instance `no_print`.
    """
    def __enter__(self):
        self.io = _strIO()
        self.old_io = sys.stdout
        sys.stdout = self.io
        return self.io
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_io

no_print = NPrint()
"""
Suppress the outputs.

Example:
----------
>>> print("something in the front")
>>> with no_print:
>>>     print("something in the middle")
>>> print("something in behind")
something in the front
something in behind
"""

class SPrint:
    """
    Print to a string.
    
    Inputs:
        init_text[str]: initial text, as the head of the resulting string.
        sep[str]: the seperator between different elements, just like the built-in function `print`.
        end[str]: the ending string of each output, just like the built-in function `print`.

    Example:
    ----------
    >>> output = SPrint("!>> ")
    >>> output("Use it", "like", 'the function', "'print'.", sep=' ')
    !>> Use it like the function 'print'.
    >>> output("A return is added automatically each time", end=".")
    !>> Use it like the function 'print'.
    A return is added automatically each time.
    >>> output.text
    !>> Use it like the function 'print'.
    A return is added automatically each time.
    """

    def __init__(self, init_text='', sep=' ', end='\n'):
        self.text = init_text
        self.def_sep = sep
        self.def_end = end

    def __call__(self, *parts, sep=None, end=None):
        if not sep: sep = self.def_sep
        if not end: end = self.def_end
        self.text += sep.join([str(x) for x in parts if str(x)]) + end
        return self.text

    def __str__(self): return self.text

    def clear(self): self.text = ""
    
    def save(self, file_path):
        with open(file_path, 'w') as fp:
            fp.write(self.text)
            
    def append_to(self, file_path):
        with open(file_path, 'a') as fp:
            fp.write(self.text)

str_print = SPrint()
"""
Print to a string.

Example:
----------
>>> str_print.clear()
>>> str_print("Use it", "like", 'the function', "'print'.", sep=' ')
Use it like the function 'print'.
>>> str_print("A return is added automatically each time", end=".")
Use it like the function 'print'.
A return is added automatically each time.
>>> str_print.text
Use it like the function 'print'.
A return is added automatically each time.
"""
