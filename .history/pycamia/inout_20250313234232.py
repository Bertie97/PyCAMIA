
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
    no_out
    no_print
    SPrint

    StrIO
    input_with_timeout
""".split()

import os, sys

class StrIO:
    def __init__(self, file_name = os.path.abspath('.null')):
        self._str_ = None
        self._file_ = open(file_name, 'w+')
        self.file_name = file_name
        self.fileno = self._file_.fileno
    def write(self, s): self._file_.write(s)
    def __str__(self):
        if self._str_ is not None: return self._str_
        self._file_.seek(0)
        self._str_ = self._file_.read()
        self.close()
        return self._str_
    def split(self, c=None): return str(self).split(c)
    def string(self): return str(self)
    def close(self):
        self._file_.close()
        if self.file_name == os.path.abspath('.null'):
            if os.path.exists(self.file_name): os.remove(self.file_name)
            self._file_ = None

class NPrint:
    """
    Suppress the outputs, designed for instance `no_print`.
    """
    def __init__(self, no_error=False):
        self.no_error = no_error
    def __enter__(self):
        self.io = StrIO()
        self.num_out = sys.stdout.fileno()
        if self.no_error: self.num_err = sys.stderr.fileno()
        self.old_out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
        if self.no_error: self.old_err = os.fdopen(os.dup(sys.stderr.fileno()), 'wb')
        os.dup2(self.io.fileno(), self.num_out)
        if self.no_error: os.dup2(self.io.fileno(), self.num_err)
        return self.io
    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.old_out.fileno(), self.num_out)
        if self.no_error: os.dup2(self.old_err.fileno(), self.num_err)
        self.io.string()

no_out = NPrint()
no_print = NPrint(no_error=True)
"""
Suppress the outputs.

Examples::
    >>> print("something in the front")
    >>> with no_print as io:
    >>>     print("something in the middle")
    >>> print("something in behind")
    >>> print(io.text)
    something in the front
    something in behind
    something in the middle
"""

class SPrint:
    """
    Print to a string.
    
    Args:
        init_text (str): initial text, as the head of the resulting string.
        sep (str): the seperator between different elements, just like the built-in function `print`.
        end (str): the ending string of each output, just like the built-in function `print`.

    Examples::
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

    def __init__(self, init_text='', sep=' ', end='\n', print_onscreen=False):
        self._text = init_text
        self.def_sep = sep
        self.def_end = end
        self.def_print = print_onscreen

    def __call__(self, *parts, sep=None, end=None, print_onscreen=None):
        if sep is None: sep = self.def_sep
        if end is None: end = self.def_end
        if print_onscreen is None: print_onscreen = self.def_print
        line = sep.join([str(x) for x in parts if str(x)])
        if print_onscreen: print(line, end=end)
        self._text += line + end
        return self._text

    def __str__(self): return self.text
    
    @property
    def text(self):
        t = self._text
        self._text = ""
        return t

    def clear(self): self._text = ""
    
    def rstrip(self, *args):
        self._text = self._text.rstrip(*args)
        return self
    
    def save(self, file_path):
        with open(file_path, 'w') as fp:
            fp.write(self._text)
            
    def append_to(self, file_path):
        with open(file_path, 'a') as fp:
            fp.write(self._text)

SP = ' '
CR = '\r'
LF = '\n'
CRLF = CR + LF

class TimeoutOccurred(Exception): ...

def echo(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def posix_inputimeout(prompt='', timeout=60):
    echo(prompt)
    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ)
    events = sel.select(timeout)

    if events:
        key, _ = events[0]
        return key.fileobj.readline().rstrip(LF)
    else:
        echo(LF)
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        raise TimeoutOccurred

def win_inputimeout(prompt='', timeout=60):
    echo(prompt)
    begin = time.monotonic()
    end = begin + timeout
    line = ''

    while time.monotonic() < end:
        if msvcrt.kbhit():
            c = msvcrt.getwche()
            if c in (CR, LF):
                echo(CRLF)
                return line
            if c == '\003':
                raise KeyboardInterrupt
            if c == '\b':
                line = line[:-1]
                cover = SP * len(prompt + line + SP)
                echo(''.join([CR, cover, CR, prompt, line]))
            else:
                line += c
        time.sleep(0.05)

    echo(CRLF)
    raise TimeoutOccurred

try: import msvcrt
except ImportError:
    import selectors
    import termios
    input_with_timeout = posix_inputimeout
else:
    import time
    input_with_timeout = win_inputimeout
