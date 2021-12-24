
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to manage the environment.",
    help = "Use `get_environ_vars` to obtain the `locals()` outside the function. "
)

__all__ = """
    get_environ_vars
""".split()

import sys

def _mid(x): return x[1] if len(x) > 1 else x[0]
def _rawname(s): return _mid(str(s).split("'"))

class get_environ_vars(dict):
    """
    get_environ_vars() -> 'dict'(a type extended from dict)

    Returns the environment variables in the frame out of the package.
        i.e. the variables defined in the most reasonable user environments.

    Note:
        Please do not use it abusively as it is currently provided for 
        private use in project PyCAMIA only. Using it outside may cause error.

    Example:
    ----------
    In file `main.py`:
        from mod import function
        x = 1
        function()
    In file `mod.py`:
        from pyoverload.utils import get_environ_vars
        def function(): return get_environ_vars()
    Output:
        {
            'function': < function 'function' >,
            'x': 1,
            '__name__': "__main__",
            ...
        }
    """

    def __new__(cls):
        self = super().__new__(cls)
        frame = sys._getframe()
        self.all_vars = []
        prev_frame = frame
        prev_frame_file = _rawname(frame)
        while frame.f_back is not None:
            frame = frame.f_back
            frame_file = _rawname(frame)
            if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>': continue
            if '<module>' not in str(frame):
                if frame_file != prev_frame_file:
                    prev_frame = frame
                    prev_frame_file = frame_file
                continue
            if frame_file != prev_frame_file: self.all_vars.extend([frame.f_locals])
            else: self.all_vars.extend([prev_frame.f_locals])
            break
        else: raise TypeError("Unexpected function stack, please contact the developer for further information. ")
        return self

    def __init__(self): pass

    def __getitem__(self, k):
        for varset in self.all_vars:
            if k in varset:
                return varset[k]
                break
        else: raise IndexError(f"No '{k}' found in the environment. ")

    def __setitem__(self, k, v):
        for varset in self.all_vars:
            if k in varset:
                varset[k] = v
                break
        else: self.all_vars[0][k] = v

    def __contains__(self, x):
        for varset in self.all_vars:
            if x in varset: break
        else: return False
        return True
    
    def __str__(self):
        return str(self.simplify())

    def update(self, x): self.all_vars[0].update(x)

    def simplify(self):
        collector = {}
        for varset in self.all_vars[::-1]: collector.update(varset)
        return collector
