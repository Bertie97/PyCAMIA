
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to record time."
)

__all__ = """
    timethis
    timer
    scope
    jump
    JUMP
    Workflow
    periodic
""".split()

with __info__:
    import time
    from functools import wraps
    from threading import Timer

def timethis(func):
    """
    A function wrapper of function `func` that outputs the time used to run it. 

    Example:
    ----------
    @timethis
    >>> def func_to_run(*args):
    ...     # inside codes
    ... 
    >>> func_to_run(*input_args)
    # some outputs
    [func_to_run takes 0.001s]
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if hasattr(getattr(func, '__wrapped__', func), '__name__'):
            print("[%s takes %lfs]"%(func.__name__, end-start))
        else:
            print("[%s takes %lfs]"%(func.__class__.__name__, end-start))
        return result
    return wrapper

class timer(object):
    """
    An environment that outputs the time used to run codes within. 

    Example:
    ----------
    >>> with timer("test"):
    ...     # inside codes
    ... 
    # some outputs
    [test takes 0.001s]
    """
    def __init__(self, name='', timing=True):
        if not timing: name = ''
        self.name = name
        self.nround = 0
    def __enter__(self):
        self.start = time.time()
        self.prevtime = self.start
        return self
    def round(self, name = ''):
        self.nround += 1
        self.end = time.time()
        if self.name:
            if not name: name = "%s(round%d)"%(self.name, self.nround)
            print("[%s takes %lfs]"%(name, self.end - self.prevtime))
        self.prevtime = self.end
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == RuntimeError and str(exc_value) == "JUMP": return True
        if self.name:
            print("[%s%s takes %lfs]"%
                  (self.name, '' if self.nround == 0 else "(all)", time.time() - self.start))

class JUMP(object):
    """
    Creates a JUMP RuntimeError, designed for instance `jump`. 
    """
    def __init__(self, jump=None): self.jump = True if jump is None else jump
    def __enter__(self):
        def dojump(): raise RuntimeError("JUMP")
        if self.jump: dojump()
        else: return dojump
    def __exit__(self, *args): pass
    def __call__(self, condition): return JUMP(condition)
    
def scope(name, timing=True):
    """
    An allias of timer to better organize the codes. 
    
    Inputs:
        name[str]: the name of the scope, used to display. 
        timing[bool]: whether to show the time span or not. 

    Example:
    ----------
    >>> with scope("test"):
    ...     # inside codes
    ... 
    # some outputs
    [scope test takes 0.001s]
    """
    return timer("scope " + name, timing)

jump = JUMP()
"""
The jumper, one can use it along with `scope`(or `timer`) to jump a chunk of codes. 

Example:
----------
>>> with scope("test"), jump:
...     # inside codes
... 
# nothing, the inside codes do not run
"""

class Workflow:
    """
    A structure to create a series of workflow. 
    
    Note:
        Remember to manually add `, *.jump` after with so that 
        we can control it. See the example. 
    
    Arrguments:
        *args: the list of scope names to run. 

    Example:
    ----------
    >>> run = Workflow("read data", "run method", "visualization")
    ... with run("read data"), run.jump:
    ...     print(1, end='')
    ... with run("pre-processing"), run.jump:
    ...     print(2, end='')
    ... with run("run method"), run.jump:
    ...     print(3, end='')
    ... with run("visualization"), run.jump:
    ...     print(4, end='')
    ... 
    1[read data takes 0.000022s]
    3[run method takes 0.000008s]
    4[visualization takes 0.000006s]
    """
    def __init__(self, *args): self.workflow = args
    def __call__(self, key): self.key=key; return timer(key)
    def __getattr__(self, k): return self(k)
    def __getitem__(self, k): return self(k)
    @property
    def j(self): return JUMP(self.key not in self.workflow)
    @property
    def jump(self): return JUMP(self.key not in self.workflow)

class TimerCtrl(Timer):
    """
    Creates a Time Handler, designed for function `periodic`. 
    """

    def __init__(self, seconds, function):
        Timer.__init__(self, seconds, function)
        self.isCanceled = False
        self.seconds = seconds
        self.function = function
        self.funcname = function.__name__
        self.startTime = time.time()

    def cancel(self):
        Timer.cancel(self)
        self.isCanceled = True

    def is_canceled(self): return self.isCanceled

    def setFunctionName(self, funcname): self.funcname = funcname

    def __str__(self):
        return "%5.3fs to run next "%(self.seconds + self.startTime -
                                      time.time()) + self.funcname


def periodic(period, maxiter=float('Inf')):
    """
    A function wrapper to repeatedly run the wrapped function `period`.
    
    Arrguments:
        maxiter[int]: the number of iterations. 

    Example:
    ----------
    >>> i = 1
    ... @periodic(1)
    ... def func():
    ...     print(i)
    ...     i+= 1
    ... 
    1
    2
    3
    [Output every 1s, and GO ON...]
    """
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global count, timer_ctrl
            try:
                if timer_ctrl.is_canceled(): return
            except NameError: pass
            timer_ctrl = TimerCtrl(period, lambda : wrapper(*args, **kwargs))
            timer_ctrl.setFunctionName(func.__name__)
            timer_ctrl.start()
            ret = func(timer_ctrl, *args, **kwargs)
            try: count += 1
            except Exception: count = 1
            if count >= maxiter: return
            return ret
        return wrapper
    return wrap
