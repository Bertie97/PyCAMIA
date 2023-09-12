
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to record time."
)

__all__ = """
    time_this
    Timer
    Jump
    scope
    jump
    Workflow
    periodic
    periodic_run
    periodic_call
    run_later
""".split()

import time
from functools import wraps
from threading import Thread, Timer as tTimer

from .exception import Error
from .decorators import alias
from .environment import get_environ_vars

def time_this(func):
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

class Timer(object):
    """
    An environment that outputs the time used to run codes within. 

    Example:
    ----------
    >>> with Timer("test"):
    ...     # inside codes
    ... 
    # some outputs
    [test takes 0.001s]
    """
    def __init__(self, name='', timing=True, log_on_screen=True):
        if not name: name = 'Unknown'
        self.name = str(name)
        self.nround = 0
        self.timing = timing
        self.log_on_screen = log_on_screen
        self.prevrecord = None
    @property
    def recorded_time(self): return self.prevrecord
    def __enter__(self):
        self.start = time.time()
        self.prevtime = self.start
        return self
    def round(self, name = ''):
        self.nround += 1
        self.end = time.time()
        if self.timing:
            if not name: name = "%s(round%d)"%(self.name, self.nround)
            self.prevrecord = self.end - self.prevtime
            print(f"[{name} takes {self.prevrecord}s]")
        self.prevtime = self.end
    def exit(self): raise Error("Jump")("Jump by calling 'exit'. ")
    def __exit__(self, exc_type, exc_value, traceback):
        if self.timing:
            self.prevrecord = time.time() - self.start
            if self.log_on_screen:
                print(f"[{self.name}{'' if self.nround == 0 else '(all)'} takes {self.prevrecord}s]")
        if exc_type == Error("Jump"): return True

class Jump(object):
    """
    Creates a Jump Error to escape scopes. 

    Example:
    ----------
    >>> with scope("test"), jump:
    ...     # inside codes
    ... 
    # nothing, the inside codes do not run
    >>> with scope("test"), Jump(False) as stop:
    ...     print("Part I")
    ...     stop()
    ...     print("Part II")
    ... 
    Part I
    """
    def __init__(self, jump=None): self.jump = True if jump is None else jump
    def __enter__(self):
        def dojump(): raise Error("Jump")("Jump by class 'Jump'. ")
        if self.jump: dojump()
        else: return dojump
    def __exit__(self, *args): pass
    def __call__(self, condition): return Jump(condition)
    
def scope(name, log_on_screen=True):
    """
    An allias of timer to better organize the codes, use .exit() to exit the scope. 
    
    Inputs:
        name (str): the name of the scope, used to display. 
        log_on_screen (bool): whether to show the time span or not. 

    Example:
    ----------
    >>> with scope("test"):
    ...     # inside codes
    ... 
    # some outputs
    [scope test takes 0.001s]
    >>> with scope("this") as s:
    ...     print("Part I")
    ...     s.exit()
    ...     print("Part II")
    ...
    Part I
    >>> with scope("this again", False) as s:
    ...     print("Part I")
    ...     print("Part II")
    ...
    Part I
    Part II
    >>> print(s.recorded_time)
    2.86102294921875e-06
    """
    return Timer("scope " + str(name), timing=True, log_on_screen=log_on_screen)

jump = Jump()
"""
The jumper, one can use it along with `scope`(or `Timer`) to jump a chunk of codes. 
"""

class Workflow:
    """
    A structure to create a series of workflow. 
    
    Note:
        Remember to manually add a behaviour for each block: 
            '*.force_run' force the block to run, without checking the workflow.
            '*.force_skip'/'*.force_jump' force the block to be skipped, without checking the workflow.
            '*.use_tag'/'*.run_as_workflow' runs the block following the workflow schedule if one tag name is provided.
            '*.all_tags'/'*.run_if_all_tags_in_workflow' runs the block when all given tags are defined in the workflow. 
            '*.any_tag'/'*.run_if_any_tag_in_workflow' runs the block when at least one tag is defined in the workflow. 
        Fogetting to add the behaviour would result in an automatic run of blocks. See the example for details of bahaviours. 
    
    Args:
        *args: the list of scope names to run. 

    Example:
    ----------
    >>> run = Workflow("read data", "run method", "visualization")
    ... with run("read data"), run.use_tag:
    ...     print(1, end='')
    ... with run("pre-processing"), run.use_tag:
    ...     print(2, end='')
    ... with run("pre-processing", "run method"), run.all_tags:
    ...     print(3, end='')
    ... with run("visualization"), run.force_skip:
    ...     print(4, end='')
    ... 
    1[read data takes 0.000022s]
    """
    def __init__(self, *args, verbose=True): self.workflow = args; self.verbose=verbose
    def __call__(self, *keys):
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)): keys = keys[0]
        self.keys=keys
        return Timer(','.join(keys), timing=self.verbose)
    def __getattr__(self, *k): return self(*k)
    def __getitem__(self, *k): return self(*k)
    @property
    def force_run(self): return Jump(False)
    @alias("force_jump")
    @property
    def force_skip(self): return Jump(True)
    @alias("run_as_workflow", "all_tags", "use_tag")
    @property
    def run_if_all_tags_in_workflow(self):
        return Jump(any(k not in self.workflow for k in self.keys))
    @alias("any_tag")
    @property
    def run_if_any_tag_in_workflow(self):
        return Jump(all(k not in self.workflow for k in self.keys))

class TimerCtrl(tTimer):
    """
    Creates a Time Handler, designed for function `periodic`. 
    """

    def __init__(self, seconds, function):
        tTimer.__init__(self, seconds, function)
        self.isCanceled = False
        self.seconds = seconds
        self.function = function
        self.funcname = function.__name__
        self.startTime = time.time()

    def cancel(self):
        tTimer.cancel(self)
        self.isCanceled = True

    def is_canceled(self): return self.isCanceled

    def setFunctionName(self, funcname): self.funcname = funcname

    def __str__(self):
        return "%5.3fs to run next "%(self.seconds + self.startTime -
                                      time.time()) + self.funcname

class STOP:
    def __init__(self): self.stopped = False; self.index = 0
    def __bool__(self): return self.stopped
    def set_index(self, index): self.index = index
    def stop(self): self.stopped = True
    def resume(self): self.stopped = False

def periodic(period, maxiter=float('Inf'), wait_before_first_call=False, wait_return=False, use_new_thread=True, process_return=None):
    """
    A function wrapper to repeatedly run the wrapped function `period`.
    The return of function being False means the periodic process should stop.
    
    Args:
        period (int / float): the number of seconds to wait between calls.
        maxiter (int): the number of iterations. 
        wait_before_first_call (bool): If True, the iterations would not start immediately, but after a prior `period` seconds.
        wait_return (bool): If True, the next iteration awaits for `period` seconds after the current is over.
            If False, the next iteration starts right after `period` seconds, unless the current job is unfinished even then
                (A warning message would be sent in this circumstance).
        use_new_thread (bool): Whether to ceate new threads for the task.
        process_return (func): a function accept arguments (timer, return_value) to deal with the values returned from the main function. 
            Only activated when `wait_return = True`. It controls the timer by default. The default function is:
                process_return_default = lambda t, r: t.stop() if r is False else None
            Note: This means the timer does not stop by default when no return is provided. 

        Note: if wait_return is False, function should accept a timer controller timer as the first argument just like 'self'.
        timer: controller with .index storing the iteration, .stop() to stop the periodic process before it is meant to stop, .resume() to continue.

    Example:
    ----------
    >>> i = 1
    ... @periodic(1)
    ... def func(timer):
    ...     global i
    ...     print(i, timer.index)
    ...     i += 1
    ...     timer.stop()
    ...     timer.resume()
    ... 
    1
    2
    3
    [Output every 1s, and GO ON...]
    """
    if process_return is None:
        process_return = lambda t, r: t.stop() if r is False else None
    if not wait_return and not use_new_thread: raise TypeError("Not waiting for results needs a new thread to work on the process.")
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fname = func.__name__ + '_' + str(time.time())
            global counts, stops, dones, waitings
            if 'counts' not in globals(): counts = {}
            if 'stops' not in globals(): stops = {}
            if 'dones' not in globals(): dones = {}
            if 'waitings' not in globals(): waitings = {}
            if fname not in counts: counts[fname] = 0
            if fname not in stops: stops[fname] = STOP()
            if fname not in dones: dones[fname] = True
            if fname not in waitings: waitings[fname] = False
            
            def run_next(fname, process_return, args, kwargs, calling):
                global counts, stops, dones, waitings
                if use_new_thread:
                    timer_ctrl = TimerCtrl(period, calling)
                    timer_ctrl.setFunctionName(fname)
                if counts[fname] == 0:
                    counts[fname] = 1
                    if wait_before_first_call:
                        if use_new_thread: timer_ctrl.start()
                        else:
                            time.sleep(period)
                            calling()
                        return
                if not wait_return:
                    if not dones[fname]:
                        print("Warning: Job not finished in one period, waiting for it to be done.")
                        waitings[fname] = True
                        return
                    dones[fname] = False
                    timer_ctrl.start()
                    stops[fname].set_index(counts[fname])
                    ret = func(stops[fname], *args, **kwargs)
                    dones[fname] = True
                    counts[fname] += 1
                    if counts[fname] >= maxiter: return
                    if waitings[fname]:
                        waitings[fname] = False
                        instant_timer_ctrl = TimerCtrl(0, calling)
                        instant_timer_ctrl.setFunctionName(fname)
                        instant_timer_ctrl.start()
                    return
                else:
                    ret = func(*args, **kwargs)
                    counts[fname] += 1
                    if counts[fname] >= maxiter: return
                    process_return(stops[fname], ret)
                    if use_new_thread: timer_ctrl.start()
                    else:
                        time.sleep(period)
                        calling()
            if use_new_thread:
                class new_thread(Thread):
                    def __init__(self, fname, *args, **kwargs):
                        super().__init__()
                        self.fname = fname
                        self.process_return = process_return
                        self.args = args
                        self.kwargs = kwargs
                        
                    def run(self):
                        global stops
                        if stops[self.fname]: return
                        run_next(self.fname, self.process_return, self.args, self.kwargs, self.run)
                new_thread(fname, *args, **kwargs).start()
            else:
                def recursive():
                    run_next(fname, process_return, args, kwargs, recursive)
                recursive()
        return wrapper
    return wrap

def periodic_run(period, maxiter=float('Inf'), use_new_thread=True):
    return periodic(period, maxiter=maxiter, wait_return=True, use_new_thread=use_new_thread)

def periodic_call(period, maxiter=float('Inf'), use_new_thread=True):
    return periodic(period, maxiter=maxiter, wait_return=False, use_new_thread=use_new_thread)

def run_later(wait, use_new_thread=True):
    return periodic(wait, 1, wait_return=True, wait_before_first_call=True, use_new_thread=use_new_thread)
            
