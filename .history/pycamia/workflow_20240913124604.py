
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2024-09",
    fileinfo = "File to set workflows."
)

__all__ = """
    Jump
    scope
    jump
    Workflow
    switch
""".split()

from .timing import Timer
from .exception import Error
from .decorators import alias

class Jump(object):
    """
    Creates a Jump Error to escape scopes. 

    Examples::
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
    
    Args:
        name (str): the name of the scope, used to display. 
        log_on_screen (bool): whether to show the time span or not. 

    Examples::
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

    Examples::
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
    def force_jump(self): ...
    def run_as_workflow(self): ...
    def all_tags(self): ...
    def use_tag(self): ...
    def any_tag(self): ...
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

class Switch:
    
    def __init__(self, variable): self.value = variable
    def __call__(self, value): return Switch(value == self.value)
    def __enter__(self): return self.value
    def __exit__(self, error_type, error_msg, traceback):
        if error_type == Error("Jump") or error_type == AssertionError: return
        raise error_type(error_msg)

switch = Switch()