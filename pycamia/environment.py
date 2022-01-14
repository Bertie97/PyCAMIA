
__info__ = dict(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    fileinfo = "File to manage the environment.",
    help = "Use `get_**` to obtain the the variables etc. outside the function. "
)

__all__ = """
    get_environ_locals
    get_environ_globals
    get_args_expression
""".split()

import sys

def _mid(x): return x[1] if len(x) > 1 else x[0]
def _rawname(s): return _mid(str(s).split("'"))

stack_error = lambda x: TypeError(f"Unexpected function stack for {x}, please contact the developer for further information. ")

def _get_frames():
    frames = []
    frame = sys._getframe()
    fname = frame.f_back.f_code.co_name
    while frame is not None:
        frame_file = _rawname(frame)
        if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>':
            frame = frame.f_back
            continue
        frames.append(frame)
        if len(frames) >= 4: return frames[2:]
        frame = frame.f_back
    raise stack_error(fname)

def get_environ_locals():
    _, client_frame = _get_frames()
    return client_frame.f_locals

def get_environ_globals():
    _, client_frame = _get_frames()
    return client_frame.f_globals
    
def get_args_expression():
    module_frame, client_frame = _get_frames()
    func_name = module_frame.f_code.co_name
    with open(client_frame.f_code.co_filename) as fp:
        for _ in range(client_frame.f_lineno-1): fp.readline()
        l = fp.readline()
        if func_name not in l: raise stack_error
        return l.split(func_name)[-1].split(';')[0].strip().strip('()')
