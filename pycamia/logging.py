
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File to log running infos. "
)

__all__ = """
    logging
    start_logging
""".split()

import re, os, sys, math
import builtins
import traceback as tb
from datetime import datetime
from .exception import avouch
from .environment import get_environ_vars, _get_frames, stack_error
from .system import Path

logger = None
builtin_print = print
    
def start_logging(file_path=None):
    logger = logging(file_path, _vars = get_environ_vars())
    logger.__enter__()
    builtins.print = logger.print
    client_frame = _get_frames(2)
    func_name = 'start_logging'
    with open(client_frame.f_code.co_filename) as fp:
        for _ in range(client_frame.f_lineno-1): fp.readline()
        l = fp.readline()
        if func_name not in l: raise stack_error

        prefix = re.search(r'^([ \t]*)[^ \t]', l).group(1)
        log_var = l[len(prefix):l.index(func_name)].strip(' =')
        if '\t' in prefix: indent = '\t'
        elif len(prefix) % 4 == 0: indent = '    '
        else: indent = '  '
        code = ""
        while True:
            l = fp.readline()
            if l == '': break
            avouch(l.startswith(prefix))
            l = indent + l[len(prefix):]
            code += l

        code = "try:\n" + code + """
except Exception as e:
    exc_type, exc_value, traceback = sys.exc_info()
    logger.print("Traceback (most recent call last):")
    for item in tb.StackSummary.from_list(tb.extract_tb(traceback)).format():
        logger.print(item, end="")
    logger.print(exc_type.__name__ + ':', exc_value)
    logger.fp.close()
finally:
    os._exit(0)
        """
        locals().update(client_frame.f_locals)
        globals().update(client_frame.f_globals)
        if log_var: exec(f"{log_var} = logger")
        exec(code)
    
def reformat(string, line_width = 72):
    if line_width is None: return string
    return '\n'.join([
        '\n'.join([
            l[i * line_width:(i+1) * line_width]
            for i in range(math.ceil(len(l) / line_width))
        ]) for l in string.split('\n')
    ])
    
class logging:
    def __init__(self, log_path=None, log_dir=None, prefix="{time}", _vars=None, exc_remove=True, line_width=100):
        if isinstance(prefix, str): self.prefix = lambda: prefix.format(time=datetime.now())
        else: self.prefix = prefix
        self.file_path = log_path
        self.file_dir = log_dir
        self.vars = _vars
        self.fp = None
        self.is_fplinestart = True
        self.exc_remove = exc_remove
        self.line_width = line_width
        self.start_time = datetime.now()

    def __enter__(self):
        if self.file_path is not None and not self.file_path: return
        if self.file_dir is not None and not self.file_dir: return
        if self.vars is None: self.vars = get_environ_vars()
        vars = self.vars
        if self.file_path is None:
            self.file_path = Path(vars['__file__']).with_ext('log')
        self.file_path = Path(self.file_path)
        if not self.file_path | 'log':
            self.file_path = self.file_path // 'log'
        time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        self.file_path ^= time_str
        if self.file_dir is not None:
            self.file_path = Path(self.file_dir).mkdir() / self.file_path.filename
        open(self.file_path, 'w').close()
        self.fp = open(self.file_path, 'a')
        builtins.print = self.print
        fill1 = fill2 = ''
        if '__info__' in vars:
            fill1 = f" in {vars['__info__'].project}{'.' + vars['__info__'].package if vars['__info__'].package else ''} ({vars['__info__'].author})"
            fill2 = '\n| ' + vars['__info__'].fileinfo
        header = '=' * 100 + f"""
Logging file for python script{fill1}: {Path(vars['__file__']).filename}{fill2}
Started at {datetime.now()} with command:
> python {Path(sys.argv[0]).filename} {' '.join(sys.argv[1:])}
"""
        # if '__info__' in vars:
        header += '=' * 100 + "\n\n"
        header = reformat(header, line_width=self.line_width)
        builtin_print(header, end='')
        self.fp.write(header)
        self.fp.flush()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.fp is None: return
        if exc_type is not None:
            exc_string = "Traceback (most recent call last): \n"
            for item in tb.StackSummary.from_list(tb.extract_tb(traceback)).format(): exc_string += str(item)
            exc_string += exc_type.__name__ + ': ' + str(exc_value) + '\n'
            self.fp_print(exc_string)
            self.fp.close()
            if self.exc_remove:
                log_file = Path(self.file_path)
                if "KeyboardInterrupt: " in exc_string: log_file.rename(log_file^"interrupted")
                else: log_file.remove()
                er_file = log_file.with_name("error_msgs")
                if er_file.exists(): mode = 'a'
                else: mode = 'w'
                with er_file.open(mode) as fp: fp.write(('\n' if mode == 'a' else '') + f"[{self.start_time} - {datetime.now()}]: {exc_string}")
        self.fp = None
        builtins.print = builtin_print
        
    def print(self, *values, sep=' ', end='\n', **kwargs):
        builtin_print(self.fp_print(*values, sep=sep, end=end, **kwargs), end='')

    def fp_print(self, *values, sep=' ', end='\n', **kwargs):
        line_prefix = self.prefix()
        if "{time}" in line_prefix: line_prefix = line_prefix.replace("{time}", str(datetime.now()))
        blank_prefix = ' ' * len(line_prefix)
        if line_prefix:
            line_prefix += ' | '
            blank_prefix += ' | '
        output_str = sep.join([str(v) for v in values if str(v)]) + \
                     sep.join([f"{k} = {str(v)}," for k, v in kwargs.items() if str(v)]) + end
        output_str = reformat(output_str, line_width=self.line_width)
        first_line, *other_lines = output_str.split('\n')
        if len(other_lines) == 0:
            self.fp.write((line_prefix if self.is_fplinestart else '') + first_line)
            self.is_fplinestart = False
        else:
            last_line = other_lines.pop(-1)
            self.fp.write((line_prefix if self.is_fplinestart else '') + first_line + '\n')
            self.fp.write(''.join([blank_prefix + l + '\n' for l in other_lines]))
            if last_line == '': self.is_fplinestart = True
            else: self.fp.write(blank_prefix + last_line); self.is_fplinestart = False
        self.fp.flush()
        return output_str

    def std_print(self, *values, sep=' ', end='\n'):
        builtin_print(*values, sep=sep, end=end)
