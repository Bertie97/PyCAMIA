
from .manager import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = '<main>',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.41',
    update = '2024-12-24 16:23:04',
    contact = 'bertiezhou@163.com',
    keywords = ['environment', 'path', 'touch'],
    description = 'The main package and a background support of project PyCAMIA. ',
    requires = ['tqdm', 'psutil']
).check()
__version__ = '1.0.41'

from .environment import get_environ_vars, get_environ_globals, get_environ_locals, update_locals_by_environ, get_args_expression, get_reference_line, get_declaration, EnvironVars #*
from .exception import touch, crashed, avouch, Error, Warn #*
from .functions import empty_function, const_function, identity_function #*
from .inout import no_out, no_print, SPrint, StrIO, input_with_timeout #*
from .manager import info_manager, Hyper, hypers, Version, Args, args #*
from .timing import time_this, Timer, periodic, periodic_run, periodic_call, run_later #*
from .workflow import scope, Jump, jump, Workflow, Switch, switch #*
from .decorators import alias, decorator, restore_type_wrapper #*
from .listop import prod, cumsum, cumprod, cartesian_prod, argmin, argmax, min_argmin, max_argmax, kth_biggest, kth_smallest, median, flatten_list, item, to_list, to_tuple, to_set, map_ele, sublist, arg_tuple, arg_extract, count, unique, infinite_itemize, cat_generator, param_join #*
from .loopop import Collector, iterate #*
from .strop import is_digits, is_alphas, is_snakename, get_digits, get_alphas, get_snakename, get_snakenames, str_len, str_slice, find_all, enclosed_object, tokenize, token_replace, refind, resplit, rereplace, columns, format #*
from .pythonop import sorted_dict_repr, dict_parse, set_indent_len, get_indent_len, tab_to_indent, no_indent, with_indent, get_num_indent, python_lines, python_line, execblock, add_lineno #*
from .system import path, Path, path_list, PathList, set_black_list, get_black_list, curdir, pardir, homedir, rootdir, pwd, ls, cp, copy, mv, move, merge, rename, is_valid_command, Shell, get_volumes, ByteSize #*
from .logging import logging, start_logging #*
from .math import GCD, isint, rational, factorial #*
from .structures import struct, odict, protocol #*
from .more import once, void, random_seed #*
from . import system as pth
