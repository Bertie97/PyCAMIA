
from .manager import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = '<main>',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.38',
    update = '2023-07-07 18:16:08',
    contact = 'bertiezhou@163.com',
    keywords = ['environment', 'path', 'touch'],
    description = 'The main package and a background support of project PyCAMIA. ',
    requires = []
).check()
__version__ = '1.0.38'

from .environment import get_environ_vars, get_environ_globals, get_environ_locals, update_locals_by_environ, get_args_expression, get_declaration, EnvironVars #*
from .exception import touch, touch_func, crashed, avouch, Error, void #*
from .functions import empty_function, const_function, identity_function #*
from .inout import no_out, no_print, SPrint, StrIO #*
from .manager import info_manager, Hyper, hypers, Version #*
from .timing import time_this, Timer, Jump, scope, jump, Workflow, periodic, periodic_run, periodic_call, run_later #*
from .decorators import alias, restore_type_wrapper #*
from .listop import prod, cartesian_prod, argmin, argmax, min_argmin, max_argmax, kth_biggest, kth_smallest, median, flatten_list, item, to_list, to_tuple, to_set, map_ele, sublist, arg_tuple, arg_extract, count, unique #*
from .loopop import Collector, iterate #*
from .strop import is_digits, is_alphas, is_snakename, get_digits, get_alphas, get_snakename, get_snakenames, str_len, str_slice, find_all, sorted_dict_repr, enclosed_object, tokenize, dict_parse, no_indent, columns, python_lines #*
from .system import path, Path, path_list, PathList, set_black_list, get_black_list, curdir, pardir, homedir, rootdir, pwd, ls, cp, copy, mv, move, rename, is_valid_command, Shell, execblock, get_volumes, ByteSize #*
from .logging import logging, start_logging #*
from .math import GCD, isint, rational #*
from .structures import struct #*
from . import system as pth
