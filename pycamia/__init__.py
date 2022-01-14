
from .manager import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = '<main>',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.24',
    contact = 'bertiezhou@163.com',
    keywords = ['environment', 'path', 'touch'],
    description = 'The main package and a background support of project PyCAMIA. ',
    requires = []
).check()

from .environment import get_environ_locals, get_environ_globals, get_args_expression #*
from .exception import touch, crashed, avouch, Error #*
from .functions import empty_function, const_function, identity_function #*
from .inout import no_print, str_print, SPrint #*
from .manager import info_manager #*
from .timing import time_this, Timer, Jump, scope, jump, Workflow, periodic #*
from .listop import prod, argmin, argmax, flat_list, item, to_list, to_tuple #*
from .strop import str_len, str_slice, find_all, sorted_dict_repr, enclosed_object, tokenize #*













































































































