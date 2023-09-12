
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'pyoverload',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '2.0.0',
    contact = 'bertiezhou@163.com',
    keywords = ['overload'],
    description = "'pyoverload' overloads the functions by simply using typehints and adding decorator '@overload'. ",
    requires = ['pycamia'],
    update = '2023-07-06 20:58:10'
).check()
__version__ = '1.1.31'

from .utils import *
from .typehint import *
from .override import *
