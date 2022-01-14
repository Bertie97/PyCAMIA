
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'pyoverload',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.1.24',
    contact = 'bertiezhou@163.com',
    keywords = ['overload'],
    description = "'pyoverload' overloads the functions by simply using typehints and adding decorator '@overload'. "
).check()

from .typehint import isofsubclass, inheritable, isitertype, iterable, isarray, isdtype, isatype, isoftype, isclassmethod, typename, Type, params, Bool, Int, Float, Str, Set, List, Tuple, Dict, Class, Callable, Function, Method, Lambda, Functional, Real, real, Null, null, Sequence, sequence, Array, Iterable, Scalar, IntScalar, FloatScalar #*
from .override import override, overload, params #*













































































































