
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
__version__ = '2.0.0'

from .typings import as_type, get_type_name, ArrayType, type, dtype, union, intersection, intersect, insct, avoid, tag, note, class_satisfies, type_satisfies, instance_satisfies, object, bool, int, short, long, float, double, real, complex, scalar, property, slice, null, callable, functional, lambda_func, method, function, builtin_function_or_method, method_descriptor, method_wrapper, generator_function, classmethod, staticmethod, str, bytearray, bytes, memoryview, map, filter, array, iterable, sequence, list, tuple, dict, set, reversed, frozenset, range, generator, zip, enumerate #*
from .typehint import typehint, TypeHintError, HintTypeError, get_arg_values, get_virtual_declaration, deprecated, params #*
from .overload import overload, OverloadError, deprecated, override #*
