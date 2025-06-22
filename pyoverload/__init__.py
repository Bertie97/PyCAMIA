
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'pyoverload',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '2.0.3',
    contact = 'bertiezhou@163.com',
    keywords = ['overload'],
    description = "'pyoverload' overloads the functions by simply using typehints and adding decorator '@overload'. ",
    requires = ['pycamia'],
    update = '2024-12-24 16:23:04'
).check()
__version__ = '2.0.3'

type = type
from .dtypes import to_dtype, to_rawdtype, to_torch_type, to_torch_dtype, to_numpy_type, to_numpy_dtype #*
from .typings import as_type, get_type_name, ArrayType, class_, class_type, dtype, union, intersection, intersect, insct, avoid, tag, note, class_satisfies, type_satisfies, instance_satisfies, object, bool, short, int, long, half, float, double, rational, real, complex, number, scalar, property, slice, null, callable, functional, lambda_, lambda_func, method, function, builtin_function_or_method, method_descriptor, method_wrapper, generator_function, classmethod, staticmethod, str, bytearray, bytes, memoryview, map, filter, array, iterable, sequence, list, tuple, dict, set, reversed, frozenset, range, generator, zip, enumerate #*
from .typehint import typehint, TypeHintError, HintTypeError, get_func_info, get_arg_values, get_virtual_declaration, deprecated, params #*
from .overload import overload, OverloadError, deprecated, override #*
