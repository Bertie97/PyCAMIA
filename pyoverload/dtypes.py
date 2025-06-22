try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "pyoverload",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "list all the dtypes. ",
    requires = ""
)

__all__ = """
    to_dtype
    to_rawdtype
    to_torch_type
    to_torch_dtype
    to_numpy_type
    to_numpy_dtype
""".split()

import re
from .typings import dtype

char_ = dtype('char')
byte_ = dtype('byte')
bool8 = bool_ = dtype('bool')
int_ = integer_ = dtype('int')
int8 = dtype('int8')
int16 = short_ = dtype('int16')
int32 = intc_ = dtype('int32')
int64 = intp_ = long_ = longlong_ = signedinteger_ = dtype('int64')
uint8 = ubyte_ = dtype('uint8')
uint16 = ushort_ = dtype('uint16')
uint32 = uintc_ = dtype('uint32')
uint64 = uint_ = uintp_ = ulonglong_ = unsignedinteger_ = dtype('uint64')
qint8 = dtype('qint8')
qint32 = dtype('qint32')
quint8 = dtype('quint8')
quint2x4 = dtype('quint2x4')
quint4x2 = dtype('quint4x2')
float_ = floating_ = dtype('float')
float16 = half_ = dtype('float16')
float32 = single_ = dtype('float32')
float64 = double_ = inexact_ = longdouble_ = longfloat_ = number_ = dtype('float64')
bfloat16 = dtype('bfloat16')
complex_ = dtype('complex')
complex32 = complexhalf_ = chalf_ = dtype('complex32')
complex64 = complexfloat_ = complexfloating_ = cfloat_ = csingle_ = singlecomplex_ = dtype('complex64')
complex128 = complexdouble_ = cdouble_ = clongfloat_ = clongdouble_ = longcomplex_ = dtype('complex128')
timedelta64 = dtype('<m8')

################################################################################
## The following chunk of code was used to generate the dtype names listed below.
################################################################################
## import torch
## import numpy
##
## torch_dtype = []
## torch_type = []
## numpy_type = []
##
## for n in dir(torch):
## 	   x = getattr(torch, n)
## 	   if isinstance(x, torch.dtype): torch_dtype.append(n)
## 	   if isinstance(x, type):
## 	   	   try:
## 			   torch.zeros(1).type(x)
## 			   torch_type.append(n)
## 		   except Exception: ...
##
## for n in dir(numpy):
## 	   x = getattr(numpy, n)
## 	   if isinstance(x, type) and issubclass(x, numpy.number): numpy_type.append(n)
################################################################################

def to_dtype(t):
    if t.__class__ == dtype: return t
    t_name = getattr(t, '__name__', str(t).split('.')[-1].split("'")[0])
    if t_name.endswith('Tensor'): t_name = t_name[:-len('Tensor')].lower()
    return globals().get(t_name, globals().get(t_name+'_', dtype(t)))

def to_rawdtype(t):
    t = to_dtype(t)
    t.module = None
    return t

torch_dtype = ['bfloat16', 'bool', 'cdouble', 'cfloat', 'chalf', 'complex128', 'complex32', 'complex64', 'double', 'float', 'float16', 'float32', 'float64', 'half', 'int', 'int16', 'int32', 'int64', 'int8', 'long', 'qint32', 'qint8', 'quint2x4', 'quint4x2', 'quint8', 'short', 'uint8']
torch_type = ['BFloat16Tensor', 'BoolTensor', 'ByteTensor', 'CharTensor', 'DoubleTensor', 'FloatTensor', 'HalfTensor', 'IntTensor', 'LongTensor', 'ShortTensor', 'Tensor', 'torch.ComplexHalfTensor', 'torch.ComplexFloatTensor', 'torch.ComplexDoubleTensor']
numpy_type = ['bool', 'bool8', 'bool_', 'byte', 'cdouble', 'cfloat', 'clongdouble', 'clongfloat', 'complex128', 'complex64', 'complex_', 'complexfloating', 'csingle', 'double', 'float16', 'float32', 'float64', 'float_', 'floating', 'half', 'inexact', 'int16', 'int32', 'int64', 'int8', 'int_', 'intc', 'integer', 'intp', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'number', 'short', 'signedinteger', 'single', 'singlecomplex', 'timedelta64', 'ubyte', 'uint', 'uint16', 'uint32', 'uint64', 'uint8', 'uintc', 'uintp', 'ulonglong', 'unsignedinteger', 'ushort']
torch_dtype_name = {to_rawdtype(n): n for n in torch_dtype}
torch_type_name = {to_rawdtype(n): n for n in torch_type}
numpy_type_name = {to_rawdtype(n): n for n in numpy_type}

to_torch_dtype_special = {'byte': 'uint8', 'char': 'int8', '<m8': 'int64'}

def to_torch_dtype(t):
    import torch
    t = to_dtype(t)
    t.module = None
    full_name = f"{t.name}{t.bits}"
    if full_name in to_torch_dtype_special:
        t.name, t.bits = re.findall(r"([<a-zA-Z]+)([x0-9]*)", to_torch_dtype_special[full_name])[0]
    if not full_name: return torch.float32
    if t.name.startswith('u'): t.name = t.name[1:]
    type_name = torch_dtype_name.get(t, None)
    if type_name is None: raise TypeError(f"Unrecognized dtype {t} for torch dtypes.")
    return getattr(torch, type_name)

def to_torch_type(t):
    import torch
    t = to_dtype(t)
    t.module = None
    if t.name.startswith('q'): t.name = t.name[1:]
    if t.name.startswith('u'): t.name = t.name[1:]
    if t.name == '<m': t.name = 'int'; t.bits = 64
    if t.bits in ('2x4', '4x2'): t.bits = 8
    type_name = torch_type_name.get(t, torch_type_name.get(dtype(t.name), None))
    if type_name is None: raise TypeError(f"Unrecognized dtype {t} for torch types.")
    return getattr(torch, type_name, type_name)

to_numpy_dtype_special = {'bfloat16': 'float16', 'complex32': 'complex64', '': 'float32'}

def to_numpy_dtype(t):
    import numpy
    t = to_dtype(t)
    full_name = f"{t.name}{t.bits}"
    return numpy.dtype(to_numpy_dtype_special.get(full_name, full_name))

to_numpy_special = {'bfloat16': 'float16', 'complex32': 'complex64', 'bool': 'uint8', 'char': 'int8'}

def to_numpy_type(t):
    import numpy
    t = to_dtype(t)
    t.module = None
    if t.name.startswith('q'): t.name = t.name[1:]
    full_name = f"{t.name}{t.bits}"
    if full_name in to_numpy_special:
        t.name, t.bits = re.findall(r"([<a-zA-Z]+)([x0-9]*)", to_numpy_special[full_name])[0]
    if not full_name: return numpy.float32
    if t.bits and isinstance(t.bits, str): t.bits = eval(t.bits.replace('x', '*'))
    type_name = numpy_type_name.get(t, None)
    if type_name is None: raise TypeError(f"Unrecognized dtype {t} for numpy types.")
    return getattr(numpy, type_name)
