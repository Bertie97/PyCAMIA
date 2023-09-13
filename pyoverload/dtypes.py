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

from .typings import dtype

byte_ = dtype('byte')
bool_ = dtype('bool')
int_ = dtype('int')
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
float_ = dtype('float')
float16 = half_ = dtype('float16')
float32 = single_ = dtype('float32')
float64 = double_ = inexact_ = longdouble_ = longfloat_ = number_ = dtype('float64')
bfloat16 = dtype('bfloat16')
complex_ = dtype('complex')
complex32 = chalf_ = dtype('complex32')
complex64 = cfloat_ = csingle_ = singlecomplex_ = dtype('complex64')
complex128 = cdouble_ = clongfloat_ = clongdouble_ = longcomplex_ = dtype('complex128')
timedelta64 = dtype('<m8')
