
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "pyoverload",
    author = "Yuncheng Zhou",
    create = "2021-12",
    version = "1.1.0",
    contact = "bertiezhou@163.com",
    keywords = ["overload"],
    description = "'pyoverload' overloads the functions by simply using typehints and adding decorator '@overload'. "
).check()

from .override import *
