
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2021-12",
    version = "1.0.0",
    contact = "bertiezhou@163.com",
    keywords = ["environment", "path", "touch"],
    description = "The main package and a background support of project PyCAMIA. ",
    requires = []
).check()

from .environment import get_environ_vars
from .exception import *
from .functions import *
from .inout import *
from .manager import *
from .timing import *
from .listop import *
from .strop import *
