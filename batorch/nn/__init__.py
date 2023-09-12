
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "batorch version of `torch.nn`",
    requires = "torch"
)

from .modules import *
from .parameter import Parameter
from . import parameter
from . import functional
from . import utils
from torch.nn import DataParallel
