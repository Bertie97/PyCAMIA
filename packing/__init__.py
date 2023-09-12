
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'packing',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.0',
    contact = 'bertiezhou@163.com',
    keywords = ['packing', 'module', 'package'],
    description = "'packing' packs module in PyCAMIA. ",
    requires = ['pycamia']
).check()

from pack import pack
from check_import import check_import
from import_all import import_all
