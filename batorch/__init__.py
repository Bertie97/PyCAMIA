
from pycamia import info_manager

__info__ = info_manager(
    project = 'PyCAMIA',
    package = 'batorch',
    author = 'Yuncheng Zhou',
    create = '2021-12',
    version = '1.0.25',
    contact = 'bertiezhou@163.com',
    keywords = ['torch', 'batch', 'batched data'],
    description = "'batorch' is an extension of package torch, for tensors with batch dimensions. ",
    requires = ['torch']
).check()

import torch
distributed = torch.distributed
autograd = torch.autograd
optim = torch.optim
utils = torch.utils
nn = torch.nn













































































































