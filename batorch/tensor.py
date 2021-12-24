
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch.",
    requires = "torch"
)

__all__ = """
    Device
    DeviceCPU
    Tensor
    Size
    set_autodevice
    unset_autodevice
    is_autodevice
""".split()

if __name__ == '__main__':
    pass
