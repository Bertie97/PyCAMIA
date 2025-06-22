
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch.",
    requires = ["pycamia", "torch", "pynvml"]
)

__all__ = """
    CPU
    AutoDevice
    SleepingDevice
    free_memory_amount
    all_memory_amount
""".split()

import os, torch
with __info__:
    import pynvml, psutil
    from pycamia import alias, ByteSize

GPU_priority = ['cuda']#, 'mps']
is_available = dict(
    cuda = hasattr(torch, 'cuda') and torch.cuda.is_available(),
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
)
GB = 1.074e+9
warning_free_memory_threshold = eval(os.environ.get('CUDA_RUN_MEMORY', '5')) # Run with at lease *GB

free_memory_amount = None
all_memory_amount = None

@alias("free_memory_amount", amount="free")
@alias("all_memory_amount", amount="total")
def memory_amount(device_id, amount="total"):
    if not isinstance(device_id, str):
        if device_id == -1: device_id = 'cpu'
        else: device_id = f'cuda:{device_id}'
    if device_id == 'cpu':
        info = psutil.virtual_memory()
    elif device_id == 'mps':
        return dict(free=0, total=0)[amount]
    elif device_id.startswith('cuda'):
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(device_id.split(':')[-1]))
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return getattr(info, amount)

def device_by_id(device_id):
    return torch.device(device_id)

class AutoDevice:
    """
    Auto find and set a device. 
    
    Args:
        fixed_device (torch.device): use a fixed device. 
        auto (bool): decide whether the device is auto selected. 
        required_memory (int): an estimated memory use which helps the selection, in GB. 
        verbose (bool): whether to print warnings or not.

    Examples::
        >>> gpu = AutoDevice()
        >>> tensor.to(gpu.device)
        <tensor>
        >>> gpu.auto = False
        >>> gpu(tensor)
        <tensor>
    """
    def __init__(self, *devices, required_devices=1, auto=True, required_memory=None, verbose=True, always_proceed=False):
        if len(devices) == 0: devices = ['cpu']
        self.required_devices = required_devices
        self.required_memory = required_memory
        self.always_proceed = always_proceed
        self.verbose = verbose
        self.auto = auto
        self.available_gpu_ids = []
        self._working_devices = None
        self.default_devices = devices

    @property
    def working_devices(self):
        if self.auto: self.auto_select()
        elif self._working_devices is None:
            self._working_devices = [f"cuda:{d}" if isinstance(d, int) else str(d) for d in self.default_devices]
        return self._working_devices

    @alias("device_id")
    @property
    def main_device_id(self): return self.working_devices[0]
        
    @alias("device")
    @property
    def main_device(self): return device_by_id(self.main_device_id)

    @property
    def working_memory(self):
        return ByteSize(free_memory_amount(self.main_device_id))

    def auto_select(self):
        self.auto = False
        for gpu_family in GPU_priority:
            if gpu_family == 'cuda' and is_available['cuda']:
                self.available_gpu_ids = list(range(torch.cuda.device_count()))
                available_gpus_memory = [free_memory_amount(i) for i in self.available_gpu_ids]

                threshold = warning_free_memory_threshold if self.required_memory is None else self.required_memory / self.required_devices / 2
                sorted_memory = sorted(enumerate(available_gpus_memory), key=lambda x: -x[1])
                candidates = [(i, m) for i, m in sorted_memory if m > threshold * GB][:self.required_devices]
                if len(candidates) > 0: most_available_gpus, most_available_gpu_memory = zip(*candidates)
                else: 
                    print(f"Warning: no gpu device is available (with memories {available_gpus_memory}), reset environment variable CUDA_RUN_MEMORY to change the memory required...")
                    print(f"Most memory: {sorted_memory[0][1] / GB:.5} GB in GPU {sorted_memory[0][0]}. ")
                    if not self.always_proceed:
                        tag = input("Do you want to proceed with CPU? [yes/no/y/n]:")
                        if 'y' not in tag.lower(): raise RuntimeError("Not enough GPU resource.")
                    continue
                if self.required_memory and sum(most_available_gpu_memory) < self.required_memory and self.verbose:
                    print(f"Warning: all remaining memory of gpu devices is not enough (on: {most_available_gpus})...")
                    print(f"Total free memory: {sum(most_available_gpu_memory) / GB:.5} GB. ")
                    if not self.always_proceed:
                        tag = input("Do you want to proceed? [yes/no/y/n]:")
                        if 'y' not in tag.lower(): raise RuntimeError("Not enough free memory left.")
                elif len(most_available_gpus) < self.required_devices and self.verbose:
                    print(f"Warning: not enough gpus available (available: {most_available_gpus})...")
                    print(f"Total free memory: {sum(most_available_gpu_memory) / GB:.5} GB. ")
                    if not self.always_proceed:
                        tag = input("Do you want to proceed? [yes/no/y/n]:")
                        if 'y' not in tag.lower(): raise RuntimeError("Not enough free devices left.")
                if self.verbose: print(f"Setting working devices: {most_available_gpus}, main device: {most_available_gpus[0]}. proceeding...")
                self._working_devices = [f"cuda:{i}" for i in most_available_gpus]
                break
            elif gpu_family == 'mps' and is_available['mps']:
                self._working_devices = ['mps']
                self.required_devices = 1
                break
        else:
            if self.verbose: print("Warning: cannot find any gpu, using cpu instead. ")
            self._working_devices = ['cpu']
            self.required_devices = 1
    
    def turn_on(self):
        if not self.auto:
            self.auto = True
            self.auto_select()
        return self
    
    def turn_off(self):
        if self.auto: self.auto = False
        return self
    
    def __eq__(self, other):
        if isinstance(other, torch.device): return self.main_device == other
        if isinstance(other, AutoDevice): return self.main_device == other.main_device
        return self == other
    
    def __call__(self, x):
        mdevice = self.main_device
        if isinstance(x, torch.Tensor):
            if x.device == mdevice: return x
            return x.to(mdevice)
        elif isinstance(x, torch.nn.Module):
            if len(self.working_devices) <= 1: return x.to(mdevice)
            return torch.nn.DataParallel(x.to(mdevice), device_ids=[int(x.split(':')[-1]) for x in self.working_devices])
        elif isinstance(x, torch.optim.Optimizer):
            if len(self.working_devices) <= 1: return x
            return torch.nn.DataParallel(x, device_ids=[int(x.split(':')[-1]) for x in self.working_devices])
        else: return torch.tensor(x, device=mdevice)
        return x
    
class SleepingDevice(AutoDevice):
    def __init__(self, *devices, required_devices=1, auto=False, required_memory=None, verbose=False):
        super().__init__(*devices, required_devices=required_devices, auto=auto, required_memory=required_memory, verbose=verbose) 

CPU = SleepingDevice(torch.device("cpu"))
if is_available['mps']:
    __all__.append('MPS')
    MPS = SleepingDevice(torch.device("mps"))
if is_available['cuda']:
    __all__.extend(['GPU', 'GPUs'])
    GPU = SleepingDevice(torch.device("cuda:0"))
    GPUs = [SleepingDevice(torch.device(f"cuda:{i}")) for i in range(torch.cuda.device_count())]
