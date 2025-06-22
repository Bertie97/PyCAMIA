
required_packages = "tqdm psutil pynvml matplotlib nibabel pydicom SimpleITK sympy".split()
"""
For [ModuleNotFoundError: No module named '_ctypes'], run
yum install libffi-devel

torch torchvision torchaudio:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
"""

# import os
# status = os.system('pip3')
# if status: pip_cmd = 'pip'
# else: pip_cmd = 'pip3'
import os, sys
pip_cmd = f"{sys.executable} -m pip"
for p in required_packages:
    status = os.system(f"{pip_cmd} install {p}")
os.system("conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    