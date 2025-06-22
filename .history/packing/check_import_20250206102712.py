
from pycamia import info_manager
__info__ = info_manager(
    project = "PyCAMIA",
    fileinfo = "File to check whether the module imports are required in __init__ file.",
    help = "Use `python check_import.py packagename` to check the python files. "
)

import sys, os
with __info__:
    from pycamia import Path, touch, Error, Warn, enclosed_object

home_path = os.pardir
builtin_module_names = list(sys.builtin_module_names) + """
abs builtins collections copy datetime functools inspect itertools logging math
numbers operator os random re shutil string sys threading time types typing warnings
__future__
""".split()

def do_package(name):
    init_path = os.path.join(home_path, name, "__init__.py")
    with open(init_path) as fp: code = fp.read()
    s = touch(lambda: code.index("info_manager("))
    if s is None: raise Error("Pack")("Please use info_manager in `__init__.py` to identify the basic information. ")
    info_str = enclosed_object(code, start=s)
    info = info_manager.parse(info_str)

    used_modules = []
    for f in Path(home_path, name).files():
        if f | 'py':
            with f.open() as fp:
                in_special = []
                for l in fp.read().split('\n'):
                    indent = 0
                    for c in l:
                        if c in ' \t': indent += 1
                        else: break
                    l = l.strip()
                    if 'if' in l and "__name__" in l and "__main__" in l: break
                    if not l: continue
                    if in_special:
                        if indent <= in_special[-1][1] and not in_special[-1][0].startswith('str'): in_special.pop(-1); continue
                        elif l.startswith('"""') and in_special[-1][0] == 'str"': in_special.pop(-1); continue
                        elif l.startswith("'''") and in_special[-1][0] == "str'": in_special.pop(-1); continue
                    if in_special and in_special[-1][0].startswith('str'): continue
                    elif l.startswith('if') or l.startswith('elif') or l.startswith('else'):
                        in_special.append(('if', indent)); continue
                    elif l.startswith('try'):
                        in_special.append(('try', indent)); continue
                    elif l.startswith('"""'):
                        in_special.append(('str"', indent)); continue
                    elif l.startswith("'''"):
                        in_special.append(("str'", indent)); continue
                    elif l.endswith(':'):
                        in_special.append(('unknown', indent))
                    if in_special and in_special[-1][0] != 'unknown': continue
                    if l.startswith('import '):
                        for x in l[len('import '):].split(','):
                            module = x.split('as')[0].strip().split('.')[0]
                            if not module: continue
                            p = (module, f)
                            if p not in used_modules: used_modules.append(p)
                    elif l.startswith('from ') and 'import' in l:
                        module = l[len('from '):].split('import')[0].strip().split('.')[0]
                        if not module: continue
                        p = (module, f)
                        if p not in used_modules: used_modules.append(p)
    extra_modules = []
    for module, p in used_modules:
        if module in builtin_module_names: continue
        if module in info.requires: continue
        if module == name: continue
        extra_modules.append((module, p - Path._curdir))
    return dict(extra_modules)

def check_import(*package_names):
    if len(package_names) == 1 and isinstance(package_names[0], (list, tuple)):
        package_names = package_names[0]
    if len(package_names) == 0: package_names = ['all']

    res = []
    if 'all' in package_names:
        if len(package_names) > 1:
            print("Warning: Indicator 'all' was found as long as ordinary package names. ")
            r = input("Shall we go on for all available packages? (Y/N)")
            if 'y' not in r.lower(): exit()
        for name in os.listdir(home_path):
            cpath = os.path.join(home_path, name)
            if os.path.isdir(cpath):
                if "__init__.py" in os.listdir(cpath) and ".ignore_pack" not in os.listdir(cpath):
                    res.append(do_package(name))
    else:
        for pn in package_names:
            if '==' in pn: pn = pn.split('==')[0]
            if os.path.exists(os.path.join(home_path, pn)):
                res.append(do_package(pn))
            else: print(f"Warning: Package '{pn}' not found. ")
            
    return res

if __name__ == "__main__":
    plist = sys.argv[1:]
    if len(plist) == 0:
        for name in os.listdir(home_path):
            cpath = os.path.join(home_path, name)
            if os.path.isdir(cpath):
                if "__init__.py" in os.listdir(cpath) and ".ignore_pack" not in os.listdir(cpath):
                    print(name, check_import(name)[0])
    else:
        for a, b in zip(plist, check_import(plist)):
            print(a, b)
