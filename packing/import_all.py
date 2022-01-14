
from pycamia import info_manager
__info__ = info_manager(
    project = "PyCAMIA",
    fileinfo = "File to repace the import * in projects.",
    help = "Use `python import_all.py packagename` to update the python files. "
)

import sys, os, re

from pycamia import tokenize, item, SPrint

home_path = os.pardir

def do_package(name):
    init_path = os.path.join(home_path, name, "__init__.py")
    with open(init_path) as fp: init_code = fp.read()
    output = SPrint()
    for l in init_code.split('\n'):
        if l.rstrip().endswith(" #*"):
            tag = l.split('import')[0] + 'import '
        elif l.rstrip().endswith("import *"):
            tag = l.rstrip()[:-1]
        else: output(l); continue
        path = tag.replace(' .', f' {name}.').split()[1].strip()
        while '..' in path: path = path.replace('..', '_.')
        path = path.replace('.', os.path.sep).replace('_', os.path.sep + os.pardir)
        path = os.path.join(home_path, path + os.path.extsep + 'py')
        with open(path) as fp:
            list_of_all = item([eval('='.join(x.split('=')[1:])) for x in tokenize(fp.read(), sep='\n') if x.split('=')[0].strip() == '__all__'])
        contents = ', '.join(list_of_all)
        output(tag + contents + ' #*')
    output.save(init_path)

def import_all(plist):
    if len(plist) > 1: package_names = plist[1:]
    else: package_names = ['all']

    if 'all' in package_names:
        if len(package_names) > 1:
            print("Warning: Indicator 'all' was found as long as ordinary package names. ")
            r = input("Shall we go on? (Y/N)")
            if 'y' not in r.lower(): exit()
        for path in os.listdir(home_path):
            cpath = os.path.join(home_path, path)
            if os.path.isdir(cpath):
                if "__init__.py" in os.listdir(cpath):
                    do_package(path)
    else:
        for pn in package_names:
            if '==' in pn: pn = pn.split('==')[0]
            if os.path.exists(os.path.join(home_path, pn)):
                do_package(pn)
            else: print(f"Warning: Package '{pn}' not found. ")

if __name__ == "__main__":
    import_all(sys.argv)
