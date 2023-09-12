
from pycamia import info_manager
__info__ = info_manager(
    project = "PyCAMIA",
    fileinfo = "File to repace the import * in projects.",
    help = "Use `python import_all.py packagename` to update the python files. "
)

import sys, os
with __info__:
    from pycamia import tokenize, item, SPrint, python_lines
home_path = os.pardir

def do_package(name):
    init_path = os.path.join(home_path, name, "__init__.py")
    with open(init_path) as fp: init_code = fp.read()
    output = SPrint()
    modified = False
    has_version = False
    version_line = ''
    for l in python_lines(init_code):
        if l.lstrip().startswith("#"): output(l); continue
        if l.lstrip().startswith("__info__"):
            raw_dict = l
            if raw_dict.endswith('.check()'): raw_dict = raw_dict.rstrip()[:-len(".check()")]
            raw_dict = raw_dict.split('=', 1)[-1].replace('info_manager', 'dict')
            version = eval(raw_dict)['version']
            output(l)
            version_line = f"__version__ = '{version}'"
            continue
        if l.lstrip().startswith("__version__"):
            has_version = True
            if l.strip() != version_line:
                modified = True
                output(l.split('__version__')[0] + version_line)
            else: output(l)
            continue
        if l.rstrip().endswith(" #*"): tail = '#*'; tag = l.split('import')[0] + 'import '
        elif l.rstrip().endswith(" #**"): tail = '#**'; tag = l.split('import')[0] + 'import '
        elif l.rstrip().endswith("import *"): tail = '#*'; tag = l.rstrip()[:-1]
        else: output(l); continue
        tokens = tag.replace(' .', f' {name}.').split()
        module = tokens[tokens.index('from') + 1].strip()
        path = module
        while '..' in path: path = path.replace('..', '_.')
        path = path.replace('.', os.path.sep).replace('_', os.path.sep + os.pardir)
        path = os.path.join(home_path, path + os.path.extsep + 'py')
        if tail == '#**':
            tmp_file_list = ['temp.py']
            with open(path) as fp_in, open(f"temp{os.path.extsep}py", 'w') as fp_out:
                for l in fp_in.read().split('\n'):
                    if l.lstrip().startswith("from ."):
                        ref_name = l.lstrip()[len("from ."):].split(maxsplit=1)[0]
                        source = os.path.join(os.path.dirname(path), ref_name + os.path.extsep + 'py')
                        if not os.path.exists(source): continue
                        # raise FileExistsError(f"Cannot find reference package .{ref_name} in {module}.")
                        l = l.replace("from .", "from temp_")
                        target = f"temp_{ref_name}{os.path.extsep}py"
                        os.system(f'cp {source} {target}')
                        tmp_file_list.append(target)
                    fp_out.write(l + '\n')
            from temp import __all__ as list_of_all
            for f in tmp_file_list: os.remove(f)
        else:
            try:
                with open(path) as fp:
                    list_of_all = item([eval('='.join(x.split('=')[1:])) for x in tokenize(fp.read(), sep='\n') if x.split('=')[0].strip() == '__all__'])
            except FileNotFoundError:
                try:
                    temp_mod = __import__(module)
                    list_of_all = [x for x in temp_mod.__all__ if not x.startswith('__') or not x.endswith('__')]
                except ModuleNotFoundError:
                    raise ImportError(f"Line '{l}' in init file of package {name} is not valid. ")
        if len(list_of_all) <= 0: new_l = tag + '*'
        else:
            contents = ', '.join(list_of_all)
            new_l = tag + contents + ' ' + tail
        if new_l != l: modified = True
        output(new_l)
    if not has_version:
        output_text = output.text
        output.clear()
        for l in python_lines(output_text):
            if l.lstrip().startswith("__info__"):
                output(l)
                output(version_line)
                modified = True
            else: output(l)
    if modified:
        output.save(init_path)

def import_all(*package_names):
    if len(package_names) == 1 and isinstance(package_names[0], (list, tuple)):
        package_names = package_names[0]
    if len(package_names) == 0: package_names = ['all']

    if 'all' in package_names:
        if len(package_names) > 1:
            print("Warning: Indicator 'all' was found as long as ordinary package names. ")
            r = input("Shall we go on for all available packages? (Y/N)")
            if 'y' not in r.lower(): exit()
        for path in os.listdir(home_path):
            cpath = os.path.join(home_path, path)
            if os.path.isdir(cpath):
                if "__init__.py" in os.listdir(cpath) and ".ignore_pack" not in os.listdir(cpath):
                    do_package(path)
    else:
        for pn in package_names:
            if '==' in pn: pn = pn.split('==')[0]
            if os.path.exists(os.path.join(home_path, pn)):
                do_package(pn)
            else: print(f"Warning: Package '{pn}' not found. ")

if __name__ == "__main__":
    import_all(sys.argv[1:])
