
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    fileinfo = "File to pack packages in the project.",
    help = "Use `python pack.py packagename` to pack and upload packages. ",
    requires = ['twine', 'wheel', 'pexpect', 'mintotp'] # mintotp: Package to generate TOTP 2FA security code. 
).check()

import sys, os, time, pexpect
from subprocess import PIPE, Popen

with __info__:
    from pycamia.manager import update_format
    from pycamia import touch, Warn, Error, enclosed_object, to_list, is_valid_command
    from import_all import import_all
    from check_import import check_import

usr, psw, api, *_ = open(os.path.join(os.pardir, os.pardir, "pypi.keychain")).read().split()
key_chain = {"usr": usr, "psw": psw, "api": api}
# key_chain = {"usr": "None", "psw": "None", "api": "None"} # One can use plain text here.
home_path = os.pardir
packing_package_path = os.path.join(os.curdir, "packing_package")
pack_type_command = dict(
    wheel = 'bdist_wheel --dist-dir dist_{dist_name}',
    zip = 'sdist --dist-dir dist_{dist_name}',
    zipwheel = 'sdist --dist-dir dist_{dist_name} bdist_wheel --dist-dir dist_{dist_name}'
)
if os.path.exists("github_main.url"):
    github_mainpage = open("github_main.url").read().strip()
else: github_mainpage = ""
update_code_list = []

def find_last_modification(cpath):
    ctime = os.path.getmtime(cpath)
    if not os.path.isdir(cpath): return ctime
    for f in os.listdir(cpath):
        if f.startswith('__'): continue
        if f.startswith('.'): continue
        npath = os.path.join(cpath, f)
        if not os.path.isdir(npath): ntime = os.path.getmtime(npath)
        else: ntime = find_last_modification(npath)
        if ntime > ctime:
            ctime = ntime
            
    return ctime

def do_package(name, version=None, pack_type="zipwheel", upload=True):
    """
    Setup, pack and upload the package. 
    Args:
        name (str): the name of the package, no version tag
        version (str): the version string, None means an auto increase
        pack_type (str): pack to zip or wheel, both are commonly used
        upload (bool): whether to upload
    """
    package_path = os.path.join(home_path, name)
    # Check if there is modification after last update. 
    last_mod_time = find_last_modification(package_path)
    
    # Read the __init__ file and get meta data. 
    with open(os.path.join(package_path, "__init__.py")) as fp: code = fp.read()
    pivot = "info_manager("
    s = touch(lambda: code.index(pivot))
    if s is None: raise Error("Pack")("Please use info_manager in `__init__.py` to identify the basic information. ")
    info_str = enclosed_object(code, start=s)
    info = info_manager.parse(info_str)
    last_version = info.get('version', '1.0.0')
    # Check if there is modification after last update. 
    if hasattr(info, 'update') and last_mod_time < time.mktime(time.strptime(info.update, update_format)) + 10: return ""
    # Upload only if the package is modified. 
    info.version_update()
    code = code.replace(info_str, str(info))
    code = code.replace(f"__version__ = '{last_version}'", f"__version__ = '{info['version']}'")
    update_code_list.append((os.path.join(package_path, "__init__.py"), code))
    import_all(name)
    missing_require = check_import(name)
    if missing_require: print(f"Warning: package {name} imported {missing_require} but didnot declare requirement. ")

    # Move package to the temporary working directory. 
    os.system(f"cp -r {package_path} {os.path.join(packing_package_path, name)}")
    os.system(f"cp {os.path.join(package_path, 'README.md')} {os.path.join(packing_package_path, 'README.md')}")
    os.system(f"cp {os.path.join(package_path, 'MANIFEST.in')} {os.path.join(packing_package_path, 'MANIFEST.in')}")

    # create the setup.py file
    setup_file_path = os.path.join(packing_package_path, f"setup_{name}.py")
    setup_file_content = "from setuptools import setup, find_packages\n\n"
    options = dict(
        name = name,
        version = version if version else info.get('version', '1.0.0'),
        keywords = ["pip", "pymyc", name] + [x.strip() for x in to_list(info.get('keywords', "")) if x.strip()],
        description = info.get('description', info.get('introduction', "")),
        long_description = touch(lambda: open(os.path.join(packing_package_path, name, "README.md")).read(), ""),
        long_description_content_type="text/markdown",
        license = "MIT Licence",

        url = github_mainpage + "/" + name,
        author = info.get('author', "anonymous"),
        author_email = info.get('email', info.get('contact', "")),

        packages = "[CODE]find_packages()",
        include_package_data = f"[CODE]{info.get('package_data', False)}",
        platforms = "any",
        install_requires = [x.strip() for x in to_list(info.get('requires', "")) if x.strip()]
    )
    setup_file_content += "setup(\n" + ',\n'.join([f"\t{k} = {v[6:] if isinstance(v, str) and v.startswith('[CODE]') else repr(v)}" for k, v in options.items()]) + "\n)\n"

    # pack the package and record the path
    open(setup_file_path, "w").write(setup_file_content)
    os.system(f"cd {packing_package_path}; python3 setup_{name}.py {pack_type_command[pack_type].format(dist_name = name)}")
        
    return f" {packing_package_path}/dist_{name}/*"

def pack(*package_names):
    if len(package_names) == 1 and isinstance(package_names[0], (list, tuple)):
        package_names = package_names[0]
    if len(package_names) == 0: package_names = ['all']

    # Pack the packages. 
    if is_valid_command('twine'): command = "twine upload"
    elif is_valid_command('python3') and is_valid_command('python3 -m twine', "No module named"): command = "python3 -m twine upload"
    elif is_valid_command('python2') and is_valid_command('python2 -m twine', "No module named"): command = "python2 -m twine upload"
    elif is_valid_command('python') and is_valid_command('python -m twine', "No module named"): command = "python -m twine upload"
    else: raise TypeError("Cannot find installed command 'twine'. ")
    if os.path.exists(packing_package_path): os.system(f"rm -r {packing_package_path}")
    os.mkdir(packing_package_path)
    if 'all' in package_names:
        if len(package_names) > 1:
            print("Warning: Indicator 'all' was found as long as ordinary package names. ")
            r = input("Shall we go on for all available packages? (Y/N)")
            if 'y' not in r.lower(): exit()
        for path in os.listdir(home_path):
            cpath = os.path.join(home_path, path)
            if os.path.isdir(cpath):
                if "__init__.py" in os.listdir(cpath) and ".ignore_pack" not in os.listdir(cpath):
                    command += do_package(path)
    else:
        for pn in package_names:
            v = None
            if '==' in pn: pn, v = pn.split('==')
            if os.path.exists(os.path.join(home_path, pn)):
                command += do_package(pn, version=v)
            else: print(f"Warning: Package '{pn}' not found. ")

    # Test command. 
    # print(command)
    # return
    # Upload using twine. 
    if command.strip().endswith('upload'): print("All mentioned packages are up to date. "); return
    p = pexpect.spawn(command)
    # new version (starts from 2024)
    p.expect("Enter your API token:")
    p.sendline(key_chain['api'])
    # p.expect("Enter your username:")
    # p.sendline(key_chain['usr'])
    # p.expect("Enter your password:")
    # p.sendline(key_chain['psw'])
    p.expect(pexpect.EOF)
    if b"403 Forbidden" in p.before: print(f"Upload Failed... {p.before.decode('utf-8')}"); return
    if b"400 Bad Request" in p.before: print(f"Upload Failed (A copy of given version found on the server, please correct the version manually)... {p.before.decode('utf-8')}"); return
    else:
        names = []
        for f, c in update_code_list:
            with open(f, 'w') as fp: fp.write(c)
            names.append(os.path.basename(os.path.dirname(f)))
        print("Uploaded Successfully:", ', '.join(names));
    os.system(f"rm -r {packing_package_path}")

if __name__ == "__main__":
    pack(sys.argv[1:])