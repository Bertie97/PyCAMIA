#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Project: PyZMyc
## File: Package packer
##############################

import sys, os, re, pexpect
from getpass import getpass
# from pyoverload import *
# from functools import wraps
# from types import GeneratorType

key_chain = {"usr": "None", "psw": "None"}
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
def do_package(name, version=None, pack_type="zipwheel", upload=True):
    """
    Setup, pack and upload the package. 
    Inputs:
        name[str]: the name of the package, no version tag
        version[str]: the version string, None means an auto increase
        pack_type[str]: pack to zip or wheel, both are commonly used
        upload[bool]: whether to upload
    """
    # Move package to the temporary working directory
    if os.path.exists(packing_package_path): os.system(f"rm -r {packing_package_path}")
    os.mkdir(packing_package_path)
    os.system(f"cp -r {os.path.join(home_path, name)} {os.path.join(packing_package_path, name)}")

    # read the init file and get meta data
    available_info = {}
    with open(os.path.join(packing_package_path, name, "__init__.py")) as fp:
        init_code = fp.read()
        for info in re.split("###+", init_code)[1].split('\n##'):
            if ':' in info:
                k, *v = info.strip().split(':')
                available_info[k.strip().lower()] = (':'.join(v)).strip()

    # update the version tag
    old_version = available_info.get('version', None)
    if version is None:
        if old_version is None:
            version = '1.0.0'
            init_code = init_code.replace('\n###', '## Version: 1.0.0\n###', 1)
            print("version 1.0.0 created")
        else:
            version = re.sub(r"((\d+\.){2})(\d+)", lambda x: x.group(1)+str(eval(x.group(3))+1), old_version)
            init_code = init_code.replace(old_version, version, 1)
            print(f"version: {old_version} => {version}")

    # write version information back to the init file
    with open(os.path.join(home_path, name, "__init__.py"), 'w') as fp:
        fp.write(init_code)

    # create the setup.py file
    setup_file_path = os.path.join(packing_package_path, "setup.py")
    setup_file_content = "from setuptools import setup, find_packages\n\n"
    options = dict(
        name = name,
        version = version,
        keywords = ["pip", "pymyc", name] + [x.strip() for x in available_info.get('keywords', "").split(',') if x.strip()],
        description = available_info.get('description', available_info.get('introduction', "")),
        long_description = open(os.path.join(packing_package_path, name, "README.md")).read(),
        long_description_content_type="text/markdown",
        license = "MIT Licence",

        url = github_mainpage + "/" + name,
        author = available_info.get('author', "anonymous"),
        author_email = available_info.get('email', available_info.get('contact', "")),

        packages = "[CODE]find_packages()",
        include_package_data = "[CODE]True",
        platforms = "any",
        install_requires = [x.strip() for x in available_info.get('requires', "").split(',') if x.strip()]
    )
    setup_file_content += "setup(\n" + ',\n'.join([f"\t{k} = {v[6:] if isinstance(v, str) and v.startswith('[CODE]') else repr(v)}" for k, v in options.items()]) + "\n)\n"

    # pack the package and record the path
    open(setup_file_path, "w").write(setup_file_content)
    os.system(f"cd {packing_package_path}; python3 setup.py {pack_type_command[pack_type].format(dist_name = name)}")
    return f" {packing_package_path}/dist_{name}/*"

if __name__ == "__main__":
    command = "twine upload"
    if len(sys.argv) > 1: package_names = sys.argv[1:]
    else: package_names = ['all']

    # pack the packages
    if 'all' in package_names:
        if len(package_names) > 1:
            print("Warning: Indicator 'all' was found as long as ordinary package names. ")
            r = input("Shall we go on? (Y/N)")
            if 'y' not in r.lower(): exit()
        for path in os.listdir(home_path):
            cpath = os.path.join(home_path, path)
            if os.path.isdir(cpath):
                if "__init__.py" in os.listdir(cpath):
                    command += do_package(path)
    else:
        for pn in package_names:
            v = None
            if '==' in pn: pn, v = pn.split('==')
            if os.path.exists(os.path.join(home_path, pn)):
                command += do_package(pn, version=v)
            else: print(f"Warning: Package '{pn}' not found. ")

    # upload using twine
    os.system(command)
    os.system(f"rm -r {packing_package_path}")
