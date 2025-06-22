
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Unit test for overload @params.",
    requires = ["pyoverload", "unittest"]
)

import unittest
with __info__:
    from pycamia import scope, jump, touch, get_declaration
    # from pyoverload.old_version_files_deprecated.typehint import params
    # from pyoverload.old_version_files_deprecated.typehint import *

class AtParamsTests(unittest.TestCase):

    def test_param_speed(self):
        with scope("create func_a"):
            def func_a(a: int, b: str, c: callable) -> str:
                return repr(c(a, len(b)))
        with scope("create func_b"):
            @params
            def func_b(a: int, b: str, c: callable) -> str:
                return repr(c(a, len(b)))
        with scope("Test param filtering", log_on_screen=False):
            with scope("func_a"):
                print(touch(lambda:func_a(10., "abcd", max), print_error_message=True))
            with scope("func_b"):
                print(touch(lambda:func_b(10., "abcd", max), print_error_message=True))
        with scope("Test param time"):
            with scope("func_a"):
                x = ''
                for _ in range(int(1e5)):
                    x = func_a(10, x, max)
            with scope("func_b"):
                x = ''
                for _ in range(int(1e5)):
                    x = func_b(10, x, max)
                
    def test_leading_to_new_typehint(self):
        from functools import wraps
        def decr(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        # @decr
        def func(a: int, b, /, c: callable, d:int =2, *e: int, bad=3, **f: int) -> str:
            x = 1
            return repr(c(a, len(b)))
        
        self.assertEqual(
            str(func.__annotations__), 
            "{'c': <built-in function callable>, 'd': <class 'int'>, 'a': <class 'int'>, 'e': <class 'int'>, 'f': <class 'int'>, 'return': <class 'str'>}"
        )
        self.assertEqual(func.__defaults__,                     (2,))
        self.assertEqual(str(func.__kwdefaults__),              "{'bad': 3}")
        self.assertEqual(func.__code__.co_argcount,             4)
        self.assertEqual(func.__code__.co_kwonlyargcount,       1)
        self.assertEqual(func.__code__.co_posonlyargcount,      2)
        self.assertEqual(func.__code__.co_name,                 "func")
        self.assertEqual(func.__code__.co_names,                ('repr', 'len'))
        self.assertEqual(func.__code__.co_varnames,             ('a', 'b', 'c', 'd', 'bad', 'e', 'f', 'x'))
        self.assertEqual(func.__code__.co_nlocals,              8)
        self.assertEqual((func.__code__.co_flags & 0x08),       8)
        self.assertEqual((func.__code__.co_flags & 0x04),       4)
        self.assertEqual(bool(func.__code__.co_flags & 0x02),   True)
        self.assertEqual(bool(func.__code__.co_flags & 0x01),   True)
        annotations = func.__annotations__
        # func_out = func
        def func_out(*args, **kwargs):
            co_nonextargcount = func.__code__.co_argcount + func.__code__.co_kwonlyargcount
            for i, var in enumerate(func.__code__.co_varnames):
                if var not in annotations: continue
                annotation = annotations[var]
                if i < func.__code__.co_argcount:
                    if i < len(args): value = args[i]
                    elif i >= func.__code__.co_posonlyargcount and var in kwargs: value = kwargs[var]
                    else:
                        index = i - func.__code__.co_argcount + len(func.__defaults__)
                        if index >= len(func.__defaults__):
                            missing_args = func.__code__.co_varnames[len(args):func.__code__.co_argcount - len(func.__defaults__)]
                            n_missing = len(missing_args)
                            missing_args = ' and '.join([', '.join([repr(x) for x in missing_args[:-1]]), missing_args[-1]])
                            raise TypeError(f"{func.__name__}() missing {n_missing} required position argument{'s' if n_missing > 1 else ''}: {missing_args}")
                        value = func.__defaults__[index]
                elif i < co_nonextargcount:
                    if var in kwargs: value = kwargs[var]
                    elif var in func.__kwdefaults__: value = func.__kwdefaults__[var]
                    else: 
                        missing_args = [v for v in func.__code__.co_varnames[func.__code__.co_argcount:func.__code__.co_argcount + func.__code__.co_kwonlyargcount] if v not in func.__kwdefaults__]
                        n_missing = len(missing_args)
                        missing_args = ' and '.join([', '.join([repr(x) for x in missing_args[:-1]]), missing_args[-1]])
                        raise TypeError(f"{func.__name__}() missing {n_missing} required keyword-only argument{'s' if n_missing > 1 else ''}: {missing_args}")
                elif i == co_nonextargcount:
                    if len(args) > func.__code__.co_argcount:
                        value = args[func.__code__.co_argcount:]
                    else: value = tuple()
                elif i == co_nonextargcount + 1:
                    remaining_kwargs = {k: v for k, v in kwargs.items() if k not in func.__code__.co_varnames[func.__code__.co_posonlyargcount:co_nonextargcount]}
                    value = remaining_kwargs
                else:
                    with open(func.__code__.co_filename) as fp:
                        for _ in range(func.__code__.co_firstlineno - 1): fp.readline()
                        while True:
                            l = fp.readline()
                            if l.strip().startswith('def'): break
                        if func.__name__ not in l:
                            raise RuntimeError(f"Unfound function {func.__name__}() with invalid annotations: {func.__annotations__}.")
                    l = l.strip()[4:].rstrip(':')
                    raise RuntimeError(f"Function {l} with unexpected annotations: {func.__annotations__}.")
                print(var, value, annotation)
            ret_value = func(*args, **kwargs)
            print('return', ret_value, annotations.get('return', None))
            return ret_value
        func_out(10., "abcd", c=max, d='', this=3, that=5)
        ans_sheet = dict(
            a = (10., int),
            c = (max, callable),
            d = ('', int),
            e = (tuple(), int),
            f = ({'this': 3, 'that': 5}, int),
            # return = (10., str)
        )
        func_out(10., "abcd", max, 4, 4, 5, this=3, that=5)
        ans_sheet = dict(
            a = (10., int),
            c = (max, callable),
            d = (4, int),
            e = ((4, 5), int),
            f = ({'this': 3, 'that': 5}, int),
            # return = (10., str)
        )

if __name__ == "__main__":
    unittest.main()

