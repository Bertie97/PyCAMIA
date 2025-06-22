# pycamia

[TOC]

#### Introduction

[`pycamia`](https://github.com/Bertie97/pycamia/tree/main/pycamia) is the base package affiliated to project [`PyCAMIA`](https://github.com/Bertie97/pycamia). It is a collection of different useful tools necessary in Python programming. `pycamia` was designed for `Python v3.6+`. The most important functions (and most frequently used) of this package include:

1. [`@alias`](#package `decorators`) decorator to create alias names for an object. 
2. [`get_environ_vars()`](#package `environment`) to get the variables from the parent scope of the current function.
3. [`avouch(bool, assertion_text_or_raised_error)`](#package `exception`) to avouch (i.e. assert) with a self-designed error message (or the input expression by default).
4. [`touch(expression_or_function, *args, default=None, ...)`](#package `exception`) to touch a dangerous function (or a string of code), return the value if no error has occurred but a default value if errors are raised.
5. [`print=Sprint()`](#package `inout`) to "print" to  a string, instead of the console.
6. [`with logging(path)`](#package `logging`) to save to log files by simply using `'print'`.
7. [`collector = Collector()`](#package `loopop`) to collect metrics in each iteration as numpy arrays. 
8. [`with TaskProgress() as iterate`](#package `loopop`) to create progress bar for each line of outputs.
9. More to come...

In detail, the package consists of the following sub-packages. 

1. **`decorators`** is the package containing the decorators: e.g. use @`alias` to call a function with multiple names; use `@restore_type_wrapper` to cast the return value as the first input argument.
2. **`environment`** is the package containing functions to inspect the context: e.g. use `get_environ_vars()` in a function to get the variables in the scope calling the function. 
3. **`exception`** is the package to handle exceptions: e.g. use `touch("a[i]")` to try a function (or code in str) and suppress the error; use `avouch(assertion, "error message")` to assert with comments; use `Error("DIY")` to create an Error.
4. **`functions`** is the package of special (and commonly trivial) functions: e.g. use `empty_function` for an empty function. 
5. **`inout`** is the package to extend the input/output: e.g. use `sprint=SPrint()` to print into a string; use `with no_print:` to suppress the console output. 
6. **`listop`** is the collection of advanced functions for lists: e.g. use `prod` to find the product of a list; use `flatten_list` to flatten a nested list.
7. **`logging`** is the package for logs: e.g. use `with logging(file_path):` to log all the printed contents (using `print`) to the log file; use `start_logging(file_path)` to serve this purpose for the rest of the program.
8. **`loopop`** is the package giving support to loops: e.g. use `Collector` to register the output of each iteration and retrieve the time sequence at the end of the loop; use `TaskProgress` (or `iterate`) to create a progress bar for the output.
9. **manager** is the package to manage file and package info: e.g. use `with __info__:` to check the dependencies and have better error message; use `info_manager` to organize the file information; use `Version` to parse the version strings and compare. 
10. **`strop`** is the collection of advanced functions for strings: e.g. use `tokenize` to tokenize a string by splitting at places not nested with brackets; use `find_all` to find all indices for matched sub-strings.
11. **`structures`** is the collection of extended data structures for python: e.g. use `struct(a=1, b=2)` to create a structure that has the properties `'a'` and `'b'`.
12. **`system`** is the package that communicates with the operating system: use `Path.curpath/"folder"/"filename.xx"` to create the path object which can be used as a string but is easy to modify; use `copy` to copy a file from a destination to another.
13. **timing** is the package to time the executions: e.g. use `with scope(scope_name):` to record time spent for a set of commands. 
14. **`math`** is not officially open access so far. It is the package of mathematic objects.
15. **`more`** is a collection of uncategorized functions, one needs to import them directly from `pycamia.more`.
16. **`numpyop`** is the package for numpy operations, which currently consists of functions changing dtype into signed/unsigned formats, **this package is currently deprecated**.

#### Installation

This package can be installed by `pip install pycamia` or moving the source code to the directory of python libraries (the source code can be downloaded on [github](https://github.com/Bertie97/pycamia) or [PyPI](https://pypi.org/project/pyoverload/)). 

```shell
pip install pycamia
```

#### Usage

##### package `decorators`

This package contains the useful decorators which is expected to extend in the future.

1. Use `@alias` to create alias names, e.g.

    ```python
    ... @other_wrappers # wrappers for function `func_b` only. 
    ... @alias("func_a", b=1) # wrappers in between are for functions `func_a` and `func_b`.
    ... @alias("func_c", b=2)
    ... @some_wrappers # wrappers for functions `func_a`, `func_b` and `func_c`.
    ... def func_b(x, b):
    ...     print(x+b)
    ...
    >>> func_a(1), func_b(2, 4), func_c(7)
    (2, 6, 9)
    ```

    or one can also use it for non-function objects,

    ```python
    >>> a = alias('b')(10)
    >>> a, b
    (10, 10)
    >>> @alias("B", "C")
    ... class A:
    ...     @alias('a')
    ...     @property
    ...     def name(self): return 1
    ...
    >>> B().a, A().name
    (1, 1)
    ```

    Note that the `@alias` won't change the `'__name__'` property.

2. Use `@restore_type_wrapper` to cast the return value back to the first input argument. 

##### package `environment`

This package fetches the surrounding environment of the call. i.e. 

```python
a.py:
    from b import a_func
    def some_func(*args):
        here_is_the_var_needed = []
        a_func()
        print(here_is_the_var_needed)

b.py:
    from pycamia import get_environ_vars
    def a_func():
        vars = get_environ_vars().locals
        cell_var = vars['here_is_the_var_needed']
        cell_var.append(sth)
        vars['here_is_the_var_needed'] = cell_var
```

1. Use `v = get_environ_vars()` to get an `EnvironVars` object which can provide access to the variables in the surrounding environment by dictionary operations.

2. Use `v = get_environ_vars().locals` and `v = get_environ_locals()` (or `v = get_environ_vars().globals` and `v = get_environ_globals()` for global variables) to get the dictionary of local or global variables in the parent environment. If the result is out of expectations, please contact the developer. 

3. Use `update_locals_by_environ()` in a scope to copy the environmental variables to the local scope. One can then directly use the variables. However, please note that this operation may cause overriding of previously defined scope variables (including function arguments). 

4. Use `get_declaration(functional)` to get the declaration line of `functional`. One is encouraged to use `get_virtual_declaration` in package `pyoverload` (a package installed by `pip install pyoverload`, which is in the same project) to get the declaration reorganized by function properties (this would be faster as it reads from memory). 

5. Use `get_args_expression()` in a function to get the input arguments when this function is called. Note that this can obtain the expression of the input argument instead of only the variable name and the value, as follows,

    ```python
    >>> def func(x):
    ...     print(get_args_expression())
    ...
    >>> func(x for x in range(100) if x % 2 == 1)
    x for x in range(100) if x % 2 == 1
    ```

##### package `exception`

This package handles exceptions with `touch` and an alias `avouch` for `assert`. 

1. Use `touch(function_or_str)` to try a function and suppress the error in the meantime. e.g. `touch(lambda: 1/a)` returns `None` to tell you that an exception occurs when `a=0`, but returns `1` when `a=1`. 

    One can use `default=...` to identify the value return when an error occurs and `error_type=TypeError` to only catch the `TypeError`s. What's more, one can use a string of expressions as the touch subject, or a function with input arguments directly sent to touch.

2. Use `crashed(function) -> bool` to check whether a function has failed or not. 

3. Use `avouch(bool_to_be_tested, assertion_text_or_error)` to avouch that the given expression is true and output your designed `assertion_text` when the test fails. e.g.

    ```python
    untitled.py:
        from pycamia import avouch
    	avouch(isinstance(1, str))
    Traceback (most recent call last):
        [...omitted...]
    AssertionError: Failure in assertion 'isinstance(1, str)'
    
    >>> from pycamia import avouch
    >>> avouch(isinstance(1, str), TypeError("1 is not a string"))
    Traceback (most recent call last):
        [...omitted...]
    TypeError: 1 is not a string
    ```

    Note that the auto-generated assertion message may be `'<unreachable arg expression>'` in the Python console due to the file-reading logic of `get_arg_expression`, please contact the developer if you have a better idea of getting the expression and are willing to share. 

4. Use `Error("name")` to create a new error type. It is equivalent to creating an Error tag by `class {name}Error(Exception): pass`.

##### package `functions`

This package contains simple functions which is the simplest package in the project so far. 

1. Use `empty_function` for a function that does nothing and returns `None`. One can put any argument to the function but nothing would happen. 
2. Use `const_function(a)` for a function that accepts any argument but does nothing and always returns `a`.
3. Use `identity_function(*x)` for a function that returns exactly the input (though multiple inputs will be returned as a tuple). 

##### package `inout`

This package manipulates the input/output. Currently, it only deals with print. Shell handler or other in & out functions will be added here in the future. 

1. Use `"with no_out:"` to suppress the console output. Although not recommended, one can use `with no_out as out_stream:` and `output = str(out_stream)` inside the `'with'` scope to fetch the current accumulated output. 

2. Use `"with no_print:"` to suppress the console output, including the error information in `std_err`. 

3. Use `sprint = SPrint()` to create a function `sprint` that collects the printed text. 

    ```python
    from pycamia import SPrint
    sprint = SPrint()
    output_from_middle = sprint(output_str, and_other_variables) # Just like 'print'
    sprint(something_else)
    output = sprint.text
    ```

4. Use `StrIO()` to create a stream 

##### package `logging`

This package deals with loggings.

1. Use `"with logging(log_file_path):"` to create a log file of name `log_file_path` and record all the the outputs using function `print` inside the scope to this log file. 

    `logging` takes the following keyword arguments:

    1. `log_dir`: the directory for the logging file, it is set when file path is not provided. The default file name would be `{python_file_name}.log`.

    2. `prefix`: the prefix for each line in the logging file, `"{time}"` by default indicating the time of print. `time` is the only supported tag currently, please contact the developer if you have suggestions of more useful tags. 

    3. `exc_remove`: whether to remove the logging file when an exception occurs. Please note that one doesn't have to worry about the lost of exception as they will be kept in an additional file `error_msgs.log`. This argument is set to `True` by default to avoid useless log files using debugging. 

        P.S. Manually terminating the python process by `'KeyboardInterrupt'` will not cause the removal of the log file (the log file will be marked as `'interrupted'` instead) so that the outputs are kept for reference although the exception will be recorded in `error_msgs.log`. One needs to manually delete the log file if it is not needed. 

    4. `line_width`: the number of characters in a single line which is $100$ by default. 

    5. `_vars`: private argument for environmental variables, one can use `_var=globals()` if a `'problem in stacks'` is raised.

2. Use `start_logging(log_file_path)` to start the logging for the rest of the program. 

##### package `loopop`

This package helps users to manage loops in python.

1. Use `collector = Collector(max_storage)` to create a metric collector. In each loop, one can 'collect' the output in the iteration and get them back as an array. Keyword argument `max_storage` determines how many values should be stored in the array. Not limit is set by default. 

    In each loop, use `collector.register(name, value, format)` to collect the value with the format follows the python formatting protocol. If the format string contains two formats seperated by a slash: `".4f/.3f"`, the first format will be used for mean value and the second for the standard deviation (in methods `as_str_tuple`, `as_pm` and `as_latex`. 

    Use negetive sign to eliminate the `0` before the decimal point, i.e. format `".4f"` will make $0.23215$ the string `"0.2322"` but `"-.4f"` will result in `".2322"`.

    One can also place the format right after a colon in name, i.e. `register("metric:.4f", 0.23215)`. 

2. Use `TaskProgress()` (or its instance alias `iterate`) to create a progress bar before the output lines.

    ```python
    from pycamia import iterate, TaskProgress
    # Usage #1
    with iterate:
        for i in iterate(100):
            ...
            print([...something...])
    # Usage #2
    with TaskProgress(progress_len=10, n_timespan_store=20, show_progress_bar=True, breakable=True, break_key=None) as iterate:
        for i in iterate([2,3,4,...]):
            ...
            print([...something...])
    # Usage #3
    for i in iterate(100):
        for j in iterate(1000):
            ...
            print(iterate.prefix, [...something...])
    ```

    The keyword arguments determine the length (number of characters) of the progress bar (`progress_len`), the recorded iteration time for remaining time estimation (`n_timespan_store`), whether to show progress bar (`show_progress_bar`), whether the loops can be terminated by hot keys (`breakable`) and the hot key for breaking the loops (`break_key`).

    The hot key is in format like `'ctrl+d'`. Use `'+'` to connect the keys and lower-cased words for control keys. 

    `iterate(*)` for each loop takes an integer of a list (or iterable with length) as the input. It can also accept the previous keyword arguments (except `progress_len`), except they will only take effect in the sole loop (neither the outer loops nor the inner ones). 

##### package `listop`

This package cope with list objects, containing useful functions for lists. 

1. Use `argmin(list, domain)` to find the indices for the minimal value in list. The function only search in the indices `domain`. A list is output as there may be multiple entries. 
2. Use `argmax` to find the indices for the maximal value, similar to `argmin`. 
3. Use `flatten_list` to unwrap the list elements to create a list with no element in type `list`. 
4. Use `prod` to obtain the product of all numbers in the list. 
5. Use `item` to fetch the only element in the list. An error will be raised if there isn't any or are more than 1 elements. 

##### package `manager`

This package manages the info of packages and files. One can use it to organize the project. 

1. Use `__info__ = info_manager(project="PyCAMIA", ...)` to list the properties at the front of files. This serve as a brief introduction to readers.
2. Use `info_manager` at the beginning of `__init__.py`, `pack.py` uses it to create the portrait of a package. 
3. Use `__info__.check_requires()` to automatically check if the dependencies in attribute `requires` exist or not. This is commonly used in `__init__.py`. One can use `__info__ = info_manager(...).check()` to perform an in-place check.
4. Use `with __info__:` before importing required dependencies as well to perform a double check. 

##### package `strop`

This package copes with str objects. 
1. Use `str_len` to find the ASCII length for a string, with a length `2` for wide characters.
2. Use `str_slice` to slice a string by given indices.
3. Use `find_all` to obtain all the indices of a given string. `str_slice(s, find_all(s, k))` is equivalent to `s.split(k)`. 
4. Use `sorted_dict_repr` to create a repr string for a dictionary with ordered key.
5. Use `enclosed_object` to find the first object enclosed by brackets. 
6. Use `tokenize` to split a string without breaking enclosed objects. This is useful in breaking text of dictionary structures or arguments. e.g. one can use `tokenize(args, sep=[',', '='])[::2]` to find the argument names if `args` is a string in the format `key1=value1, key2 = value2, ...`.

##### package `structures`

##### package `system`

##### package `timing`

This package use the time spent of commands to perform useful inspection or organization of the codes.
1. Use `@time_this` to time a function.
2. Use `with Timer("name"):` to time a series of commands.
3. Use `with scope("name"):` to nest a series of commands. It is an alias of `Timer`. 
4. Use `with scope("name"), jump:` to jump a series of commands. 
5. Use `with scope("name"), Jump(False):` to disable the jump.
6. Use `wf = Workflow("step1", "step2")` and `with wf("step1(2)"), wf.use_tag:` before commands of "step1(2)" to create a workflow. One can change the arguments in the init function to decide which steps to run. 
7. Use `@periodic(seconds, max_repeat)` to run a function repeatedly. 

##### package `math`

##### package `more`

Currently, only `once` is contained in the `more` package. 
Adding it in a function to check if the function is run once or not. 

##### package `numpyop`

#### Waiting to Be Improved

1. More functions will be added in the future, including path handler, tools for shell and so on. 
2. Contact us to make suggestions. 

#### Acknowledgment

@Yuncheng Zhou: Developer