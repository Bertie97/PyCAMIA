# pyoverload `I`

## Introduction

[`pyoverload`](https://github.com/Bertie97/pycamia/tree/main/pyoverload) is a package affiliated to project [`PyCAMIA`](https://github.com/Bertie97/pycamia). It is a powerful overloading tool to enable function overloading for `Python v3.6+` which provides `@overload` and `@typehint` decorators to perform function multiple usages. The simplest one, however, can be easily implemented as follows. 

```python
>>> from pyoverload import overload
>>> @overload
... def func(x: int):
...     print("func1", x)
...
>>> @overload
... def func(x: str):
...     print("func2", x)
...
>>> func(1)
func1 1
>>> func("1")
func2 1
```

`pyoverload` has all of following appealing features:

1. Support of **`Jedi` auto-completion** by keyword decorator `@overload`. This means all main-stream python IDE can hint you the overloaded functions you have defined. 
2. **Multiple usages** that are user friendly for all kinds of users, including `C/Java` language system users and those who are used to `singledispatch` based overload. Also, easy collector of ordinary python functions is also provided. 
3. Support of **all kinds of functions**, including functions, methods, class methods and static methods. One simple implementation for all.
4. **String types** supported. This means that one can use `"numpy.ndarray"` to mark a numpy array without importing the whole package. 
5. Sufficient **built-in types are provided** for easy representations such as `List[Int]`, `Dict@{str: int}` or `List<<int>>[10]`. 
6. **Available usage listing** when no overload function matches the input arguments. 
7. **Type constraint for an ordinary function** using `@params` decorator. 

## Installation

This package can be installed by `pip install pyoverload` or moving the source code to the directory of python libraries (the source code can be downloaded on [github](https://github.com/Bertie97/pycamia) or [PyPI](https://pypi.org/project/pyoverload/)). 

```shell
pip install pyoverload
```

## Usages

### Usage 1: Decorator Fashion

One can use `@overload` before the functions with the same function name to build an overloaded function. When the function is called, the inputs will be handed out to the suitable implementation. 

The types of the input arguments are specified by the typehints available in `python3.6+`. All known types can be added after the colon. For package classes like `np.ndarray`, please use a string to represent it. For more types, one can use types from package `types` or `pyoverload.typehint`. 

For usage of `pyoverload.typehint`, please refer to section **Typehints** for more information.

All implementations of the overloaded function are referenced in the order of definition, but the implementation ends with `__default__` or `__0__` will be used when no usage is available. Note that there are **four** underlines for this notation, **two** on each side. 

```python
>>> from pyoverload import *
>>> import numpy as np
>>> @overload
... def func__default__(x):
... 	print("func1", x)
...
>>> @overload
... def func(x: int):
... 	print("func2", x)
...
>>> @overload
... def func(x: str):
... 	print("func3", x)
...
>>> @overload
... def func(x: List<<Int>>[4]):
... 	print("func4", x)
...
>>> @overload
... def func(x: 'np.ndarray'):
... 	print("func5", x)
...
>>> func(1)
func2 1
>>> func("1")
func3 1
>>> func([1,2,3,4])
func4 [1, 2, 3, 4]
>>> func(np.array([1,2,3,4]))
func5 [1 2 3 4]
>>> func(1.)
func1 1.0
```

Note that the auto-completion by `Jedi` can only work for this usage. 

<img src="https://github.com/Bertie97/pycamia/raw/main/pyoverload/Jedi.jpg" alt="Jedi" style="zoom:50%;" />

### Usage 2: Registering Fashion

After using `@overload` decorator, apart from using `@overload` to decorate functions with the same name, one can also use the decorator with the function name `@{fill in the function name}` to decorate other functions with relevant names like `func1`, `func_str`, `first_func` for function `func`. However, omitting sign `_` is recommended for these functions. 

In this fashion, the default function is the one decorated with `@overload` though it can still be changed by adding `__default__` or `__0__` tags in the decorated function names. All typehints are the same as the first usage. 

The following example realized the first three functions in the usage 1 example in a reimplementation. 

```python
>>> from pyoverload import overload
>>> @overload
... def func(x):
... 	print("func1", x)
... 
>>> @func
... def func2(x: int):
... 	print("func2", x)
... 
>>> @func
... def _(x: str):
... 	print("func3", x)
... 
```

Note that **usage 1** and **usage 2** can be used together though you may need to specify the default function manually like in usage 1 if needed. The last example is rewrote in a combined style. 

```python
>>> from pyoverload import overload
>>> @overload
... def func__default__(x):
... 	print("func1", x)
...
>>> @overload
... def func(x: int):
... 	print("func2", x)
... 
>>> @func
... def _(x: str):
... 	print("func3", x)
... 
```

### Usage 3: Collector Fashion

The last possible usage can not be used along with the previous two, or at least this is not tested by the developer and is not recommended. Another decorator `@override` is used as a collector. 

In this usage, all functions should have different names and all functions with typehints should add decorator `@params` to activate the typehint regularization. Collector `@override` takes additional function list as the arguments indicating these functions should be packed into a single function. 

Note that the last function in the function list is the default function. 

```python
>>> from pyoverload import override, params
>>> @params
... def func_all(x):
... 	print("func1", x)
... 
>>> @params
... def func_int(x: int):
... 	print("func2", x)
... 
>>> @params
... def func_str(x: str):
... 	print("func3", x)
... 
>>> @override(func_int, func_str, func_all)
... def func(): pass
... 
```

Theoretically, decorator `@override` can also be used in **usages 1&2**, but this is not recommended either. 

## Implementation List

When an overloaded function receives arguments that are not suitable for all implementations, the error information will tell you which ones are correct. 

```python
>>> from pyoverload import overload
>>> @overload
... def func(x: int):
...     print("func1", x)
...
>>> @overload
... def func(x: str):
...     print("func2", x)
...
>>> func(1.)
Traceback (most recent call last):
  [...omitted...]
NameError: No func() matches arguments 1.0. All available usages are:
func(x:int)
func(x:str)
```

This function is available for all two Implementations but none of them takes `1.`. 

## Typehints

Decorator `@params` enables functions to reject inputs with wrong types by raising **`TypeHintError`**. One can use it directly to decorate functions with python typehints or one can add some arguments to it. In the following example, we apply the condition that `a` is a function, `b` is an integer, `k` is a series of integers while function `test_func` needs to return an iterable type of real numbers with length `2`. 

```python
>>> from pyoverload import *
>>> @params(Func, Int, +Int, __return__ = Real[2])
... def test_func(a, b=2, *k):
...     print(a, b, k)
...     return k
...
>>> test_func(lambda x: 1, 3, 4, 5)
<function <lambda> at 0x7fbdb2027f70> 3 (4, 5)
(4, 5)
```

The basic types in `pyoverload.typehint` are `Bool`, `Int`, `Float`, `Str`, `Set`, `List`, `Tuple`, `Dict`, `Callable`, `Function`, `Method`, `Lambda`, `Functional`, `Real`, `Null`, `Sequence`, `Array`, `Iterable`, `Scalar`, `IntScalar`, `FloatScalar`. Among which, 

1. `callable` contains all callable objects including callable classes while `Func` consists all kinds of actual functions. `Function`, however, only refer to explicitly defined functions, and `Method` and `Lambda` refer to the class methods and anonymous functions respectively. These three types are all contained in type `Functional`. 
2. `Real` is a `pyoverload.typehint.Type` while `real`, a list `[int, float]`, is not.
3. `null` is the type of element `None` while `Null` is the `pyoverload.typehint.Type` form of it. 
4. `sequence` is `[tuple, list, set]`, while `Sequence` is the `pyoverload.typehint.Type` form.
5. `Array` is the type of package based tensors. Only the tensors of `numpy`, `torch`, `tensorflow` and `batorch` are currently supported. Use `Array.Numpy`, `Array.Torch`, `Array.Tensorflow` and `Array.Batorch` to identify specific packages. 
6. `Iterable` includes `Array`, `Sequence` and `dict`. 
7. Three types of `Scalar`s support the array variables.

All these types are subclasses of `type` and instances of `pyoverload.typehint.Type` which will be abbreviated as `Type` in the following introduction. 

One can use `Type(int)` to convert a python type like `int` to a `Type` or use `Type(int, float)` to combine multiple existing types. Recurrence is also acceptable, `Type(Type(int, float), Type(str))` is equivalent to `Type(int, float, str)`. It will be better if a keyword `name` is also assigned. 

For a `Type`, `List` for example, we can do the following operations. Except the first four, these usages are designed for iterable types only. 

1. `+List`: This indicates that this is an extendable argument, which means it decorates arguments after `*`. It was used to specify `*args` arguments in `@params` but currently deprecated though adding `+` would not lead to failures. 

2. `~List`: This invert the typehint, meaning that all non-list types. 

3. `List|Tuple`: This is equivalent to `Type(List, Tuple)` indicating list or tuples. 

4. `Type(A)&Callable`: The *and* operator. Note that both indicator should be `Type`s. 

   ```python
   >>> from pyoverload import *
   >>> class A: pass
   ...
   >>> class B(A): pass
   ...
   >>> class C(A):
   ...     def __call__(self): pass
   ...
   >>> isoftype(B(), Type(A)&Callable)
   False
   >>> isoftype(C(), Type(A)&Callable)
   True
   ```

5. `List[5]`: This indicates the length or shape of the variable. For other types such as `Int[5]`, this indicates an iterable of length 5 with integer elements. Note that `Array[3, 4]` represents an array with shape `3 x 4`, but this representation is not available for nested lists. Use `List[List[4]][3]` or `List<<List[4]>>[3]` instead.

6. `List@[int, str]`: This only works for ordered sequence `List` and `Tuple`. It indicates that the list has 2 elements, and the first one is an integer while the second is a string. Note that this is **not** equivalent to `List[int, str]` which represents a list with elements either integer or string.

7. `List@int` or `Dict@{str: int}`: For all Iterables, `A@B` indicates type `A` with all elements of type `B`. `Dict@{str:int}` represents a dictionary with string keys and integer values.  

8. `List[int]`: `=List@int`. A list with integer elements. Note that one can use `List[Int|float]` to directly identify a list of elements that are either `int` or `float`. `List[int, float]` and `Dict[str: int, int:str]` are also valid for multiple element types or key-value pairs. Using list in the blankets will return to the effect of `@` operator: `List[[int, float]]=List@[int, float]`. 

9. `List<<int>>[10]`: `=(List@int)[10]=List[int][10]` The first one of the two types should be a `Type`. The representation refers to an integer list of length 10 which is equivalent to `List[10]@int`. The length or shape is not specified if a pair of empty blankets is given. Use `List<<Type(int, Float)>>[]` to represent multiple candidates and `Dict<<(str,int)>>[]=Dict[str:int]` to represent a dictionary. 

10. `len(List[10, 20])`: Function `len` returns the length of the array. `200` should be the result for the given example. 

## Suggestions

1. The speed of type check for `pyoverload.typehint.Type` is not very fast, hence please try your best to use builtin types, types from module `types` or list of types to do the typehint. 
2. The `overload` takes extra time for delivering the arguments, hence using it for functions require fast speed is not recommended. 
3. Do use `@params` for functions not overloaded but needs typehint constraints instead of using `@overload` without actually has multiple implementations. This is because `@params` is way faster. 

## Acknowledgment

@Yuncheng Zhou: Developer