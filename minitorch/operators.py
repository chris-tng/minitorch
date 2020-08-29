# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: minitorch
#     language: python
#     name: minitorch
# ---

# Basic math operations as well as a few warmup problems for testing out your functional programming chops in python. (Please ignore the @jit decorator for now. It will come back in later assignments.)

try:
    from .util import jit
except:
    from util import jit
import math


@jit
def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y


@jit
def id(x):
    ":math:`f(x) = x`"
    return x


@jit
def add(x, y):
    ":math:`f(x, y) = x + y`"
    return float(x + y)


@jit
def neg(x):
    ":math:`f(x) = -x`"
    return -float(x)


@jit
def lt(x, y):
    ":math:`f(x) =` 1.0 if x is greater then y else 0.0"
    return 1.0 if x > y else 0.


# +
EPS = 1e-6

@jit
def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)

@jit
def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)

@jit
def log_back(a, b):
    return b / (a + EPS)


# -

@jit
def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-a})}`
    (See https://en.wikipedia.org/wiki/Sigmoid_function .)
    """
    return 1.0 / add(1.0, exp(-x))


@jit
def relu(x):
    """
    :math:`f(x) =` x if x is greater then y else 0
    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks).)
    """
    return x if x > 0. else 0.


@jit
def relu_back(x, y):
    ":math:`f(x) =` y if x is greater then 0 else 0"
    return y if x > 0. else 0.


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    return 1. if x == y else 0.


# ### Higher-order functions

def map(fn):
    """
    Higher-order map.
    .. image:: figs/Ops/maplist.png
    See https://en.wikipedia.org/wiki/Map_(higher-order_function)
    Args:
        fn (one-arg function): process one value
    Returns:
        function : a function that takes a list and applies `fn` to each element
    """
    def _fn(ls):
        return [fn(e) for e in ls]
    return _fn


def negList(ls):
    "Use :func:`map` and :func:`neg` negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).
    .. image:: figs/Ops/ziplist.png
    See https://en.wikipedia.org/wiki/Map_(higher-order_function)
    Args:
        fn (two-arg function): combine two values
    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
    applying fn(x, y) one each pair of elements.
    """
    def _fn(ls1, ls2):
        return [fn(e1, e2) for e1, e2 in zip(ls1, ls2)]
    return _fn


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.
    .. image:: figs/Ops/reducelist.png
    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`
    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def _fn(ls):
        r = start
        for e in ls:
            r = fn(r, e)
        return r
    return _fn


def sum(ls):
    """
    Sum up a list using :func:`reduce` and :func:`add`.
    """
    return reduce(add, 0)(ls)


def prod(ls):
    """
    Product of a list using :func:`reduce` and :func:`mul`.
    """
    return reduce(mul, 1)(ls)
