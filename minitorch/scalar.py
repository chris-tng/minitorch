# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

# Implementation of the scalar object for autodifferentiation.


try:
    from util import assert_close
    import operators
    from autodiff import FunctionBase, Variable, History
except:
    from .util import assert_close
    from . import operators    
    from .autodiff import FunctionBase, Variable, History

from typing import Tuple


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
       f : arbitrary function from n-args to one value
       *vals (floats): n-float values :math:`x_1 \ldots x_n`
       arg (int): the number :math:`i` of the arg to compute the derivative
       epsilon (float): a small constant

    Returns:
       float : An approximation of :math:`f'_i(x_1, \ldots, x_n)`
    """
    vals1 = list(vals)
    vals1[arg] += epsilon
    vals2 = list(vals)
    vals2[arg] -= epsilon
    return (f(*vals1) - f(*vals2))/ (2.* epsilon)


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    numbers creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.

    """

    def __init__(self, v, back=History(), name=None):
        super().__init__(back, name=name)
        self.data = v

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __add__(self, b):
        return Add.apply(self, b)
        
    def __lt__(self, b):
        return LT.apply(self, b)
        
    def __gt__(self, b):
        raise NotImplementedError
        
    def __sub__(self, b):
        return Add.apply(self, Neg.apply(b))

    def __neg__(self):
        return Neg.apply(self)
        
    def log(self):
        return Log.apply(self)
        
    def sigmoid(self):
        return Sigmoid.apply(self)
        
    def relu(self):
        return ReLU.apply(self)
        
    def get_data(self):
        return self.data


class ScalarFunction(FunctionBase):
    "A function that processes and produces Scalar variables."

    @staticmethod
    def forward(ctx, *inputs):
        """Args:

           ctx (:class:`Context`): A special container object to save
                                   any information that may be needed for the call to backward.
           *inputs (list of numbers): Numerical arguments.

        Returns:
            number : The computation of the function f.

        """
        pass

    @staticmethod
    def backward(ctx, d_output):
        """
        Args:
            ctx (Context): A special container object holding any information saved during in the corresponding `forward` call.
            d_output (number):
        Returns:
            numbers : The computation of the derive function f' for each inputs times `d_output`.
        """
        pass

    # checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    "Implements additions"

    @staticmethod
    def forward(ctx, a, b):
        return operators.add(a, b)

    @staticmethod
    def backward(ctx, d_output) -> Tuple:
        return (d_output, d_output)


class Log(ScalarFunction):
    "Implements log"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return (
            operators.log_back(a, d_output), 
        )


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx, a, b):
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx, d_output):
        return (0.0, )


# To implement.


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)
    
    @staticmethod
    def backward(ctx, d_output):
        a, b = ctx.saved_values
        return (
            operators.mul(d_output, b), 
            operators.mul(d_output, a)
        )


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return 1./a
        
    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return (
            operators.neg(d_output) / operators.mul(a, a), 
        )


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        return operators.neg(a)
        
    @staticmethod
    def backward(ctx, d_output):
        return (
            operators.neg(d_output),
        )


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        s = operators.sigmoid(a)
        ctx.save_for_backward(s)
        return s
        
    @staticmethod
    def backward(ctx, d_output):
        s = ctx.saved_values
        return (
            operators.mul(d_output, operators.mul(s, 1 - s)),
        )


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.relu(a)
        
    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return (
            operators.relu_back(a, d_output),
        )


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        e = operators.exp(a)
        ctx.save_for_backward(e)
        return e
        
    @staticmethod
    def backward(ctx, d_output):
        e = ctx.saved_values
        return (
            operators.mul(e, d_output),
        )


def derivative_check(f, *scalars):

    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]

    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print(x.derivative, check)
        assert_close(x.derivative, check.data)
