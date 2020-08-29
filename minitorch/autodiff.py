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

# + [markdown] heading_collapsed=true
# ### Set-up

# + hidden=true
try:
    from .util import wrap_tuple, unwrap_tuple
except:
    from util import wrap_tuple, unwrap_tuple

import uuid

# + hidden=true
from typing import List, Dict, Optional

# + hidden=true
nx = None


# + [markdown] heading_collapsed=true
# ### `Context`

# + hidden=true
class Context:
    """
    Simple storage to keep values in the forward pass that are required 
    in backward pass.
    Used by History
    
    Attributes:
        no_grad (bool)
    """

    def __init__(self, no_grad: bool=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)


# -

# ### `FunctionBase`

class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Subclass needs to implement
    - forward: perform the forward pass
    - backward: perform the backward pass
    - variable: attach a :class:`Variable` as output
    
    Base class uses `apply` method to wrap subclass's `forward`

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        pass

    @classmethod
    def apply(cls, *vals):
        """
        Wrapper around `forward` of subclass to
        - attach :class:`Context` object as storage of `forward` pass
        """
        raw_vals = []
        need_grad = False
        # unpack Variable to "raw" values
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)
        ctx = Context(not need_grad)
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output) -> List["VariableWithDeriv"]:
        """
        Implement the derivative chain-rule.
        Used by `backpropagate` <- `Variable.backward()`

        Args:
            cls (:class:`FunctionBase`): The function
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of :class:`VariableWithDeriv`: A list of variables with their derivatives
            for each :class:`Variable` object in input (other inputs should be ignored)
        """
        d_vars = cls.backward(ctx, d_output)
        return [VariableWithDeriv(var, d_var) for var, d_var in zip(inputs, d_vars) 
                if isinstance(var, Variable)]


# + [markdown] heading_collapsed=true
# ### `History`

# + hidden=true
class History:
    """
    `History` stores the last `Function` operations that were used to
    construct an autodiff object.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last function that was called.
        ctx (:class:`Context`): The context for that function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.
    """

    def __init__(self, last_fn: FunctionBase=None, ctx: Context=None, 
                 inputs: Optional[List]=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def is_leaf(self):
        return self.last_fn is None

    def chain_rule(self, d_output):
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)


# -

# ### `Variable`

# + [markdown] heading_collapsed=true
# #### Motivation

# + [markdown] hidden=true
# `Variable` consists of
# - `History` : 
#     - last func `FunctionBase` - similar to `grad_fn` in pytorch
#     - ctx: `Context`      
#     - inputs

# + [markdown] hidden=true
#     x ----- [ Mul ] ------> z 
#                |
#     y ---------

# + [markdown] hidden=true
# the goal is `auto-differentiation` : abstraction is computation graph where
# - nodes are operations
# - edges are tensors flowing in and out of nodes
#
# `forward` pass is simple
#
# `backward` pass: 
# - starting from z: z needs to keep track of the op that creates it: link from z to Mul
# - `z.backward(d_out or 1.0)` : send the upstream grad to Mul
#
# - at `Mul`: notice the following
# $$ \frac{dz}{dx} = y $$
# $$ \frac{dz}{dy} = x $$
#
# this means we need to save x, y for backward pass. 
#
# - `Mul` need to perform the gradient computation with respect to all of its inputs, this means it needs to keep track all of its inputs
# - `Mul.backward()` performs the computation needed and returns the grads wrt inputs
# - `Mul.backward()` recursively asks its inputs to backward the gradients

# + [markdown] heading_collapsed=true
# #### Potential Implementation

# + [markdown] hidden=true
# - Option 1: implement the computation as instance method
#
# ```python
# class Mul:
#
#     def __init__(self):
#         super().__init__()
#         self._inputs = None
#
#     def forward(self, x, y):
#         self._inputs = [x, y]
#         return x * y
#
#     def backward(self, d_out):
#         x, y = self._inputs
#         return d_out * y, d_out * x
# ```
#
# - Option 2: implement the computation as class method. For this, we can not keep a reference to the input, defer this to an object called `History`.
# -

# #### Impl

class Variable:
    """
    
    Attributes:
        history (:class:`History`) : The sequence of function calls that led to this variable.
        derivative (number): The derivative with respect to this variable.
        name (string) : an optional name for debugging.
    """

    def __init__(self, history: "History", name: str = None):
        assert history is None or isinstance(history, History), history
        self.history = history
        self._derivative = None

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

    def requires_grad_(self, val):
        self.history = History(None, None, None)

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(VariableWithDeriv(self, d_output))

    @property
    def derivative(self):
        return self._derivative

    ## IGNORE
    def __hash__(self):
        return hash(self._name)

    def _add_deriv(self, val):
        assert self.history.is_leaf(), "Only leaf variables can have derivatives."
        assert type(val) == float, f"Only supports scalar float so far"
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_grad_(self):
        self._derivative = self.zeros()

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0

    def expand(self, x):
        return x

    # def make_graph(self, graphfile, grad_output=1.0):
    #     global nx
    #     import networkx as nx
    #     import graphviz

    #     G = AutodiffRunner().run(self, grad_output, make_graph=True)
    #     nx.nx_pydot.write_dot(G, graphfile)
    #     graphviz.render(filepath="graph.dot", format="png", engine="dot")

    ## IGNORE


# + [markdown] heading_collapsed=true
# #### `Variable with Derivative`

# + [markdown] hidden=true
# Intuitively, this class represents the `backward` tensor (which is the Variable and its upstream derivative) while the Variable class represents the `forward` tensor.
#
# `backprop` is a multi-step process where at each step, `VariableWithDeriv` holds the backward tensor to an op, the op performs the grad computation and returns the output backward tensor.
#
# `forward`
#
#             forward
#     x ----> [ Mul ] ----> z
#                |
#     y ---------|
#     
# `backward`
#
#                              backward
#     x <---- (x, y*dL/dz) --- [ Mul ] <---- (z, dL/dz) --- z
#                                 |
#     y <---- (y, x*dL/dz) -------|
#

# + hidden=true
class VariableWithDeriv:
    "Holder for a variable with it derivative."

    def __init__(self, variable, deriv):
        self.variable = variable
        self.deriv = variable.expand(deriv)


# + hidden=true
def is_leaf(val):
    return isinstance(val, Variable) and val.history.is_leaf()


# -

# ### `Backprop`

# + [markdown] heading_collapsed=true
# #### Potential Impl

# + [markdown] hidden=true
# - recursive (local) `backprop` which has the tensor to call backward on the op, the op in turn calls backward on the input tensor. The limitation of this approach is that it's a local procedure, it doesn't have a global view to optimize the computation
#
# - static global `backprop`: has the potential to optimize
# -

# #### Impl

# `Breath-first search` : https://en.wikipedia.org/wiki/Breadth-first_search
#
# > Breadth-first search (BFS) is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root (or some arbitrary node of a graph, sometimes referred to as a 'search key'[1]), and explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.
#
# > It uses the opposite strategy as depth-first search, which instead explores the node branch as far as possible before being forced to backtrack and expand other nodes
#
# initiated by `Var.backward`
#
# - why it uses `VariableWithDeriv`, see the section `VariableWithDeriv`

# +
from collections import OrderedDict

def backpropagate(final_variable_with_deriv: VariableWithDeriv):
    """
    Runs a breadth-first search on the computation graph in order to
    propagate derivative to the leaves.

    See :doc:`backpropagate` for details on the algorithm

    Args:
       final_variable_with_deriv (:class:`VariableWithDeriv`): The final value
           and its derivative that we want to propagate backward to the leaves.
    """
    q = OrderedDict(
        {final_variable_with_deriv.variable.name: final_variable_with_deriv}
    )
    while len(q) > 0:
        _, v = q.popitem(last=False) # pop the first item
        var, deriv = v.variable, v.deriv
        if is_leaf(var): # no history, can't backprop further
            var._add_deriv(deriv)
        else:
            vars_with_deriv: List[VariableWithDeriv] = var.history.chain_rule(deriv)
            for in_var in vars_with_deriv:
                if in_var.variable.name in q:
                    # if already in queue, accumulate the grad
                    q[in_var.variable.name].deriv += in_var.deriv
                else:
                    # if not, append to the queue
                    q.update({in_var.variable.name: in_var})


# + tags=["active-ipynb"]
# # using orderdict as a LIFO queue
# v1 = Variable(None, name="v1")
# v2 = Variable(None, name="v2")
#
# d = OrderedDict({v.name: v for v in [v1, v2]})
#
# print(d)
#
# v3 = Variable(None, name="v3")
# d.update({v3.name: v3})
#
# print(d)
#
# d.popitem(last=False)
