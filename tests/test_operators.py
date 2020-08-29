# +
import sys
sys.path.append("..")
from minitorch import assert_close, operators
    
from hypothesis import given
from hypothesis.strategies import lists
import pytest

try:
    from .strategies import small_floats
except:
    from strategies import small_floats


# -

# `Task 0.1`

@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_add_and_mul(x, y):
    assert_close(operators.mul(x, y), x * y)
    assert_close(operators.add(x, y), x + y)
    assert_close(operators.neg(x), -x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert operators.relu(a) == a
    else:
        assert operators.relu(a) == 0.0


# `Task 0.2`

@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a, b):
    """
    Ensures :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    assert operators.mul(a, b) == operators.mul(b, a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    r"""
    Ensures operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    assert_close(operators.mul(z, operators.add(x, y)), 
                 operators.add(operators.mul(z, x), operators.mul(z, y))) # < 1e-20


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_order(x, y, z):
    r"""
    :math:`(x \times y) \times z = x \times (y \times z)`
    """
    assert_close(operators.mul(operators.mul(x, y), z), 
                 operators.mul(x, operators.mul(y, z))) # <1e-8


# HIGHER ORDER


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    assert_close(operators.addLists([a, b], [c, d]), [a + c, b + d])


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_property(ls1, ls2):
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    assert_close(operators.add(operators.sum(ls1), operators.sum(ls2)), 
                 operators.sum(ls1 + ls2))


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(operators.sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(operators.prod([x, y, z]), x * y * z)
