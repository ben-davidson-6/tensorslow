import numpy as np
from numpy.testing import assert_almost_equal

from tensorslow.operations import get_op, ADDITION
from tensorslow.tensor import Tensor
from tests.types import TensorFactory


def test_forward(tensor_factory: TensorFactory) -> None:
    x = tensor_factory((1,))
    y = tensor_factory((1,))
    addition_op = get_op(ADDITION)
    xpy = addition_op._forward(x, y)
    assert isinstance(xpy, Tensor)
    assert xpy.value == (x.value + y.value)


def test_backward(one: Tensor, tensor_factory: TensorFactory) -> None:
    x = tensor_factory((1,))
    y = tensor_factory((1,))
    addition_op = get_op(ADDITION)
    xpy = addition_op._forward(x, y)
    closure = addition_op._backward(xpy, x, y)
    dxpy_dx, dxpy_dy = closure(one)
    assert isinstance(dxpy_dx, Tensor)
    assert isinstance(dxpy_dy, Tensor)
    assert_almost_equal(dxpy_dx.value, np.ones((1,)))
    assert_almost_equal(dxpy_dy.value, np.ones((1,)))
