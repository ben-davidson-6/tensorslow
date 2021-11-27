import numpy as np
from numpy.testing import assert_almost_equal

from tensorslow.operations import get_op, MAT_MUL
from tensorslow.tensor import Tensor
from tests.types import TensorFactory


def test_forward(tensor_factory: TensorFactory) -> None:
    x = tensor_factory((3, 2))
    y = tensor_factory((2, 3))
    op = get_op(MAT_MUL)
    xmy = op._forward(x, y)
    assert isinstance(xmy, Tensor)
    assert_almost_equal(xmy.value, np.matmul(x.value, y.value))


def test_backward(one: Tensor, tensor_factory: TensorFactory) -> None:
    x = tensor_factory((1,))
    y = tensor_factory((1,))
    op = get_op(MAT_MUL)
    xmy = op._forward(x, y)
    closure = op._backward(xmy, x, y)
    dxmy_dx, dxmy_dy = closure(one)
    assert isinstance(dxmy_dx, Tensor)
    assert isinstance(dxmy_dy, Tensor)
    assert_almost_equal(dxmy_dx.value, y.value.T)
    assert_almost_equal(dxmy_dy.value, x.value)
