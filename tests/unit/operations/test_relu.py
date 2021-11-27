import numpy as np
from numpy.testing import assert_almost_equal

from tensorslow.operations import get_op, RELU
from tensorslow.tensor import Tensor
from tests.types import TensorFactory


def test_forward(tensor_factory: TensorFactory) -> None:
    x = tensor_factory((3, 3))
    op = get_op(RELU)
    rx = op._forward(x)
    assert isinstance(rx, Tensor)
    assert_almost_equal(rx.value, (rx.value > 0.)*rx.value)


def test_backward(one: Tensor, tensor_factory: TensorFactory) -> None:
    x = tensor_factory((1,))
    y = tensor_factory((1,))
    op = get_op(RELU)
    rx = op._forward(x)    
    closure = op._backward(rx, x)
    drx_dx, = closure(one)
    assert isinstance(drx_dx, Tensor)
    assert_almost_equal(drx_dx.value, (rx.value > 0).astype(np.float32))