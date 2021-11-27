import numpy as np
from numpy.testing import assert_almost_equal

from tensorslow.operations import get_op, CROSS_ENTROPY
from tensorslow.tensor import Tensor
from tests.conftest import CentredDifferenceApproximator
from tests.types import TensorFactory


def test_forward(tensor_factory: TensorFactory) -> None:
    batch_size = 3
    classes = 4
    logits = tensor_factory((batch_size, classes))
    labels = Tensor(np.random.randint(0, classes, (batch_size, 1)))
    op = get_op(CROSS_ENTROPY)
    log_sum = op._forward(logits, labels)
    assert isinstance(log_sum, Tensor)

    # use naive summation without stability fix
    e_x = np.exp(logits.value)
    softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
    softmax = np.take_along_axis(softmax, labels.value, axis=1)
    naive_sum = -np.sum(np.log(softmax))

    assert_almost_equal(log_sum.value, naive_sum)


def test_backward(
    approx_df: CentredDifferenceApproximator, one: Tensor, tensor_factory: TensorFactory
) -> None:

    # setup
    batch_size = 3
    classes = 4
    logits = tensor_factory((batch_size, classes))
    labels = Tensor(np.random.randint(0, classes, (batch_size, 1)))
    op = get_op(CROSS_ENTROPY)

    # forward
    log_sum = op._forward(logits, labels)
    assert isinstance(log_sum, Tensor)

    # backward
    closure = op._backward(log_sum, logits, labels)
    dlog_sum_dlogit, dlog_sum_dlabel = closure(one)

    # approximate
    approx_df.set_function(
        lambda x: op._forward(x, labels),
        [logits]
        )

    assert_almost_equal(approx_df.centred_difference()[0], dlog_sum_dlogit.value)
    assert isinstance(dlog_sum_dlogit, Tensor)
    assert dlog_sum_dlabel is None