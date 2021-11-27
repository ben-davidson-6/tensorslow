from typing import Callable, Tuple

import numpy as np

from tensorslow.operations.base import Op, OperationType
from tensorslow.tensor import Tensor

OP_NAME = __name__.rsplit(".")[-1]
Grads = Tuple[Tensor, Tensor]
Closure = Callable[[Tensor], Grads]


class _MatrixMultiplication(Op, operation_type=OperationType.NUMPY, operation_name=OP_NAME):
    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_val = x.value
        y_val = y.value
        return Tensor(
            np.matmul(x_val, y_val),
            name=f"{x.name}.{y.name}",
        )

    def _backward(self, _: Tensor, x: Tensor, y: Tensor) -> Closure:
        def closure(dl_dout: Tensor) -> Grads:
            dl_dout_val = np.atleast_2d(dl_dout.value)
            dl_dx = np.matmul(dl_dout_val, y.value.T)
            dl_dy = np.matmul(x.value.T, dl_dout_val)
            return Tensor(dl_dx), Tensor(dl_dy)

        return closure
