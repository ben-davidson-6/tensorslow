from typing import Callable, Tuple

import numpy as np

from tensorslow.operations.base import Op, OperationType
from tensorslow.tensor import Tensor

OP_NAME = __name__.rsplit(".")[-1]
Grads = Tuple[Tensor]
Closure = Callable[[Tensor], Grads]


class _ReluActivation(Op, operation_type=OperationType.NUMPY, operation_name=OP_NAME):
    def _forward(self, x: Tensor) -> Tensor:
        x_val = x.value
        return Tensor(
            x_val * (x_val > 0),
            name=f"rel({x.name})",
        )

    def _backward(self, _: Tensor, x: Tensor) -> Closure:
        def closure(dl_dout: Tensor) -> Grads:
            x_val = x.value
            val = (x_val > 0).astype(np.float32) * dl_dout.value
            return (Tensor(val),)

        return closure
