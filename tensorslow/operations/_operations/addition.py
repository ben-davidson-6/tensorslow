from typing import Callable, Tuple

from tensorslow.operations.base import Op, OperationType
from tensorslow.tensor import Tensor

OP_NAME = __name__.rsplit(".")[-1]
Grads = Tuple[Tensor, Tensor]
Closure = Callable[[Tensor], Grads]


class _Addition(Op, operation_type=OperationType.NUMPY, operation_name=OP_NAME):
    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_val = x.value
        y_val = y.value
        return Tensor(
            x_val + y_val,
            name=f"{x.name}+{y.name}",
        )

    def _backward(self, _: Tensor, x: Tensor, y: Tensor) -> Closure:
        def closure(dl_dout: Tensor) -> Grads:
            dl_dout_val = dl_dout.value
            return Tensor(dl_dout_val), Tensor(dl_dout_val)

        return closure

