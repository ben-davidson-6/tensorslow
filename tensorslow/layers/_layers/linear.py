import numpy as np

from tensorslow.layers.base import Layer
from tensorslow.operations import MAT_MUL, get_op
from tensorslow.tensor import Tensor


class Linear(Layer):
    def __init__(self, units: int) -> None:
        super().__init__()
        self.units = units
        self.weights: Tensor

    def build(self, x: Tensor) -> None:
        n_in = x.value.shape[-1]
        n_out = self.units
        delta = 1 / np.sqrt(n_in)
        weight = np.random.uniform(-delta, delta, (n_in, n_out))
        self.weights = Tensor(weight, name="linear_kernel")

    def forward(self, x: Tensor) -> Tensor:
        op = get_op(MAT_MUL)
        return op.forward(x, self.weights)
