from tensorslow.layers.base import Layer
from tensorslow.operations.base import OperationRegister
from tensorslow.tensor import Tensor
from tensorslow.operations import RELU, get_op


class Relu(Layer):
    def forward(self, x: Tensor) -> Tensor:
        relu = get_op(RELU)
        return relu.forward(x)
