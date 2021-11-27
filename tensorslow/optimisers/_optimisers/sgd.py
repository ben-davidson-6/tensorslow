from typing import Dict

from tensorslow.model import Model
from tensorslow.optimisers.base import Optimiser
from tensorslow.tensor import Tensor


class SGD(Optimiser):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def minimise(self, gradients: Dict[str, Tensor], model: Model) -> None:
        for tensor_name, tensor in model.trainable_variables().items():
            tensor.value -= self.learning_rate * gradients[tensor_name].value