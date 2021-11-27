from tensorslow.layers.base import Layer
from tensorslow.operations import CROSS_ENTROPY, get_op
from tensorslow.tensor import Tensor


class SoftmaxCrossEntropyLogits(Layer):
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        loss = get_op(CROSS_ENTROPY)
        return loss.forward(logits, labels)
