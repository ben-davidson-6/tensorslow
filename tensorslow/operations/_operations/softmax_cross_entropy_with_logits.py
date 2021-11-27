from typing import Callable, Tuple

import numpy as np

from tensorslow.operations.base import Op, OperationType
from tensorslow.tensor import Tensor

OP_NAME = __name__.rsplit(".")[-1]
Grads = Tuple[Tensor, None]
Closure = Callable[[Tensor], Grads]


class _SoftmaxCrossEntropyWithLogits(Op, operation_type=OperationType.NUMPY, operation_name=OP_NAME):
    def _forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
        logits_val = logits.value
        labels_val = labels.value

        max_logit = np.max(logits_val, axis=-1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(logits_val - max_logit), axis=1))
        x = np.take_along_axis(logits_val, labels_val, axis=1)[..., 0]
        individual = x - max_logit[..., 0] - log_sum
        
        return Tensor(-np.sum(individual))

    def _backward(self, _: Tensor, logits: Tensor, labels: Tensor) -> Closure:
        logits_val = logits.value
        labels_val = labels.value

        max_logit = np.max(logits_val, axis=-1, keepdims=True)
        expped = np.exp(logits_val - max_logit)
        softmax = expped / np.sum(expped, axis=-1, keepdims=True)
        ones = np.zeros_like(softmax)
        np.put_along_axis(ones, labels_val, 1, axis=1)

        def closure(dl_dout: Tensor) -> Grads:
            dl_dlogit = (softmax - ones) * dl_dout.value
            dl_dlabels = None
            return Tensor(dl_dlogit), dl_dlabels

        return closure
