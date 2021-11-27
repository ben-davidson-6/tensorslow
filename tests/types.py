from typing import Callable, Tuple

from tensorslow.tensor import Tensor

TensorFactory = Callable[[Tuple[int, ...]], Tensor]

