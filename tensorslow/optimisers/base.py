from typing import Any, Dict

from tensorslow.model import Model
from tensorslow.tensor import Tensor


class Optimiser:
    def minimise(self, gradients: Dict[str, Tensor], model: Model) -> Any:
        raise NotImplementedError
