from typing import Any, Callable, Dict, Optional
from tensorslow.tensor import Tensor


class Layer:
    forward: Callable
    build: Callable

    def __init__(self) -> None:
        """Base from which all other layers should inherit.

        This provides functionality to get all training variables.
        """
        self._trainable_variables: Dict[str, Tensor] = {}
        self._sub_layers: Dict[str, Layer] = {}
        self.built = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.built and hasattr(self, "build"):
            self.build(*args, **kwargs)
            self.built = True
        return self.forward(*args, **kwargs)

    def trainable_variables(
        self, trainable: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, Tensor]:
        if trainable is None:
            trainable = self._trainable_variables
        else:
            trainable.update(self._trainable_variables)

        for layer in self._sub_layers:
            self._sub_layers[layer].trainable_variables(trainable)

        return trainable

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Whenever you assign to self we capture the trainable tensors.

        Note this doesn't work for self.layer = [layer_0, layer_1]

        Args:
            name (str): attribute name
            value (Any): whatever you are assigning
        """
        is_layer = isinstance(value, Layer)
        is_trainable = isinstance(value, Tensor) and value.trainable
        if is_trainable:
            self._trainable_variables[value.name] = value
        elif is_layer:
            self._sub_layers[name] = value
        super().__setattr__(name, value)
