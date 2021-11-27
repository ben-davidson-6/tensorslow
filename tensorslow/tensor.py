from typing import Optional, Set, Union

from numpy.typing import NDArray


class Tensor:
    count = 0
    _existing_names: Set[str] = set()

    def __init__(
        self, value: NDArray, name: Optional[str] = None, trainable: bool = True
    ) -> None:
        Tensor.count += 1
        self._name: str
        self._set_name(name)
        self._value = value
        self._trainable = trainable

    def _set_name(self, name: Union[str, None]) -> None:
        if name is None:
            self._name = f"tensor_{Tensor.count}"
        elif name not in Tensor._existing_names:
            self._name = name
            Tensor._existing_names.add(name)
        else:
            self._name = f"{name}_{Tensor.count}"

    @property
    def value(self) -> NDArray:
        return self._value

    @value.setter
    def value(self, value: NDArray) -> None:
        self._value = value

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def name(self) -> str:
        return self._name
