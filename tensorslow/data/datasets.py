from typing import Any


class Dataset:
    def __iter__(self) -> Any:
        raise NotImplementedError
