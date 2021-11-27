from typing import Tuple

import numpy as np
import pytest

from tensorslow.tensor import Tensor
from tensorslow.approx import CentredDifferenceApproximator
from tests.types import (
    TensorFactory
)


@pytest.fixture
def tensor_factory() -> TensorFactory:
    def factory(shape: Tuple[int, ...]) -> Tensor:
        return Tensor(np.random.random(shape))

    return factory


@pytest.fixture
def one() -> Tensor:
    return Tensor(np.array(1.0))


@pytest.fixture
def approx_df() -> CentredDifferenceApproximator:
    return CentredDifferenceApproximator()
