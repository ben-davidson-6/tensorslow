import pytest

from typing import Type, Optional
from tensorslow.layers import Layer
from tensorslow.tensor import Tensor

import numpy as np
@pytest.fixture
def empty_layer() -> Type[Layer]:
    class MyLayer(Layer):
        def __init__(self) -> None:
            super().__init__()
            self.weight = Tensor(np.zeros([1]))

    return MyLayer


@pytest.fixture
def empty_layer_with_() -> Type[Layer]:
    class MyLayer(Layer):
        def __init__(self) -> None:
            super().__init__()
    return MyLayer


def test_sub_layer_is_assigned(empty_layer: Layer) -> None:
    class LayerWithLayer(Layer):
        def __init__(self) -> None:
            super().__init__()
            self.layer_0 = empty_layer()

    layer_0 = LayerWithLayer()

    assert len(layer_0._sub_layers) == 1


def test_finds_trainable_variables(empty_layer: Layer) -> None:
    class LayerWithLayer(Layer):
        def __init__(self, other_layer: Optional[Layer] = None) -> None:
            super().__init__()
            self.other_layer = other_layer
            self.layer_0 = empty_layer()

    layer_0 = LayerWithLayer()
    layer_1 = LayerWithLayer(layer_0)

    assert len(layer_1.trainable_variables()) == 2

    
