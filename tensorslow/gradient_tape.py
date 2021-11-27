from collections import defaultdict
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from tensorslow.tensor import Tensor

__global_tape = None


T = TypeVar("T", bound="GradientTape")


class GradientTape:
    def __init__(self) -> None:
        self._tape: List[Node] = []

    def __enter__(self: T) -> T:
        set_active_tape(self)
        return self

    def __exit__(
        self,
        ex_type: Optional[Type[BaseException]],
        ex_inst: Optional[BaseException],
        ex_tb: Optional[TracebackType],
    ) -> None:
        set_active_tape(None)

    def add_node(
        self,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
        backward_closure: Callable[..., Any],
    ) -> None:
        self._tape.append(Node(inputs, outputs, backward_closure))

    def gradients(self, y: Tensor) -> Dict[str, Tensor]:
        dy_d: Dict[str, Tensor] = defaultdict(lambda: Tensor(np.array(0.)))
        dy_d[y.name] = Tensor(np.ones_like(y.value))
        for node in reversed(self._tape):
            dy_dout = dy_d[node.outputs]
            dy_dinputs = node.backward_closure(dy_dout)
            for x, dy_din in zip(node.inputs, dy_dinputs):
                if x in dy_d:
                    dy_d[x].value += dy_din.value
                else:
                    dy_d[x] = dy_din
        return dy_d

    def clear(self) -> None:
        self._tape = []


class Node:
    def __init__(
        self,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
        backward_closure: Callable[..., Any],
    ) -> None:
        self.inputs = tuple(x.name for x in inputs)
        self.outputs = outputs.name
        self.backward_closure = backward_closure


def active_tape() -> Union[None, GradientTape]:
    return __global_tape


def set_active_tape(gradient_tape: Union[GradientTape, None]) -> None:
    global __global_tape
    __global_tape = gradient_tape
