import enum
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from tensorslow.gradient_tape import active_tape
from tensorslow.tensor import Tensor


class OperationType(enum.Enum):
    BASE = -1
    NUMPY = 0
    NUMBA = 1


class OperationRegister(type):
    operations: Dict[OperationType, Dict[str, "Op"]] = {}
    default_operations: Dict[str, "Op"] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):  # type: ignore
        return super().__new__(mcs, name, bases, namespace)

    def __init__(
        self,
        name: str,
        bases: Tuple[type, ...],
        dct: Dict[str, Any],
        operation_type: OperationType,
        operation_name: str
    ) -> None:
        if operation_type not in OperationRegister.operations:
            OperationRegister.operations[operation_type] = {}
        OperationRegister.operations[operation_type][operation_name] = self()

    @staticmethod
    def load_operations(operation_type: OperationType) -> Dict[str, "Op"]:
        ops = OperationRegister.operations[operation_type]
        OperationRegister.default_operations = ops
        return ops

    @staticmethod
    def find_op(op_name: str) -> "Op":
        for op_type in OperationRegister.operations:
            for op_name_ in OperationRegister.operations[op_type]:
                if op_name == op_name_:
                    return OperationRegister.operations[op_type][op_name_]
        raise RuntimeError(f"Could not load op {op_name}, as no implementation")

    @staticmethod
    def get_operation(op_name: str) -> "Op":
        try:
            op = OperationRegister.default_operations[op_name]
        except KeyError:
            op = OperationRegister.find_op(op_name)
        return op


class Op(metaclass=OperationRegister, operation_type=OperationType.BASE, operation_name="Base"):
    _forward: Callable[..., Tensor]
    _backward: Callable[..., Callable[..., Sequence[Union[Tensor, None]]]]

    def forward(self, *args: Tensor) -> Tensor:
        x = self._forward(*args)
        gradient_tape = active_tape()
        if gradient_tape is not None:
            backward_closure = self._backward(x, *args)
            gradient_tape.add_node(
                inputs=args,
                outputs=x,
                backward_closure=backward_closure,
            )
        return x


def get_op(op_name: str) -> Op:
    return OperationRegister.get_operation(op_name)
