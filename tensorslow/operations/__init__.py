from tensorslow.operations._operations.addition import OP_NAME as ADDITION
from tensorslow.operations._operations.matrix_multiplication import OP_NAME as MAT_MUL
from tensorslow.operations._operations.multiplication import OP_NAME as MULTIPLICATION
from tensorslow.operations._operations.relu import OP_NAME as RELU
from tensorslow.operations._operations.softmax_cross_entropy_with_logits import (
    OP_NAME as CROSS_ENTROPY,
)
from tensorslow.operations.base import OperationRegister, OperationType, get_op

OperationRegister.load_operations(OperationType.NUMPY)


__all__ = [
    "OperationRegister",
    "OperationType",
    "get_op",
    "MAT_MUL",
    "ADDITION",
    "MULTIPLICATION",
    "RELU",
    "CROSS_ENTROPY",
]
