from tensorslow.operations.base import Op, OperationRegister


def test_operation_register() -> None:

    class Foo(Op):
        pass

    assert Foo in OperationRegister.operations

