from tensorslow.gradient_tape import GradientTape, Node, active_tape


def test_tape_active() -> None:
    assert active_tape() is None
    with GradientTape():
        assert active_tape() is not None
    assert active_tape() is None
