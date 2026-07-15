import pytest

from xpdtools.motors import get_encoder_value_from_pos


@pytest.mark.parametrize(
    "position, encoder_resolution, encoder_pos_at_zero, expected",
    [
        (0.0, 10.0, 1000, 1000),
        (90.0, 10.0, 1000, 1009),
        (180.0, 10.0, 1000, 1018),
        (360.0, 10.0, 1000, 1036),
        (-90.0, 10.0, 1000, 991),
    ],
)
def test_get_encoder_value_from_pos(
    position, encoder_resolution, encoder_pos_at_zero, expected
):
    actual = get_encoder_value_from_pos(
        position, encoder_resolution, encoder_pos_at_zero
    )
    assert actual == expected, f"Expected {expected}, got {actual}"
