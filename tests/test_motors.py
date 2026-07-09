import pytest
from xpdtools.motors import get_encoder_value_from_angle

@pytest.mark.parametrize(
    "angle, encoder_resolution, encoder_pos_at_zero, expected",
    [
        (0.0, 10.0, 1000, 1000),
        (90.0, 10.0, 1000, 1900),
        (180.0, 10.0, 1000, 2800),
        (360.0, 10.0, 1000, 4600),
        (-90.0, 10.0, 1000, 100),
    ],
)
def test_get_encoder_value_from_angle(angle, encoder_resolution, encoder_pos_at_zero, expected):
    actual = get_encoder_value_from_angle(angle, encoder_resolution, encoder_pos_at_zero)
    assert actual == expected, f"Expected {expected}, got {actual}"
