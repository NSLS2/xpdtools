import pytest
from ophyd_async.core import FlyMotorInfo

from xpdtools.flyers import (
    PandaPcompDirection,
    SingleAxisFlyscanInfo,
    calculate_move_time_for_flyscan,
    construct_fly_info_models,
)


@pytest.mark.parametrize(
    "travel_distance, max_motor_velocity, num_images,"
    " exposure_time, acq_time_overhead, expected_time",
    [
        (10.0, 2.0, 3, 0.2, 0.001, 5.0),
        (5.0, 1.0, 2, 0.1, 0.002, 5.0),
        (20.0, 5.0, 3, 0.3, 0.003, 4.0),
        (15.0, 3.0, 2, 0.25, 0.004, 5.0),
        (8.0, 4.0, 10, 0.5, 0.005, 5.05),
    ],
)
def test_calculate_move_time_for_flyscan(
    travel_distance,
    max_motor_velocity,
    num_images,
    exposure_time,
    acq_time_overhead,
    expected_time,
):
    result = calculate_move_time_for_flyscan(
        travel_distance,
        max_motor_velocity,
        num_images,
        exposure_time,
        acq_time_overhead,
    )
    assert pytest.approx(result, rel=1e-3) == expected_time


@pytest.mark.parametrize(
    "num_pulses, max_exposure_time, start_position,"
    " stop_position, encoder_resolution, max_motor_velocity,"
    " encoder_pos_at_zero, time_based,"
    " expected_flyer_info, expected_motor_info",
    [
        (
            11,
            0.1,
            0.0,
            100.0,
            0.1,
            50.0,
            0,
            False,
            SingleAxisFlyscanInfo(
                start=0,
                num_pulses=11,
                direction=PandaPcompDirection.POSITIVE,
                pulse_width=1,
                pulse_step=100,
                time_based=False,
            ),
            FlyMotorInfo(start_position=0.0, end_position=100.0, time_for_move=2.0),
        ),
        (
            5,
            0.2,
            90.0,
            0.0,
            0.1,
            30.0,
            0,
            False,
            SingleAxisFlyscanInfo(
                start=900,
                num_pulses=5,
                direction=PandaPcompDirection.NEGATIVE,
                pulse_width=1,
                pulse_step=225,
                time_based=False,
            ),
            FlyMotorInfo(start_position=90.0, end_position=0.0, time_for_move=3.0),
        ),
        (
            10,
            0.1,
            0.0,
            50.0,
            10.0,
            25.0,
            0,
            True,
            SingleAxisFlyscanInfo(
                start=0,
                num_pulses=10,
                direction=PandaPcompDirection.POSITIVE,
                pulse_width=0.101,
                pulse_step=0.2,
                time_based=True,
            ),
            FlyMotorInfo(start_position=0.0, end_position=50.0, time_for_move=2.0),
        ),
        (
            1801,
            0.05,
            0.0,
            180.0,
            360 / 70000,
            60.0,
            39240,
            True,
            SingleAxisFlyscanInfo(
                start=39240,
                num_pulses=1801,
                direction=PandaPcompDirection.POSITIVE,
                pulse_width=0.051,
                pulse_step=0.051,
                time_based=True,
            ),
            FlyMotorInfo(start_position=0.0, end_position=180.0, time_for_move=91.851),
        ),
    ],
)
def test_construct_fly_info_models(
    num_pulses,
    max_exposure_time,
    start_position,
    stop_position,
    encoder_resolution,
    max_motor_velocity,
    encoder_pos_at_zero,
    time_based,
    expected_flyer_info,
    expected_motor_info,
):
    flyer_info, motor_info = construct_fly_info_models(
        num_pulses,
        max_exposure_time,
        start_position,
        stop_position,
        encoder_resolution,
        max_motor_velocity,
        encoder_pos_at_zero,
        time_based=time_based,
    )
    assert flyer_info.start == expected_flyer_info.start
    assert flyer_info.num_pulses == expected_flyer_info.num_pulses
    assert flyer_info.direction == expected_flyer_info.direction
    assert flyer_info.pulse_width == pytest.approx(expected_flyer_info.pulse_width)
    assert flyer_info.pulse_step == pytest.approx(expected_flyer_info.pulse_step)
    assert flyer_info.time_based == expected_flyer_info.time_based
    assert motor_info.start_position == pytest.approx(
        expected_motor_info.start_position
    )
    assert motor_info.end_position == pytest.approx(expected_motor_info.end_position)
    assert motor_info.time_for_move == pytest.approx(expected_motor_info.time_for_move)


@pytest.mark.parametrize(
    "num_pulses, max_exposure_time, start_position,"
    " stop_position, encoder_resolution, max_motor_velocity,"
    " encoder_pos_at_zero, expected_match",
    [
        # travel_counts=1000, num_pulses-1=6 -> 1000 % 6 != 0
        (7, 0.1, 0.0, 100.0, 0.1, 50.0, 0, "not evenly divisible"),
        # travel_counts=10, num_pulses-1=10 -> 10 < 10*2=20
        (11, 0.1, 0.0, 1.0, 0.1, 50.0, 0, "less than the minimum required"),
    ],
)
def test_construct_fly_info_models_raises(
    num_pulses,
    max_exposure_time,
    start_position,
    stop_position,
    encoder_resolution,
    max_motor_velocity,
    encoder_pos_at_zero,
    expected_match,
):
    with pytest.raises(ValueError, match=expected_match):
        construct_fly_info_models(
            num_pulses,
            max_exposure_time,
            start_position,
            stop_position,
            encoder_resolution,
            max_motor_velocity,
            encoder_pos_at_zero,
        )
