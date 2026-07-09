"""Acquisition plans for XPD beamline at NSLS-II."""

import bluesky.plan_stubs as bps
from ophyd_async.core import (
    DetectorTrigger,
    FlyMotorInfo,
    StandardFlyer,
    TriggerInfo,
)
from ophyd_async.epics.adcore import ADBaseIO, AreaDetector
from ophyd_async.fastcs.panda import HDFPanda as PandABox
from ophyd_async.fastcs.panda import (
    PandaPcompDirection,
)

from .flyers import (
    SingleAxisFlyscanController,
    SingleAxisFlyscanInfo,
    SingleAxisFlyscanType,
)
from .motors import RotationMotor


def single_axis_flyscan(
    detectors: list[AreaDetector[ADBaseIO]],
    panda: PandABox,
    motor: RotationMotor,
    num_images: int = 1801,
    start: float = 0.0,
    stop: float = 180.0,
    stream_name: str = "tomo",
    acq_time_overhead: float = 0.001,
):
    """Perform a single axis flyscan with the given detectors, PandABox, and motor.

    Parameters
    ----------
    detectors : list[AreaDetector[ADBaseIO]]
        list of area detectors to use in the scan
    panda : PandABox
        PandABox to use for triggering the detectors and motor
    motor : RotationMotor
        RotationMotor to use for the scan
    num_images : int, optional
        Number of images to acquire, by default 1801
    start : float, optional
        Start motor position of the scan, by default 0.0
    stop : float, optional
        Stop motor position of the scan, by default 180.0
    stream_name : str, optional
        Name of the data stream, by default "tomo"
    acq_time_overhead : float, optional
        Overhead time for acquisition for each frame in seconds, by default 0.001
    """

    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    single_axis_panda_flyer = StandardFlyer(SingleAxisFlyscanController(panda))
    all_devices = [*all_detectors, single_axis_panda_flyer, motor]

    # Get the start position in encoder counts
    # encoder_res = yield from bps.rd(motor.encoder_resolution)
    # max_velocity = yield from bps.rd(motor.max_velocity)
    acquisition_periods = []
    for det in detectors:
        acq_time = yield from bps.rd(det.driver.acquire_time)
        acq_period = yield from bps.rd(det.driver.acquire_period)
        if acq_period < acq_time:
            acq_period = None
        acquisition_periods.append((acq_time, acq_period))

    # flyer_info, motor_info = construct_fly_info_models(
    #     num_pulses=num_images,
    #     acquisition_periods=[
    #         (yield from bps.rd(det.driver.acquire_time), None) for det in detectors
    #     ],
    #     start_position=start,
    #     stop_position=stop,
    #     encoder_resolution=encoder_res,
    #     max_motor_velocity=max_velocity,
    #     encoder_pos_at_zero=(yield from bps.rd(motor.encoder_pos_at_zero)),
    # )

    # Get the currently configured exposure times
    exposure_times = []
    for det in detectors:
        exposure_time = yield from bps.rd(det.driver.acquire_time)
        exposure_times.append(exposure_time)

    det_trigger_info = TriggerInfo(
        number_of_events=num_images,
        trigger=DetectorTrigger.EXTERNAL_EDGE,
    )

    panda_trigger_info = TriggerInfo(
        number_of_events=num_images,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
    )

    single_axis_flyscan_info = SingleAxisFlyscanInfo(
        pulse_width=1,
        pulse_step=2,
        start=start_in_counts,
        scan_type=SingleAxisFlyscanType.POSITION_BASED,
        num_pulses=num_images,
        direction=PandaPcompDirection.POSITIVE,
    )

    rotation_motor_info = FlyMotorInfo(
        start_position=start,
        end_position=stop,
        time_for_move=max(exposure_times) * num_images,
    )

    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num_images,
        "plan_name": "single_axis_flyscan",
        "hints": {},
    }
    yield from bps.open_run(md=_md)

    # Stage All!
    yield from bps.stage_all(*all_detectors)

    yield from bps.prepare(motor, rotation_motor_info, group="prepare")

    yield from bps.prepare(
        single_axis_panda_flyer, single_axis_flyscan_info, group="prepare"
    )

    for det in detectors:
        yield from bps.prepare(det, det_trigger_info, group="prepare")

    yield from bps.prepare(panda, panda_trigger_info, group="prepare")

    # TODO: Come up with a way to set a timeout automatically based on the
    # motor move to start position time.
    yield from bps.wait(group="prepare")

    yield from bps.declare_stream(*all_detectors, name=stream_name)

    yield from bps.kickoff_all(*all_devices, wait=True)
    yield from bps.collect_while_completing(
        all_devices,
        all_detectors,
        flush_period=max(1, max(exposure_times)),
        stream_name=stream_name,
    )
    yield from bps.unstage_all(*all_detectors)

    yield from bps.close_run()
