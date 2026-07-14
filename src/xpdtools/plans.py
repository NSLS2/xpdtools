"""Acquisition plans for XPD beamline at NSLS-II."""

import bluesky.plan_stubs as bps
from ophyd_async.core import (
    DetectorTrigger,
    StandardFlyer,
    TriggerInfo,
)
from ophyd_async.epics.adcore import ADBaseIO, AreaDetector
from ophyd_async.fastcs.panda import HDFPanda as PandABox

from xpdtools.detectors.utils import get_detector_acq_times

from .flyers import (
    SingleAxisFlyscanController,
    SingleAxisFlyscanType,
    construct_fly_info_models,
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
    flyscan_type: SingleAxisFlyscanType = SingleAxisFlyscanType.POSITION_BASED,
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
    flyscan_type : SingleAxisFlyscanType, default SingleAxisFlyscanType.POSITION_BASED
        Type of flyscan to perform, by default SingleAxisFlyscanType.POSITION_BASED
    """
    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    single_axis_panda_flyer = StandardFlyer(SingleAxisFlyscanController(panda))
    all_devices = [*all_detectors, single_axis_panda_flyer, motor]

    # Get the start position in encoder counts
    encoder_res = yield from bps.rd(motor.encoder_resolution)
    max_velocity = yield from bps.rd(motor.max_velocity)

    acquisition_periods = yield from get_detector_acq_times(detectors)
    max_acq_period = max(acquisition_periods)

    det_trigger_info = TriggerInfo(
        number_of_events=num_images,
        trigger=DetectorTrigger.EXTERNAL_EDGE,
    )

    panda_trigger_info = TriggerInfo(
        number_of_events=num_images,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
    )

    flyer_info, motor_info = construct_fly_info_models(
        num_pulses=num_images,
        max_exposure_time=max_acq_period,
        start_position=start,
        stop_position=stop,
        encoder_resolution=encoder_res,
        max_motor_velocity=max_velocity,
        encoder_pos_at_zero=motor.encoder_pos_at_zero,
        acq_time_overhead=acq_time_overhead,
        flyscan_type=flyscan_type,
    )

    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num_images,
        "plan_name": "single_axis_flyscan",
        "hints": {},
    }
    yield from bps.open_run(md=_md)

    yield from bps.stage_all(*all_detectors)

    yield from bps.prepare(motor, motor_info, group="prepare")
    yield from bps.prepare(single_axis_panda_flyer, flyer_info, group="prepare")

    for det in detectors:
        yield from bps.prepare(det, det_trigger_info, group="prepare")

    yield from bps.prepare(panda, panda_trigger_info, group="prepare")

    # TODO: Come up with a way to set a timeout automatically based on the
    # motor move to start position time.
    yield from bps.wait(group="prepare")

    yield from bps.declare_stream(*all_detectors, name=stream_name)

    yield from bps.kickoff_all(*all_devices, wait=True)

    flush_period = max(1, max_acq_period)
    yield from bps.collect_while_completing(
        all_devices,
        all_detectors,
        flush_period=flush_period,
        stream_name=stream_name,
    )
    yield from bps.unstage_all(*all_detectors)

    yield from bps.close_run()
