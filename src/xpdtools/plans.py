from .motors import RotationMotor, get_encoder_value_from_angle
from .flyers import SingleAxisFlyscanType, SingleAxisFlyscanInfo, SingleAxisFlyscanController
from ophyd_async.fastcs.panda import HDFPanda as PandABox
from bluesky.protocols import Flyable, Collectable
from ophyd_async.core import StandardDetector, TriggerInfo, FlyMotorInfo, DetectorTrigger, DEFAULT_TIMEOUT, StandardFlyer
from ophyd_async.epics.adcore import ADBaseIO, AreaDetector
from ophyd_async.fastcs.panda import PcompInfo, StaticPcompTriggerLogic, PandaPcompDirection


import bluesky.plan_stubs as bps
import bluesky.plans as bp


def single_axis_flyscan(
    detectors: list[AreaDetector[ADBaseIO]],
    panda: PandABox,
    motor: RotationMotor,
    num_images: int = 1801,
    start: float = 0.0,
    stop: float = 180.0,
    stream_name: str = "tomo",
):
    """Simple hardware triggered flyscan tomography

    Parameters
    ----------
    detectors : list[AreaDetector[ADBaseIO]]
        list of area detectors to use in the scan
    num_images : int
        total number of camera images to collect during the scan
    start_deg : float (optional)
        starting point in degrees
    stop_deg : float (optional)
        stopping point in degrees
    lead_angle : float (optional)
        the angle in degrees to be used to move motor to -lead_angle before 'start_deg' and +lead_angle after 'stop_deg'
    reset_speed : float
        speed of the rotary motor during reset movements, in deg/s
    use_shutter : bool
        whether to use/check the shutter during the scan
    """

    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    single_axis_panda_flyer = StandardFlyer(SingleAxisFlyscanController(panda))
    all_devices = [*all_detectors, single_axis_panda_flyer, motor]

    # Get the start position in encoder counts
    eres = yield from bps.rd(motor.encoder_resolution)
    start_in_counts = get_encoder_value_from_angle(start, eres, motor.encoder_pos_at_zero)

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
        pulse_width = 1,
        pulse_step = 2,
        start = start_in_counts,
        scan_type = SingleAxisFlyscanType.POSITION_BASED,
        num_pulses = num_images,
        direction = PandaPcompDirection.POSITIVE,
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

    yield from bps.prepare(single_axis_panda_flyer, single_axis_flyscan_info, group="prepare")

    for det in detectors:
        yield from bps.prepare(det, det_trigger_info, group="prepare")

    yield from bps.prepare(panda, panda_trigger_info, group="prepare")

    # TODO: Come up with a way to set a timeout automatically based on the 
    # motor move to start position time.
    yield from bps.wait(group="prepare")

    yield from bps.declare_stream(*all_detectors, name=stream_name)

    yield from bps.kickoff_all(*all_devices, wait=True)
    yield from bps.collect_while_completing(
        all_devices, all_detectors, flush_period=max(1, max(exposure_times)), stream_name=stream_name
    )
    yield from bps.unstage_all(*all_detectors)

    yield from bps.close_run()
