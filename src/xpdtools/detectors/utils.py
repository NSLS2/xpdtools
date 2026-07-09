from ophyd_async.epics.adcore import AreaDetector


def get_detector_acq_times(detectors: list[AreaDetector]) -> list[float]:
    """Get the acquisition times and periods for a list of detectors.

    Parameters
    ----------
    detectors : list[AreaDetector]
        List of area detectors to query.

    Returns
    -------
    list[tuple[float, float | None]]
        List of tuples containing the acquisition time and period for each detector.
        If the acquisition period is less than the acquisition time, it will be set to None.
    """
    acquisition_periods = []
    for det in detectors:
        acq_time: float = yield from bps.rd(det.driver.acquire_time)
        acq_period: float = yield from bps.rd(det.driver.acquire_period)
        if acq_period < acq_time:
            acq_period = acq_time
    return acquisition_periods
