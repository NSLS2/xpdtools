"""Utility functions for xpdtools detectors."""

from collections.abc import Generator
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.utils import Msg
from ophyd_async.epics.adcore import AreaDetector


def get_detector_acq_times(
    detectors: list[AreaDetector],
) -> Generator[Msg, Any, list[tuple[float, float]]]:
    """Get the acquisition times and periods for a list of detectors.

    Parameters
    ----------
    detectors : list[AreaDetector]
        List of area detectors to query.

    Returns
    -------
    list[tuple[float, float]]
        List of tuples containing the acquisition time and period
        for each detector. If the acquisition period is less than
        the acquisition time, it will be set to the acquisition time.
    """
    acquisition_periods: list[tuple[float, float]] = []
    for det in detectors:
        acq_time: float = yield from bps.rd(det.driver.acquire_time)
        acq_period: float = yield from bps.rd(det.driver.acquire_period)
        if acq_period < acq_time:
            acq_period = acq_time
        acquisition_periods.append((acq_time, acq_period))
    return acquisition_periods
