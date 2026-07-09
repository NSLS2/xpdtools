"""Flyer classes for XPD beamline at NSLS-II."""

import asyncio
from enum import StrEnum

from ophyd_async.core import (
    ConfinedModel,
    FlyerController,
    wait_for_value,
)
from ophyd_async.fastcs.panda import CommonPandaBlocks, PandaPcompDirection


class SingleAxisFlyscanType(StrEnum):
    """Single axis flyscan type.

    Attributes
    ----------
    POSITION_BASED : str
        Position based flyscan
    TIME_BASED : str
        Time based flyscan
    """

    POSITION_BASED = "position_based"
    TIME_BASED = "time_based"


class SingleAxisFlyscanInfo(ConfinedModel):
    """Information for a single axis flyscan.

    Attributes
    ----------
    start : int
        The start position for the flyscan, in encoder counts
    num_pulses : int
        The number of pulses to send during the flyscan
    direction : PandaPcompDirection
        The direction of the flyscan, either positive or negative
    pulse_width : float | int
        The width of each pulse, in counts for position based scans, s for time based
    pulse_step : float | int
        The step between pulses, in counts for position based scans, s for time based
    scan_type : SingleAxisFlyscanType
        The type of flyscan, either position based or time based
    """

    start: int
    num_pulses: int
    direction: PandaPcompDirection
    pulse_width: float | int
    pulse_step: float | int
    scan_type: SingleAxisFlyscanType


class SingleAxisFlyscanController(FlyerController[SingleAxisFlyscanInfo]):
    """Controller for a single axis flyscan."""

    def __init__(self, panda: CommonPandaBlocks) -> None:
        self.panda = panda

    async def prepare(self, value: SingleAxisFlyscanInfo):
        pcomp = self.panda.pcomp[1]
        pulse = self.panda.pulse[1]
        coros = [
            pcomp.dir.set(value.direction),
            pcomp.start.set(value.start),
        ]
        if value.scan_type == SingleAxisFlyscanType.POSITION_BASED:
            coros.extend(
                [
                    pcomp.pulses.set(value.num_pulses),
                    pcomp.width.set(int(value.pulse_width)),
                    pcomp.step.set(int(value.pulse_step)),
                ]
            )
        else:
            coros.extend(
                [
                    pcomp.pulses.set(1),
                    pcomp.width.set(1),
                    pcomp.step.set(2),
                    pulse.pulses.set(value.num_pulses),
                    pulse.width.set(int(value.pulse_width)),
                    pulse.step.set(int(value.pulse_step)),
                ]
            )
        await asyncio.gather(*coros)

    async def kickoff(self) -> None:
        await wait_for_value(self.panda.pcomp[1].active, True, timeout=1)

    async def complete(self, timeout: float | None = None) -> None:
        await wait_for_value(self.panda.pcomp[1].active, False, timeout=timeout)

    async def stop(self):
        await wait_for_value(self.panda.pcomp[1].active, False, timeout=1)
