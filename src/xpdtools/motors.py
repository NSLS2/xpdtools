"""Additional motor classes and helper functions for XPD beamline at NSLS-II."""

import asyncio

from ophyd_async.core import (
    DeviceMock,
    callback_on_mock_put,
    default_mock_class,
    derived_signal_r,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.epics.core import epics_signal_r
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


def get_encoder_value_from_pos(
    current_position: float, encoder_resolution: float, encoder_pos_at_zero: int
) -> int:
    """Calculate the encoder value from a motor position.

    Parameters
    ----------
    current_position : float
        The current position of the motor.
    encoder_resolution : float
        The resolution of the encoder in counts per degree.
    encoder_pos_at_zero : int
        The encoder position corresponding to 0 degrees.

    Returns
    -------
    int
        The encoder value corresponding to the given motor position.
    """
    return int(current_position * encoder_resolution + encoder_pos_at_zero)


class VelocityRespectingMotorMock(DeviceMock[AsyncEpicsMotor]):
    """Mock behaviour that respects motor velocity and acceleration time."""

    async def connect(self, device: AsyncEpicsMotor) -> None:
        """Mock signals to simulate a move respecting velocity and acceleration."""
        set_mock_value(device.velocity, 10)
        set_mock_value(device.max_velocity, 100)
        set_mock_value(device.acceleration_time, 1)

        # Motor starts in "done" state (not moving)
        set_mock_value(device.motor_done_move, 1)

        async def _do_move(target: float):
            current = await device.user_readback.get_value()
            velocity = await device.velocity.get_value()
            acceleration_time = await device.acceleration_time.get_value()
            move_time = abs(target - current) / velocity + 2 * acceleration_time
            set_mock_value(device.motor_done_move, 0)
            elapsed = 0.0
            while elapsed < move_time:
                await asyncio.sleep(min(1.0, move_time - elapsed))
                elapsed += 1.0
                fraction = min(elapsed / move_time, 1.0)
                position = current + (target - current) * fraction
                set_mock_value(device.user_readback, position)
            set_mock_value(device.user_readback, target)
            set_mock_value(device.motor_done_move, 1)
            set_mock_put_proceeds(device.user_setpoint, True)

        def _on_setpoint_write(value):
            set_mock_put_proceeds(device.user_setpoint, False)
            asyncio.ensure_future(_do_move(value))

        callback_on_mock_put(device.user_setpoint, _on_setpoint_write)


@default_mock_class(VelocityRespectingMotorMock)
class RotationMotor(AsyncEpicsMotor):
    """A motor that can be used for rotation scans.

    This class is a subclass of the AsyncEpicsMotor class and is used to represent
    a motor that can be used for rotation scans. It has additional attributes and
    methods that are specific to rotation scans.
    """

    def __init__(self, prefix: str, encoder_pos_at_zero: int = 0, name: str = ""):
        super().__init__(prefix, name=name)
        self.encoder_resolution = epics_signal_r(float, prefix + ".ERES")
        self.encoder_counts_per_rev = derived_signal_r(
            self.get_encoder_counts_per_rev,
            derived_units="counts",
            derived_precision=0,
            encoder_resolution=self.encoder_resolution,
        )
        self.encoder_pos_at_zero = encoder_pos_at_zero

    def get_encoder_counts_per_rev(self, encoder_resolution: float) -> int:
        """Calculate the number of encoder counts per revolution.

        Parameters
        ----------
        encoder_resolution : float
            The resolution of the encoder in counts per degree.

        Returns
        -------
        int
            The number of encoder counts per revolution.
        """
        return int(360.0 * encoder_resolution)

    async def get_encoder_value_from_angle(self, angle: float) -> int:
        """Calculate the encoder value from an angle.

        Parameters
        ----------
        angle : float
            The angle in degrees.

        Returns
        -------
        int
            The encoder value corresponding to the given angle.
        """
        encoder_resolution = await self.encoder_resolution.get_value()
        return get_encoder_value_from_pos(
            current_position=angle,
            encoder_resolution=encoder_resolution,
            encoder_pos_at_zero=self.encoder_pos_at_zero,
        )
