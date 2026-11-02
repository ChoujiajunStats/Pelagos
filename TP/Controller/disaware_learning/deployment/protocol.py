"""Pure command framing for the external ROV TCP protocol.

The device accepts a fixed 24-byte motor frame.  The model still owns six
thruster commands; the protocol has two unused slots (L4/R4) on this vehicle.
The channel map is explicit and configurable because motor order and sign
must be verified on the real hardware before arming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FRAME_SIZE = 24
PWM_MIN = 1000
PWM_NEUTRAL = 1500
PWM_MAX = 2000
MOTOR_CHANNELS = ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")
DEFAULT_SIX_MOTOR_CHANNELS = ("L1", "L2", "L3", "R1", "R2", "R3")
COMMAND_WIDTH = 4


@dataclass(frozen=True)
class SixMotorChannelMap:
    """Map model motor order ``m1..m6`` to protocol motor names.

    ``signs`` permits a later per-motor direction calibration without
    changing the learned model.  A zero sign is rejected to prevent silently
    disabling a motor.
    """

    channels: tuple[str, ...] = DEFAULT_SIX_MOTOR_CHANNELS
    signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.channels) != 6 or len(set(self.channels)) != 6:
            raise ValueError("six distinct protocol motor channels are required")
        if any(channel not in MOTOR_CHANNELS for channel in self.channels):
            raise ValueError("unknown protocol motor channel")
        if len(self.signs) != 6 or any(float(sign) not in (-1.0, 1.0) for sign in self.signs):
            raise ValueError("motor signs must contain six values of +1 or -1")


def _pwm(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError("motor PWM must contain six finite values")
    return np.rint(np.clip(result, PWM_MIN, PWM_MAX)).astype(np.uint16)


def command_to_six_motor_normalized(
    command: Iterable[float],
    *,
    roll: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the retained BlueROV2 mixer to ``[surge, sway, heave, yaw]``.

    The first result is the saturated six-motor command used for transmission;
    the second is the raw mixer output for safety/diagnostic logging.  This is
    the same allocation used by the trained six-thruster controller.  Roll is
    exposed for completeness but defaults to zero because both deployment
    controllers currently command four axes.
    """
    values = np.asarray(list(command), dtype=np.float64)
    if values.shape != (COMMAND_WIDTH,) or not np.all(np.isfinite(values)):
        raise ValueError("command must contain four finite axes")
    if not np.isfinite(roll):
        raise ValueError("roll must be finite")
    surge, sway, heave, yaw = values
    raw = np.asarray(
        (
            yaw - surge + sway,
            -yaw - surge - sway,
            -yaw + surge + sway,
            yaw + surge - sway,
            float(roll) - heave,
            -float(roll) - heave,
        ),
        dtype=np.float64,
    )
    return np.clip(raw, -1.0, 1.0), raw


def normalized_to_protocol_pwm(motor_normalized: Iterable[float]) -> np.ndarray:
    """Convert six normalized motor values to the vendor's 1000--2000 PWM."""
    values = np.asarray(list(motor_normalized), dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("motor command must contain six finite values")
    return np.rint(PWM_NEUTRAL + 500.0 * np.clip(values, -1.0, 1.0)).astype(np.int32)


def pack_motor_command(
    motor_pwm: Iterable[float],
    *,
    channel_map: SixMotorChannelMap = SixMotorChannelMap(),
    light: int = 0,
    gimbal_1: int = 0,
    gimbal_2: int = 0,
    gripper: int = 0,
    arm: int = 0,
) -> bytes:
    """Pack six PWM values into the complete 24-byte motor frame.

    The protocol examples encode ``1500`` as ``05 DC``, so motor PWM is sent
    big-endian.  L4/R4 default to neutral and all optional actuator fields
    default to safe zero values.
    """
    values = _pwm(motor_pwm).astype(np.float64)
    values = 1500.0 + np.asarray(channel_map.signs) * (values - 1500.0)
    slots = np.full(8, PWM_NEUTRAL, dtype=np.uint16)
    for value, channel in zip(values, channel_map.channels):
        slots[MOTOR_CHANNELS.index(channel)] = np.uint16(np.clip(value, PWM_MIN, PWM_MAX))
    fields = (light, gimbal_1, gimbal_2, gripper, arm)
    if any(not 0 <= int(value) <= 255 for value in fields):
        raise ValueError("auxiliary protocol fields must be bytes")
    frame = bytearray((0xFF,))
    for value in slots:
        frame.extend(int(value).to_bytes(2, byteorder="big", signed=False))
    frame.extend(int(value) for value in fields)
    frame.append(0)
    frame.append(0xBB)
    if len(frame) != FRAME_SIZE:
        raise AssertionError(f"motor frame has invalid size: {len(frame)}")
    return bytes(frame)


def pack_command(
    command: Iterable[float],
    *,
    channel_map: SixMotorChannelMap = SixMotorChannelMap(),
    roll: float = 0.0,
    light: int = 0,
    gimbal_1: int = 0,
    gimbal_2: int = 0,
    gripper: int = 0,
    arm: int = 0,
) -> bytes:
    """Mix one controller command and pack the complete motor frame."""
    motors, _ = command_to_six_motor_normalized(command, roll=roll)
    return pack_motor_command(
        normalized_to_protocol_pwm(motors),
        channel_map=channel_map,
        light=light,
        gimbal_1=gimbal_1,
        gimbal_2=gimbal_2,
        gripper=gripper,
        arm=arm,
    )


def pack_heartbeat() -> bytes:
    """Pack the protocol heartbeat frame."""
    return bytes((0xFF,)) + bytes(22) + bytes((0x11,))
