from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


FRAME_SIZE = 24
MAIN_TELEMETRY_SIZE = 32
STATUS_TELEMETRY_SIZE = 29
PWM_MIN = 1000
PWM_NEUTRAL = 1500
PWM_MAX = 2000
MOTOR_CHANNELS = ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")
SIX_THRUSTER_CHANNELS = ("L1", "R1", "L3", "R3", "L2", "R2")


def _bounded_number(name: str, value: float, low: float, high: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _byte(name: str, value: int, low: int = 0, high: int = 255) -> int:
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


@dataclass(frozen=True)
class ControlCommand:
    surge: float = 0.0
    yaw: float = 0.0
    sway: float = 0.0
    heave: float = 0.0
    light: int = 0
    heading_hold: bool = False
    depth_hold: bool = False
    gear: int = 1
    gimbal_1: int = 0
    gimbal_2: int = 0
    gripper: int = 0
    armed: bool = False

    def __post_init__(self) -> None:
        for name in ("surge", "yaw", "sway", "heave"):
            _bounded_number(name, getattr(self, name), -1.0, 1.0)
        _byte("light", self.light, 0, 100)
        _byte("gear", self.gear, 1, 3)
        _byte("gimbal_1", self.gimbal_1)
        _byte("gimbal_2", self.gimbal_2)
        _byte("gripper", self.gripper, 0, 1)

    @classmethod
    def neutral(cls) -> "ControlCommand":
        return cls()


def normalized_to_pwm(value: float) -> int:
    value = _bounded_number("axis", value, -1.0, 1.0)
    return round(PWM_NEUTRAL + 500 * value)


def pack_channel_command(command: ControlCommand) -> bytes:
    frame = bytearray(FRAME_SIZE)
    frame[0] = 0xFF
    axes = (command.surge, command.yaw, command.sway, command.heave)
    for index, value in enumerate(axes):
        offset = 1 + index * 2
        frame[offset : offset + 2] = normalized_to_pwm(value).to_bytes(2, "big")
    frame[9] = command.light
    frame[10] = int(command.heading_hold)
    frame[11] = int(command.depth_hold)
    frame[12] = command.gear
    frame[13] = command.gimbal_1
    frame[14] = command.gimbal_2
    frame[15] = command.gripper
    frame[18] = int(command.armed)
    frame[23] = 0xAA
    return bytes(frame)


def mix_six_thrusters(command: ControlCommand) -> tuple[float, ...]:
    """Mix four motion axes into four horizontal and two vertical thrusters."""
    surge, sway, heave, yaw = command.surge, command.sway, command.heave, command.yaw
    raw = (
        yaw - surge + sway,
        -yaw - surge - sway,
        -yaw + surge + sway,
        yaw + surge - sway,
        -heave,
        -heave,
    )
    return tuple(max(-1.0, min(1.0, value)) for value in raw)


def pack_motor_command(
    command: ControlCommand,
    channels: tuple[str, ...] = SIX_THRUSTER_CHANNELS,
    signs: tuple[int, ...] = (1, 1, 1, 1, 1, 1),
) -> bytes:
    """Pack the six installed thrusters into the vendor's eight-slot frame.

    L4 and R4 are not installed on this vehicle and always remain at neutral
    PWM. The frame is still exactly 24 bytes as required by the protocol.
    """
    installed = {"L1", "L2", "L3", "R1", "R2", "R3"}
    if len(channels) != 6 or set(channels) != installed:
        raise ValueError("channels must map the six installed thrusters exactly once")
    if len(signs) != 6 or any(int(sign) not in (-1, 1) for sign in signs):
        raise ValueError("signs must contain six values of +1 or -1")

    slots = [PWM_NEUTRAL] * len(MOTOR_CHANNELS)
    for value, channel, sign in zip(mix_six_thrusters(command), channels, signs):
        slots[MOTOR_CHANNELS.index(channel)] = normalized_to_pwm(value * int(sign))

    frame = bytearray(FRAME_SIZE)
    frame[0] = 0xFF
    for index, pwm in enumerate(slots):
        offset = 1 + index * 2
        frame[offset : offset + 2] = pwm.to_bytes(2, "big")
    frame[17] = command.light
    frame[18] = command.gimbal_1
    frame[19] = command.gimbal_2
    frame[20] = command.gripper
    frame[21] = int(command.armed)
    frame[23] = 0xBB
    return bytes(frame)


def pack_heartbeat() -> bytes:
    return bytes((0xFF,)) + bytes(22) + bytes((0x11,))


def _signed_angle(sign: int, magnitude: int) -> int | None:
    if sign == 0x10:
        return -magnitude
    if sign in (0x00, 0x01):
        return magnitude
    return None


def _bcd4(high: int, low: int) -> int | None:
    nibbles = (high >> 4, high & 0x0F, low >> 4, low & 0x0F)
    if any(value > 9 for value in nibbles):
        return None
    return nibbles[0] * 1000 + nibbles[1] * 100 + nibbles[2] * 10 + nibbles[3]


def _acceleration(sign: int, high: int, low: int) -> float | None:
    magnitude = _bcd4(high, low)
    if magnitude is None:
        return None
    # Real frames use 00 for positive values; the document's positive example
    # uses 01. Both are accepted until the vendor format is fully calibrated.
    if sign not in (0x00, 0x01, 0x10):
        return None
    return (-1 if sign == 0x10 else 1) * magnitude / 1000.0


def _angular_rate(sign_and_hundreds: int, high: int, low: int) -> float | None:
    """Decode the vendor's provisional gyro representation.

    The document's examples disagree, so callers must retain raw_hex and treat
    these values as provisional until a real-device calibration is complete.
    """
    magnitude = _bcd4(high, low)
    if magnitude is None:
        return None
    sign_nibble = sign_and_hundreds >> 4
    hundreds = sign_and_hundreds & 0x0F
    if sign_nibble not in (0, 1) or hundreds > 9:
        return None
    value = hundreds * 100 + magnitude / 100.0
    return -value if sign_nibble == 1 else value


@dataclass(frozen=True)
class MainTelemetry:
    heading_deg: int
    pitch_deg: int | None
    roll_deg: int | None
    depth_m: float
    water_temperature_c: int
    leak_detected: bool
    acceleration_x_g: float | None
    acceleration_y_g: float | None
    acceleration_z_g: float | None
    angular_rate_x_dps: float | None
    angular_rate_y_dps: float | None
    angular_rate_z_dps: float | None
    battery_percent: int
    raw_hex: str
    provisional_imu_decode: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatusTelemetry:
    armed: bool
    heading_hold: bool
    depth_hold: bool
    heading_pid: tuple[int, int, int]
    depth_pid: tuple[int, int, int]
    heading_output: int
    depth_output: int
    thruster_reversed: tuple[bool, ...]
    runtime_seconds: int
    light: int
    gear: int
    gimbal_1: int
    gimbal_2: int
    gripper_open: bool
    raw_hex: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decode_depth_counts(payload: bytes) -> int:
    """The real ROV zero-extends signed 16-bit negative centimetres (00 FF F5 = -11)."""
    if len(payload) != 3:
        raise ValueError("depth field must contain three bytes")
    value = int.from_bytes(payload, "big")
    return value - 0x10000 if 0x8000 <= value <= 0xFFFF else value


def decode_main_telemetry(frame: bytes) -> MainTelemetry:
    if len(frame) != MAIN_TELEMETRY_SIZE or frame[0] != 0xAA or frame[-1] != 0xFF:
        raise ValueError("invalid 32-byte main telemetry frame")
    return MainTelemetry(
        heading_deg=int.from_bytes(frame[1:3], "big"),
        pitch_deg=_signed_angle(frame[3], frame[4]),
        roll_deg=_signed_angle(frame[5], frame[6]),
        depth_m=decode_depth_counts(frame[7:10]) / 100.0,
        water_temperature_c=frame[10],
        leak_detected=frame[11] == 0,
        acceleration_x_g=_acceleration(*frame[12:15]),
        acceleration_y_g=_acceleration(*frame[15:18]),
        acceleration_z_g=_acceleration(*frame[18:21]),
        angular_rate_x_dps=_angular_rate(*frame[21:24]),
        angular_rate_y_dps=_angular_rate(*frame[24:27]),
        angular_rate_z_dps=_angular_rate(*frame[27:30]),
        battery_percent=frame[30],
        raw_hex=frame.hex(" "),
    )


def decode_status_telemetry(frame: bytes) -> StatusTelemetry:
    if len(frame) != STATUS_TELEMETRY_SIZE or frame[0] != 0xAA or frame[-1] != 0xEE:
        raise ValueError("invalid 29-byte status telemetry frame")
    return StatusTelemetry(
        armed=frame[1] != 0,
        heading_hold=frame[2] != 0,
        depth_hold=frame[3] != 0,
        heading_pid=tuple(frame[4:7]),
        depth_pid=tuple(frame[7:10]),
        heading_output=frame[10],
        depth_output=frame[11],
        thruster_reversed=tuple(value != 0 for value in frame[12:20]),
        runtime_seconds=int.from_bytes(frame[20:23], "big"),
        light=frame[23],
        gear=frame[24],
        gimbal_1=frame[25],
        gimbal_2=frame[26],
        gripper_open=frame[27] != 0,
        raw_hex=frame.hex(" "),
    )


def plausible_status_frame(frame: bytes | bytearray) -> bool:
    if len(frame) < STATUS_TELEMETRY_SIZE or frame[0] != 0xAA or frame[28] != 0xEE:
        return False
    return (
        all(value <= 100 for value in frame[4:10])
        and all(value <= 30 for value in frame[10:12])
        and all(value <= 1 for value in frame[12:20])
        and frame[23] <= 100
        and frame[24] <= 3
        and frame[27] <= 1
    )
