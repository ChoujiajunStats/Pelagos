"""External-ROV sensor decoding and VIO observation construction.

The vendor frame has no horizontal position or linear velocity.  Those fields
must come from an external VIO/odometry source.  This module keeps that
boundary explicit: protocol data supplies depth, gyro, and accelerometer;
VIO supplies position, orientation, and body velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from disaware_learning.estimation.vio_state import VioState
from disaware_learning.learning.motor_jepa_data import (
    GRAVITY_M_S2,
    gravity_compensated_body_acceleration,
    quaternion_to_rotation6d,
)


def _vector(values: Iterable[float], width: int, name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {width} finite values")
    return result


def _bcd_integer(values: bytes) -> int:
    digits: list[int] = []
    for value in values:
        high, low = value >> 4, value & 0x0F
        if high > 9 or low > 9:
            raise ValueError("invalid packed-BCD sensor field")
        digits.extend((high, low))
    output = 0
    for digit in digits:
        output = output * 10 + digit
    return output


def _signed_bcd(sign: int, magnitude: bytes, *, scale: float) -> float:
    if sign not in (0x00, 0x01, 0x10, 0x11):
        raise ValueError(f"unknown signed sensor marker: 0x{sign:02x}")
    value = _bcd_integer(magnitude) * float(scale)
    return -value if sign in (0x10, 0x11) else value


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RovSensorDecodeConfig:
    """Explicit units for the vendor sensor frame.

    The document specifies depth as centimeters and gives packed-BCD examples
    for acceleration/gyro.  Yaw has no decimal-scale example, so its scale is
    configurable and defaults to one degree per raw count.
    """

    yaw_degrees_per_count: float = 1.0
    depth_meters_per_count: float = 0.01
    acceleration_g_per_bcd_count: float = 0.001
    gyro_deg_s_per_bcd_count: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "yaw_degrees_per_count",
            "depth_meters_per_count",
            "acceleration_g_per_bcd_count",
            "gyro_deg_s_per_bcd_count",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class RovSensorSample:
    """Decoded values from one 32-byte vendor sensor frame."""

    yaw_rad: float
    pitch_rad: float
    roll_rad: float
    depth_m: float
    acceleration_specific_force_body_m_s2: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    battery_percent: float

    def __post_init__(self) -> None:
        scalar_names = ("yaw_rad", "pitch_rad", "roll_rad", "depth_m", "battery_percent")
        for name in scalar_names:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(
            self,
            "acceleration_specific_force_body_m_s2",
            _vector(self.acceleration_specific_force_body_m_s2, 3, "acceleration"),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_rad_s",
            _vector(self.angular_velocity_body_rad_s, 3, "angular velocity"),
        )


def decode_sensor_frame(
    frame: bytes,
    *,
    config: RovSensorDecodeConfig = RovSensorDecodeConfig(),
) -> RovSensorSample:
    """Decode one documented 32-byte ``0xAA ... 0xFF`` sensor frame."""
    if not isinstance(frame, bytes) or len(frame) != 32:
        raise ValueError("sensor frame must be exactly 32 bytes")
    if frame[0] != 0xAA or frame[-1] != 0xFF:
        raise ValueError("sensor frame has invalid header or tail")
    yaw = int.from_bytes(frame[1:3], "big") * config.yaw_degrees_per_count
    # Pitch/roll are documented as sign byte + one packed-BCD degree byte
    # (``10 13`` means -13 degrees), unlike the binary depth field.
    pitch = _signed_bcd(frame[3], frame[4:5], scale=1.0)
    roll = _signed_bcd(frame[5], frame[6:7], scale=1.0)
    depth = int.from_bytes(frame[7:10], "big") * config.depth_meters_per_count
    accel_g = np.asarray(
        (
            _signed_bcd(frame[12], frame[13:15], scale=config.acceleration_g_per_bcd_count),
            _signed_bcd(frame[15], frame[16:18], scale=config.acceleration_g_per_bcd_count),
            _signed_bcd(frame[18], frame[19:21], scale=config.acceleration_g_per_bcd_count),
        ),
        dtype=np.float64,
    )
    gyro_deg_s = np.asarray(
        (
            _signed_bcd(frame[21], frame[22:24], scale=config.gyro_deg_s_per_bcd_count),
            _signed_bcd(frame[24], frame[25:27], scale=config.gyro_deg_s_per_bcd_count),
            _signed_bcd(frame[27], frame[28:30], scale=config.gyro_deg_s_per_bcd_count),
        ),
        dtype=np.float64,
    )
    return RovSensorSample(
        yaw_rad=math.radians(yaw),
        pitch_rad=math.radians(pitch),
        roll_rad=math.radians(roll),
        depth_m=float(depth),
        acceleration_specific_force_body_m_s2=accel_g * GRAVITY_M_S2,
        angular_velocity_body_rad_s=np.deg2rad(gyro_deg_s),
        battery_percent=float(frame[30]),
    )


@dataclass(frozen=True)
class ExternalVioPose:
    """The position/orientation/velocity portion absent from the ROV frame."""

    position_ned_m: np.ndarray
    orientation_body_to_ned_xyzw: np.ndarray
    linear_velocity_body_frd_m_s: np.ndarray
    timestamp_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_ned_m", _vector(self.position_ned_m, 3, "position"))
        orientation = _vector(self.orientation_body_to_ned_xyzw, 4, "orientation")
        norm = float(np.linalg.norm(orientation))
        if norm <= 1e-9:
            raise ValueError("orientation quaternion cannot be zero")
        object.__setattr__(self, "orientation_body_to_ned_xyzw", orientation / norm)
        object.__setattr__(self, "linear_velocity_body_frd_m_s", _vector(self.linear_velocity_body_frd_m_s, 3, "velocity"))
        if not math.isfinite(float(self.timestamp_seconds)):
            raise ValueError("timestamp_seconds must be finite")


class ExternalRovObservationAdapter:
    """Combine a vendor frame and external VIO into deployable model inputs."""

    def __init__(self, *, depth_anchor_m: float | None = None) -> None:
        self.depth_anchor_m = depth_anchor_m

    def reset(self) -> None:
        self.depth_anchor_m = None

    def observe(
        self,
        sensor: RovSensorSample,
        vio: ExternalVioPose,
        *,
        source: str = "external_vio+rov_tcp_sensor",
    ) -> tuple[VioState, np.ndarray]:
        if self.depth_anchor_m is None:
            self.depth_anchor_m = float(sensor.depth_m)
        rotation6d = quaternion_to_rotation6d(
            vio.orientation_body_to_ned_xyzw.reshape(1, 4)
        )[0]
        dynamic_accel = gravity_compensated_body_acceleration(
            sensor.acceleration_specific_force_body_m_s2.reshape(1, 3),
            vio.orientation_body_to_ned_xyzw.reshape(1, 4),
        )[0]
        observation = np.concatenate(
            (
                rotation6d,
                vio.linear_velocity_body_frd_m_s,
                sensor.angular_velocity_body_rad_s,
                dynamic_accel,
                [sensor.depth_m - self.depth_anchor_m],
            )
        ).astype(np.float32)
        if observation.shape != (16,) or not np.all(np.isfinite(observation)):
            raise ValueError("constructed JEPA observation is invalid")
        state = VioState(
            position_ned_m=vio.position_ned_m,
            orientation_body_to_ned_xyzw=vio.orientation_body_to_ned_xyzw,
            linear_velocity_body_frd_m_s=vio.linear_velocity_body_frd_m_s,
            angular_velocity_body_frd_rad_s=sensor.angular_velocity_body_rad_s,
            timestamp_seconds=vio.timestamp_seconds,
            source=source,
        )
        return state, observation
