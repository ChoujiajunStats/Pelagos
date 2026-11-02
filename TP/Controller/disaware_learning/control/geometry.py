"""Coordinate and attitude helpers shared by online controllers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry


def body_velocity_to_ned(
    quaternion_xyzw: np.ndarray,
    velocity_body_frd: np.ndarray,
) -> np.ndarray:
    """Rotate a body-FRD velocity into Stonefish's NED world frame."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    velocity = np.asarray(velocity_body_frd, dtype=np.float64)
    if quaternion.shape != (4,) or velocity.shape != (3,):
        raise ValueError("quaternion must be (4,) and velocity must be (3,)")
    norm = float(np.linalg.norm(quaternion))
    if (
        not math.isfinite(norm)
        or norm <= 1e-9
        or not np.all(np.isfinite(velocity))
    ):
        raise ValueError("quaternion and velocity must be finite")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )
    return rotation @ velocity


def odometry_rpy(odometry: Odometry) -> tuple[float, float, float]:
    """Convert an odometry quaternion to roll, pitch, and yaw radians."""
    q = odometry.pose.pose.orientation
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z),
        1.0 - 2.0 * (q.x * q.x + q.y * q.y),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    )
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    return roll, pitch, yaw
