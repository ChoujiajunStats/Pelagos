"""State boundary shared by deployable VIO/SLAM and simulation audit data."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def _finite_vector(value: Any, width: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (width,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {width} finite values")
    return vector


@dataclass(frozen=True)
class VioState:
    """Minimal state expected from VIO/SLAM by the collection controller."""

    position_ned_m: np.ndarray
    orientation_body_to_ned_xyzw: np.ndarray
    linear_velocity_body_frd_m_s: np.ndarray
    angular_velocity_body_frd_rad_s: np.ndarray
    timestamp_seconds: float
    source: str

    def __post_init__(self) -> None:
        position = _finite_vector(self.position_ned_m, 3, "position_ned_m")
        orientation = _finite_vector(
            self.orientation_body_to_ned_xyzw,
            4,
            "orientation_body_to_ned_xyzw",
        )
        linear_velocity = _finite_vector(
            self.linear_velocity_body_frd_m_s,
            3,
            "linear_velocity_body_frd_m_s",
        )
        angular_velocity = _finite_vector(
            self.angular_velocity_body_frd_rad_s,
            3,
            "angular_velocity_body_frd_rad_s",
        )
        norm = float(np.linalg.norm(orientation))
        if norm <= 1e-9:
            raise ValueError("orientation quaternion cannot be zero")
        timestamp = float(self.timestamp_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_seconds must be finite")
        source = str(self.source).strip()
        if not source:
            raise ValueError("state source is required")
        object.__setattr__(self, "position_ned_m", position.copy())
        object.__setattr__(
            self,
            "orientation_body_to_ned_xyzw",
            orientation / norm,
        )
        object.__setattr__(
            self,
            "linear_velocity_body_frd_m_s",
            linear_velocity.copy(),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_frd_rad_s",
            angular_velocity.copy(),
        )
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class SimulatedVioConfig:
    """Conservative stereo VIO/SLAM error model for simulation collection."""

    latency_seconds: float = 0.08
    correlation_time_seconds: float = 0.50
    position_noise_std_m: tuple[float, float, float] = (0.010, 0.010, 0.015)
    position_drift_std_m_per_sqrt_s: tuple[float, float, float] = (
        0.002,
        0.002,
        0.003,
    )
    orientation_noise_std_rad: tuple[float, float, float] = (
        math.radians(0.20),
        math.radians(0.20),
        math.radians(0.50),
    )
    orientation_drift_std_rad_per_sqrt_s: tuple[float, float, float] = (
        math.radians(0.02),
        math.radians(0.02),
        math.radians(0.05),
    )
    linear_velocity_noise_std_m_s: tuple[float, float, float] = (
        0.015,
        0.015,
        0.020,
    )
    linear_velocity_bias_std_m_s_per_sqrt_s: tuple[float, float, float] = (
        0.001,
        0.001,
        0.002,
    )
    angular_velocity_noise_std_rad_s: tuple[float, float, float] = (
        0.003,
        0.003,
        0.005,
    )

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.latency_seconds)
            or self.latency_seconds < 0.0
            or not math.isfinite(self.correlation_time_seconds)
            or self.correlation_time_seconds <= 0.0
        ):
            raise ValueError("VIO latency/correlation time is invalid")
        for name in (
            "position_noise_std_m",
            "position_drift_std_m_per_sqrt_s",
            "orientation_noise_std_rad",
            "orientation_drift_std_rad_per_sqrt_s",
            "linear_velocity_noise_std_m_s",
            "linear_velocity_bias_std_m_s_per_sqrt_s",
            "angular_velocity_noise_std_rad_s",
        ):
            value = _finite_vector(getattr(self, name), 3, name)
            if np.any(value < 0.0):
                raise ValueError(f"{name} must be nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def _rotation_vector_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1e-12:
        return np.asarray(
            [0.5 * rotation_vector[0], 0.5 * rotation_vector[1],
             0.5 * rotation_vector[2], 1.0],
            dtype=np.float64,
        )
    axis = rotation_vector / angle
    return np.concatenate((axis * math.sin(0.5 * angle), [math.cos(0.5 * angle)]))


class SimulatedVioEstimator:
    """Causal, repeatable GT-to-VIO proxy with latency and correlated drift."""

    SOURCE = "simulated_stereo_vio_slam_from_gt"

    def __init__(
        self,
        config: SimulatedVioConfig = SimulatedVioConfig(),
        *,
        seed: int = 20260804,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._last_timestamp: float | None = None
        self._position_drift = np.zeros(3, dtype=np.float64)
        self._orientation_drift = np.zeros(3, dtype=np.float64)
        self._velocity_bias = np.zeros(3, dtype=np.float64)
        self._position_noise = np.zeros(3, dtype=np.float64)
        self._orientation_noise = np.zeros(3, dtype=np.float64)
        self._velocity_noise = np.zeros(3, dtype=np.float64)
        self._angular_noise = np.zeros(3, dtype=np.float64)
        self._latency_queue: deque[VioState] = deque()

    def _correlated_noise(
        self,
        previous: np.ndarray,
        standard_deviation: tuple[float, float, float],
        dt: float,
    ) -> np.ndarray:
        alpha = math.exp(-dt / self.config.correlation_time_seconds)
        innovation_scale = math.sqrt(max(0.0, 1.0 - alpha * alpha))
        return alpha * previous + innovation_scale * np.asarray(
            standard_deviation, dtype=np.float64
        ) * self._rng.normal(size=3)

    def update(self, truth: VioState) -> VioState:
        timestamp = truth.timestamp_seconds
        if self._last_timestamp is not None and timestamp < self._last_timestamp - 1e-9:
            raise ValueError("VIO truth timestamps must be monotonic")
        # A repeated simulator odometry sample is not a new camera frame.  It
        # must be idempotent: advancing the random walk on duplicate input
        # would create estimator state that depends on callback scheduling.
        if self._last_timestamp is not None and timestamp <= self._last_timestamp + 1e-9:
            if not self._latency_queue:
                raise RuntimeError("VIO estimator has no cached estimate")
            return self._latency_queue[0]
        dt = (
            0.04
            if self._last_timestamp is None
            else max(timestamp - self._last_timestamp, 1e-6)
        )
        self._last_timestamp = timestamp
        sqrt_dt = math.sqrt(dt)
        self._position_drift += np.asarray(
            self.config.position_drift_std_m_per_sqrt_s
        ) * sqrt_dt * self._rng.normal(size=3)
        self._orientation_drift += np.asarray(
            self.config.orientation_drift_std_rad_per_sqrt_s
        ) * sqrt_dt * self._rng.normal(size=3)
        self._velocity_bias += np.asarray(
            self.config.linear_velocity_bias_std_m_s_per_sqrt_s
        ) * sqrt_dt * self._rng.normal(size=3)
        self._position_noise = self._correlated_noise(
            self._position_noise, self.config.position_noise_std_m, dt
        )
        self._orientation_noise = self._correlated_noise(
            self._orientation_noise, self.config.orientation_noise_std_rad, dt
        )
        self._velocity_noise = self._correlated_noise(
            self._velocity_noise, self.config.linear_velocity_noise_std_m_s, dt
        )
        self._angular_noise = self._correlated_noise(
            self._angular_noise, self.config.angular_velocity_noise_std_rad_s, dt
        )

        orientation_error = _rotation_vector_quaternion(
            self._orientation_drift + self._orientation_noise
        )
        estimate = VioState(
            position_ned_m=(
                truth.position_ned_m + self._position_drift + self._position_noise
            ),
            orientation_body_to_ned_xyzw=_quaternion_multiply(
                orientation_error, truth.orientation_body_to_ned_xyzw
            ),
            linear_velocity_body_frd_m_s=(
                truth.linear_velocity_body_frd_m_s
                + self._velocity_bias
                + self._velocity_noise
            ),
            angular_velocity_body_frd_rad_s=(
                truth.angular_velocity_body_frd_rad_s + self._angular_noise
            ),
            timestamp_seconds=timestamp,
            source=self.SOURCE,
        )
        self._latency_queue.append(estimate)
        cutoff = timestamp - self.config.latency_seconds
        while (
            len(self._latency_queue) > 1
            and self._latency_queue[1].timestamp_seconds <= cutoff
        ):
            self._latency_queue.popleft()
        if self._latency_queue[0].timestamp_seconds <= cutoff:
            return self._latency_queue[0]
        # Startup returns the earliest available estimate until the configured
        # history has accumulated; no future truth sample is ever used.
        return self._latency_queue[0]

    def provenance(self) -> dict[str, Any]:
        return {
            "source": self.SOURCE,
            "seed": self.seed,
            "config": self.config.as_dict(),
            "causal": True,
            "uses_future_ground_truth": False,
        }


def ground_truth_odometry_as_vio(message: Any) -> VioState:
    """Adapt Stonefish NED/body-FRD odometry to the future VIO contract."""
    frame_id = str(getattr(message.header, "frame_id", ""))
    if frame_id and frame_id != "world_ned":
        raise ValueError(f"ground-truth odometry frame must be world_ned, got {frame_id}")
    pose = message.pose.pose
    twist = message.twist.twist
    stamp = message.header.stamp
    timestamp = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
    return VioState(
        position_ned_m=np.asarray(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        ),
        orientation_body_to_ned_xyzw=np.asarray(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        ),
        # Stonefish odometry publishes twist in the body FRD frame.
        linear_velocity_body_frd_m_s=np.asarray(
            [twist.linear.x, twist.linear.y, twist.linear.z],
            dtype=np.float64,
        ),
        angular_velocity_body_frd_rad_s=np.asarray(
            [twist.angular.x, twist.angular.y, twist.angular.z],
            dtype=np.float64,
        ),
        timestamp_seconds=timestamp,
        source="stonefish_ground_truth_as_vio_no_noise",
    )


def canonical_odometry_as_vio(
    message: Any,
    *,
    source: str,
    expected_body_frame: str,
) -> VioState:
    """Parse canonical local-NED/body-FRD odometry without frame guessing."""
    frame_id = str(getattr(message.header, "frame_id", ""))
    child_frame_id = str(getattr(message, "child_frame_id", ""))
    if frame_id != "world_ned":
        raise ValueError(f"VIO world frame must be world_ned, got {frame_id}")
    if child_frame_id != expected_body_frame:
        raise ValueError(
            f"VIO body frame must be {expected_body_frame}, got {child_frame_id}"
        )
    pose = message.pose.pose
    twist = message.twist.twist
    stamp = message.header.stamp
    return VioState(
        position_ned_m=np.asarray(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        ),
        orientation_body_to_ned_xyzw=np.asarray(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        ),
        linear_velocity_body_frd_m_s=np.asarray(
            [twist.linear.x, twist.linear.y, twist.linear.z],
            dtype=np.float64,
        ),
        angular_velocity_body_frd_rad_s=np.asarray(
            [twist.angular.x, twist.angular.y, twist.angular.z],
            dtype=np.float64,
        ),
        timestamp_seconds=float(stamp.sec) + 1e-9 * float(stamp.nanosec),
        source=source,
    )
