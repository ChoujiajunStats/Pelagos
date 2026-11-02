"""Fixed body-velocity and yaw-rate PID for MANUAL data collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from disaware_learning.control.contract import intent_to_manual_control
from disaware_learning.estimation.vio_state import VioState


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("quaternion cannot be zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
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


def quaternion_rpy(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    return np.asarray(
        (
            math.atan2(
                2.0 * (w * x + y * z),
                1.0 - 2.0 * (x * x + y * y),
            ),
            math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            ),
        ),
        dtype=np.float64,
    )


def _vector(value: tuple[float, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def summarize_body_twist_tracking(
    reference: np.ndarray,
    actual: np.ndarray,
    command: np.ndarray,
    dt: float,
    *,
    trim_seconds: float = 2.0,
    maximum_lag_seconds: float = 5.0,
) -> dict[str, Any]:
    """Summarize per-axis response gain, lag, and command smoothness.

    The summary is intentionally descriptive: it does not turn a slow
    excitation into a pass/fail criterion.  A positive lag means the measured
    response follows the reference after that many seconds.
    """
    reference_array = np.asarray(reference, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    command_array = np.asarray(command, dtype=np.float64)
    if (
        reference_array.ndim != 2
        or reference_array.shape[1] != 4
        or actual_array.shape != reference_array.shape
        or command_array.shape != reference_array.shape
        or reference_array.shape[0] < 8
        or not np.all(np.isfinite(reference_array))
        or not np.all(np.isfinite(actual_array))
        or not np.all(np.isfinite(command_array))
    ):
        raise ValueError("tracking arrays must be finite [N, 4] matrices")
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or not math.isfinite(trim_seconds)
        or trim_seconds < 0.0
        or not math.isfinite(maximum_lag_seconds)
        or maximum_lag_seconds < 0.0
    ):
        raise ValueError("tracking timing arguments must be finite and nonnegative")

    trim = min(int(round(trim_seconds / dt)), (reference_array.shape[0] - 4) // 2)
    start, stop = trim, reference_array.shape[0] - trim
    reference_view = reference_array[start:stop]
    actual_view = actual_array[start:stop]
    lag_count = min(int(round(maximum_lag_seconds / dt)), len(reference_view) - 3)
    names = ("surge", "sway", "heave", "yaw")
    axes: dict[str, dict[str, float]] = {}
    for axis, name in enumerate(names):
        reference_axis = reference_view[:, axis]
        actual_axis = actual_view[:, axis]
        reference_rms = float(np.sqrt(np.mean(reference_axis**2)))
        actual_rms = float(np.sqrt(np.mean(actual_axis**2)))
        best_correlation = -2.0
        best_lag = 0
        for lag in range(lag_count + 1):
            if lag:
                reference_segment = reference_axis[:-lag]
                actual_segment = actual_axis[lag:]
            else:
                reference_segment = reference_axis
                actual_segment = actual_axis
            if (
                np.std(reference_segment) <= 1e-12
                or np.std(actual_segment) <= 1e-12
            ):
                correlation = 0.0
            else:
                correlation = float(
                    np.corrcoef(reference_segment, actual_segment)[0, 1]
                )
            if correlation > best_correlation:
                best_correlation = correlation
                best_lag = lag
        if best_lag:
            reference_segment = reference_axis[:-best_lag]
            actual_segment = actual_axis[best_lag:]
        else:
            reference_segment = reference_axis
            actual_segment = actual_axis
        slope = float(
            np.dot(reference_segment, actual_segment)
            / (np.dot(reference_segment, reference_segment) + 1e-12)
        )
        command_rate = np.diff(command_array[:, axis]) / dt
        axes[name] = {
            "reference_rms": reference_rms,
            "actual_rms": actual_rms,
            "rms_gain": actual_rms / (reference_rms + 1e-12),
            "best_lag_seconds": best_lag * dt,
            "best_lag_correlation": best_correlation,
            "lag_aligned_slope": slope,
            "lag_aligned_rmse": float(
                np.sqrt(np.mean((reference_segment - actual_segment) ** 2))
            ),
            "command_peak": float(np.max(np.abs(command_array[:, axis]))),
            "command_rate_p95_per_second": float(
                np.percentile(np.abs(command_rate), 95)
            ),
            "command_rate_max_per_second": float(np.max(np.abs(command_rate))),
        }
    return {
        "trim_seconds": float(trim_seconds),
        "maximum_lag_seconds": float(maximum_lag_seconds),
        "axes": axes,
    }


def _nearest_path_rms(query: np.ndarray, path: np.ndarray) -> float:
    squared_distance_sum = 0.0
    for start in range(0, len(query), 512):
        difference = query[start : start + 512, None, :] - path[None, :, :]
        squared_distance_sum += float(
            np.sum(np.min(np.sum(difference**2, axis=2), axis=1))
        )
    return math.sqrt(squared_distance_sum / len(query))


def summarize_trajectory_tracking(
    reference_position_ned_m: np.ndarray,
    actual_position_ned_m: np.ndarray,
    reference_yaw_rad: np.ndarray,
    actual_yaw_rad: np.ndarray,
) -> dict[str, Any]:
    """Report both time-aligned trajectory error and geometric path similarity."""
    reference_position = np.asarray(reference_position_ned_m, dtype=np.float64)
    actual_position = np.asarray(actual_position_ned_m, dtype=np.float64)
    reference_yaw = np.asarray(reference_yaw_rad, dtype=np.float64)
    actual_yaw = np.asarray(actual_yaw_rad, dtype=np.float64)
    sample_count = reference_position.shape[0]
    if (
        reference_position.ndim != 2
        or reference_position.shape[1] != 3
        or actual_position.shape != reference_position.shape
        or reference_yaw.shape != (sample_count,)
        or actual_yaw.shape != (sample_count,)
        or sample_count < 2
        or not np.all(np.isfinite(reference_position))
        or not np.all(np.isfinite(actual_position))
        or not np.all(np.isfinite(reference_yaw))
        or not np.all(np.isfinite(actual_yaw))
    ):
        raise ValueError("trajectory arrays must be finite [N, 3] and [N] arrays")

    position_error = actual_position - reference_position
    horizontal_error = np.linalg.norm(position_error[:, :2], axis=1)
    position_error_3d = np.linalg.norm(position_error, axis=1)
    yaw_error = np.arctan2(
        np.sin(actual_yaw - reference_yaw),
        np.cos(actual_yaw - reference_yaw),
    )

    def path_length(values: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1)))

    reference_horizontal_length = path_length(reference_position[:, :2])
    actual_horizontal_length = path_length(actual_position[:, :2])
    reference_3d_length = path_length(reference_position)
    actual_3d_length = path_length(actual_position)
    return {
        "time_aligned_position_error": {
            "rmse_ned_m": np.sqrt(np.mean(position_error**2, axis=0)).tolist(),
            "horizontal_rms_m": float(np.sqrt(np.mean(horizontal_error**2))),
            "horizontal_p95_m": float(np.percentile(horizontal_error, 95)),
            "horizontal_max_m": float(np.max(horizontal_error)),
            "position_3d_rms_m": float(np.sqrt(np.mean(position_error_3d**2))),
            "position_3d_p95_m": float(np.percentile(position_error_3d, 95)),
            "position_3d_max_m": float(np.max(position_error_3d)),
            "endpoint_error_ned_m": position_error[-1].tolist(),
            "endpoint_3d_error_m": float(position_error_3d[-1]),
        },
        "path_geometry": {
            "reference_horizontal_length_m": reference_horizontal_length,
            "actual_horizontal_length_m": actual_horizontal_length,
            "horizontal_length_ratio": actual_horizontal_length
            / (reference_horizontal_length + 1e-12),
            "reference_3d_length_m": reference_3d_length,
            "actual_3d_length_m": actual_3d_length,
            "length_3d_ratio": actual_3d_length / (reference_3d_length + 1e-12),
            "actual_to_reference_nearest_xy_rms_m": _nearest_path_rms(
                actual_position[:, :2], reference_position[:, :2]
            ),
            "reference_to_actual_nearest_xy_rms_m": _nearest_path_rms(
                reference_position[:, :2], actual_position[:, :2]
            ),
            "actual_to_reference_nearest_3d_rms_m": _nearest_path_rms(
                actual_position, reference_position
            ),
            "reference_to_actual_nearest_3d_rms_m": _nearest_path_rms(
                reference_position, actual_position
            ),
        },
        "yaw_error": {
            "rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
            "p95_rad": float(np.percentile(np.abs(yaw_error), 95)),
            "max_rad": float(np.max(np.abs(yaw_error))),
            "endpoint_rad": float(yaw_error[-1]),
        },
    }


def summarize_state_estimation_audit(
    actual_position_ned_m: np.ndarray,
    estimated_position_ned_m: np.ndarray,
    actual_yaw_rad: np.ndarray,
    estimated_yaw_rad: np.ndarray,
) -> dict[str, Any]:
    """Compare deployable state with truth without mixing coordinate gauges.

    VINS horizontal position and yaw have an arbitrary local-world gauge. A
    fixed first-sample SE(2) transform maps the truth trajectory into that
    gauge for audit only. The deployable D coordinate comes from pressure and
    shares Stonefish's water-surface datum, so depth is compared directly and
    is deliberately not rebased.
    """
    actual_position = np.asarray(actual_position_ned_m, dtype=np.float64)
    estimated_position = np.asarray(estimated_position_ned_m, dtype=np.float64)
    actual_yaw = np.asarray(actual_yaw_rad, dtype=np.float64)
    estimated_yaw = np.asarray(estimated_yaw_rad, dtype=np.float64)
    sample_count = actual_position.shape[0]
    if (
        actual_position.ndim != 2
        or actual_position.shape[1] != 3
        or estimated_position.shape != actual_position.shape
        or actual_yaw.shape != (sample_count,)
        or estimated_yaw.shape != (sample_count,)
        or sample_count < 2
        or not np.all(np.isfinite(actual_position))
        or not np.all(np.isfinite(estimated_position))
        or not np.all(np.isfinite(actual_yaw))
        or not np.all(np.isfinite(estimated_yaw))
    ):
        raise ValueError("state audit arrays must be finite [N, 3] and [N] arrays")

    yaw_gauge = wrap_angle(float(estimated_yaw[0] - actual_yaw[0]))
    c, s = math.cos(yaw_gauge), math.sin(yaw_gauge)
    truth_to_estimate = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    aligned_actual_horizontal = estimated_position[0, :2] + (
        truth_to_estimate
        @ (actual_position[:, :2] - actual_position[0, :2]).T
    ).T
    horizontal_axis_error = estimated_position[:, :2] - aligned_actual_horizontal
    depth_error = estimated_position[:, 2] - actual_position[:, 2]
    position_error = np.column_stack((horizontal_axis_error, depth_error))
    horizontal_norm = np.linalg.norm(horizontal_axis_error, axis=1)
    position_norm = np.linalg.norm(position_error, axis=1)
    aligned_actual_yaw = actual_yaw + yaw_gauge
    yaw_error = np.arctan2(
        np.sin(estimated_yaw - aligned_actual_yaw),
        np.cos(estimated_yaw - aligned_actual_yaw),
    )
    return {
        "affects_control": False,
        "affects_tracking_score": False,
        "alignment": {
            "horizontal_and_yaw": "fixed first-sample SE(2) gauge alignment",
            "depth": "no alignment; pressure and truth share water-surface datum",
            "yaw_gauge_rotation_rad": yaw_gauge,
        },
        "position_rmse_estimate_frame_ned_m": np.sqrt(
            np.mean(position_error**2, axis=0)
        ).tolist(),
        "horizontal_position_rmse_m": float(
            np.sqrt(np.mean(horizontal_norm**2))
        ),
        "pressure_depth_rmse_m": float(np.sqrt(np.mean(depth_error**2))),
        "position_rmse_3d_m": float(np.sqrt(np.mean(position_norm**2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
    }


@dataclass(frozen=True)
class BodyTwistPidConfig:
    """Transparent fixed gains shared by every physics domain."""

    position_kp_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    maximum_position_correction_m_s: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    velocity_kp: tuple[float, float, float] = (0.45, 0.45, 0.45)
    velocity_ki: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Calibrated on the nominal six-thruster plant with the deployable
    # reference scaled to the available command and motor envelope.
    velocity_feedforward: tuple[float, float, float] = (2.45, 2.65, 3.05)
    velocity_acceleration_feedforward: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_integral_limit: tuple[float, float, float] = (0.35, 0.35, 0.30)
    yaw_position_kp: float = 0.0
    maximum_yaw_rate_correction_rad_s: float = 0.0
    yaw_rate_kp: float = 0.25
    yaw_rate_ki: float = 0.0
    yaw_rate_feedforward: float = 0.29
    yaw_rate_integral_limit: float = 0.35
    yaw_command_slew_rate_per_second: float = 0.80
    roll_kp: float = 0.55
    roll_rate_kd: float = 0.14
    command_limit: float = 0.50

    def __post_init__(self) -> None:
        vector_names = (
            "position_kp_body",
            "maximum_position_correction_m_s",
            "velocity_kp",
            "velocity_ki",
            "velocity_feedforward",
            "velocity_acceleration_feedforward",
            "velocity_integral_limit",
        )
        for name in vector_names:
            values = _vector(getattr(self, name), name)
            if np.any(values < 0.0):
                raise ValueError(f"{name} cannot contain negative values")
        scalar_names = (
            "yaw_position_kp",
            "maximum_yaw_rate_correction_rad_s",
            "yaw_rate_kp",
            "yaw_rate_ki",
            "yaw_rate_feedforward",
            "yaw_rate_integral_limit",
            "yaw_command_slew_rate_per_second",
            "roll_kp",
            "roll_rate_kd",
            "command_limit",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0.0 < self.command_limit <= 1.0:
            raise ValueError("command_limit must be in (0, 1]")
        if self.yaw_command_slew_rate_per_second <= 0.0:
            raise ValueError("yaw_command_slew_rate_per_second must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def zero_twist_settling_pid_config() -> BodyTwistPidConfig:
    """Conservative MANUAL controller for a short stationary-history window."""

    return BodyTwistPidConfig(
        position_kp_body=(0.0, 0.0, 0.75),
        maximum_position_correction_m_s=(0.0, 0.0, 0.12),
        velocity_feedforward=(0.0, 0.0, 0.0),
        velocity_kp=(1.0, 1.0, 2.50),
        velocity_ki=(0.05, 0.05, 0.25),
        velocity_integral_limit=(0.20, 0.20, 0.30),
        yaw_position_kp=0.25,
        maximum_yaw_rate_correction_rad_s=0.08,
        yaw_rate_kp=0.15,
        yaw_rate_ki=0.02,
        yaw_rate_feedforward=0.0,
        roll_kp=0.55,
        roll_rate_kd=0.14,
        command_limit=0.35,
    )


@dataclass(frozen=True)
class BodyTwistReference:
    position_preview_ned_m: np.ndarray
    yaw_preview_rad: float
    linear_velocity_body_frd_m_s: np.ndarray
    yaw_rate_rad_s: float
    linear_acceleration_body_frd_m_s2: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        position = np.asarray(self.position_preview_ned_m, dtype=np.float64)
        velocity = np.asarray(self.linear_velocity_body_frd_m_s, dtype=np.float64)
        acceleration = np.asarray(
            self.linear_acceleration_body_frd_m_s2, dtype=np.float64
        )
        if position.shape != (3,) or velocity.shape != (3,) or acceleration.shape != (3,):
            raise ValueError(
                "reference position, velocity, and acceleration must be three-vectors"
            )
        if not (
            np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))
            and np.all(np.isfinite(acceleration))
            and math.isfinite(float(self.yaw_preview_rad))
            and math.isfinite(float(self.yaw_rate_rad_s))
        ):
            raise ValueError("reference must be finite")
        object.__setattr__(self, "position_preview_ned_m", position.copy())
        object.__setattr__(self, "linear_velocity_body_frd_m_s", velocity.copy())
        object.__setattr__(
            self,
            "linear_acceleration_body_frd_m_s2",
            acceleration.copy(),
        )


@dataclass(frozen=True)
class BodyTwistPidOutput:
    manual_control_normalized: np.ndarray
    diagnostics: dict[str, Any]


class BodyTwistVelocityRatePid:
    """Body-velocity/yaw-rate tracker with optional disabled-by-default outer loops."""

    def __init__(self, config: BodyTwistPidConfig = BodyTwistPidConfig()) -> None:
        self.config = config
        self.linear_velocity_integral = np.zeros(3, dtype=np.float64)
        self.yaw_rate_integral = 0.0
        self.previous_yaw_command = 0.0

    def reset(self) -> None:
        self.linear_velocity_integral.fill(0.0)
        self.yaw_rate_integral = 0.0
        self.previous_yaw_command = 0.0

    def step(
        self,
        reference: BodyTwistReference,
        state: VioState,
        dt: float,
    ) -> BodyTwistPidOutput:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("PID dt must be positive and finite")
        config = self.config
        rotation_body_to_ned = quaternion_rotation_matrix(
            state.orientation_body_to_ned_xyzw
        )
        position_error_ned = reference.position_preview_ned_m - state.position_ned_m
        position_error_body = rotation_body_to_ned.T @ position_error_ned
        position_correction = np.clip(
            _vector(config.position_kp_body, "position_kp_body") * position_error_body,
            -_vector(
                config.maximum_position_correction_m_s,
                "maximum_position_correction_m_s",
            ),
            _vector(
                config.maximum_position_correction_m_s,
                "maximum_position_correction_m_s",
            ),
        )
        desired_velocity = reference.linear_velocity_body_frd_m_s + position_correction
        velocity_error = desired_velocity - state.linear_velocity_body_frd_m_s
        previous_linear_velocity_integral = self.linear_velocity_integral.copy()
        integral_limit = _vector(
            config.velocity_integral_limit,
            "velocity_integral_limit",
        )
        self.linear_velocity_integral = np.clip(
            self.linear_velocity_integral + velocity_error * dt,
            -integral_limit,
            integral_limit,
        )
        linear_feedforward_term = (
            _vector(config.velocity_feedforward, "velocity_feedforward")
            * reference.linear_velocity_body_frd_m_s
        )
        linear_acceleration_feedforward_term = (
            _vector(
                config.velocity_acceleration_feedforward,
                "velocity_acceleration_feedforward",
            )
            * reference.linear_acceleration_body_frd_m_s2
        )
        linear_proportional_term = (
            _vector(config.velocity_kp, "velocity_kp") * velocity_error
        )
        linear_integral_term = (
            _vector(config.velocity_ki, "velocity_ki") * self.linear_velocity_integral
        )
        intent = (
            linear_feedforward_term
            + linear_acceleration_feedforward_term
            + linear_proportional_term
            + linear_integral_term
        )

        rpy = quaternion_rpy(state.orientation_body_to_ned_xyzw)
        yaw_error = wrap_angle(float(reference.yaw_preview_rad) - float(rpy[2]))
        previous_yaw_rate_integral = float(self.yaw_rate_integral)
        yaw_correction = float(
            np.clip(
                config.yaw_position_kp * yaw_error,
                -config.maximum_yaw_rate_correction_rad_s,
                config.maximum_yaw_rate_correction_rad_s,
            )
        )
        desired_yaw_rate = float(reference.yaw_rate_rad_s) + yaw_correction
        yaw_rate_error = desired_yaw_rate - float(
            state.angular_velocity_body_frd_rad_s[2]
        )
        self.yaw_rate_integral = float(
            np.clip(
                self.yaw_rate_integral + yaw_rate_error * dt,
                -config.yaw_rate_integral_limit,
                config.yaw_rate_integral_limit,
            )
        )
        yaw_feedforward_term = config.yaw_rate_feedforward * float(
            reference.yaw_rate_rad_s
        )
        yaw_proportional_term = config.yaw_rate_kp * yaw_rate_error
        yaw_integral_term = config.yaw_rate_ki * self.yaw_rate_integral
        yaw_command = yaw_feedforward_term + yaw_proportional_term + yaw_integral_term

        # Do not keep integrating while a command is saturated in the same
        # direction as its tracking error.  The collector applies a separate
        # slew limit, so this conditional anti-windup is the only protection
        # available inside the controller when a fast reference exceeds the
        # vehicle bandwidth.
        linear_saturated = np.abs(intent) > config.command_limit
        linear_pushes_out = intent * velocity_error > 0.0
        linear_rollback = linear_saturated & linear_pushes_out
        if np.any(linear_rollback):
            self.linear_velocity_integral[linear_rollback] = (
                previous_linear_velocity_integral[linear_rollback]
            )
            linear_integral_term = (
                _vector(config.velocity_ki, "velocity_ki")
                * self.linear_velocity_integral
            )
            intent = (
                linear_feedforward_term
                + linear_acceleration_feedforward_term
                + linear_proportional_term
                + linear_integral_term
            )

        yaw_saturated = abs(yaw_command) > config.command_limit
        yaw_pushes_out = yaw_command * yaw_rate_error > 0.0
        if yaw_saturated and yaw_pushes_out:
            self.yaw_rate_integral = previous_yaw_rate_integral
            yaw_integral_term = config.yaw_rate_ki * self.yaw_rate_integral
            yaw_command = (
                yaw_feedforward_term + yaw_proportional_term + yaw_integral_term
            )
        yaw_command_before_slew = float(yaw_command)
        yaw_command_step = config.yaw_command_slew_rate_per_second * dt
        yaw_command = self.previous_yaw_command + np.clip(
            yaw_command - self.previous_yaw_command,
            -yaw_command_step,
            yaw_command_step,
        )
        self.previous_yaw_command = float(yaw_command)
        four_dof_intent = np.asarray(
            # Body FRD velocity uses +z down, while ArduSub's positive
            # throttle convention is heave-up.
            [intent[0], intent[1], -intent[2], yaw_command],
            dtype=np.float64,
        )
        manual = intent_to_manual_control(four_dof_intent)
        manual[1] = -config.roll_kp * float(rpy[0]) - config.roll_rate_kd * float(
            state.angular_velocity_body_frd_rad_s[0]
        )
        unclipped = manual.copy()
        manual = np.clip(manual, -config.command_limit, config.command_limit)
        diagnostics: dict[str, Any] = {
            "position_error_ned_m": position_error_ned.copy(),
            "position_error_body_m": position_error_body.copy(),
            "position_correction_body_m_s": position_correction.copy(),
            "position_correction_saturated": bool(
                np.any(
                    np.abs(
                        _vector(config.position_kp_body, "position_kp_body")
                        * position_error_body
                    )
                    > _vector(
                        config.maximum_position_correction_m_s,
                        "maximum_position_correction_m_s",
                    )
                )
            ),
            "reference_velocity_body_m_s": (
                reference.linear_velocity_body_frd_m_s.copy()
            ),
            "reference_acceleration_body_m_s2": (
                reference.linear_acceleration_body_frd_m_s2.copy()
            ),
            "desired_velocity_body_m_s": desired_velocity.copy(),
            "measured_velocity_body_m_s": (state.linear_velocity_body_frd_m_s.copy()),
            "velocity_error_body_m_s": velocity_error.copy(),
            "velocity_integral": self.linear_velocity_integral.copy(),
            "velocity_integral_saturated": bool(
                np.any(np.abs(self.linear_velocity_integral) >= integral_limit - 1e-12)
            ),
            "linear_intent_unclipped": intent.copy(),
            "linear_feedforward_term": linear_feedforward_term.copy(),
            "linear_acceleration_feedforward_term": (
                linear_acceleration_feedforward_term.copy()
            ),
            "linear_proportional_term": linear_proportional_term.copy(),
            "linear_integral_term": linear_integral_term.copy(),
            "linear_anti_windup_axes": np.flatnonzero(linear_rollback).tolist(),
            "yaw_error_rad": yaw_error,
            "reference_yaw_rate_rad_s": float(reference.yaw_rate_rad_s),
            "desired_yaw_rate_rad_s": desired_yaw_rate,
            "measured_yaw_rate_rad_s": float(state.angular_velocity_body_frd_rad_s[2]),
            "yaw_rate_error_rad_s": yaw_rate_error,
            "yaw_rate_integral": self.yaw_rate_integral,
            "yaw_command_unclipped": float(yaw_command),
            "yaw_command_before_slew": yaw_command_before_slew,
            "yaw_command_slew_rate_per_second": config.yaw_command_slew_rate_per_second,
            "yaw_feedforward_term": float(yaw_feedforward_term),
            "yaw_proportional_term": float(yaw_proportional_term),
            "yaw_integral_term": float(yaw_integral_term),
            "yaw_correction_saturated": bool(
                abs(config.yaw_position_kp * yaw_error)
                >= config.maximum_yaw_rate_correction_rad_s - 1e-12
            ),
            "yaw_anti_windup": bool(yaw_saturated and yaw_pushes_out),
            "roll_rad": float(rpy[0]),
            "pitch_rad": float(rpy[1]),
            "yaw_rad": float(rpy[2]),
            "unclipped_manual_control": unclipped,
            "manual_saturated": bool(np.any(np.abs(unclipped) > config.command_limit)),
            "state_source": state.source,
        }
        return BodyTwistPidOutput(
            manual_control_normalized=manual,
            diagnostics=diagnostics,
        )
