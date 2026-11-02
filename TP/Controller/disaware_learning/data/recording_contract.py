"""Pure quality checks for one v15 six-thruster VIO dynamics episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from disaware_learning.data.episode_schema import DEPLOYABLE_OBSERVATION_ARRAYS


@dataclass(frozen=True)
class RecordingContract:
    actual_sample_rate_hz: float
    manual_control_stream_valid: bool
    motor_output_stream_valid: bool
    observation_stream_valid: bool
    applied_thrust_stream_valid: bool
    simulation_truth_audit: dict | None


def _sample_rate(time_values: np.ndarray) -> float:
    steps = np.diff(np.asarray(time_values, dtype=np.float64))
    if not steps.size or not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        return 0.0
    return float(1.0 / np.median(steps))


def _simulation_truth_audit(arrays: Mapping[str, np.ndarray]) -> dict:
    truth_depth_delta = (
        arrays["ground_truth_position_ned_m"][:, 2]
        - arrays["ground_truth_position_ned_m"][0, 2]
    )
    observed_depth_delta = arrays["depth_down_m"] - arrays["depth_down_m"][0]
    depth_error = observed_depth_delta - truth_depth_delta
    rate_error = (
        arrays["ahrs_angular_velocity_body_frd_rad_s"]
        - arrays["ground_truth_angular_velocity_body_frd_rad_s"]
    )
    applied_thrust = arrays["ground_truth_applied_thrust_n"]
    position_error = (
        arrays["vio_position_ned_m"] - arrays["ground_truth_position_ned_m"]
    )
    velocity_error = (
        arrays["vio_linear_velocity_body_frd_m_s"]
        - arrays["ground_truth_velocity_body_frd_m_s"]
    )
    angular_velocity_error = (
        arrays["vio_angular_velocity_body_frd_rad_s"]
        - arrays["ground_truth_angular_velocity_body_frd_rad_s"]
    )
    quaternion_dot = np.abs(
        np.sum(
            arrays["vio_orientation_body_to_ned_xyzw"]
            * arrays["ground_truth_orientation_body_to_ned_xyzw"],
            axis=1,
        )
    )
    orientation_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
    return {
        "affects_pass_fail": False,
        "depth_delta_rmse_m": float(np.sqrt(np.mean(depth_error**2))),
        "angular_velocity_body_rmse_rad_s": np.sqrt(
            np.mean(rate_error**2, axis=0)
        ).tolist(),
        "vio_position_rmse_ned_m": np.sqrt(
            np.mean(position_error**2, axis=0)
        ).tolist(),
        "vio_velocity_rmse_body_frd_m_s": np.sqrt(
            np.mean(velocity_error**2, axis=0)
        ).tolist(),
        "vio_angular_velocity_rmse_body_frd_rad_s": np.sqrt(
            np.mean(angular_velocity_error**2, axis=0)
        ).tolist(),
        "vio_orientation_angle_rmse_rad": float(
            np.sqrt(np.mean(orientation_error**2))
        ),
        "applied_thrust_rms_n": np.sqrt(
            np.mean(applied_thrust**2, axis=0)
        ).tolist(),
        "maximum_abs_applied_thrust_n": float(np.max(np.abs(applied_thrust))),
    }


def evaluate_recording_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    max_control_age: float,
    max_motor_age: float,
    max_vio_age: float,
    max_ahrs_age: float,
    max_imu_age: float,
    max_depth_age: float,
    max_pressure_age: float,
    record_ground_truth: bool,
) -> RecordingContract:
    """Evaluate only signals available on the physical vehicle."""
    manual_valid = bool(
        np.all(np.isfinite(arrays["manual_control_normalized"]))
        and np.all(np.isfinite(arrays["manual_control_pwm"]))
        and np.max(np.abs(arrays["manual_control_normalized"])) <= 1.0001
        and np.allclose(
            arrays["manual_control_normalized"],
            np.clip(
                (arrays["manual_control_pwm"][:, :6] - 1500.0) / 400.0,
                -1.0,
                1.0,
            ),
            atol=1e-12,
        )
        and np.all(arrays["manual_control_age_seconds"] <= max_control_age)
    )
    motor_valid = bool(
        np.all(np.isfinite(arrays["motor_output_normalized"]))
        and np.all(np.isfinite(arrays["motor_output_pwm"]))
        and arrays["motor_output_normalized"].shape[1:] == (6,)
        and arrays["motor_output_pwm"].shape[1:] == (6,)
        and np.max(np.abs(arrays["motor_output_normalized"])) <= 1.0001
        and np.allclose(
            arrays["motor_output_normalized"],
            np.clip(
                (arrays["motor_output_pwm"] - 1500.0) / 400.0,
                -1.0,
                1.0,
            ),
            atol=1e-12,
        )
        and np.all(arrays["motor_output_age_seconds"] <= max_motor_age)
    )
    age_limits = {
        "vio_age_seconds": max_vio_age,
        "ahrs_age_seconds": max_ahrs_age,
        "imu_age_seconds": max_imu_age,
        "depth_age_seconds": max_depth_age,
        "pressure_age_seconds": max_pressure_age,
    }
    observation_valid = bool(
        all(np.all(arrays[name] <= limit) for name, limit in age_limits.items())
        and all(
            np.all(arrays[name])
            for name in (
                "vio_valid",
                "ahrs_valid",
                "imu_valid",
                "depth_valid",
                "pressure_valid",
            )
        )
        and all(
            np.all(np.isfinite(arrays[name]))
            for name in DEPLOYABLE_OBSERVATION_ARRAYS
        )
    )
    applied_thrust_valid = bool(
        not record_ground_truth
        or (
            arrays.get("ground_truth_applied_thrust_n", np.empty((0, 6))).shape
            == (arrays["time"].size, 6)
            and np.all(np.isfinite(arrays["ground_truth_applied_thrust_n"]))
        )
    )
    return RecordingContract(
        actual_sample_rate_hz=_sample_rate(arrays["time"]),
        manual_control_stream_valid=manual_valid,
        motor_output_stream_valid=motor_valid,
        observation_stream_valid=observation_valid,
        applied_thrust_stream_valid=applied_thrust_valid,
        simulation_truth_audit=(
            _simulation_truth_audit(arrays) if record_ground_truth else None
        ),
    )
