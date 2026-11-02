"""Validate one v15 MANUAL six-thruster VIO dynamics episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from disaware_learning.collection.physics_domain import (
    DOMAIN_BANK_PROTOCOL_ID,
    DOMAIN_PROTOCOL_ID,
)
from disaware_learning.collection.current_generator import CURRENT_MODEL_ID
from disaware_learning.control.contract import (
    ACTUATOR_COUNT,
    ARDUSUB_NOMINAL_PARAMETERS,
    ARDUSUB_SITL_PARAMETERS,
    EXECUTION_MODE,
    MANUAL_CONTROL_AUTHORITY_MASK,
    MANUAL_CONTROL_ORDER,
)
from disaware_learning.data.episode_io import load_episode
from disaware_learning.data.episode_schema import (
    ALIGNED_ARRAYS,
    FORMAT_VERSION,
    GROUND_TRUTH_ARRAYS,
    NATIVE_STREAM_ARRAYS,
    PRIVILEGED_ACTUATOR_ARRAYS,
    PRIVILEGED_ACTUATOR_NATIVE_ARRAYS,
    PROTOCOL_ID,
    REQUIRED_ARRAYS,
)
from disaware_learning.estimation.pressure_depth import (
    PRESSURE_DEPTH_CONVERSION_ID,
    PRESSURE_DEPTH_SOURCE_ID,
    STANDARD_GRAVITY_M_S2,
)


FORBIDDEN_NAME_PARTS = (
    "dvl",
    "estimated_position",
    "estimated_velocity",
    "action_thruster",
    "applied_thruster",
    "data_engine",
)
TYPICAL_TRAVEL_SPEED_BAND_M_S = (0.1, 0.4)


def _pressure_depth_contract_valid(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> bool:
    sensor = metadata.get("depth_sensor")
    if not isinstance(sensor, dict):
        return False
    try:
        density = float(sensor.get("fluid_density_kg_m3"))
        surface_pressure = float(sensor.get("surface_pressure_pa"))
        aligned_pressure = np.asarray(arrays["pressure_pa"], dtype=np.float64)
        aligned_depth = np.asarray(arrays["depth_down_m"], dtype=np.float64)
        raw_pressure = np.asarray(arrays["pressure_raw_pa"], dtype=np.float64)
        raw_depth = np.asarray(arrays["depth_raw_down_m"], dtype=np.float64)
        depth_time = np.asarray(arrays["depth_time"], dtype=np.float64)
        pressure_time = np.asarray(arrays["pressure_time"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(density) or density <= 0.0 or not math.isfinite(surface_pressure):
        return False
    expected_aligned = (aligned_pressure - surface_pressure) / (
        density * STANDARD_GRAVITY_M_S2
    )
    expected_raw = (raw_pressure - surface_pressure) / (
        density * STANDARD_GRAVITY_M_S2
    )
    controller_sensor = metadata.get("controller_depth_sensor")
    controller_valid = bool(
        controller_sensor is None
        or (
            isinstance(controller_sensor, dict)
            and controller_sensor.get("source_id") == PRESSURE_DEPTH_SOURCE_ID
            and controller_sensor.get("topic") == sensor.get("topic")
            and controller_sensor.get("conversion") == PRESSURE_DEPTH_CONVERSION_ID
            and controller_sensor.get("replaces_vio_position_z_for_control") is True
            and controller_sensor.get("ground_truth_position_z_used") is False
        )
    )
    return bool(
        sensor.get("source_id") == PRESSURE_DEPTH_SOURCE_ID
        and sensor.get("topic") == sensor.get("pressure_topic")
        and str(sensor.get("topic", "")).endswith("/pressure")
        and sensor.get("message_type") == "sensor_msgs/msg/FluidPressure"
        and sensor.get("pressure_reference") == "gauge"
        and sensor.get("conversion") == PRESSURE_DEPTH_CONVERSION_ID
        and sensor.get("derived_from_archive_array") == "pressure_pa"
        and sensor.get("vio_position_z_used") is False
        and sensor.get("ground_truth_position_z_used") is False
        and aligned_pressure.shape == aligned_depth.shape
        and raw_pressure.shape == raw_depth.shape
        and depth_time.shape == pressure_time.shape
        and np.allclose(depth_time, pressure_time, rtol=0.0, atol=1e-9)
        and np.allclose(aligned_depth, expected_aligned, rtol=0.0, atol=1e-9)
        and np.allclose(raw_depth, expected_raw, rtol=0.0, atol=1e-9)
        and controller_valid
    )


def _physics_domain_provenance_valid(metadata: dict[str, Any]) -> bool:
    provenance = metadata.get("physics_domain_provenance")
    if provenance is None:
        return True
    if not isinstance(provenance, dict):
        return False
    parameters = provenance.get("parameters", {})
    return bool(
        provenance.get("domain_protocol_id") == DOMAIN_PROTOCOL_ID
        and provenance.get("domain_bank_protocol_id") == DOMAIN_BANK_PROTOCOL_ID
        and str(provenance.get("domain_id", ""))
        and provenance.get("domain_split") in {"train", "validation", "test"}
        and provenance.get("sensor_robustness_included") is False
        and isinstance(parameters, dict)
        and set(parameters)
        == {"actuator", "vehicle", "hydrodynamics", "environment"}
        and metadata.get("domain_split") == provenance.get("domain_split")
    )


def _physics_domain_runtime_valid(metadata: dict[str, Any]) -> bool:
    provenance = metadata.get("physics_domain_provenance")
    runtime = metadata.get("physics_domain_runtime")
    if provenance is None:
        return runtime is None
    if not isinstance(provenance, dict) or not isinstance(runtime, dict):
        return False
    environment = provenance.get("parameters", {}).get("environment", {})
    configured = environment.get("static_current_velocity_ned_m_s")
    activation = runtime.get("current_activation")
    external = runtime.get("external_current")
    if not isinstance(activation, dict) or not isinstance(external, dict):
        return False
    try:
        configured_velocity = np.asarray(configured, dtype=np.float64)
        activated_velocity = np.asarray(
            activation.get("configured_static_velocity_ned_m_s"),
            dtype=np.float64,
        )
        amplitude = float(environment.get("current_amplitude_m_s"))
        maximum_published = float(external.get("maximum_published_speed_m_s"))
    except (TypeError, ValueError):
        return False
    return bool(
        configured_velocity.shape == (3,)
        and activated_velocity.shape == (3,)
        and np.all(np.isfinite(configured_velocity))
        and np.allclose(
            activated_velocity,
            configured_velocity,
            rtol=0.0,
            atol=1e-12,
        )
        and activation.get("required") is True
        and activation.get("service") == "/stonefish_ros2/enable_currents"
        and activation.get("requested") is True
        and activation.get("response_received") is True
        and activation.get("passed") is True
        and "enabled" in str(activation.get("response_message", "")).lower()
        and environment.get("current_model") == CURRENT_MODEL_ID
        and external.get("model") == environment.get("current_model")
        and external.get("topic") == environment.get("current_topic")
        and external.get("message_type") == "geometry_msgs/msg/Vector3"
        and external.get("clock_source") == "formal_reference_time_seconds"
        and external.get("zero_during_reset_vins_and_preroll") is True
        and external.get("initialization_zero_published") is True
        and external.get("terminal_zero_published") is True
        and int(external.get("formal_samples_published", 0)) > 0
        and math.isfinite(amplitude)
        and math.isfinite(maximum_published)
        and 0.0 <= maximum_published <= amplitude + 1e-9
    )


def _strictly_increasing(values: np.ndarray) -> bool:
    return bool(values.size >= 2 and np.all(np.diff(values) > 0.0))


def _stream_shape(
    arrays: dict[str, np.ndarray],
    time_name: str,
    value_name: str,
    width: int | None,
) -> bool:
    count = arrays[time_name].size
    values = arrays[value_name]
    return bool(
        values.shape == ((count,) if width is None else (count, width))
    )


def _parameter_contract_valid(metadata: dict[str, Any]) -> bool:
    contract = metadata.get("ardusub_contract")
    if not isinstance(contract, dict):
        return False
    snapshot = contract.get("parameter_snapshot", {})
    expected = dict(ARDUSUB_NOMINAL_PARAMETERS)
    # Real-vehicle episodes intentionally do not require SITL-only markers.
    # When the marker is present, every collection-only setting is checked.
    if next(iter(ARDUSUB_SITL_PARAMETERS)) in snapshot:
        expected.update(ARDUSUB_SITL_PARAMETERS)
    log_suppression = contract.get("collection_log_suppression", {})
    log_suppression_valid = bool(
        not isinstance(log_suppression, dict)
        or log_suppression.get("required") is not True
        or (
            log_suppression.get("parameter") == "LOG_BITMASK"
            and np.isclose(float(log_suppression.get("expected", np.nan)), 0.0)
            and np.isclose(float(log_suppression.get("readback", np.nan)), 0.0)
            and log_suppression.get("passed") is True
            and np.isclose(float(snapshot.get("LOG_BITMASK", np.nan)), 0.0)
        )
    )
    return bool(
        contract.get("execution_mode") == EXECUTION_MODE
        and contract.get("continuous_heartbeat_gate", {}).get("passed") is True
        and log_suppression_valid
        and all(
            name in snapshot
            and np.isclose(float(snapshot[name]), float(expected_value), atol=1e-3)
            for name, expected_value in expected.items()
        )
    )


def inspect_episode(path: str | Path) -> dict[str, Any]:
    """Return a machine-readable v15 protocol report."""
    try:
        episode_path, arrays, metadata = load_episode(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {"pass": False, "path": str(path), "error": str(error)}

    required = set(REQUIRED_ARRAYS)
    names = set(arrays)
    missing = sorted(required - names)
    forbidden = sorted(
        name
        for name in names
        if any(part in name.lower() for part in FORBIDDEN_NAME_PARTS)
    )
    time_values = arrays.get("time", np.empty(0))
    sample_count = int(time_values.size)
    aligned_shapes = bool(
        sample_count >= 2
        and all(
            arrays[name].shape[0] == sample_count
            for name in ALIGNED_ARRAYS
            if name in arrays
        )
    )
    finite = bool(
        not missing
        and all(
            np.all(np.isfinite(arrays[name]))
            for name in REQUIRED_ARRAYS
        )
    )
    ahrs_quaternion = arrays.get(
        "ahrs_orientation_body_to_ned_xyzw", np.empty((0, 4))
    )
    vio_quaternion = arrays.get(
        "vio_orientation_body_to_ned_xyzw", np.empty((0, 4))
    )
    quaternion_valid = bool(
        ahrs_quaternion.shape == (sample_count, 4)
        and vio_quaternion.shape == (sample_count, 4)
        and np.allclose(np.linalg.norm(ahrs_quaternion, axis=1), 1.0, atol=1e-3)
        and np.allclose(np.linalg.norm(vio_quaternion, axis=1), 1.0, atol=1e-3)
    )
    control = arrays.get("manual_control_normalized", np.empty((0, 6)))
    control_pwm = arrays.get("manual_control_pwm", np.empty((0, 8)))
    motors = arrays.get("motor_output_normalized", np.empty((0, 6)))
    motor_pwm = arrays.get("motor_output_pwm", np.empty((0, 6)))
    six_thruster_shapes = bool(
        control.shape == (sample_count, 6)
        and control_pwm.shape == (sample_count, 8)
        and motors.shape == (sample_count, ACTUATOR_COUNT)
        and motor_pwm.shape == (sample_count, ACTUATOR_COUNT)
    )
    bounded = bool(
        six_thruster_shapes
        and np.max(np.abs(control)) <= 1.0001
        and np.max(np.abs(motors)) <= 1.0001
        and np.min(control_pwm) >= 1100.0
        and np.max(control_pwm) <= 1900.0
        and np.min(motor_pwm) >= 1100.0
        and np.max(motor_pwm) <= 1900.0
    )
    pwm_conversion_consistent = bool(
        six_thruster_shapes
        and np.allclose(
            control,
            np.clip((control_pwm[:, :6] - 1500.0) / 400.0, -1.0, 1.0),
            atol=1e-12,
        )
        and np.allclose(
            motors,
            np.clip((motor_pwm - 1500.0) / 400.0, -1.0, 1.0),
            atol=1e-12,
        )
    )

    native_time_names = tuple(
        name for name in NATIVE_STREAM_ARRAYS if name.endswith("_time")
    )
    native_nonempty = bool(
        not missing and all(arrays[name].size for name in native_time_names)
    )
    native_times_valid = bool(
        native_nonempty
        and all(_strictly_increasing(arrays[name]) for name in native_time_names)
    )
    source_stamp_names = tuple(
        name for name in NATIVE_STREAM_ARRAYS if name.endswith("_source_stamp_seconds")
    )
    source_stamps_valid = bool(
        native_nonempty
        and all(_strictly_increasing(arrays[name]) for name in source_stamp_names)
    )
    native_shapes = bool(
        not missing
        and all(
            _stream_shape(arrays, "vio_time", name, width)
            for name, width in (
                ("vio_source_stamp_seconds", None),
                ("vio_raw_position_ned_m", 3),
                ("vio_raw_orientation_body_to_ned_xyzw", 4),
                ("vio_raw_linear_velocity_body_frd_m_s", 3),
                ("vio_raw_angular_velocity_body_frd_rad_s", 3),
            )
        )
        and all(
            (
                _stream_shape(arrays, "ahrs_time", name, width)
                for name, width in (
                    ("ahrs_source_stamp_seconds", None),
                    ("ahrs_raw_orientation_body_to_ned_xyzw", 4),
                    ("ahrs_raw_angular_velocity_body_frd_rad_s", 3),
                )
            )
        )
        and all(
            _stream_shape(arrays, "imu_time", name, width)
            for name, width in (
                ("imu_source_stamp_seconds", None),
                ("imu_raw_angular_velocity_body_frd_rad_s", 3),
                ("imu_raw_linear_acceleration_body_frd_m_s2", 3),
            )
        )
        and _stream_shape(arrays, "depth_time", "depth_raw_down_m", None)
        and all(
            _stream_shape(arrays, "pressure_time", name, None)
            for name in (
                "pressure_source_stamp_seconds",
                "pressure_raw_pa",
                "pressure_raw_variance_pa2",
            )
        )
        and all(
            _stream_shape(arrays, "manual_control_time", name, width)
            for name, width in (
                ("manual_control_raw_normalized", 6),
                ("manual_control_raw_pwm", 8),
            )
        )
        and all(
            _stream_shape(arrays, "motor_output_time", name, 6)
            for name in (
                "motor_output_raw_normalized",
                "motor_output_raw_pwm",
            )
        )
    )

    requested_rate = float(metadata.get("requested_sample_rate_hz", 0.0))
    actual_rate = (
        float(1.0 / np.median(np.diff(time_values)))
        if _strictly_increasing(time_values)
        else 0.0
    )
    rate_valid = bool(
        requested_rate >= 2.0
        and abs(actual_rate - requested_rate) / requested_rate <= 0.15
    )
    quality = metadata.get("stream_quality", {})
    ground_truth = metadata.get("ground_truth_recording", {})
    truth_enabled = bool(ground_truth.get("enabled"))
    controller = metadata.get("controller_provenance", {})
    vio_provenance = controller.get("simulated_vio_provenance")
    simulated_vio_valid = bool(
        (
            controller.get("state_source") != "simulated_stereo_vio_slam_from_gt"
            and ground_truth.get("simulation_state_generation_allowed") is not True
        )
        or (
            controller.get("state_source") == "simulated_stereo_vio_slam_from_gt"
            and isinstance(vio_provenance, dict)
            and vio_provenance.get("source")
            == "simulated_stereo_vio_slam_from_gt"
            and isinstance(vio_provenance.get("seed"), int)
            and isinstance(vio_provenance.get("config"), dict)
            and vio_provenance.get("causal") is True
            and vio_provenance.get("uses_future_ground_truth") is False
            and controller.get("feedback_access") == "deployable_vio_odometry"
            and ground_truth.get("feedback_access_allowed") is False
            and ground_truth.get("simulation_state_generation_allowed") is True
        )
    )
    truth_shapes = bool(
        all(name not in arrays for name in GROUND_TRUTH_ARRAYS)
        if not truth_enabled
        else all(
            name in arrays and arrays[name].shape[0] == sample_count
            for name in GROUND_TRUTH_ARRAYS
        )
    )
    applied = metadata.get("applied_thrust_recording", {})
    applied_enabled = bool(applied.get("enabled"))
    privileged_names = set(PRIVILEGED_ACTUATOR_ARRAYS) | set(
        PRIVILEGED_ACTUATOR_NATIVE_ARRAYS
    )
    if not applied_enabled:
        applied_thrust_valid = bool(
            not truth_enabled and privileged_names.isdisjoint(arrays)
        )
    else:
        applied_thrust = arrays.get(
            "ground_truth_applied_thrust_n", np.empty((0, ACTUATOR_COUNT))
        )
        thruster_time = arrays.get("thruster_state_time", np.empty(0))
        applied_thrust_valid = bool(
            truth_enabled
            and privileged_names.issubset(arrays)
            and applied.get("source") == "Stonefish ThrusterState.getThrust()"
            and applied.get("deployable_observation_access_allowed") is False
            and applied.get("feedback_access_allowed") is False
            and applied.get("jepa_action_input_allowed") is False
            and applied.get("affects_pass_fail") is False
            and applied_thrust.shape == (sample_count, ACTUATOR_COUNT)
            and np.all(np.isfinite(applied_thrust))
            and _strictly_increasing(thruster_time)
            and _stream_shape(
                arrays,
                "thruster_state_time",
                "thruster_state_source_stamp_seconds",
                None,
            )
            and all(
                _stream_shape(arrays, "thruster_state_time", name, ACTUATOR_COUNT)
                for name in (
                    "thruster_setpoint_raw",
                    "thruster_rpm_raw",
                    "thruster_thrust_n_raw",
                    "thruster_torque_nm_raw",
                )
            )
            and all(
                np.all(np.isfinite(arrays[name]))
                for name in privileged_names
            )
        )
    checks = {
        "format_v15": metadata.get("format_version") == FORMAT_VERSION,
        "protocol_identified": metadata.get("protocol_id") == PROTOCOL_ID,
        "required_arrays_present": not missing,
        "old_protocol_arrays_absent": not forbidden,
        "aligned_sample_shapes": aligned_shapes,
        "numeric_values_finite": finite,
        "time_strictly_increasing": _strictly_increasing(time_values),
        "sample_rate_matches_request": rate_valid,
        "attitude_quaternions_normalized": quaternion_valid,
        "manual_and_six_motor_shapes": six_thruster_shapes,
        "commands_and_motor_outputs_bounded": bounded,
        "normalized_pwm_pairs_consistent": pwm_conversion_consistent,
        "native_streams_nonempty": native_nonempty,
        "native_stream_times_increasing": native_times_valid,
        "native_source_timestamps_increasing": source_stamps_valid,
        "native_stream_shapes": native_shapes,
        "pressure_depth_sensor_chain_valid": _pressure_depth_contract_valid(
            metadata, arrays
        ),
        "manual_control_stream_valid": quality.get(
            "manual_control_stream_valid"
        )
        is True,
        "motor_output_stream_valid": quality.get(
            "motor_output_stream_valid"
        )
        is True,
        "observation_stream_valid": quality.get(
            "observation_stream_valid"
        )
        is True,
        "applied_thrust_privileged_truth_valid": applied_thrust_valid
        and (
            quality.get("applied_thrust_stream_valid") is True
            if applied_enabled
            else quality.get("applied_thrust_stream_valid", True) is True
        ),
        "manual_execution_mode": (
            metadata.get("configured_execution_mode") == EXECUTION_MODE
        ),
        "standard_frame_authority_declared": (
            tuple(metadata.get("manual_control_order", ()))
            == MANUAL_CONTROL_ORDER
            and tuple(metadata.get("manual_control_authority_mask", ()))
            == MANUAL_CONTROL_AUTHORITY_MASK
        ),
        "horizontal_state_available_from_vio": (
            metadata.get("horizontal_state_available") is True
        ),
        "motor_output_not_claimed_as_force": (
            "not measured thrust or force"
            in str(metadata.get("motor_output_semantics", ""))
        ),
        "ardusub_parameter_snapshot_complete": _parameter_contract_valid(
            metadata
        ),
        "ground_truth_optional_and_separate": truth_shapes
        and ground_truth.get("deployable_observation_access_allowed") is False
        and ground_truth.get("feedback_access_allowed") is False
        and ground_truth.get("affects_pass_fail") is False,
        "simulated_vio_provenance_valid_when_present": simulated_vio_valid,
        "physics_domain_provenance_valid_when_present": (
            _physics_domain_provenance_valid(metadata)
        ),
        "physics_domain_runtime_valid_when_present": (
            _physics_domain_runtime_valid(metadata)
        ),
    }
    result = {
        "pass": bool(all(checks.values())),
        "path": str(episode_path),
        "checks": checks,
        "missing_arrays": missing,
        "forbidden_arrays": forbidden,
        "samples": sample_count,
        "duration_seconds": float(time_values[-1]) if sample_count else 0.0,
        "actual_sample_rate_hz": actual_rate,
        "maximum_motor_output": float(np.max(np.abs(motors)))
        if motors.size
        else 0.0,
        "depth_change_m": float(arrays["depth_down_m"][-1] - arrays["depth_down_m"][0])
        if sample_count and "depth_down_m" in arrays
        else 0.0,
    }
    if truth_enabled and "ground_truth_position_ned_m" in arrays:
        position = arrays["ground_truth_position_ned_m"]
        velocity = arrays["ground_truth_velocity_body_frd_m_s"]
        position_step = np.diff(position, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
        duration = float(time_values[-1] - time_values[0])
        path_length = float(np.sum(np.linalg.norm(position_step, axis=1)))
        horizontal_path_length = float(
            np.sum(np.linalg.norm(position_step[:, :2], axis=1))
        )
        minimum_path_length = TYPICAL_TRAVEL_SPEED_BAND_M_S[0] * duration
        maximum_path_length = TYPICAL_TRAVEL_SPEED_BAND_M_S[1] * duration
        recipe = metadata.get("command_provenance", {}).get("recipe") or {}
        active_axes = set(recipe.get("active_axes", ()))
        path_band_applicable = bool(active_axes & {"surge", "sway"})
        result["simulation_truth_audit"] = {
            **ground_truth.get("simulation_truth_audit", {}),
            "path_length_m": path_length,
            "horizontal_path_length_m": horizontal_path_length,
            "net_displacement_m": float(np.linalg.norm(position[-1] - position[0])),
            "net_horizontal_displacement_m": float(
                np.linalg.norm(position[-1, :2] - position[0, :2])
            ),
            "mean_speed_m_s": float(np.mean(speed)),
            "median_speed_m_s": float(np.median(speed)),
            "p95_speed_m_s": float(np.quantile(speed, 0.95)),
            "maximum_speed_m_s": float(np.max(speed)),
            "mean_horizontal_speed_m_s": float(np.mean(horizontal_speed)),
            "typical_speed_band_m_s": list(TYPICAL_TRAVEL_SPEED_BAND_M_S),
            "duration_scaled_path_length_band_m": [
                minimum_path_length,
                maximum_path_length,
            ],
            "path_length_band_applicable": path_band_applicable,
            "within_typical_path_length_band": (
                minimum_path_length <= path_length <= maximum_path_length
                if path_band_applicable
                else None
            ),
            "final_displacement_ned_m": (position[-1] - position[0]).tolist(),
            "affects_pass_fail": False,
        }
    return result


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="v15 episode .npz or .json")
    parsed = parser.parse_args(args)
    report = inspect_episode(parsed.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
