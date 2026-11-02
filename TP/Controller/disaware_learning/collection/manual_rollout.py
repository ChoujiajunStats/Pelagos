"""Collect one slow-multisine dynamics episode through ArduSub MANUAL."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure, Imu
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from disaware_learning.collection.ardusub_collector import ArduSubCollectorBase
from disaware_learning.collection.physics_domain import (
    EXTERNAL_CURRENT_TOPIC,
    PhysicsDomain,
    load_domain_from_bank,
)
from disaware_learning.collection.rich_reference import (
    load_rich_reference,
    rich_reference_provenance,
)
from disaware_learning.control.body_twist_pid import (
    BodyTwistPidConfig,
    BodyTwistReference,
    BodyTwistVelocityRatePid,
    quaternion_rotation_matrix,
    quaternion_rpy,
    summarize_body_twist_tracking,
    summarize_trajectory_tracking,
)
from disaware_learning.control.rc import clamp_manual_control, slew_towards
from disaware_learning.data.episode_io import load_metadata, write_metadata
from disaware_learning.data.episode_validation import inspect_episode
from disaware_learning.estimation.vio_state import (
    VioState,
    canonical_odometry_as_vio,
    ground_truth_odometry_as_vio,
)
from disaware_learning.estimation.pressure_depth import (
    PRESSURE_DEPTH_CONVERSION_ID,
    PRESSURE_DEPTH_SOURCE_ID,
    pressure_to_depth_down_m,
)


def _roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or norm <= 1e-9 or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite values")
    x, y, z, w = values / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _resample_reference_arrays(
    arrays: dict[str, np.ndarray],
    sample_rate_hz: float,
) -> dict[str, np.ndarray]:
    """Resample a slow reference bank without changing its time duration."""

    rate = float(sample_rate_hz)
    source_time = np.asarray(arrays["time_seconds"], dtype=np.float64)
    if rate <= 0.0 or source_time.ndim != 1 or np.any(np.diff(source_time) <= 0.0):
        raise ValueError("reference resampling requires a positive rate and valid time")
    step = 1.0 / rate
    count = int(math.floor((source_time[-1] - source_time[0]) / step + 1e-9)) + 1
    target_time = source_time[0] + np.arange(count, dtype=np.float64) * step
    result: dict[str, np.ndarray] = {"time_seconds": target_time}
    for name, value in arrays.items():
        if name == "time_seconds":
            continue
        data = np.asarray(value)
        if data.ndim < 1 or data.shape[0] != len(source_time):
            result[name] = data.copy()
            continue
        flat = data.reshape(len(source_time), -1).astype(np.float64)
        sampled = np.empty((len(target_time), flat.shape[1]), dtype=np.float64)
        angular = name == "yaw_preview_rad"
        for column in range(flat.shape[1]):
            source = np.unwrap(flat[:, column]) if angular else flat[:, column]
            sampled[:, column] = np.interp(target_time, source_time, source)
        result[name] = sampled.reshape((len(target_time), *data.shape[1:])).astype(
            data.dtype, copy=False
        )
    return result


class ManualRollout(ArduSubCollectorBase):
    """Replay a rich reference with canonical stereo-VINS feedback and fixed PID."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("command_path", "")
        self.declare_parameter("resample_reference_to_command_rate", False)
        self.declare_parameter("domain_bank_path", "")
        self.declare_parameter("domain_id", "")
        self.declare_parameter("scenario_path", "")
        self.declare_parameter("simulation_reset", True)
        self.declare_parameter("reset_spawn_yaw_rad", 0.0)
        self.declare_parameter("minimum_depth_down_m", 0.20)
        self.declare_parameter("maximum_depth_down_m", 11.0)
        self.declare_parameter("depth_source", "fluid_pressure")
        self.declare_parameter("pressure_topic", "")
        self.declare_parameter("fluid_density_kg_m3", 1025.0)
        self.declare_parameter("surface_pressure_pa", 0.0)
        self.declare_parameter("reset_convergence_seconds", 1.0)
        self.declare_parameter("reset_convergence_timeout_seconds", 12.0)
        self.declare_parameter("reset_max_depth_error_m", 0.35)
        self.declare_parameter("reset_max_angular_speed_rad_s", 0.15)
        self.declare_parameter("pre_record_settle_seconds", 2.0)
        self.declare_parameter("pid_trace_path", "")
        self.declare_parameter("pid_config_path", "")
        self.declare_parameter("maximum_accepted_motor_output", 0.95)
        self.declare_parameter("pid_velocity_kp", [0.45, 0.45, 0.45])
        self.declare_parameter("pid_velocity_ki", [0.0, 0.0, 0.0])
        # First closed-loop calibration: surge/heave were under-amplified while
        # yaw was over-amplified on the nominal six-thruster plant.
        self.declare_parameter("pid_velocity_feedforward", [2.00, 2.20, 2.60])
        self.declare_parameter(
            "pid_velocity_acceleration_feedforward", [0.0, 0.0, 0.0]
        )
        self.declare_parameter("pid_velocity_integral_limit", [0.35, 0.35, 0.30])
        self.declare_parameter("pid_position_kp_body", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "pid_maximum_position_correction_m_s", [0.0, 0.0, 0.0]
        )
        self.declare_parameter("pid_yaw_rate_kp", 0.25)
        self.declare_parameter("pid_yaw_rate_ki", 0.0)
        self.declare_parameter("pid_yaw_rate_feedforward", 0.24)
        self.declare_parameter("pid_yaw_rate_integral_limit", 0.35)
        self.declare_parameter("pid_yaw_position_kp", 0.0)
        self.declare_parameter("pid_maximum_yaw_rate_correction_rad_s", 0.0)
        self.declare_parameter("pid_yaw_command_slew_rate", 0.80)
        self.declare_parameter("pid_roll_kp", 0.55)
        self.declare_parameter("pid_roll_rate_kd", 0.14)
        self.declare_parameter("pid_command_limit", 0.50)
        self.declare_parameter("vio_odometry_topic", "")
        self.declare_parameter("vio_source_id", "vins_fusion_ros2_imu_propagate")
        self.declare_parameter("maximum_vio_age_seconds", 0.10)
        self.declare_parameter("synchronize_vins_reset", True)
        self.declare_parameter("vins_restart_topic", "/vins_restart")
        self.declare_parameter(
            "vins_reset_origin_service", "/vins_fusion_adapter/reset_origin"
        )
        self.declare_parameter("vins_restart_silence_seconds", 0.25)
        self.declare_parameter("vins_reinitialization_timeout_seconds", 25.0)
        # A fresh VINS origin is not sufficient evidence that its state is
        # usable.  Reject a reset generation that starts with an impossible
        # velocity or attitude before any feedback controller can consume it.
        self.declare_parameter("vins_initial_max_linear_speed_m_s", 0.50)
        self.declare_parameter("vins_initial_max_angular_speed_rad_s", 0.50)
        self.declare_parameter("vins_initial_max_tilt_degrees", 10.0)
        self.declare_parameter("vins_initial_stable_seconds", 0.50)

        self.maximum_accepted_motor_output = float(
            self.get_parameter("maximum_accepted_motor_output").value
        )
        if not 0.0 < self.maximum_accepted_motor_output <= 1.0:
            raise ValueError("maximum_accepted_motor_output must be in (0, 1]")

        command_path = str(self.get_parameter("command_path").value).strip()
        if not command_path:
            raise ValueError("command_path is required")
        self.command_path = Path(command_path).expanduser().resolve()
        self.command_arrays, self.command_metadata = load_rich_reference(
            self.command_path
        )
        self.command_provenance = rich_reference_provenance(
            self.command_path, self.command_metadata
        )

        domain_bank_path = str(self.get_parameter("domain_bank_path").value).strip()
        domain_id = str(self.get_parameter("domain_id").value).strip()
        scenario_path = str(self.get_parameter("scenario_path").value).strip()
        if bool(domain_bank_path) != bool(domain_id):
            raise ValueError("domain_bank_path and domain_id must be provided together")
        self.domain_provenance: dict[str, Any] | None = None
        self.physics_domain: PhysicsDomain | None = None
        self.physics_domain_runtime: dict[str, Any] | None = None
        self.enable_currents = None
        self.external_current_publisher = None
        self._external_current_active = False
        if domain_bank_path:
            self.physics_domain, self.domain_provenance = load_domain_from_bank(
                domain_bank_path, domain_id
            )
            expected = Path(self.domain_provenance["scenario_path"]).resolve()
            if (
                not scenario_path
                or Path(scenario_path).expanduser().resolve() != expected
            ):
                raise ValueError(
                    "scenario_path does not match the selected physics domain"
                )
            generator = self.physics_domain.current_generator()
            generator_data = generator.as_dict()
            self.physics_domain_runtime = {
                "current_activation": {
                    "required": True,
                    "service": "/stonefish_ros2/enable_currents",
                    "requested": False,
                    "response_received": False,
                    "passed": False,
                    "configured_static_velocity_ned_m_s": [0.0, 0.0, 0.0],
                },
                "external_current": {
                    "required": True,
                    "topic": EXTERNAL_CURRENT_TOPIC,
                    "message_type": "geometry_msgs/msg/Vector3",
                    "model": generator_data["current_model"],
                    "clock_source": "formal_reference_time_seconds",
                    "amplitude_m_s": generator.amplitude_m_s,
                    "direction_rad": generator.direction_rad,
                    "period_seconds": generator.period_seconds,
                    "phase_rad": generator.phase_rad,
                    "ramp_seconds": generator.ramp_seconds,
                    "secondary_period_seconds": generator.secondary_period_seconds,
                    "secondary_phase_rad": generator.secondary_phase_rad,
                    "secondary_weight": generator_data["current_secondary_weight"],
                    "direction_wobble_amplitude_rad": (
                        generator.direction_wobble_amplitude_rad
                    ),
                    "direction_wobble_period_seconds": (
                        generator.direction_wobble_period_seconds
                    ),
                    "direction_wobble_phase_rad": generator.direction_wobble_phase_rad,
                    "zero_during_reset_vins_and_preroll": True,
                    "initialization_zero_published": False,
                    "formal_samples_published": 0,
                    "maximum_published_speed_m_s": 0.0,
                    "terminal_zero_published": False,
                }
            }
            self.enable_currents = self.create_client(
                Trigger, "/stonefish_ros2/enable_currents"
            )
            self.external_current_publisher = self.create_publisher(
                Vector3, EXTERNAL_CURRENT_TOPIC, 10
            )

        command_steps = np.diff(self.command_arrays["time_seconds"])
        command_rate = float(1.0 / np.median(command_steps))
        if (
            not math.isclose(command_rate, self.command_rate, rel_tol=0.02)
            and bool(self.get_parameter("resample_reference_to_command_rate").value)
        ):
            self.command_arrays = _resample_reference_arrays(
                self.command_arrays, self.command_rate
            )
            self.command_provenance["runtime_reference_resampling"] = {
                "source_rate_hz": command_rate,
                "runtime_rate_hz": self.command_rate,
                "method": "linear_slow_reference_v1",
            }
            command_steps = np.diff(self.command_arrays["time_seconds"])
            command_rate = float(1.0 / np.median(command_steps))
        if not math.isclose(command_rate, self.command_rate, rel_tol=0.02):
            raise ValueError(
                f"command rate {command_rate:.3f} does not match runtime {self.command_rate:.3f}"
            )

        self.simulation_reset = bool(self.get_parameter("simulation_reset").value)
        self.reset_spawn_yaw_rad = float(
            self.get_parameter("reset_spawn_yaw_rad").value
        )
        if not math.isfinite(self.reset_spawn_yaw_rad):
            raise ValueError("reset_spawn_yaw_rad must be finite")
        self.minimum_depth = float(self.get_parameter("minimum_depth_down_m").value)
        self.maximum_depth = float(self.get_parameter("maximum_depth_down_m").value)
        if self.minimum_depth >= self.maximum_depth:
            raise ValueError("minimum depth must be below maximum depth")
        self.depth_source = str(self.get_parameter("depth_source").value).strip()
        if self.depth_source != "fluid_pressure":
            raise ValueError(
                "formal rollout requires pressure-derived depth; "
                "set depth_source='fluid_pressure'"
            )
        configured_pressure_topic = str(
            self.get_parameter("pressure_topic").value
        ).strip()
        self.pressure_topic = configured_pressure_topic or (
            f"/{self.vehicle_name}/pressure"
        )
        self.fluid_density = float(
            self.get_parameter("fluid_density_kg_m3").value
        )
        self.surface_pressure = float(
            self.get_parameter("surface_pressure_pa").value
        )
        pressure_to_depth_down_m(
            self.surface_pressure,
            fluid_density_kg_m3=self.fluid_density,
            surface_pressure_pa=self.surface_pressure,
        )
        self.reset_convergence_seconds = self._positive_parameter(
            "reset_convergence_seconds"
        )
        self.reset_convergence_timeout = self._positive_parameter(
            "reset_convergence_timeout_seconds"
        )
        self.reset_max_depth_error = self._positive_parameter("reset_max_depth_error_m")
        self.reset_max_angular_speed = self._positive_parameter(
            "reset_max_angular_speed_rad_s"
        )
        self.pre_record_settle_seconds = float(
            self.get_parameter("pre_record_settle_seconds").value
        )
        if (
            not math.isfinite(self.pre_record_settle_seconds)
            or self.pre_record_settle_seconds < 0.0
        ):
            raise ValueError("pre_record_settle_seconds must be nonnegative and finite")

        self.pid_config_path = str(
            self.get_parameter("pid_config_path").value
        ).strip()
        self.pid_config_id: str | None = None
        self.pid_config_parameters: dict[str, Any] = {}
        if self.pid_config_path:
            config_path = Path(self.pid_config_path).expanduser().resolve()
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"failed to load pid_config_path {config_path}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError("PID config must contain a JSON object")
            parameters = payload.get("parameters", payload)
            if not isinstance(parameters, dict):
                raise ValueError("PID config parameters must contain a JSON object")
            allowed = {
                "pid_velocity_kp",
                "pid_velocity_ki",
                "pid_velocity_feedforward",
                "pid_velocity_acceleration_feedforward",
                "pid_velocity_integral_limit",
                "pid_position_kp_body",
                "pid_maximum_position_correction_m_s",
                "pid_yaw_rate_kp",
                "pid_yaw_rate_ki",
                "pid_yaw_rate_feedforward",
                "pid_yaw_rate_integral_limit",
                "pid_yaw_position_kp",
                "pid_maximum_yaw_rate_correction_rad_s",
                "pid_yaw_command_slew_rate",
                "pid_roll_kp",
                "pid_roll_rate_kd",
                "pid_command_limit",
            }
            unknown = sorted(set(parameters) - allowed)
            if unknown:
                raise ValueError(f"PID config contains unknown parameters: {unknown}")
            self.pid_config_parameters = dict(parameters)
            self.pid_config_id = str(payload.get("config_id", config_path.stem))
            self.pid_config_path = str(config_path)

        def configured(name: str) -> Any:
            return self.pid_config_parameters.get(
                name, self.get_parameter(name).value
            )

        def vector(name: str) -> tuple[float, ...]:
            return tuple(float(value) for value in configured(name))

        def scalar(name: str) -> float:
            return float(configured(name))

        configured_vio_topic = str(
            self.get_parameter("vio_odometry_topic").value
        ).strip()
        self.vio_odometry_topic = configured_vio_topic or (
            f"/{self.vehicle_name}/vio/odometry"
        )
        self.vio_source_id = str(self.get_parameter("vio_source_id").value).strip()
        self.maximum_vio_age = self._positive_parameter("maximum_vio_age_seconds")
        if not self.vio_source_id:
            raise ValueError("vio_source_id cannot be empty")
        self.synchronize_vins_reset = bool(
            self.get_parameter("synchronize_vins_reset").value
        )
        self.vins_restart_silence = self._positive_parameter(
            "vins_restart_silence_seconds"
        )
        self.vins_reinitialization_timeout = self._positive_parameter(
            "vins_reinitialization_timeout_seconds"
        )
        self.vins_initial_max_linear_speed = self._positive_parameter(
            "vins_initial_max_linear_speed_m_s"
        )
        self.vins_initial_max_angular_speed = self._positive_parameter(
            "vins_initial_max_angular_speed_rad_s"
        )
        self.vins_initial_max_tilt_degrees = self._positive_parameter(
            "vins_initial_max_tilt_degrees"
        )
        self.vins_initial_stable_seconds = self._positive_parameter(
            "vins_initial_stable_seconds"
        )
        vins_restart_topic = str(
            self.get_parameter("vins_restart_topic").value
        ).strip()
        vins_reset_origin_service = str(
            self.get_parameter("vins_reset_origin_service").value
        ).strip()
        if self.synchronize_vins_reset and (
            not vins_restart_topic or not vins_reset_origin_service
        ):
            raise ValueError("VINS reset topic and adapter reset service are required")
        self._last_vio_state: VioState | None = None
        self._last_vio_received_at = -math.inf
        self.vins_restart_publisher = self.create_publisher(
            Bool, vins_restart_topic, 10
        )
        self.vins_reset_origin = self.create_client(
            Trigger, vins_reset_origin_service
        )
        self.vins_reset_lifecycle: dict[str, Any] = {
            "required": self.synchronize_vins_reset,
            "restart_topic": vins_restart_topic,
            "adapter_reset_origin_service": vins_reset_origin_service,
            "silence_seconds": self.vins_restart_silence,
            "reinitialization_timeout_seconds": self.vins_reinitialization_timeout,
            "state_sanity_gate": {
                "max_linear_speed_m_s": self.vins_initial_max_linear_speed,
                "max_angular_speed_rad_s": self.vins_initial_max_angular_speed,
                "max_tilt_degrees": self.vins_initial_max_tilt_degrees,
                "required_stable_seconds": self.vins_initial_stable_seconds,
                "ground_truth_used": False,
            },
            "attempts": [],
        }

        self.pid_controller = BodyTwistVelocityRatePid(
            BodyTwistPidConfig(
                position_kp_body=vector("pid_position_kp_body"),
                maximum_position_correction_m_s=vector(
                    "pid_maximum_position_correction_m_s"
                ),
                velocity_kp=vector("pid_velocity_kp"),
                velocity_ki=vector("pid_velocity_ki"),
                velocity_feedforward=vector("pid_velocity_feedforward"),
                velocity_acceleration_feedforward=vector(
                    "pid_velocity_acceleration_feedforward"
                ),
                velocity_integral_limit=vector("pid_velocity_integral_limit"),
                yaw_position_kp=scalar("pid_yaw_position_kp"),
                maximum_yaw_rate_correction_rad_s=scalar(
                    "pid_maximum_yaw_rate_correction_rad_s"
                ),
                yaw_rate_kp=scalar("pid_yaw_rate_kp"),
                yaw_rate_ki=scalar("pid_yaw_rate_ki"),
                yaw_rate_feedforward=scalar("pid_yaw_rate_feedforward"),
                yaw_rate_integral_limit=scalar("pid_yaw_rate_integral_limit"),
                yaw_command_slew_rate_per_second=scalar(
                    "pid_yaw_command_slew_rate"
                ),
                roll_kp=scalar("pid_roll_kp"),
                roll_rate_kd=scalar("pid_roll_rate_kd"),
                command_limit=scalar("pid_command_limit"),
            )
        )

        prefix = f"/{self.vehicle_name}"
        self.create_subscription(
            Odometry,
            self.vio_odometry_topic,
            self._vio_callback,
            qos_profile_sensor_data,
        )
        self.latest_quaternion: np.ndarray | None = None
        self.latest_angular_velocity: np.ndarray | None = None
        self.latest_depth: float | None = None
        self.ahrs_received_at = -math.inf
        self.depth_received_at = -math.inf
        self.create_subscription(
            Imu, f"{prefix}/ardusub/ahrs", self._ahrs_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            FluidPressure,
            self.pressure_topic,
            self._pressure_depth_callback,
            qos_profile_sensor_data,
        )
        self.pid_anchor_position_ned_m: np.ndarray | None = None
        self.pid_anchor_yaw_rad: float | None = None
        self.pid_trace: dict[str, list[Any]] = {}
        self.pid_trace_path = str(self.get_parameter("pid_trace_path").value).strip()
        self.reset_convergence_gate: dict[str, Any] = {
            "passed": False,
            "required_continuous_seconds": self.reset_convergence_seconds,
            "timeout_seconds": self.reset_convergence_timeout,
            "depth_reference": "configured_respawn_depth",
            "ground_truth_state_used": False,
            "privileged_actuator_state_used": False,
        }

    def _ahrs_callback(self, message: Imu) -> None:
        quaternion = np.asarray(
            [
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            ],
            dtype=np.float64,
        )
        rate = np.asarray(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        if (
            np.all(np.isfinite(quaternion))
            and np.linalg.norm(quaternion) > 1e-9
            and np.all(np.isfinite(rate))
        ):
            self.latest_quaternion = quaternion
            self.latest_angular_velocity = rate
            self.ahrs_received_at = time.monotonic()

    def _vio_callback(self, message: Odometry) -> None:
        try:
            state = canonical_odometry_as_vio(
                message,
                source=self.vio_source_id,
                expected_body_frame=f"{self.vehicle_name}/base_link_frd",
            )
        except ValueError as error:
            self.get_logger().warning(f"Rejected canonical VIO state: {error}")
            return
        if (
            self._last_vio_state is not None
            and state.timestamp_seconds
            <= self._last_vio_state.timestamp_seconds + 1e-9
        ):
            return
        self._last_vio_state = state
        self._last_vio_received_at = time.monotonic()

    def _respawn_vehicle(self, spawn_yaw: float) -> None:
        super()._respawn_vehicle(spawn_yaw)
        if self.synchronize_vins_reset:
            self._synchronize_vins_after_respawn()

    def _spin_until_fresh_vio(
        self,
        *,
        received_after: float,
        timestamp_after: float | None,
        timeout_seconds: float,
        failure_message: str,
    ) -> VioState:
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            state = self._last_vio_state
            if (
                state is not None
                and self._last_vio_received_at > received_after
                and (
                    timestamp_after is None
                    or state.timestamp_seconds > timestamp_after + 1e-9
                )
            ):
                return state
        raise RuntimeError(failure_message)

    def _wait_for_vio_state_sanity(self, *, timeout_seconds: float) -> VioState:
        """Require a physically plausible, deployable VIO state after reset.

        Origin anchoring only checks position.  VINS can still publish a fresh
        estimate with a bad initial rotation/velocity; consuming that estimate
        in PID or MPPI immediately creates a large command.  This gate uses
        only the canonical VIO message and deliberately does not consult the
        Stonefish truth odometry.
        """
        deadline = time.monotonic() + timeout_seconds
        stable_since: float | None = None
        latest: dict[str, float | bool] = {}
        while rclpy.ok() and time.monotonic() < deadline:
            state = self._last_vio_state
            fresh = bool(
                state is not None
                and time.monotonic() - self._last_vio_received_at
                <= self.maximum_vio_age
            )
            if fresh and state is not None:
                velocity = np.asarray(state.linear_velocity_body_frd_m_s)
                angular = np.asarray(state.angular_velocity_body_frd_rad_s)
                try:
                    roll, pitch, _ = quaternion_rpy(
                        state.orientation_body_to_ned_xyzw
                    )
                    tilt_degrees = float(np.degrees(np.hypot(roll, pitch)))
                except (TypeError, ValueError):
                    tilt_degrees = math.inf
                speed = float(np.linalg.norm(velocity))
                angular_speed = float(np.linalg.norm(angular))
                finite = bool(
                    np.all(np.isfinite(velocity))
                    and np.all(np.isfinite(angular))
                    and math.isfinite(tilt_degrees)
                )
                sane = bool(
                    finite
                    and speed <= self.vins_initial_max_linear_speed
                    and angular_speed <= self.vins_initial_max_angular_speed
                    and tilt_degrees <= self.vins_initial_max_tilt_degrees
                )
                latest = {
                    "linear_speed_m_s": speed,
                    "angular_speed_rad_s": angular_speed,
                    "tilt_degrees": tilt_degrees,
                    "finite": finite,
                    "sane": sane,
                }
                stable_since = (
                    stable_since if stable_since is not None else time.monotonic()
                ) if sane else None
                if (
                    stable_since is not None
                    and time.monotonic() - stable_since
                    >= self.vins_initial_stable_seconds
                ):
                    return state
            rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
        raise RuntimeError(
            "VINS state sanity gate did not converge after reset: "
            f"{latest}"
        )

    def _synchronize_vins_after_respawn(self) -> None:
        """Restart VINS and re-anchor its canonical frame after a hard reset."""
        attempt: dict[str, Any] = {
            "passed": False,
            "started_monotonic_seconds": time.monotonic(),
        }
        self.vins_reset_lifecycle["attempts"].append(attempt)
        subscriber_deadline = time.monotonic() + 5.0
        while (
            rclpy.ok()
            and self.vins_restart_publisher.get_subscription_count() < 1
            and time.monotonic() < subscriber_deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        subscribers = self.vins_restart_publisher.get_subscription_count()
        attempt["restart_subscriber_count"] = subscribers
        if subscribers < 1:
            attempt["failure"] = "VINS restart subscriber is unavailable"
            raise RuntimeError(attempt["failure"])
        if not self.vins_reset_origin.wait_for_service(timeout_sec=5.0):
            attempt["failure"] = "VINS adapter reset-origin service is unavailable"
            raise RuntimeError(attempt["failure"])

        restart = Bool()
        restart.data = True
        restart_published_at = time.monotonic()
        self.vins_restart_publisher.publish(restart)
        attempt["restart_published_monotonic_seconds"] = restart_published_at

        # The fork publishes imu_propagate only in NON_LINEAR mode. A genuine
        # restart therefore creates a short output gap before initialization
        # completes. Waiting for that gap prevents a queued pre-reset estimate
        # from being mistaken for the new estimator generation.
        silence_deadline = restart_published_at + min(
            5.0, self.vins_reinitialization_timeout
        )
        while rclpy.ok() and time.monotonic() < silence_deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.monotonic()
            if (
                now - restart_published_at >= self.vins_restart_silence
                and now - self._last_vio_received_at >= self.vins_restart_silence
            ):
                attempt["output_silence_observed_seconds"] = now - restart_published_at
                break
        else:
            attempt["failure"] = "VINS output did not stop after restart"
            raise RuntimeError(attempt["failure"])

        self._last_vio_state = None
        self._last_vio_received_at = -math.inf
        reinitialized = self._spin_until_fresh_vio(
            received_after=restart_published_at,
            timestamp_after=None,
            timeout_seconds=self.vins_reinitialization_timeout,
            failure_message="VINS did not reinitialize after Stonefish respawn",
        )
        reinitialized_received_at = self._last_vio_received_at
        attempt["reinitialized_vio_timestamp_seconds"] = (
            reinitialized.timestamp_seconds
        )
        attempt["reinitialization_wait_seconds"] = (
            reinitialized_received_at - restart_published_at
        )

        reset_requested_at = time.monotonic()
        reset_future = self.vins_reset_origin.call_async(Trigger.Request())
        self._spin_until(
            reset_future,
            5.0,
            "VINS adapter reset-origin service timed out",
        )
        reset_response = reset_future.result()
        if reset_response is None or not reset_response.success:
            attempt["failure"] = "VINS adapter rejected reset-origin request"
            raise RuntimeError(attempt["failure"])

        self._last_vio_state = None
        self._last_vio_received_at = -math.inf
        anchored = self._spin_until_fresh_vio(
            received_after=reset_requested_at,
            timestamp_after=reinitialized.timestamp_seconds,
            timeout_seconds=min(5.0, self.vins_reinitialization_timeout),
            failure_message="canonical VIO did not publish after origin reset",
        )
        anchor_norm = float(np.linalg.norm(anchored.position_ned_m))
        attempt.update(
            {
                "passed": True,
                "adapter_response": str(reset_response.message),
                "anchored_vio_timestamp_seconds": anchored.timestamp_seconds,
                "anchored_position_norm_m": anchor_norm,
                "completed_monotonic_seconds": time.monotonic(),
            }
        )
        if anchor_norm > 0.05:
            attempt["passed"] = False
            attempt["failure"] = (
                "canonical VIO origin did not re-anchor near zero: "
                f"{anchor_norm:.3f} m"
            )
            raise RuntimeError(attempt["failure"])
        try:
            sane = self._wait_for_vio_state_sanity(
                timeout_seconds=min(5.0, self.vins_reinitialization_timeout)
            )
        except RuntimeError as error:
            attempt["passed"] = False
            attempt["failure"] = str(error)
            raise
        roll, pitch, _ = quaternion_rpy(sane.orientation_body_to_ned_xyzw)
        attempt["state_sanity"] = {
            "linear_speed_m_s": float(np.linalg.norm(sane.linear_velocity_body_frd_m_s)),
            "angular_speed_rad_s": float(
                np.linalg.norm(sane.angular_velocity_body_frd_rad_s)
            ),
            "tilt_degrees": float(np.degrees(np.hypot(roll, pitch))),
            "passed": True,
        }

    def _pressure_depth_callback(self, message: FluidPressure) -> None:
        try:
            depth = pressure_to_depth_down_m(
                float(message.fluid_pressure),
                fluid_density_kg_m3=self.fluid_density,
                surface_pressure_pa=self.surface_pressure,
            )
        except ValueError:
            return
        self.latest_depth = depth
        self.depth_received_at = time.monotonic()

    def _run_for(self, target: np.ndarray, duration: float) -> None:
        bounded = clamp_manual_control(target, self.action_limit)
        end = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end:
            started = time.monotonic()
            self.current_command = slew_towards(
                self.current_command, bounded, self.action_slew_rate, self.period
            )
            self._send_command(self.current_command)
            rclpy.spin_once(self, timeout_sec=min(0.01, self.period))
            for _ in range(12):
                rclpy.spin_once(self, timeout_sec=0.0)
            self._verify_execution_heartbeat()
            remaining = self.period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)

    def _startup_background_velocity_ned(self) -> np.ndarray:
        # External current is held at zero until the formal episode starts.
        return np.zeros(3, dtype=np.float64)

    def _publish_external_current(self, elapsed_seconds: float | None) -> np.ndarray:
        if self.physics_domain is None or self.external_current_publisher is None:
            return np.zeros(3, dtype=np.float64)
        if elapsed_seconds is None or not self._external_current_active:
            velocity = np.zeros(3, dtype=np.float64)
        else:
            velocity = np.asarray(
                self.physics_domain.external_current_velocity_ned_m_s(
                    elapsed_seconds
                ),
                dtype=np.float64,
            )
        message = Vector3()
        message.x, message.y, message.z = (float(value) for value in velocity)
        self.external_current_publisher.publish(message)
        if self.physics_domain_runtime is not None:
            runtime = self.physics_domain_runtime["external_current"]
            if elapsed_seconds is None:
                if self._external_current_active:
                    runtime["terminal_zero_published"] = True
                else:
                    runtime["initialization_zero_published"] = True
            else:
                runtime["formal_samples_published"] += 1
                runtime["maximum_published_speed_m_s"] = max(
                    float(runtime["maximum_published_speed_m_s"]),
                    float(np.linalg.norm(velocity)),
                )
        return velocity

    def _begin_external_current_episode(self) -> None:
        self._external_current_active = True

    def _formal_episode_time_seconds(self, index: int) -> float:
        """Return the deterministic time used to evaluate episode profiles."""
        values = np.asarray(self.command_arrays["time_seconds"], dtype=np.float64)
        if index < 0 or index >= values.size:
            raise IndexError(f"episode reference index out of range: {index}")
        value = float(values[index])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("episode reference time must be finite and nonnegative")
        return value

    def _end_external_current_episode(self) -> None:
        if self._external_current_active:
            # The subscriber retains its last command, so explicitly clear it.
            self._publish_external_current(None)
        self._external_current_active = False

    def _wait_for_observation(self) -> None:
        deadline = time.monotonic() + 5.0
        while (
            rclpy.ok()
            and (
                self.latest_quaternion is None
                or self.latest_depth is None
                or self._last_vio_state is None
            )
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        if (
            self.latest_quaternion is None
            or self.latest_depth is None
            or self._last_vio_state is None
        ):
            raise RuntimeError("AHRS/depth/canonical VIO observation is unavailable")

    def _wait_for_startup_stability(self) -> None:
        """Gate formal startup using only deployable estimator/sensor streams."""
        deadline = time.monotonic() + self.startup_settle_timeout_seconds
        stable_since: float | None = None
        latest: dict[str, Any] = {}
        while rclpy.ok() and time.monotonic() < deadline:
            self.current_command.fill(0.0)
            self._send_command(self.current_command)
            rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.0)
            state = self._last_vio_state
            if (
                state is None
                or time.monotonic() - self._last_vio_received_at > self.maximum_vio_age
                or self.latest_quaternion is None
                or self.latest_angular_velocity is None
                or self.latest_depth is None
            ):
                continue
            speed = float(np.linalg.norm(state.linear_velocity_body_frd_m_s))
            angular_speed = float(np.linalg.norm(self.latest_angular_velocity))
            roll, pitch = _roll_pitch(self.latest_quaternion)
            tilt_degrees = float(np.degrees(np.hypot(roll, pitch)))
            depth_error = float(self.latest_depth - self.spawn_depth)
            latest = {
                "vio_body_speed_m_s": speed,
                "ahrs_angular_speed_rad_s": angular_speed,
                "ahrs_tilt_degrees": tilt_degrees,
                "pressure_depth_down_m": float(self.latest_depth),
                "configured_respawn_depth_down_m": self.spawn_depth,
                "pressure_depth_error_m": depth_error,
                "ground_truth_used": False,
            }
            stable = bool(
                speed <= self.startup_max_linear_speed
                and angular_speed <= self.startup_max_angular_speed
                and tilt_degrees <= self.startup_max_tilt_degrees
                and abs(depth_error) <= self.reset_max_depth_error
            )
            stable_since = (stable_since or time.monotonic()) if stable else None
            if (
                stable_since is not None
                and time.monotonic() - stable_since >= self.startup_settle_seconds
            ):
                self.startup_motion_gate.update(
                    {
                        "passed": True,
                        "observed": latest,
                        "linear_speed_basis": "canonical_vins_body_frd",
                        "attitude_rate_basis": "ardusub_ahrs",
                        "depth_basis": PRESSURE_DEPTH_SOURCE_ID,
                        "ground_truth_used": False,
                        "settled_for_seconds": time.monotonic() - stable_since,
                    }
                )
                return
            time.sleep(max(0.0, self.period - 0.02))
        self.startup_motion_gate.update(
            {"passed": False, "observed": latest, "ground_truth_used": False}
        )
        raise RuntimeError(
            "deployable startup state did not settle after respawn: "
            f"{latest or 'required streams unavailable'}"
        )

    def _activate_physics_domain(self) -> None:
        if self.physics_domain_runtime is None:
            return
        activation = self.physics_domain_runtime["current_activation"]
        assert self.enable_currents is not None
        if not self.enable_currents.wait_for_service(timeout_sec=10.0):
            activation["failure"] = "enable_currents service unavailable"
            raise RuntimeError("Stonefish enable_currents service is unavailable")
        activation["requested"] = True
        future = self.enable_currents.call_async(Trigger.Request())
        self._spin_until(future, 10.0, "Stonefish enable_currents service timed out")
        response = future.result()
        activation["response_received"] = response is not None
        activation["response_message"] = (
            str(response.message) if response is not None else ""
        )
        activation["passed"] = bool(response is not None and response.success)
        if not activation["passed"]:
            raise RuntimeError(
                f"Stonefish current activation failed: {activation['response_message']}"
            )
        subscriber_deadline = time.monotonic() + 2.0
        assert self.external_current_publisher is not None
        while (
            rclpy.ok()
            and self.external_current_publisher.get_subscription_count() < 1
            and time.monotonic() < subscriber_deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.02)
        if self.external_current_publisher.get_subscription_count() < 1:
            raise RuntimeError("Stonefish external-current subscriber is unavailable")
        self._publish_external_current(None)

    def _wait_for_reset_convergence(self) -> None:
        deadline = time.monotonic() + self.reset_convergence_timeout
        stable_since: float | None = None
        latest: dict[str, float] = {}
        while rclpy.ok() and time.monotonic() < deadline:
            self._send_command(np.zeros(6))
            rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            if (
                self.latest_depth is None
                or self.latest_angular_velocity is None
            ):
                continue
            depth_error = abs(self.latest_depth - self.spawn_depth)
            angular_speed = float(np.linalg.norm(self.latest_angular_velocity))
            latest = {
                "sensor_depth_down_m": self.latest_depth,
                "depth_source": self.depth_source,
                "configured_respawn_depth_down_m": self.spawn_depth,
                "depth_error_m": depth_error,
                "angular_speed_rad_s": angular_speed,
            }
            stable = (
                time.monotonic() - self.ahrs_received_at <= 0.35
                and time.monotonic() - self.depth_received_at <= 0.35
                and depth_error <= self.reset_max_depth_error
                and angular_speed <= self.reset_max_angular_speed
            )
            stable_since = (stable_since or time.monotonic()) if stable else None
            if (
                stable_since is not None
                and time.monotonic() - stable_since >= self.reset_convergence_seconds
            ):
                self.reset_convergence_gate.update({"passed": True, "observed": latest})
                return
            time.sleep(max(0.0, self.period - 0.02))
        self.reset_convergence_gate.update({"passed": False, "observed": latest})
        raise RuntimeError(
            f"estimator/actuator reset did not converge: {latest or 'required streams unavailable'}"
        )

    def _initialize_pid_anchor(self) -> None:
        state = self._deployable_control_state()
        self.pid_anchor_position_ned_m = state.position_ned_m.copy()
        self.pid_anchor_yaw_rad = float(
            quaternion_rpy(state.orientation_body_to_ned_xyzw)[2]
        )
        self.pid_controller.reset()
        self.pid_trace = {
            "time_seconds": [],
            "external_current_time_seconds": [],
            "position_preview_ned_m": [],
            "reference_velocity_body_frd_m_s": [],
            "reference_acceleration_body_frd_m_s2": [],
            "yaw_preview_rad": [],
            "reference_yaw_rate_rad_s": [],
            "actual_position_ned_m": [],
            "actual_orientation_body_to_ned_xyzw": [],
            "actual_velocity_body_frd_m_s": [],
            "actual_yaw_rad": [],
            "actual_yaw_rate_rad_s": [],
            "estimated_position_ned_m": [],
            "estimated_orientation_body_to_ned_xyzw": [],
            "estimated_velocity_body_frd_m_s": [],
            "estimated_angular_velocity_body_frd_rad_s": [],
            "estimated_yaw_rad": [],
            "external_current_velocity_ned_m_s": [],
            "vio_measurement_latency_seconds": [],
            "requested_4dof_intent": [],
            "requested_manual_control_normalized": [],
            "executed_4dof_intent": [],
            "executed_manual_control_normalized": [],
            "velocity_error_body_frd_m_s": [],
            "yaw_rate_error_rad_s": [],
            "linear_feedforward_term": [],
            "linear_acceleration_feedforward_term": [],
            "linear_proportional_term": [],
            "linear_integral_term": [],
            "yaw_feedforward_term": [],
            "yaw_proportional_term": [],
            "yaw_integral_term": [],
            "roll_rad": [],
            "pitch_rad": [],
            "manual_saturated": [],
            "state_source": [],
        }

    def _current_vio_state(self) -> VioState:
        if (
            self._last_vio_state is None
            or time.monotonic() - self._last_vio_received_at > self.maximum_vio_age
        ):
            raise RuntimeError("canonical VIO state became stale")
        return self._last_vio_state

    def _deployable_control_state(self) -> VioState:
        """Fuse pressure depth into VIO state without consulting simulator truth."""
        state = self._current_vio_state()
        if self.latest_depth is None or not math.isfinite(self.latest_depth):
            raise RuntimeError("pressure-derived depth is unavailable")
        position = state.position_ned_m.copy()
        position[2] = float(self.latest_depth)
        return VioState(
            position_ned_m=position,
            orientation_body_to_ned_xyzw=state.orientation_body_to_ned_xyzw.copy(),
            linear_velocity_body_frd_m_s=state.linear_velocity_body_frd_m_s.copy(),
            angular_velocity_body_frd_rad_s=state.angular_velocity_body_frd_rad_s.copy(),
            timestamp_seconds=state.timestamp_seconds,
            source=f"{state.source}+{PRESSURE_DEPTH_SOURCE_ID}",
        )

    def _trajectory_reference(self, index: int) -> BodyTwistReference:
        if self.pid_anchor_position_ned_m is None or self.pid_anchor_yaw_rad is None:
            raise RuntimeError("PID reference anchor has not been initialized")
        source = np.asarray(
            self.command_arrays["position_preview_ned_m"][index], dtype=np.float64
        )
        origin = np.asarray(
            self.command_arrays["position_preview_ned_m"][0], dtype=np.float64
        )
        relative = source - origin
        c, s = math.cos(self.pid_anchor_yaw_rad), math.sin(self.pid_anchor_yaw_rad)
        world_relative = np.asarray(
            (
                c * relative[0] - s * relative[1],
                s * relative[0] + c * relative[1],
                relative[2],
            ),
            dtype=np.float64,
        )
        reference_velocity = np.asarray(
            self.command_arrays["body_twist_reference"][index, :3], dtype=np.float64
        )
        if index == 0:
            reference_acceleration = np.zeros(3, dtype=np.float64)
        else:
            previous_velocity = np.asarray(
                self.command_arrays["body_twist_reference"][index - 1, :3],
                dtype=np.float64,
            )
            delta_time = float(
                self.command_arrays["time_seconds"][index]
                - self.command_arrays["time_seconds"][index - 1]
            )
            reference_acceleration = (
                reference_velocity - previous_velocity
            ) / delta_time
        return BodyTwistReference(
            position_preview_ned_m=self.pid_anchor_position_ned_m + world_relative,
            yaw_preview_rad=self.pid_anchor_yaw_rad
            + float(
                self.command_arrays["yaw_preview_rad"][index]
                - self.command_arrays["yaw_preview_rad"][0]
            ),
            linear_velocity_body_frd_m_s=reference_velocity,
            yaw_rate_rad_s=float(self.command_arrays["body_twist_reference"][index, 3]),
            linear_acceleration_body_frd_m_s2=reference_acceleration,
        )

    def _simulation_audit_state(self) -> VioState:
        """Return Stonefish truth for the separate RMSE trace only."""
        if (
            self.validation_odometry is None
            or time.monotonic() - self.validation_odometry_received_at > 0.20
        ):
            raise RuntimeError("simulation truth audit source became stale")
        return ground_truth_odometry_as_vio(self.validation_odometry)

    def _pid_command(self, index: int, elapsed_seconds: float) -> np.ndarray:
        state = self._deployable_control_state()
        reference = self._trajectory_reference(index)
        output = self.pid_controller.step(reference, state, self.period)
        truth = self._simulation_audit_state()
        diagnostics = output.diagnostics
        manual = output.manual_control_normalized
        for name, value in {
            "time_seconds": elapsed_seconds,
            "position_preview_ned_m": reference.position_preview_ned_m.copy(),
            "reference_velocity_body_frd_m_s": reference.linear_velocity_body_frd_m_s.copy(),
            "reference_acceleration_body_frd_m_s2": (
                reference.linear_acceleration_body_frd_m_s2.copy()
            ),
            "yaw_preview_rad": reference.yaw_preview_rad,
            "reference_yaw_rate_rad_s": reference.yaw_rate_rad_s,
            "actual_position_ned_m": truth.position_ned_m.copy(),
            "actual_orientation_body_to_ned_xyzw": (
                truth.orientation_body_to_ned_xyzw.copy()
            ),
            "actual_velocity_body_frd_m_s": truth.linear_velocity_body_frd_m_s.copy(),
            "actual_yaw_rad": quaternion_rpy(
                truth.orientation_body_to_ned_xyzw
            )[2],
            "actual_yaw_rate_rad_s": truth.angular_velocity_body_frd_rad_s[2],
            "estimated_position_ned_m": state.position_ned_m.copy(),
            "estimated_orientation_body_to_ned_xyzw": (
                state.orientation_body_to_ned_xyzw.copy()
            ),
            "estimated_velocity_body_frd_m_s": (
                state.linear_velocity_body_frd_m_s.copy()
            ),
            "estimated_angular_velocity_body_frd_rad_s": (
                state.angular_velocity_body_frd_rad_s.copy()
            ),
            "estimated_yaw_rad": diagnostics["yaw_rad"],
            "vio_measurement_latency_seconds": max(
                0.0, truth.timestamp_seconds - state.timestamp_seconds
            ),
            "requested_4dof_intent": manual[[4, 5, 2, 3]].copy(),
            "requested_manual_control_normalized": manual.copy(),
            "velocity_error_body_frd_m_s": diagnostics[
                "velocity_error_body_m_s"
            ].copy(),
            "yaw_rate_error_rad_s": diagnostics["yaw_rate_error_rad_s"],
            "linear_feedforward_term": diagnostics["linear_feedforward_term"].copy(),
            "linear_acceleration_feedforward_term": diagnostics[
                "linear_acceleration_feedforward_term"
            ].copy(),
            "linear_proportional_term": diagnostics["linear_proportional_term"].copy(),
            "linear_integral_term": diagnostics["linear_integral_term"].copy(),
            "yaw_feedforward_term": diagnostics["yaw_feedforward_term"],
            "yaw_proportional_term": diagnostics["yaw_proportional_term"],
            "yaw_integral_term": diagnostics["yaw_integral_term"],
            "roll_rad": diagnostics["roll_rad"],
            "pitch_rad": diagnostics["pitch_rad"],
            "manual_saturated": diagnostics["manual_saturated"],
            "state_source": diagnostics["state_source"],
        }.items():
            self.pid_trace[name].append(value)
        return clamp_manual_control(output.manual_control_normalized, self.action_limit)

    def _apply_command_slew(self, requested: np.ndarray) -> np.ndarray:
        """Apply the collector's default uniform MANUAL command-rate limit."""
        return slew_towards(
            self.current_command, requested, self.action_slew_rate, self.period
        )

    def _save_pid_trace(self) -> None:
        if not self.pid_trace.get("time_seconds"):
            return
        path = self.pid_trace_path
        if not path and self.report_path:
            path = str(
                Path(self.report_path)
                .expanduser()
                .resolve()
                .with_suffix(".pid_trace.npz")
            )
        if not path:
            return
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(values) for name, values in self.pid_trace.items()}
        np.savez_compressed(target, **arrays)
        self.pid_trace_path = str(target)

    def _safety_gate(self) -> None:
        if (
            time.monotonic() - self.ahrs_received_at > 0.35
            or time.monotonic() - self.depth_received_at > 0.35
        ):
            raise RuntimeError("AHRS/depth observation became stale")
        self._current_vio_state()
        assert self.latest_quaternion is not None and self.latest_depth is not None
        roll, pitch = _roll_pitch(self.latest_quaternion)
        tilt_degrees = math.degrees(math.hypot(roll, pitch))
        if tilt_degrees > self.envelope["tilt_degrees"]:
            raise RuntimeError(f"tilt safety limit exceeded: {tilt_degrees:.2f} deg")
        if not self.minimum_depth <= self.latest_depth <= self.maximum_depth:
            raise RuntimeError(
                f"depth safety limit exceeded: {self.latest_depth:.3f} m"
            )

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        metadata = load_metadata(episode_path)
        duration = float(self.command_provenance["duration_seconds"])
        warmup = float(self.command_provenance["warmup_seconds"])
        cooldown = float(self.command_provenance["cooldown_seconds"])
        metadata["data_split"] = self.command_provenance["data_split"]
        metadata["command_provenance"] = self.command_provenance
        metadata["episode_reset_convergence"] = self.reset_convergence_gate
        metadata["vins_reset_lifecycle"] = self.vins_reset_lifecycle
        metadata["controller_depth_sensor"] = {
            "source_id": PRESSURE_DEPTH_SOURCE_ID,
            "topic": self.pressure_topic,
            "message_type": "sensor_msgs/msg/FluidPressure",
            "surface_pressure_pa": self.surface_pressure,
            "fluid_density_kg_m3": self.fluid_density,
            "conversion": PRESSURE_DEPTH_CONVERSION_ID,
            "replaces_vio_position_z_for_control": True,
            "ground_truth_position_z_used": False,
        }
        metadata["episode_timing"] = {
            "simulation_paused_during_startup": False,
            "recording_starts_after_reset_convergence": True,
            "pre_record_settle_seconds": self.pre_record_settle_seconds,
            "command_warmup_seconds": warmup,
            "command_cooldown_seconds": cooldown,
            "raw_samples_retained": True,
            "training_target_window": {
                "start_seconds": warmup,
                "end_seconds": duration - cooldown,
            },
        }
        metadata["controller_provenance"] = {
            "controller_id": "fixed_body_twist_velocity_rate_pid_v2",
            "execution_mode": self.mode,
            "state_source": self.vio_source_id,
            "state_topic": self.vio_odometry_topic,
            "state_contract": "world_ned pose + body_frd linear/angular velocity",
            "feedback_access": "deployable_vio_odometry",
            "pressure_depth_replaces_vio_position_z": True,
            "ground_truth_feedback": False,
            "simulation_truth_use": "separate_rmse_audit_trace_only",
            "position_outer_loop_enabled": bool(
                np.any(np.asarray(self.pid_controller.config.position_kp_body) > 0.0)
            ),
            "yaw_outer_loop_enabled": bool(
                self.pid_controller.config.yaw_position_kp > 0.0
            ),
            "preview_geometry_used_for_feedback": False,
            "normalized_4dof_intent_order": [
                "surge",
                "sway",
                "heave_up",
                "yaw",
            ],
            "pid_config": self.pid_controller.config.as_dict(),
            "pid_config_id": self.pid_config_id,
            "pid_config_path": self.pid_config_path or None,
            "trace_path": self.pid_trace_path or None,
        }
        ground_truth = metadata.get("ground_truth_recording", {})
        ground_truth["enabled"] = False
        ground_truth["arrays"] = []
        ground_truth["ground_truth_fields_excluded"] = True
        ground_truth["feedback_access_allowed"] = False
        ground_truth["simulation_state_generation_allowed"] = False
        ground_truth["simulation_state_generation_use"] = None
        ground_truth["supervisor_and_metric_access_only"] = True
        metadata["ground_truth_recording"] = ground_truth
        if self.domain_provenance is not None:
            metadata["domain_split"] = self.domain_provenance["domain_split"]
            metadata["physics_domain_provenance"] = self.domain_provenance
            metadata["physics_domain_runtime"] = self.physics_domain_runtime
            metadata["joint_split_cell"] = (
                f"{self.command_provenance['data_split']}_command__{self.domain_provenance['domain_split']}_domain"
            )
        write_metadata(episode_path, metadata)

    def run(self) -> dict[str, Any]:
        self._connect_ardusub()
        recorder_started = False
        episode_path = ""
        motor_samples: list[np.ndarray] = []
        started = time.monotonic()
        failure: str | None = None
        try:
            self._activate_physics_domain()
            if self.simulation_reset:
                self._prepare_vehicle()
            else:
                self._arm()
                self._run_for(np.zeros(6), 1.0)
            self._wait_for_observation()
            self._wait_for_reset_convergence()
            if self.pre_record_settle_seconds:
                self._run_for(np.zeros(6), self.pre_record_settle_seconds)
            self._initialize_pid_anchor()
            if self.manage_recorder:
                self._start_recorder()
                recorder_started = True
            self._begin_external_current_episode()
            count = len(self.command_arrays["body_twist_reference"])
            for index in range(count):
                tick_started = time.monotonic()
                elapsed_seconds = self._formal_episode_time_seconds(index)
                current_velocity = self._publish_external_current(elapsed_seconds)
                self.pid_trace["external_current_velocity_ned_m_s"].append(
                    current_velocity.copy()
                )
                self.pid_trace["external_current_time_seconds"].append(
                    elapsed_seconds
                )
                requested = self._pid_command(index, elapsed_seconds)
                self.current_command = self._apply_command_slew(requested)
                self._send_command(self.current_command)
                executed = self.current_command.copy()
                # Report saturation of the command actually sent.  A raw
                # controller request may be at a limit while the slew limiter
                # keeps the executed command safely inside it.
                self.pid_trace["manual_saturated"][-1] = bool(
                    np.any(np.abs(executed) >= self.action_limit - 1e-9)
                )
                self.pid_trace["executed_manual_control_normalized"].append(executed)
                self.pid_trace["executed_4dof_intent"].append(
                    executed[[4, 5, 2, 3]].copy()
                )
                rclpy.spin_once(self, timeout_sec=min(0.01, self.period))
                for _ in range(12):
                    rclpy.spin_once(self, timeout_sec=0.0)
                self._verify_execution_heartbeat()
                self._safety_gate()
                if self.applied_motors is not None:
                    motor_samples.append(self.applied_motors.copy())
                remaining = self.period - (time.monotonic() - tick_started)
                if remaining > 0.0:
                    time.sleep(remaining)
        except Exception as error:
            failure = str(error)
        finally:
            self._end_external_current_episode()
            try:
                self._save_pid_trace()
            except OSError as error:
                failure = failure or f"could not save PID trace: {error}"
            if recorder_started:
                try:
                    episode_path = self._stop_recorder()
                    self._annotate_rollout_provenance(episode_path)
                except RuntimeError as error:
                    failure = failure or str(error)
            self._neutral_and_disarm(release=True)

        motors = (
            np.stack(motor_samples)
            if motor_samples
            else np.empty((0, 6), dtype=np.float64)
        )
        saturation_fraction = (
            float(np.mean(np.any(np.abs(motors) >= 0.999, axis=1)))
            if motors.size
            else 1.0
        )
        episode_report = (
            inspect_episode(episode_path)
            if episode_path
            else {"pass": False, "error": "episode was not saved"}
        )
        pid_summary: dict[str, Any] = {}
        if self.pid_trace.get("time_seconds"):
            velocity_error = np.asarray(self.pid_trace["velocity_error_body_frd_m_s"])
            commands = np.asarray(self.pid_trace["executed_manual_control_normalized"])
            reference_series = np.column_stack(
                (
                    np.asarray(self.pid_trace["reference_velocity_body_frd_m_s"]),
                    np.asarray(self.pid_trace["reference_yaw_rate_rad_s"]),
                )
            )
            actual_series = np.column_stack(
                (
                    np.asarray(self.pid_trace["actual_velocity_body_frd_m_s"]),
                    np.asarray(self.pid_trace["actual_yaw_rate_rad_s"]),
                )
            )
            command_series = np.asarray(self.pid_trace["executed_4dof_intent"])
            estimated_position = np.asarray(
                self.pid_trace["estimated_position_ned_m"]
            )
            actual_position = np.asarray(self.pid_trace["actual_position_ned_m"])
            estimated_velocity = np.asarray(
                self.pid_trace["estimated_velocity_body_frd_m_s"]
            )
            actual_velocity = np.asarray(
                self.pid_trace["actual_velocity_body_frd_m_s"]
            )
            estimated_orientation = np.asarray(
                self.pid_trace["estimated_orientation_body_to_ned_xyzw"]
            )
            actual_orientation = np.asarray(
                self.pid_trace["actual_orientation_body_to_ned_xyzw"]
            )
            estimated_rotations = np.stack(
                [quaternion_rotation_matrix(value) for value in estimated_orientation]
            )
            actual_rotations = np.stack(
                [quaternion_rotation_matrix(value) for value in actual_orientation]
            )
            # VINS has an arbitrary local world origin and yaw gauge. Align GT
            # once at the first controller sample for audit metrics only. This
            # transform is never exposed to the controller.
            estimate_from_truth = estimated_rotations[0] @ actual_rotations[0].T
            aligned_actual_position = estimated_position[0] + (
                estimate_from_truth @ (actual_position - actual_position[0]).T
            ).T
            aligned_actual_rotations = np.einsum(
                "ij,njk->nik", estimate_from_truth, actual_rotations
            )
            relative_rotations = np.einsum(
                "nij,njk->nik",
                np.transpose(estimated_rotations, (0, 2, 1)),
                aligned_actual_rotations,
            )
            orientation_cosine = np.clip(
                (np.trace(relative_rotations, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
            orientation_error = np.arccos(orientation_cosine)
            aligned_actual_yaw = np.arctan2(
                aligned_actual_rotations[:, 1, 0],
                aligned_actual_rotations[:, 0, 0],
            )
            estimated_yaw = np.arctan2(
                estimated_rotations[:, 1, 0], estimated_rotations[:, 0, 0]
            )
            yaw_estimate_error = np.arctan2(
                np.sin(estimated_yaw - aligned_actual_yaw),
                np.cos(estimated_yaw - aligned_actual_yaw),
            )
            sample_count = len(reference_series)
            tracking_metrics = (
                summarize_body_twist_tracking(
                    reference_series,
                    actual_series,
                    command_series,
                    self.period,
                )
                if sample_count >= 8
                else {
                    "available": False,
                    "reason": "fewer than 8 controller samples",
                    "sample_count": sample_count,
                }
            )
            trajectory_metrics = (
                summarize_trajectory_tracking(
                    np.asarray(self.pid_trace["position_preview_ned_m"]),
                    aligned_actual_position,
                    np.asarray(self.pid_trace["yaw_preview_rad"]),
                    aligned_actual_yaw,
                )
                if sample_count >= 2
                else {
                    "available": False,
                    "reason": "fewer than 2 controller samples",
                    "sample_count": sample_count,
                }
            )
            pid_summary = {
                "velocity_error_rms_m_s": np.sqrt(
                    np.mean(velocity_error**2, axis=0)
                ).tolist(),
                "yaw_rate_error_rms_rad_s": float(
                    np.sqrt(
                        np.mean(np.asarray(self.pid_trace["yaw_rate_error_rad_s"]) ** 2)
                    )
                ),
                "manual_command_rms": np.sqrt(np.mean(commands**2, axis=0)).tolist(),
                "manual_command_peak": np.max(np.abs(commands), axis=0).tolist(),
                "manual_saturation_fraction": float(
                    np.mean(np.asarray(self.pid_trace["manual_saturated"], dtype=bool))
                ),
                "state_source": sorted(set(self.pid_trace["state_source"])),
                "tracking_metrics": tracking_metrics,
                "trajectory_metrics": trajectory_metrics,
                "vio_state_error_metrics": {
                    "audit_alignment": "fixed first-sample SE(3); GT metrics only",
                    "position_rmse_ned_m": np.sqrt(
                        np.mean(
                            (estimated_position - aligned_actual_position) ** 2,
                            axis=0,
                        )
                    ).tolist(),
                    "linear_velocity_rmse_body_frd_m_s": np.sqrt(
                        np.mean((estimated_velocity - actual_velocity) ** 2, axis=0)
                    ).tolist(),
                    "yaw_rmse_rad": float(
                        np.sqrt(np.mean(yaw_estimate_error**2))
                    ),
                    "orientation_angle_rmse_rad": float(
                        np.sqrt(np.mean(orientation_error**2))
                    ),
                    "latency_p50_seconds": float(
                        np.percentile(
                            self.pid_trace["vio_measurement_latency_seconds"], 50
                        )
                    ),
                    "latency_p95_seconds": float(
                        np.percentile(
                            self.pid_trace["vio_measurement_latency_seconds"], 95
                        )
                    ),
                    "latency_max_seconds": float(
                        np.max(self.pid_trace["vio_measurement_latency_seconds"])
                    ),
                },
            }
        report: dict[str, Any] = {
            "protocol_id": "slow_multisine_vins_rate_pid_rollout_v6",
            "execution_mode": self.mode,
            "execution_pass": bool(
                failure is None
                and motors.size
                and saturation_fraction <= self.envelope["motor_saturation_ratio"]
                and pid_summary.get("manual_saturation_fraction", 1.0) <= 0.01
                and float(episode_report.get("maximum_motor_output", 1.0))
                <= self.maximum_accepted_motor_output
                and episode_report.get("pass")
            ),
            "failure": failure,
            "command_path": str(self.command_path),
            "data_split": self.command_provenance["data_split"],
            "command_recipe": self.command_provenance["recipe"],
            "training_bank_protocol_id": self.command_provenance[
                "training_bank_protocol_id"
            ],
            "physics_domain": self.domain_provenance,
            "physics_domain_runtime": self.physics_domain_runtime,
            "episode_reset_convergence": self.reset_convergence_gate,
            "vins_reset_lifecycle": self.vins_reset_lifecycle,
            "episode_path": episode_path,
            "episode_contract_pass": bool(episode_report.get("pass")),
            "episode_contract_failed_checks": [
                name
                for name, passed in episode_report.get("checks", {}).items()
                if not passed
            ],
            "elapsed_seconds": time.monotonic() - started,
            "motor_sample_count": int(motors.shape[0]),
            "motor_saturation_fraction": saturation_fraction,
            "manual_saturation_fraction": float(
                pid_summary.get("manual_saturation_fraction", 1.0)
            ),
            "maximum_episode_motor_output": float(
                episode_report.get("maximum_motor_output", 1.0)
            ),
            "maximum_accepted_motor_output": self.maximum_accepted_motor_output,
            "simulation_truth_audit": episode_report.get("simulation_truth_audit"),
            "controller_provenance": {
                "controller_id": "fixed_body_twist_velocity_rate_pid_v2",
                "state_source": self.vio_source_id,
                "state_topic": self.vio_odometry_topic,
                "position_outer_loop_enabled": bool(
                    np.any(
                        np.asarray(self.pid_controller.config.position_kp_body)
                        > 0.0
                    )
                ),
                "yaw_outer_loop_enabled": bool(
                    self.pid_controller.config.yaw_position_kp > 0.0
                ),
                "preview_geometry_used_for_feedback": False,
                "normalized_4dof_intent_order": [
                    "surge",
                    "sway",
                    "heave_up",
                    "yaw",
                ],
                "pid_config": self.pid_controller.config.as_dict(),
                "pid_config_id": self.pid_config_id,
                "pid_config_path": self.pid_config_path or None,
                "trace_path": self.pid_trace_path or None,
                "tracking_summary": pid_summary,
            },
        }
        if self.report_path:
            report_path = Path(self.report_path).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return report


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = ManualRollout()
    try:
        report = node.run()
        node.get_logger().info(
            f"SLOW MULTISINE ROLLOUT COMPLETE: execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
