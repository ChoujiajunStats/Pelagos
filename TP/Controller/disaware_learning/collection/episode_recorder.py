"""Record MANUAL, six motors, simulated/real VIO, IMU/AHRS, and depth."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import time

from nav_msgs.msg import Odometry
import numpy as np
from rclpy.executors import ExternalShutdownException
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure, Imu
from stonefish_ros2.msg import ThrusterState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

from disaware_learning.control.ardusub_runtime import ArduSubRuntimeContract
from disaware_learning.control.contract import (
    ACTUATOR_COUNT,
    EXECUTION_MODE,
    MANUAL_CONTROL_AUTHORITY_MASK,
    MANUAL_CONTROL_ORDER,
    MANUAL_CONTROL_SEMANTICS,
)
from disaware_learning.control.rc import pwm_to_normalized
from disaware_learning.data.episode_buffers import create_sample_buffers
from disaware_learning.data.episode_io import write_metadata
from disaware_learning.data.episode_schema import (
    DEPLOYABLE_OBSERVATION_ARRAYS,
    FORMAT_VERSION,
    GROUND_TRUTH_ARRAYS,
    OBSERVATION_FEATURE_CONTRACT_ID,
    PRIVILEGED_ACTUATOR_ARRAYS,
    PRIVILEGED_ACTUATOR_NATIVE_ARRAYS,
    PROTOCOL_ID,
)
from disaware_learning.data.recording_contract import evaluate_recording_contract
from disaware_learning.estimation.pressure_depth import (
    PRESSURE_DEPTH_CONVERSION_ID,
    PRESSURE_DEPTH_SOURCE_ID,
    pressure_to_depth_down_m,
)


def sample_due(last_sample: float, now: float, period: float) -> bool:
    """Return whether a source timestamp reached the next sample boundary."""
    return now - last_sample >= period * 0.95


def _message_stamp(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class EpisodeRecorder(Node):
    """Own the v15 deployable sensor/state boundary."""

    def __init__(self) -> None:
        super().__init__("episode_recorder")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("output_dir", "~/disaware_datasets")
        self.declare_parameter("sample_rate", 25.0)
        self.declare_parameter("max_duration", 0.0)
        self.declare_parameter("autostart", True)
        self.declare_parameter("record_ground_truth", False)
        self.declare_parameter("vio_odometry_topic", "")
        self.declare_parameter("vio_source_id", "canonical_vio_odometry")
        self.declare_parameter("depth_source", "fluid_pressure")
        self.declare_parameter("pressure_topic", "")
        self.declare_parameter("fluid_density_kg_m3", 1025.0)
        self.declare_parameter("surface_pressure_pa", 0.0)
        self.declare_parameter("configured_execution_mode", EXECUTION_MODE)
        self.declare_parameter("enforce_runtime_contract", True)
        self.declare_parameter("ardusub_contract_endpoint", "udpin:0.0.0.0:14553")
        # ArduSub multiplexes RC_CHANNELS with pressure/depth telemetry and can
        # deliver them in 250 ms bursts even when a higher rate is requested.
        self.declare_parameter("max_control_age", 0.30)
        self.declare_parameter("max_motor_age", 0.15)
        self.declare_parameter("max_vio_age", 0.20)
        self.declare_parameter("max_ahrs_age", 0.25)
        self.declare_parameter("max_imu_age", 0.15)
        self.declare_parameter("max_depth_age", 0.35)
        self.declare_parameter("max_pressure_age", 0.35)

        self.vehicle_name = str(self.get_parameter("vehicle_name").value).strip("/")
        self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.sample_rate = float(self.get_parameter("sample_rate").value)
        if not math.isfinite(self.sample_rate) or self.sample_rate < 2.0:
            raise ValueError("sample_rate must be finite and at least 2 Hz")
        self.period = 1.0 / self.sample_rate
        self.max_duration = max(0.0, float(self.get_parameter("max_duration").value))
        self.record_ground_truth = bool(self.get_parameter("record_ground_truth").value)
        configured_vio_topic = str(
            self.get_parameter("vio_odometry_topic").value
        ).strip()
        self.vio_odometry_topic = configured_vio_topic or (
            f"/{self.vehicle_name}/vio/odometry"
        )
        self.vio_source_id = str(self.get_parameter("vio_source_id").value).strip()
        if not self.vio_odometry_topic or not self.vio_source_id:
            raise ValueError("VIO topic and source id are required")
        self.depth_source = str(self.get_parameter("depth_source").value).strip()
        if self.depth_source != "fluid_pressure":
            raise ValueError(
                "v15 collection requires depth_source='fluid_pressure'; "
                "derived or navigation-state depth is not admissible"
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
        self.configured_execution_mode = (
            str(self.get_parameter("configured_execution_mode").value).strip().upper()
        )
        if self.configured_execution_mode != EXECUTION_MODE:
            raise ValueError(f"configured_execution_mode must be {EXECUTION_MODE}")
        self.enforce_runtime_contract = bool(
            self.get_parameter("enforce_runtime_contract").value
        )
        self.age_limits = {
            "control": self._positive_parameter("max_control_age"),
            "motor": self._positive_parameter("max_motor_age"),
            "vio": self._positive_parameter("max_vio_age"),
            "ahrs": self._positive_parameter("max_ahrs_age"),
            "imu": self._positive_parameter("max_imu_age"),
            "depth": self._positive_parameter("max_depth_age"),
            "pressure": self._positive_parameter("max_pressure_age"),
        }

        endpoint = str(self.get_parameter("ardusub_contract_endpoint").value).strip()
        self.runtime = ArduSubRuntimeContract(endpoint) if endpoint else None

        self.samples = create_sample_buffers(
            record_ground_truth=self.record_ground_truth
        )
        self.events: dict[str, list] = {
            "vio": [],
            "ahrs": [],
            "imu": [],
            "depth": [],
            "pressure": [],
            "manual": [],
            "motor": [],
            "thruster_state": [],
        }
        self.latest: dict[str, tuple[float, object] | None] = {
            "vio": None,
            "ahrs": None,
            "imu": None,
            "depth": None,
            "pressure": None,
            "manual_normalized": None,
            "manual_pwm": None,
            "motor_normalized": None,
            "motor_pwm": None,
            "ground_truth": None,
            "thruster_state": None,
        }
        self.recording = False
        self.started_at: float | None = None
        self.first_sample_at: float | None = None
        self.last_sample_at = -math.inf
        self.last_path: Path | None = None
        self._last_vio_source_stamp: float | None = None

        prefix = f"/{self.vehicle_name}"
        self.create_subscription(
            Odometry,
            self.vio_odometry_topic,
            self._vio_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            f"{prefix}/ardusub/ahrs",
            self._ahrs_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            f"{prefix}/ardusub/imu",
            self._imu_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            FluidPressure,
            self.pressure_topic,
            self._fluid_pressure_depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float64MultiArray,
            f"{prefix}/ardusub/manual_control_pwm",
            self._manual_control_pwm_callback,
            20,
        )
        self.create_subscription(
            Float64MultiArray,
            f"{prefix}/ardusub/motor_output_pwm",
            self._motor_output_pwm_callback,
            20,
        )
        if self.record_ground_truth:
            self.create_subscription(
                Odometry,
                f"{prefix}/odometry",
                self._ground_truth_callback,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                ThrusterState,
                f"{prefix}/simulation/thruster_state",
                self._thruster_state_callback,
                qos_profile_sensor_data,
            )

        self.create_service(Trigger, "~/start", self._start_service)
        self.create_service(Trigger, "~/stop", self._stop_service)
        self.create_service(Trigger, "~/status", self._status_service)
        self.create_timer(min(0.01, 0.25 * self.period), self._tick)
        if bool(self.get_parameter("autostart").value):
            self.start_recording()

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    @staticmethod
    def _pwm_vector(message, width: int) -> np.ndarray | None:
        values = np.asarray(message.data, dtype=np.float64)
        if (
            values.shape != (width,)
            or not np.all(np.isfinite(values))
            or np.min(values) < 1100.0
            or np.max(values) > 1900.0
        ):
            return None
        return values

    def _ahrs_callback(self, message: Imu) -> None:
        now = time.monotonic()
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
        if not np.all(np.isfinite(quaternion)) or not np.all(np.isfinite(rate)):
            return
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-9:
            return
        value = (quaternion / norm, rate, _message_stamp(message))
        self.latest["ahrs"] = (now, value)
        if self.recording:
            self.events["ahrs"].append((now, *value))

    def _vio_callback(self, message: Odometry) -> None:
        if message.header.frame_id and message.header.frame_id != "world_ned":
            return
        pose = message.pose.pose
        twist = message.twist.twist
        position = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
        )
        orientation = np.asarray(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        linear_velocity = np.asarray(
            [twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float64
        )
        angular_velocity = np.asarray(
            [twist.angular.x, twist.angular.y, twist.angular.z], dtype=np.float64
        )
        if (
            not np.all(np.isfinite(position))
            or not np.all(np.isfinite(orientation))
            or not np.all(np.isfinite(linear_velocity))
            or not np.all(np.isfinite(angular_velocity))
            or np.linalg.norm(orientation) <= 1e-9
        ):
            return
        orientation /= np.linalg.norm(orientation)
        now = time.monotonic()
        source_stamp = _message_stamp(message)
        if (
            self._last_vio_source_stamp is not None
            and source_stamp <= self._last_vio_source_stamp + 1e-9
        ):
            return
        self._last_vio_source_stamp = source_stamp
        value = (
            position,
            orientation,
            linear_velocity,
            angular_velocity,
            source_stamp,
        )
        self.latest["vio"] = (now, value)
        if self.recording:
            self.events["vio"].append((now, *value))

    def _imu_callback(self, message: Imu) -> None:
        now = time.monotonic()
        rate = np.asarray(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        acceleration = np.asarray(
            [
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(rate)) or not np.all(np.isfinite(acceleration)):
            return
        value = (rate, acceleration, _message_stamp(message))
        self.latest["imu"] = (now, value)
        if self.recording:
            self.events["imu"].append((now, *value))

    def _depth_callback(self, message: Float64) -> None:
        value = float(message.data)
        if not math.isfinite(value):
            return
        now = time.monotonic()
        self.latest["depth"] = (now, value)
        if self.recording:
            self.events["depth"].append((now, value))

    def _pressure_callback(self, message: FluidPressure) -> None:
        pressure = float(message.fluid_pressure)
        variance = float(message.variance)
        if (
            not math.isfinite(pressure)
            or pressure <= 0.0
            or not math.isfinite(variance)
            or variance < 0.0
        ):
            return
        now = time.monotonic()
        value = (pressure, variance, _message_stamp(message))
        self.latest["pressure"] = (now, value)
        if self.recording:
            self.events["pressure"].append((now, *value))

    def _fluid_pressure_depth_callback(self, message: FluidPressure) -> None:
        pressure = float(message.fluid_pressure)
        variance = float(message.variance)
        try:
            depth = pressure_to_depth_down_m(
                pressure,
                fluid_density_kg_m3=self.fluid_density,
                surface_pressure_pa=self.surface_pressure,
            )
        except ValueError:
            return
        if not math.isfinite(variance) or variance < 0.0:
            return
        now = time.monotonic()
        pressure_value = (pressure, variance, _message_stamp(message))
        self.latest["pressure"] = (now, pressure_value)
        self.latest["depth"] = (now, depth)
        if self.recording:
            self.events["pressure"].append((now, *pressure_value))
            self.events["depth"].append((now, depth))

    def _manual_control_pwm_callback(self, message: Float64MultiArray) -> None:
        values = self._pwm_vector(message, 8)
        if values is None:
            return
        now = time.monotonic()
        normalized = pwm_to_normalized(values[:6])
        self.latest["manual_normalized"] = (now, normalized)
        self.latest["manual_pwm"] = (now, values)
        if self.recording:
            self.events["manual"].append((now, normalized.copy(), values.copy()))

    def _motor_output_pwm_callback(self, message: Float64MultiArray) -> None:
        values = self._pwm_vector(message, ACTUATOR_COUNT)
        if values is None:
            return
        now = time.monotonic()
        normalized = pwm_to_normalized(values)
        self.latest["motor_normalized"] = (now, normalized)
        self.latest["motor_pwm"] = (now, values)
        if self.recording:
            self.events["motor"].append((now, normalized.copy(), values.copy()))

    def _ground_truth_callback(self, message: Odometry) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        value = (
            np.asarray([pose.position.x, pose.position.y, pose.position.z]),
            np.asarray(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            ),
            np.asarray([twist.linear.x, twist.linear.y, twist.linear.z]),
            np.asarray([twist.angular.x, twist.angular.y, twist.angular.z]),
        )
        if all(np.all(np.isfinite(item)) for item in value):
            self.latest["ground_truth"] = (time.monotonic(), value)

    def _thruster_state_callback(self, message: ThrusterState) -> None:
        values = tuple(
            np.asarray(field, dtype=np.float64)
            for field in (
                message.setpoint,
                message.rpm,
                message.thrust,
                message.torque,
            )
        )
        if any(
            value.shape != (ACTUATOR_COUNT,) or not np.all(np.isfinite(value))
            for value in values
        ):
            return
        now = time.monotonic()
        state = (*values, _message_stamp(message))
        self.latest["thruster_state"] = (now, state)
        if self.recording:
            self.events["thruster_state"].append((now, *state))

    def _ready(self) -> bool:
        required = (
            "vio",
            "ahrs",
            "imu",
            "depth",
            "pressure",
            "manual_normalized",
            "manual_pwm",
            "motor_normalized",
            "motor_pwm",
        )
        return all(self.latest[name] is not None for name in required) and (
            not self.record_ground_truth
            or (
                self.latest["ground_truth"] is not None
                and self.latest["thruster_state"] is not None
            )
        )

    def _tick(self) -> None:
        if self.runtime is not None:
            self.runtime.poll()
        if not self.recording or not self._ready():
            return
        now = time.monotonic()
        if not sample_due(self.last_sample_at, now, self.period):
            return
        if self.enforce_runtime_contract and self.runtime is not None:
            valid, reason = self.runtime.execution_gate(
                expected_mode=self.configured_execution_mode
            )
            if not valid:
                if self.samples["time"]:
                    self.runtime.mark_failure(reason or "runtime gate failed")
                return
        self._append_sample(now)
        if (
            self.max_duration > 0.0
            and self.first_sample_at is not None
            and now - self.first_sample_at >= self.max_duration
        ):
            self.stop_and_save()

    def _append_sample(self, now: float) -> None:
        if self.first_sample_at is None:
            self.first_sample_at = now
        self.last_sample_at = now
        ahrs_at, ahrs = self.latest["ahrs"]
        vio_at, vio = self.latest["vio"]
        imu_at, imu = self.latest["imu"]
        depth_at, depth = self.latest["depth"]
        pressure_at, pressure = self.latest["pressure"]
        manual_at, manual = self.latest["manual_normalized"]
        manual_pwm_at, manual_pwm = self.latest["manual_pwm"]
        motor_at, motor = self.latest["motor_normalized"]
        motor_pwm_at, motor_pwm = self.latest["motor_pwm"]

        self.samples["time"].append(now - self.first_sample_at)
        self.samples["vio_position_ned_m"].append(vio[0].copy())
        self.samples["vio_orientation_body_to_ned_xyzw"].append(vio[1].copy())
        self.samples["vio_linear_velocity_body_frd_m_s"].append(vio[2].copy())
        self.samples["vio_angular_velocity_body_frd_rad_s"].append(vio[3].copy())
        self.samples["ahrs_orientation_body_to_ned_xyzw"].append(ahrs[0].copy())
        self.samples["ahrs_angular_velocity_body_frd_rad_s"].append(ahrs[1].copy())
        self.samples["imu_angular_velocity_body_frd_rad_s"].append(imu[0].copy())
        self.samples["imu_linear_acceleration_body_frd_m_s2"].append(imu[1].copy())
        self.samples["depth_down_m"].append(depth)
        self.samples["pressure_pa"].append(pressure[0])
        ages = {
            "vio": now - vio_at,
            "ahrs": now - ahrs_at,
            "imu": now - imu_at,
            "depth": now - depth_at,
            "pressure": now - pressure_at,
            "manual": max(now - manual_at, now - manual_pwm_at),
            "motor": max(now - motor_at, now - motor_pwm_at),
        }
        for name in ("vio", "ahrs", "imu", "depth", "pressure"):
            self.samples[f"{name}_valid"].append(ages[name] <= self.age_limits[name])
            self.samples[f"{name}_age_seconds"].append(ages[name])
        self.samples["manual_control_normalized"].append(manual.copy())
        self.samples["manual_control_pwm"].append(manual_pwm.copy())
        self.samples["manual_control_age_seconds"].append(ages["manual"])
        self.samples["motor_output_normalized"].append(motor.copy())
        self.samples["motor_output_pwm"].append(motor_pwm.copy())
        self.samples["motor_output_age_seconds"].append(ages["motor"])
        if self.record_ground_truth:
            _, truth = self.latest["ground_truth"]
            for name, value in zip(GROUND_TRUTH_ARRAYS, truth):
                self.samples[name].append(value.copy())
            _, thruster_state = self.latest["thruster_state"]
            self.samples["ground_truth_applied_thrust_n"].append(
                thruster_state[2].copy()
            )

    def start_recording(self) -> None:
        self.samples = create_sample_buffers(
            record_ground_truth=self.record_ground_truth
        )
        self.events = {name: [] for name in self.events}
        self.recording = True
        self.started_at = time.monotonic()
        self.first_sample_at = None
        self.last_sample_at = -math.inf
        self._last_vio_source_stamp = None
        if self.runtime is not None:
            self.runtime.begin_episode()

    def _native_arrays(self, duration: float) -> dict[str, np.ndarray]:
        assert self.first_sample_at is not None

        def selected(name: str) -> list:
            return [
                event
                for event in self.events[name]
                if -1e-6 <= event[0] - self.first_sample_at <= duration + 1e-6
            ]

        def times(events: list) -> np.ndarray:
            return np.asarray(
                [event[0] - self.first_sample_at for event in events],
                dtype=np.float64,
            )

        vio = selected("vio")
        ahrs = selected("ahrs")
        imu = selected("imu")
        depth = selected("depth")
        pressure = selected("pressure")
        manual = selected("manual")
        motor = selected("motor")
        thruster_state = selected("thruster_state")
        stack = lambda events, index, width: (
            np.stack([event[index] for event in events]).astype(np.float64)
            if events
            else np.empty((0, width), dtype=np.float64)
        )
        vector = lambda events, index: np.asarray(
            [event[index] for event in events], dtype=np.float64
        )
        arrays = {
            "vio_time": times(vio),
            "vio_source_stamp_seconds": vector(vio, 5),
            "vio_raw_position_ned_m": stack(vio, 1, 3),
            "vio_raw_orientation_body_to_ned_xyzw": stack(vio, 2, 4),
            "vio_raw_linear_velocity_body_frd_m_s": stack(vio, 3, 3),
            "vio_raw_angular_velocity_body_frd_rad_s": stack(vio, 4, 3),
            "ahrs_time": times(ahrs),
            "ahrs_source_stamp_seconds": vector(ahrs, 3),
            "ahrs_raw_orientation_body_to_ned_xyzw": stack(ahrs, 1, 4),
            "ahrs_raw_angular_velocity_body_frd_rad_s": stack(ahrs, 2, 3),
            "imu_time": times(imu),
            "imu_source_stamp_seconds": vector(imu, 3),
            "imu_raw_angular_velocity_body_frd_rad_s": stack(imu, 1, 3),
            "imu_raw_linear_acceleration_body_frd_m_s2": stack(imu, 2, 3),
            "depth_time": times(depth),
            "depth_raw_down_m": vector(depth, 1),
            "pressure_time": times(pressure),
            "pressure_source_stamp_seconds": vector(pressure, 3),
            "pressure_raw_pa": vector(pressure, 1),
            "pressure_raw_variance_pa2": vector(pressure, 2),
            "manual_control_time": times(manual),
            "manual_control_raw_normalized": stack(manual, 1, 6),
            "manual_control_raw_pwm": stack(manual, 2, 8),
            "motor_output_time": times(motor),
            "motor_output_raw_normalized": stack(motor, 1, 6),
            "motor_output_raw_pwm": stack(motor, 2, 6),
        }
        if self.record_ground_truth:
            arrays.update(
                {
                    "thruster_state_time": times(thruster_state),
                    "thruster_state_source_stamp_seconds": vector(
                        thruster_state, 5
                    ),
                    "thruster_setpoint_raw": stack(thruster_state, 1, 6),
                    "thruster_rpm_raw": stack(thruster_state, 2, 6),
                    "thruster_thrust_n_raw": stack(thruster_state, 3, 6),
                    "thruster_torque_nm_raw": stack(thruster_state, 4, 6),
                }
            )
        return arrays

    def stop_and_save(self) -> Path:
        if not self.recording:
            raise RuntimeError("recorder is not active")
        self.recording = False
        if self.first_sample_at is None or len(self.samples["time"]) < 2:
            raise RuntimeError("episode has fewer than two valid samples")
        arrays = {name: np.asarray(values) for name, values in self.samples.items()}
        duration = float(arrays["time"][-1])
        arrays.update(self._native_arrays(duration))
        contract = evaluate_recording_contract(
            arrays,
            max_control_age=self.age_limits["control"],
            max_motor_age=self.age_limits["motor"],
            max_vio_age=self.age_limits["vio"],
            max_ahrs_age=self.age_limits["ahrs"],
            max_imu_age=self.age_limits["imu"],
            max_depth_age=self.age_limits["depth"],
            max_pressure_age=self.age_limits["pressure"],
            record_ground_truth=self.record_ground_truth,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"episode_v{FORMAT_VERSION}_{stamp}.npz"
        np.savez_compressed(path, **arrays)
        metadata = {
            "format_version": FORMAT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at": datetime.now().astimezone().isoformat(),
            "episode_path": str(path.resolve()),
            "sample_count": int(arrays["time"].size),
            "duration_seconds": duration,
            "requested_sample_rate_hz": self.sample_rate,
            "actual_sample_rate_hz": contract.actual_sample_rate_hz,
            "configured_execution_mode": self.configured_execution_mode,
            "vehicle_profile": "standard_bluerov2_six_thruster_stereo_vio_no_dvl",
            "actuator_count": ACTUATOR_COUNT,
            "manual_control_order": list(MANUAL_CONTROL_ORDER),
            "manual_control_authority_mask": list(MANUAL_CONTROL_AUTHORITY_MASK),
            "manual_control_semantics": dict(MANUAL_CONTROL_SEMANTICS),
            "deployable_observation_arrays": list(DEPLOYABLE_OBSERVATION_ARRAYS),
            "horizontal_state_available": True,
            "vio_recording": {
                "topic": self.vio_odometry_topic,
                "source_id": self.vio_source_id,
                "world_frame": "world_ned",
                "body_frame": "body_frd",
                "pose_and_velocity_deployable": True,
                "ground_truth_fields_excluded": not self.record_ground_truth,
            },
            "depth_sensor": {
                "source_id": PRESSURE_DEPTH_SOURCE_ID,
                "topic": self.pressure_topic,
                "pressure_topic": self.pressure_topic,
                "message_type": "sensor_msgs/msg/FluidPressure",
                "pressure_reference": "gauge",
                "surface_pressure_pa": self.surface_pressure,
                "fluid_density_kg_m3": self.fluid_density,
                "conversion": PRESSURE_DEPTH_CONVERSION_ID,
                "derived_from_archive_array": "pressure_pa",
                "vio_position_z_used": False,
                "ground_truth_position_z_used": False,
            },
            "motor_output_semantics": (
                "ArduSub mixer PWM demand normalized around 1500 us; "
                "not measured thrust or force"
            ),
            "training_contract": {
                "history_observation": "VIO/IMU/AHRS/pressure-derived depth",
                "history_action": "motor_output_normalized[6]",
                "future_candidate_action": "motor_output_normalized[6]",
                "target": "future response or response-aligned latent",
                "future_position_delta_ned": (
                    "VINS north/east delta + pressure-derived down delta"
                ),
                "observation_feature_contract": OBSERVATION_FEATURE_CONTRACT_ID,
                "imu_acceleration_feature": (
                    "gravity_compensated_dynamic_body_acceleration_m_s2"
                ),
                "imu_acceleration_archive_field": (
                    "imu_linear_acceleration_body_frd_m_s2_specific_force"
                ),
            },
            "stream_quality": {
                "manual_control_stream_valid": (contract.manual_control_stream_valid),
                "motor_output_stream_valid": (contract.motor_output_stream_valid),
                "observation_stream_valid": (contract.observation_stream_valid),
                "applied_thrust_stream_valid": (
                    contract.applied_thrust_stream_valid
                ),
            },
            "ground_truth_recording": {
                "enabled": self.record_ground_truth,
                "arrays": list(GROUND_TRUTH_ARRAYS) if self.record_ground_truth else [],
                "ground_truth_fields_excluded": not self.record_ground_truth,
                "deployable_observation_access_allowed": False,
                "feedback_access_allowed": False,
                "affects_pass_fail": False,
                "simulation_truth_audit": contract.simulation_truth_audit,
            },
            "applied_thrust_recording": {
                "enabled": self.record_ground_truth,
                "source": "Stonefish ThrusterState.getThrust()",
                "aligned_arrays": list(PRIVILEGED_ACTUATOR_ARRAYS)
                if self.record_ground_truth
                else [],
                "native_arrays": list(PRIVILEGED_ACTUATOR_NATIVE_ARRAYS)
                if self.record_ground_truth
                else [],
                "deployable_observation_access_allowed": False,
                "feedback_access_allowed": False,
                "jepa_action_input_allowed": False,
                "stream_presence_required": self.record_ground_truth,
                "affects_pass_fail": False,
            },
            "ardusub_contract": (
                self.runtime.metadata() if self.runtime is not None else None
            ),
        }
        write_metadata(path, metadata)
        self.last_path = path
        self.get_logger().info(
            f"Saved v15 episode with {arrays['time'].size} samples: {path}"
        )
        return path

    def _start_service(self, request, response):
        del request
        if self.recording:
            response.success = False
            response.message = "already recording"
            return response
        self.start_recording()
        response.success = True
        response.message = "recording started"
        return response

    def _stop_service(self, request, response):
        del request
        try:
            path = self.stop_and_save()
        except RuntimeError as error:
            response.success = False
            response.message = str(error)
            return response
        response.success = True
        response.message = str(path)
        return response

    def _status_service(self, request, response):
        del request
        response.success = self.recording
        response.message = (
            f"recording samples={len(self.samples['time'])}"
            if self.recording
            else f"idle last_path={self.last_path or ''}"
        )
        return response

    def destroy_node(self) -> bool:
        if self.runtime is not None:
            self.runtime.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        if node.recording and len(node.samples["time"]) >= 2:
            node.stop_and_save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
