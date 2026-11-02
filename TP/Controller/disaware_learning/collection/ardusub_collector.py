"""Shared ArduSub MANUAL lifecycle for bounded excitation collectors."""

from __future__ import annotations

import math
import time

from nav_msgs.msg import Odometry
import numpy as np
from pymavlink import mavutil
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from stonefish_ros2.srv import Respawn

from disaware_learning.control.contract import (
    ACTUATOR_COUNT,
    ARDUSUB_NOMINAL_PARAMETERS,
    ARDUSUB_SITL_PARAMETERS,
    EXECUTION_MODE,
    MANUAL_CONTROL_ORDER,
)
from disaware_learning.control.geometry import body_velocity_to_ned, odometry_rpy
from disaware_learning.control.rc import (
    clamp_manual_control,
    override_channels,
    pwm_to_normalized,
    slew_towards,
)
from disaware_learning.data.episode_io import load_metadata, write_metadata


class ArduSubCollectorBase(Node):
    """Own the deployable MANUAL command boundary and episode lifecycle."""

    def __init__(self) -> None:
        super().__init__("ardusub_collector")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("mavlink_endpoint", "udpin:0.0.0.0:14550")
        self.declare_parameter("source_system", 250)
        self.declare_parameter("spawn_depth", 9.0)
        self.declare_parameter("action_limit", 0.50)
        self.declare_parameter("action_slew_rate", 1.20)
        self.declare_parameter("command_rate", 25.0)
        self.declare_parameter("manage_recorder", True)
        self.declare_parameter("require_collection_log_suppression", False)
        self.declare_parameter("report_path", "")
        self.declare_parameter("max_tilt_degrees", 15.0)
        self.declare_parameter("max_motor_saturation_ratio", 0.01)
        self.declare_parameter("max_continuous_saturation", 0.50)
        self.declare_parameter("startup_settle_seconds", 1.0)
        self.declare_parameter("startup_settle_timeout_seconds", 20.0)
        self.declare_parameter("startup_max_linear_speed_m_s", 0.08)
        self.declare_parameter("startup_max_angular_speed_rad_s", 0.10)
        self.declare_parameter("startup_max_tilt_degrees", 5.0)

        self.vehicle_name = str(self.get_parameter("vehicle_name").value).strip("/")
        self.endpoint = str(self.get_parameter("mavlink_endpoint").value)
        self.source_system = int(self.get_parameter("source_system").value)
        self.spawn_depth = float(self.get_parameter("spawn_depth").value)
        self.mode = EXECUTION_MODE
        self.action_limit = max(
            0.01,
            min(1.0, float(self.get_parameter("action_limit").value)),
        )
        self.action_slew_rate = self._positive_parameter("action_slew_rate")
        self.command_rate = max(2.0, float(self.get_parameter("command_rate").value))
        self.period = 1.0 / self.command_rate
        self.manage_recorder = bool(self.get_parameter("manage_recorder").value)
        self.require_collection_log_suppression = bool(
            self.get_parameter("require_collection_log_suppression").value
        )
        self.report_path = str(self.get_parameter("report_path").value).strip()
        self.startup_settle_seconds = self._positive_parameter("startup_settle_seconds")
        self.startup_settle_timeout_seconds = self._positive_parameter(
            "startup_settle_timeout_seconds"
        )
        self.startup_max_linear_speed = self._positive_parameter(
            "startup_max_linear_speed_m_s"
        )
        self.startup_max_angular_speed = self._positive_parameter(
            "startup_max_angular_speed_rad_s"
        )
        self.startup_max_tilt_degrees = self._positive_parameter(
            "startup_max_tilt_degrees"
        )
        self.envelope = {
            "tilt_degrees": float(self.get_parameter("max_tilt_degrees").value),
            "motor_saturation_ratio": float(
                self.get_parameter("max_motor_saturation_ratio").value
            ),
            "continuous_motor_saturation_seconds": float(
                self.get_parameter("max_continuous_saturation").value
            ),
        }

        prefix = f"/{self.vehicle_name}"
        self.validation_odometry: Odometry | None = None
        self.validation_odometry_received_at = -math.inf
        self.applied_motors: np.ndarray | None = None
        self.create_subscription(
            Odometry,
            f"{prefix}/odometry",
            self._validation_odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float64MultiArray,
            f"{prefix}/ardusub/motor_output",
            self._motor_callback,
            20,
        )
        self.command_publisher = self.create_publisher(
            Float64MultiArray, f"{prefix}/ardusub/manual_command", 10
        )
        self.command_pwm_publisher = self.create_publisher(
            Float64MultiArray, f"{prefix}/ardusub/manual_command_pwm", 10
        )
        self.respawn = self.create_client(Respawn, "/stonefish_ros2/respawn_robot")
        self.recorder_start = self.create_client(Trigger, "/episode_recorder/start")
        self.recorder_stop = self.create_client(Trigger, "/episode_recorder/stop")
        self.mavlink = None
        self.ardusub_parameters: dict[str, float] = {}
        self.last_heartbeat_received = -math.inf
        self.last_heartbeat_mode = "UNKNOWN"
        self.last_heartbeat_armed = False
        self.last_heartbeat_system_status: int | None = None
        self.execution_heartbeat_checks = 0
        self.execution_heartbeat_failure: str | None = None
        self.current_command = np.zeros(len(MANUAL_CONTROL_ORDER), dtype=np.float64)
        self.startup_motion_gate: dict[str, object] = {
            "passed": False,
            "settle_seconds": self.startup_settle_seconds,
            "timeout_seconds": self.startup_settle_timeout_seconds,
            "max_linear_speed_m_s": self.startup_max_linear_speed,
            "linear_speed_basis": "water_relative_ned",
            "max_angular_speed_rad_s": self.startup_max_angular_speed,
            "max_tilt_degrees": self.startup_max_tilt_degrees,
        }

    def _validation_odometry_callback(self, message: Odometry) -> None:
        self.validation_odometry = message
        self.validation_odometry_received_at = time.monotonic()

    def _motor_callback(self, message: Float64MultiArray) -> None:
        values = np.asarray(message.data, dtype=np.float64)
        if values.shape == (ACTUATOR_COUNT,) and np.all(np.isfinite(values)):
            self.applied_motors = values

    def _startup_background_velocity_ned(self) -> np.ndarray:
        """Return known simulation background flow for reset-only settling."""
        return np.zeros(3, dtype=np.float64)

    def _connect_ardusub(self) -> None:
        self.mavlink = mavutil.mavlink_connection(
            self.endpoint, source_system=self.source_system
        )
        heartbeat = self.mavlink.wait_heartbeat(timeout=15.0)
        if heartbeat is None:
            raise RuntimeError("ArduSub heartbeat timed out")
        if self.source_system != int(ARDUSUB_NOMINAL_PARAMETERS["MAV_GCS_SYSID"]):
            raise RuntimeError("collector source_system must match MAV_GCS_SYSID=250")
        required = ARDUSUB_NOMINAL_PARAMETERS
        self.ardusub_parameters = {
            name: self._read_parameter(name) for name in required
        }
        sitl_marker = next(iter(ARDUSUB_SITL_PARAMETERS))
        try:
            self._read_parameter(sitl_marker)
        except RuntimeError:
            pass
        else:
            for name, expected in ARDUSUB_SITL_PARAMETERS.items():
                value = self._read_parameter(name)
                self.ardusub_parameters[name] = value
                if not np.isclose(value, expected, atol=1e-3):
                    raise RuntimeError(
                        "ArduSub SITL parameter does not match: " f"{name}={value}"
                    )
        if self.require_collection_log_suppression:
            value = self._read_parameter("LOG_BITMASK")
            self.ardusub_parameters["LOG_BITMASK"] = value
            if not np.isclose(value, 0.0, atol=1e-3):
                raise RuntimeError(
                    "ArduSub collection log suppression does not match: "
                    f"LOG_BITMASK={value}"
                )
        mismatches = {
            name: self.ardusub_parameters[name]
            for name, expected in required.items()
            if not np.isclose(self.ardusub_parameters[name], expected, atol=1e-3)
        }
        if mismatches:
            raise RuntimeError(
                "ArduSub control or sensor parameters do not match: " f"{mismatches}"
            )

    def _read_parameter(self, name: str) -> float:
        assert self.mavlink is not None
        self.mavlink.mav.param_request_read_send(
            self.mavlink.target_system,
            self.mavlink.target_component,
            name.encode(),
            -1,
        )
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            message = self.mavlink.recv_match(
                type="PARAM_VALUE", blocking=True, timeout=0.25
            )
            if message is not None and message.param_id == name:
                return float(message.param_value)
        raise RuntimeError(f"ArduSub parameter could not be read: {name}")

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def _prepare_vehicle(self) -> None:
        """Reset the Stonefish vehicle before a simulation excitation."""
        self._neutral_and_disarm()
        if self.has_parameter("reset_spawn_yaw_rad"):
            spawn_yaw = float(self.get_parameter("reset_spawn_yaw_rad").value)
        else:
            self._wait_for_validation_odometry()
            assert self.validation_odometry is not None
            spawn_yaw = odometry_rpy(self.validation_odometry)[2]
        if not self.respawn.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Stonefish respawn service is unavailable")
        reset_attempts: list[dict[str, object]] = []
        last_error: RuntimeError | None = None
        maximum_attempts = 2
        for attempt in range(maximum_attempts):
            attempt_audit: dict[str, object] = {
                "attempt": attempt + 1,
                "passed": False,
                "started_monotonic_seconds": time.monotonic(),
            }
            reset_attempts.append(attempt_audit)
            try:
                # A derived collector may synchronize its estimator from
                # _respawn_vehicle(). Keep that synchronization inside this
                # transaction so an estimator restart failure receives the
                # same full hard-reset retry as a motion-gate failure.
                self._respawn_vehicle(spawn_yaw)
                attempt_audit["respawn_and_estimator_reset_passed"] = True
                self._wait_for_startup_stability()
                attempt_audit["startup_stability_passed"] = True
                attempt_audit["passed"] = True
                last_error = None
                break
            except RuntimeError as error:
                last_error = error
                attempt_audit["failure"] = str(error)
                if attempt + 1 < maximum_attempts:
                    self._neutral_and_disarm()
            finally:
                attempt_audit["completed_monotonic_seconds"] = time.monotonic()
        self.startup_motion_gate.update(
            {
                "respawn_reset_attempts": len(reset_attempts),
                "reset_attempts": reset_attempts,
                "recovered_after_retry": bool(
                    len(reset_attempts) > 1 and last_error is None
                ),
            }
        )
        if last_error is not None:
            raise last_error
        self._arm()
        self.current_command.fill(0.0)
        self._run_for(np.zeros(6), 1.5)

    def _respawn_vehicle(self, spawn_yaw: float) -> None:
        request = Respawn.Request()
        request.name = self.vehicle_name
        request.origin.position.z = self.spawn_depth
        request.origin.orientation.z = math.sin(0.5 * spawn_yaw)
        request.origin.orientation.w = math.cos(0.5 * spawn_yaw)
        future = self.respawn.call_async(request)
        self._spin_until(future, 5.0, "Stonefish respawn timed out")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Stonefish respawn failed")

    def _wait_for_startup_stability(self) -> None:
        deadline = time.monotonic() + self.startup_settle_timeout_seconds
        stable_since: float | None = None
        latest: dict[str, float] = {}
        while rclpy.ok() and time.monotonic() < deadline:
            self.current_command.fill(0.0)
            self._send_command(self.current_command)
            rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.0)
            if self.validation_odometry is None:
                continue
            pose = self.validation_odometry.pose.pose
            twist = self.validation_odometry.twist.twist
            rpy = odometry_rpy(self.validation_odometry)
            velocity_body = np.asarray(
                [twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float64
            )
            quaternion = np.asarray(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float64,
            )
            velocity_ned = body_velocity_to_ned(quaternion, velocity_body)
            background_velocity_ned = np.asarray(
                self._startup_background_velocity_ned(), dtype=np.float64
            )
            if (
                background_velocity_ned.shape != (3,)
                or not np.all(np.isfinite(background_velocity_ned))
            ):
                raise RuntimeError("startup background velocity must be finite NED xyz")
            water_relative_velocity_ned = velocity_ned - background_velocity_ned
            inertial_speed = float(np.linalg.norm(velocity_ned))
            water_relative_speed = float(np.linalg.norm(water_relative_velocity_ned))
            angular_speed = float(
                np.linalg.norm([twist.angular.x, twist.angular.y, twist.angular.z])
            )
            tilt_degrees = float(np.degrees(np.linalg.norm(rpy[:2])))
            latest = {
                "inertial_speed_m_s": inertial_speed,
                "water_relative_speed_m_s": water_relative_speed,
                "background_velocity_ned_m_s": background_velocity_ned.tolist(),
                "angular_speed_rad_s": angular_speed,
                "tilt_degrees": tilt_degrees,
                "position_down_m": float(pose.position.z),
            }
            stable = bool(
                water_relative_speed <= self.startup_max_linear_speed
                and angular_speed <= self.startup_max_angular_speed
                and tilt_degrees <= self.startup_max_tilt_degrees
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
                        "settled_for_seconds": (time.monotonic() - stable_since),
                    }
                )
                return
            time.sleep(max(0.0, self.period - 0.02))
        self.startup_motion_gate.update({"passed": False, "observed": latest})
        raise RuntimeError(
            "vehicle did not settle after respawn: "
            f"{latest or 'no validation odometry'}"
        )

    def _wait_for_validation_odometry(self) -> None:
        end = time.monotonic() + 3.0
        while self.validation_odometry is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.validation_odometry is None:
            raise RuntimeError("Stonefish validation odometry is unavailable")

    def _arm(self) -> None:
        assert self.mavlink is not None
        self.mavlink.set_mode_apm("MANUAL")
        time.sleep(0.35)
        self.mavlink.set_mode_apm(self.mode)
        time.sleep(0.5)
        self.mavlink.arducopter_arm()
        end = time.monotonic() + 10.0
        actual_mode = "UNKNOWN"
        while time.monotonic() < end:
            heartbeat = self.mavlink.recv_match(
                type="HEARTBEAT", blocking=True, timeout=1.0
            )
            if heartbeat:
                self._update_heartbeat(heartbeat)
                actual_mode = self.last_heartbeat_mode
            if heartbeat and self.last_heartbeat_armed:
                if actual_mode == self.mode:
                    self.get_logger().info(f"ArduSub armed in {actual_mode}")
                    return
        raise RuntimeError(
            f"ArduSub did not arm in requested mode {self.mode}; "
            f"last mode was {actual_mode}"
        )

    def _neutral_and_disarm(self, release: bool = False) -> None:
        if self.mavlink is None:
            return
        try:
            self.mavlink.set_mode_apm("MANUAL")
            time.sleep(0.35)
            self.current_command.fill(0.0)
            self._send_command(self.current_command)
            time.sleep(0.15)
            self.mavlink.arducopter_disarm()
            time.sleep(0.35)
            if release:
                self.mavlink.mav.rc_channels_override_send(
                    self.mavlink.target_system,
                    self.mavlink.target_component,
                    *([0] * 8),
                )
        except (OSError, AttributeError):
            pass

    def _start_recorder(self) -> None:
        if not self.recorder_start.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("episode recorder start service is unavailable")
        response = self._call_trigger(self.recorder_start)
        if not response.success and "already recording" in response.message:
            self._call_trigger(self.recorder_stop)
            response = self._call_trigger(self.recorder_start)
        if not response.success:
            raise RuntimeError(f"could not start recorder: {response.message}")

    def _stop_recorder(self) -> str:
        response = self._call_trigger(self.recorder_stop)
        if not response.success:
            raise RuntimeError(f"could not stop recorder: {response.message}")
        self._annotate_ardusub_contract(response.message)
        return response.message

    def _annotate_ardusub_contract(self, episode_path: str) -> None:
        metadata = load_metadata(episode_path)
        metadata["ardusub_contract"] = {
            "parameter_snapshot": self.ardusub_parameters,
            "parameter_snapshot_source": "MAVLink PARAM_VALUE readback",
            "collection_log_suppression": {
                "required": self.require_collection_log_suppression,
                "parameter": "LOG_BITMASK",
                "expected": 0.0,
                "readback": self.ardusub_parameters.get("LOG_BITMASK"),
                "passed": bool(
                    not self.require_collection_log_suppression
                    or np.isclose(
                        self.ardusub_parameters.get("LOG_BITMASK", math.nan),
                        0.0,
                        atol=1e-3,
                    )
                ),
            },
            "control_source_system": self.source_system,
            "rc_override_accepted_source_system": self.source_system,
            "execution_mode": self.mode,
            "execution_mode_verified_from_heartbeat": True,
            "normalized_axes_are_physical_rates": False,
            "continuous_heartbeat_gate": {
                "checks": self.execution_heartbeat_checks,
                "last_mode": self.last_heartbeat_mode,
                "last_armed": self.last_heartbeat_armed,
                "last_system_status": self.last_heartbeat_system_status,
                "maximum_heartbeat_age_seconds": 2.5,
                "passed": self.execution_heartbeat_failure is None,
                "failure": self.execution_heartbeat_failure,
            },
            "startup_motion_gate": self.startup_motion_gate,
        }
        write_metadata(episode_path, metadata)

    def _call_trigger(self, client) -> Trigger.Response:
        future = client.call_async(Trigger.Request())
        self._spin_until(future, 10.0, "recorder service timed out")
        response = future.result()
        if response is None:
            raise RuntimeError("recorder service returned no response")
        return response

    def _spin_until(self, future, timeout: float, message: str) -> None:
        end = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            raise RuntimeError(message)

    def _run_for(self, target, duration: float) -> None:
        bounded = clamp_manual_control(target, self.action_limit)
        end = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end:
            tick_started = time.monotonic()
            self.current_command = slew_towards(
                self.current_command,
                bounded,
                self.action_slew_rate,
                self.period,
            )
            self._send_command(self.current_command)
            rclpy.spin_once(self, timeout_sec=min(0.01, self.period))
            for _ in range(16):
                rclpy.spin_once(self, timeout_sec=0.0)
            self._verify_execution_heartbeat()
            remaining = self.period - (time.monotonic() - tick_started)
            if remaining > 0.0:
                time.sleep(remaining)

    def _update_heartbeat(self, heartbeat) -> None:
        self.last_heartbeat_received = time.monotonic()
        self.last_heartbeat_mode = mavutil.mode_string_v10(heartbeat)
        self.last_heartbeat_armed = bool(
            heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        self.last_heartbeat_system_status = int(heartbeat.system_status)

    def _verify_execution_heartbeat(self) -> None:
        assert self.mavlink is not None
        while True:
            heartbeat = self.mavlink.recv_match(type="HEARTBEAT", blocking=False)
            if heartbeat is None:
                break
            self._update_heartbeat(heartbeat)
        self.execution_heartbeat_checks += 1
        if time.monotonic() - self.last_heartbeat_received > 2.5:
            self.execution_heartbeat_failure = (
                "ArduSub heartbeat became stale during execution"
            )
            raise RuntimeError(self.execution_heartbeat_failure)
        if not self.last_heartbeat_armed:
            self.execution_heartbeat_failure = (
                "ArduSub disarmed during an active episode"
            )
            raise RuntimeError(self.execution_heartbeat_failure)
        if self.last_heartbeat_mode != self.mode:
            self.execution_heartbeat_failure = (
                "ArduSub left the fixed execution mode: "
                f"expected {self.mode}, got {self.last_heartbeat_mode}"
            )
            raise RuntimeError(self.execution_heartbeat_failure)

    def _send_command(self, command: np.ndarray) -> None:
        assert self.mavlink is not None
        channels = override_channels(command)
        effective_command = pwm_to_normalized(channels[:6])
        self.current_command = effective_command
        self.mavlink.mav.rc_channels_override_send(
            self.mavlink.target_system,
            self.mavlink.target_component,
            *channels,
        )
        self.command_publisher.publish(
            Float64MultiArray(data=effective_command.tolist())
        )
        self.command_pwm_publisher.publish(
            Float64MultiArray(data=[float(value) for value in channels])
        )
