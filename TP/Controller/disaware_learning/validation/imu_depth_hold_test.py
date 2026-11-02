"""Closed-loop depth smoke test using only deployable ArduSub observations."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
from pymavlink import mavutil
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

from disaware_learning.control.contract import ACTUATOR_COUNT, EXECUTION_MODE
from disaware_learning.control.rc import override_channels, pwm_to_normalized


def depth_feedback_command(
    depth_down_m: float,
    target_depth_down_m: float,
    gain: float,
    limit: float,
) -> float:
    """Return MANUAL throttle; positive commands upward on this vehicle."""
    values = (depth_down_m, target_depth_down_m, gain, limit)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("depth controller inputs must be finite")
    if gain <= 0.0 or not 0.0 < limit <= 1.0:
        raise ValueError("gain must be positive and limit must be in (0, 1]")
    return float(np.clip(gain * (depth_down_m - target_depth_down_m), -limit, limit))


def _tilt_degrees(message: Imu) -> float:
    q = message.orientation
    sin_roll = 2.0 * (q.w * q.x + q.y * q.z)
    cos_roll = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    return math.degrees(max(abs(roll), abs(pitch)))


class ImuDepthHoldTest(Node):
    """Step and hold relative depth through ArduSub's standard RC boundary."""

    def __init__(self) -> None:
        super().__init__("imu_depth_hold_test")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("mavlink_endpoint", "udpin:0.0.0.0:14550")
        self.declare_parameter("source_system", 250)
        self.declare_parameter("control_rate", 25.0)
        self.declare_parameter("baseline_seconds", 2.0)
        self.declare_parameter("hold_seconds", 12.0)
        self.declare_parameter("depth_step_down_m", 0.30)
        self.declare_parameter("depth_gain", 0.65)
        self.declare_parameter("action_limit", 0.30)
        self.declare_parameter("settle_window_seconds", 3.0)
        self.declare_parameter("max_settled_error_m", 0.10)
        self.declare_parameter("max_tilt_degrees", 10.0)
        self.declare_parameter("max_sensor_age", 0.25)
        self.declare_parameter("manage_recorder", True)
        self.declare_parameter("report_path", "")

        vehicle = str(self.get_parameter("vehicle_name").value).strip("/")
        prefix = f"/{vehicle}"
        self.endpoint = str(self.get_parameter("mavlink_endpoint").value)
        self.source_system = int(self.get_parameter("source_system").value)
        self.rate = max(5.0, float(self.get_parameter("control_rate").value))
        self.period = 1.0 / self.rate
        self.baseline_seconds = max(
            0.5, float(self.get_parameter("baseline_seconds").value)
        )
        self.hold_seconds = max(3.0, float(self.get_parameter("hold_seconds").value))
        self.step_down = float(self.get_parameter("depth_step_down_m").value)
        self.gain = float(self.get_parameter("depth_gain").value)
        self.action_limit = float(self.get_parameter("action_limit").value)
        self.settle_window = max(
            1.0, float(self.get_parameter("settle_window_seconds").value)
        )
        self.max_settled_error = max(
            0.01, float(self.get_parameter("max_settled_error_m").value)
        )
        self.max_tilt = max(1.0, float(self.get_parameter("max_tilt_degrees").value))
        self.max_sensor_age = max(
            0.05, float(self.get_parameter("max_sensor_age").value)
        )
        self.manage_recorder = bool(self.get_parameter("manage_recorder").value)
        self.report_path = str(self.get_parameter("report_path").value).strip()
        if self.step_down <= 0.0:
            raise ValueError("depth_step_down_m must be positive")
        depth_feedback_command(0.0, self.step_down, self.gain, self.action_limit)

        self.depth: float | None = None
        self.depth_received = -math.inf
        self.ahrs: Imu | None = None
        self.ahrs_received = -math.inf
        self.motors: np.ndarray | None = None
        self.create_subscription(
            Float64, f"{prefix}/ardusub/depth", self._depth_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu, f"{prefix}/ardusub/ahrs", self._ahrs_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Float64MultiArray,
            f"{prefix}/ardusub/motor_output",
            self._motor_callback,
            20,
        )
        self.command_publisher = self.create_publisher(
            Float64MultiArray, f"{prefix}/ardusub/manual_control", 10
        )
        self.command_pwm_publisher = self.create_publisher(
            Float64MultiArray, f"{prefix}/ardusub/manual_control_pwm", 10
        )
        self.recorder_start = self.create_client(Trigger, "/episode_recorder/start")
        self.recorder_stop = self.create_client(Trigger, "/episode_recorder/stop")
        self.mavlink = None

    def _depth_callback(self, message: Float64) -> None:
        value = float(message.data)
        if math.isfinite(value):
            self.depth = value
            self.depth_received = time.monotonic()

    def _ahrs_callback(self, message: Imu) -> None:
        self.ahrs = message
        self.ahrs_received = time.monotonic()

    def _motor_callback(self, message: Float64MultiArray) -> None:
        values = np.asarray(message.data, dtype=np.float64)
        if values.shape == (ACTUATOR_COUNT,) and np.all(np.isfinite(values)):
            self.motors = values

    def _call_trigger(self, client) -> Trigger.Response:
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 8.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        response = future.result() if future.done() else None
        if response is None:
            raise RuntimeError("recorder service timed out")
        return response

    def _send(self, throttle: float) -> None:
        assert self.mavlink is not None
        command = np.zeros(6, dtype=np.float64)
        command[2] = throttle
        channels = override_channels(command)
        effective = pwm_to_normalized(channels[:6])
        self.mavlink.mav.rc_channels_override_send(
            self.mavlink.target_system, self.mavlink.target_component, *channels
        )
        self.command_publisher.publish(Float64MultiArray(data=effective.tolist()))
        self.command_pwm_publisher.publish(
            Float64MultiArray(data=[float(value) for value in channels])
        )

    def _fresh_observation(self) -> tuple[float, float]:
        now = time.monotonic()
        if self.depth is None or self.ahrs is None:
            raise RuntimeError("IMU/depth observation is unavailable")
        if not self._observations_are_fresh(now):
            raise RuntimeError("IMU/depth observation became stale")
        tilt = _tilt_degrees(self.ahrs)
        if tilt > self.max_tilt:
            raise RuntimeError(f"tilt safety limit exceeded: {tilt:.2f} deg")
        return self.depth, tilt

    def _drain_ros_callbacks(self, first_timeout: float = 0.0) -> None:
        rclpy.spin_once(self, timeout_sec=first_timeout)
        for _ in range(16):
            rclpy.spin_once(self, timeout_sec=0.0)

    def _observations_are_fresh(self, now: float | None = None) -> bool:
        if self.depth is None or self.ahrs is None:
            return False
        checked_at = time.monotonic() if now is None else float(now)
        return (
            max(
                checked_at - self.depth_received,
                checked_at - self.ahrs_received,
            )
            <= self.max_sensor_age
        )

    def _wait_armed(self, timeout: float) -> None:
        assert self.mavlink is not None
        deadline = time.monotonic() + timeout
        armed = False
        while time.monotonic() < deadline:
            # The MAVLink wait must not starve the deployable ROS observations.
            self._drain_ros_callbacks(first_timeout=0.02)
            heartbeat = self.mavlink.recv_match(
                type="HEARTBEAT", blocking=False
            )
            if heartbeat is not None and bool(
                heartbeat.base_mode
                & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ):
                armed = True
            if armed and self._observations_are_fresh():
                return
        if armed:
            raise RuntimeError("IMU/depth observation did not become fresh after arming")
        raise RuntimeError("ArduSub did not arm before timeout")

    def run(self) -> bool:
        self.mavlink = mavutil.mavlink_connection(
            self.endpoint, source_system=self.source_system
        )
        if self.mavlink.wait_heartbeat(timeout=15.0) is None:
            raise RuntimeError("ArduSub heartbeat timed out")
        deadline = time.monotonic() + 5.0
        while (self.depth is None or self.ahrs is None or self.motors is None) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        start_depth, _ = self._fresh_observation()
        target_depth = start_depth + self.step_down

        recorder_active = False
        episode_path = None
        samples: list[tuple[float, float, float, float, float, float]] = []
        duration = self.baseline_seconds + self.hold_seconds
        try:
            self.mavlink.set_mode_apm(EXECUTION_MODE)
            time.sleep(0.5)
            self.mavlink.arducopter_arm()
            self._wait_armed(10.0)
            if self.manage_recorder:
                if not self.recorder_start.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError("episode recorder is unavailable")
                response = self._call_trigger(self.recorder_start)
                if not response.success:
                    raise RuntimeError(f"could not start recorder: {response.message}")
                recorder_active = True

            started = time.monotonic()
            while rclpy.ok() and time.monotonic() - started < duration:
                cycle = time.monotonic()
                self._drain_ros_callbacks()
                elapsed = cycle - started
                depth, tilt = self._fresh_observation()
                target = start_depth if elapsed < self.baseline_seconds else target_depth
                throttle = depth_feedback_command(
                    depth, target, self.gain, self.action_limit
                )
                self._send(throttle)
                motor_peak = (
                    float(np.max(np.abs(self.motors)))
                    if self.motors is not None
                    else float("nan")
                )
                samples.append((elapsed, depth, target, throttle, tilt, motor_peak))
                time.sleep(max(0.0, self.period - (time.monotonic() - cycle)))
        finally:
            self._send(0.0)
            time.sleep(0.25)
            self.mavlink.arducopter_disarm()
            if recorder_active:
                response = self._call_trigger(self.recorder_stop)
                if response.success:
                    episode_path = response.message

        values = np.asarray(samples, dtype=np.float64)
        settled = values[:, 0] >= values[-1, 0] - self.settle_window
        errors = values[:, 1] - values[:, 2]
        moved_down = float(np.max(values[:, 1]) - start_depth)
        motor_peak = float(np.max(values[:, 5]))
        checks = {
            "deployable_depth_is_feedback_input": True,
            "horizontal_state_is_not_feedback_input": True,
            "depth_step_response_present": moved_down >= 0.60 * self.step_down,
            "settled_depth_error_within_limit": float(np.max(np.abs(errors[settled])))
            <= self.max_settled_error,
            "six_motor_telemetry_present": self.motors is not None
            and self.motors.shape == (ACTUATOR_COUNT,),
            "motor_outputs_bounded": motor_peak <= 1.0001,
        }
        report = {
            "generated_at": datetime.now().isoformat(),
            "pass": bool(all(checks.values())),
            "checks": checks,
            "control_observation": ["ardusub_depth", "ardusub_ahrs_for_safety"],
            "depth_convention": "relative NED down, from -AHRS2.altitude",
            "start_depth_down_m": start_depth,
            "target_depth_down_m": target_depth,
            "maximum_depth_down_m": float(np.max(values[:, 1])),
            "settled_error_max_abs_m": float(np.max(np.abs(errors[settled]))),
            "settled_error_rmse_m": float(np.sqrt(np.mean(errors[settled] ** 2))),
            "maximum_abs_throttle_command": float(np.max(np.abs(values[:, 3]))),
            "maximum_tilt_degrees": float(np.max(values[:, 4])),
            "maximum_abs_motor_output": motor_peak,
            "episode_path": episode_path,
        }
        path = (
            Path(self.report_path).expanduser().resolve()
            if self.report_path
            else Path("/tmp/imu_depth_hold_report.json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.get_logger().info(f"IMU/depth hold report: {path}")
        return report["pass"]

    def destroy_node(self) -> bool:
        if self.mavlink is not None:
            self.mavlink.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = ImuDepthHoldTest()
    success = False
    try:
        success = node.run()
    except Exception as error:
        node.get_logger().error(f"IMU/depth hold test failed: {error}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
