"""Excite each actuated standard-BlueROV2 axis and check Stonefish response."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from stonefish_ros2.srv import Respawn


# ArduSub FRAME_CONFIG=1 motor factors, standard motor order 1..6.
AXIS_PATTERNS = {
    "surge": [-1, -1, 1, 1, 0, 0],
    "sway": [1, -1, 1, -1, 0, 0],
    "heave_up": [0, 0, 0, 0, -1, -1],
    "roll": [0, 0, 0, 0, 1, -1],
    "yaw": [1, -1, -1, 1, 0, 0],
}
COMBINATION_PATTERNS = {
    # Long holds isolate the coupled action combinations that a closed-loop
    # controller can accumulate even when each single axis looks healthy.
    "direct_forward": AXIS_PATTERNS["surge"],
    "direct_reverse": [-value for value in AXIS_PATTERNS["surge"]],
    "direct_lateral": AXIS_PATTERNS["sway"],
    "direct_diagonal": [0, -1, 1, 0, 0, 0],
    "forward_lateral_throttle": [0, -1, 1, 0, -1, -1],
}
COMBINATION_EXPECTED_VELOCITY_SIGN = {
    "direct_forward": {0: 1.0},
    "direct_reverse": {0: -1.0},
    "direct_lateral": {1: 1.0},
    "direct_diagonal": {0: 1.0, 1: 1.0},
    "forward_lateral_throttle": {0: 1.0, 1: 1.0, 2: -1.0},
}
STATE_TARGET_INDEX = {
    "surge": 0,
    "sway": 1,
    "heave_up": 2,
    "roll": 6,
    "yaw": 8,
}
EXPECTED_SIGN = {
    "surge": 1.0,
    "sway": 1.0,
    "heave_up": -1.0,
    "roll": 1.0,
    "yaw": 1.0,
}


class DOFDynamicsTest(Node):
    """Run reset-isolated checks for all five physically actuated axes."""

    def __init__(self) -> None:
        super().__init__("dof_dynamics_test")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("amplitude", 0.22)
        self.declare_parameter("settle_time", 1.2)
        self.declare_parameter("pulse_time", 1.0)
        self.declare_parameter("combination_hold_time", 6.0)
        self.declare_parameter("tail_time", 0.25)
        self.declare_parameter("spawn_depth", 0.5)
        self.declare_parameter("min_linear_response", 0.004)
        self.declare_parameter("min_angular_response", 0.03)
        self.declare_parameter("max_cross_axis_ratio", 0.40)
        self.declare_parameter("max_combination_tilt_degrees", 8.0)
        self.declare_parameter("report_path", "")
        self.declare_parameter(
            "respawn_service", "/stonefish_ros2/respawn_robot"
        )

        self.vehicle_name = str(self.get_parameter("vehicle_name").value).strip("/")
        self.amplitude = max(0.01, min(0.8, float(self.get_parameter("amplitude").value)))
        self.settle_time = max(0.2, float(self.get_parameter("settle_time").value))
        self.pulse_time = max(0.2, float(self.get_parameter("pulse_time").value))
        self.combination_hold_time = max(
            1.0, float(self.get_parameter("combination_hold_time").value)
        )
        self.tail_time = max(0.1, float(self.get_parameter("tail_time").value))
        self.spawn_depth = float(self.get_parameter("spawn_depth").value)
        self.min_linear_response = max(
            0.0, float(self.get_parameter("min_linear_response").value)
        )
        self.min_angular_response = max(
            0.0, float(self.get_parameter("min_angular_response").value)
        )
        self.max_cross_axis_ratio = max(
            0.0, float(self.get_parameter("max_cross_axis_ratio").value)
        )
        self.max_combination_tilt_degrees = max(
            0.0, float(self.get_parameter("max_combination_tilt_degrees").value)
        )
        self.report_path = str(self.get_parameter("report_path").value).strip()
        respawn_service = str(
            self.get_parameter("respawn_service").value
        ).strip()
        prefix = f"/{self.vehicle_name}"
        self.publisher = self.create_publisher(
            Float64MultiArray, f"{prefix}/setpoint/pwm", 10
        )
        self.create_subscription(
            Odometry,
            f"{prefix}/odometry",
            self._odometry_callback,
            qos_profile_sensor_data,
        )
        self.respawn = self.create_client(Respawn, respawn_service)
        self.respawn_service = respawn_service
        self.latest_state: np.ndarray | None = None

    def _odometry_callback(self, message: Odometry) -> None:
        twist = message.twist.twist
        orientation = message.pose.pose.orientation
        roll, pitch, yaw = _quaternion_to_rpy(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.latest_state = np.asarray(
            [
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
                roll,
                pitch,
                yaw,
            ],
            dtype=np.float64,
        )

    def run(self) -> bool:
        if not self.respawn.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f"{self.respawn_service} service is unavailable"
            )
            return False
        results = {}
        overall_pass = True
        for axis, pattern in AXIS_PATTERNS.items():
            if not self._reset_vehicle():
                return False
            baseline_samples = self._publish_for([0.0] * 6, self.settle_time)
            command = (self.amplitude * np.asarray(pattern, dtype=float)).tolist()
            response_samples = self._publish_for(command, self.pulse_time)
            baseline = self._tail_mean(baseline_samples)
            response = self._tail_mean(response_samples) - baseline
            response[6:9] = np.asarray(
                [_wrap_angle(value) for value in response[6:9]]
            )
            target_index = STATE_TARGET_INDEX[axis]
            family = [0, 1, 2] if target_index < 3 else [6, 7, 8]
            cross_indices = [index for index in family if index != target_index]
            target = float(response[target_index])
            cross = max(abs(float(response[index])) for index in cross_indices)
            cross_ratio = cross / max(abs(target), 1e-9)
            minimum = (
                self.min_linear_response
                if target_index < 3
                else self.min_angular_response
            )
            finite = bool(np.all(np.isfinite(response)))
            passed = bool(
                finite
                and abs(target) >= minimum
                and EXPECTED_SIGN[axis] * target > 0.0
                and cross_ratio <= self.max_cross_axis_ratio
            )
            overall_pass = overall_pass and passed
            results[axis] = {
                "passed": passed,
                "velocity_response": response[:6].tolist(),
                "attitude_response_rad": response[6:9].tolist(),
                "target": target,
                "cross_axis_ratio": cross_ratio,
            }
            status = "PASS" if passed else "FAIL"
            self.get_logger().info(
                f"{status} {axis}: velocity={np.array2string(response[:6], precision=4)}, "
                f"attitude={np.array2string(response[6:9], precision=4)}, "
                f"same-family cross/target={cross_ratio:.3f}"
            )
        combination_results = {}
        for name, pattern in COMBINATION_PATTERNS.items():
            if not self._reset_vehicle():
                return False
            baseline_samples = self._publish_for([0.0] * 6, self.settle_time)
            command = (self.amplitude * np.asarray(pattern, dtype=float)).tolist()
            response_samples = self._publish_for(
                command, self.combination_hold_time
            )
            baseline = self._tail_mean(baseline_samples)
            response = self._tail_mean(response_samples) - baseline
            response[6:9] = np.asarray(
                [_wrap_angle(value) for value in response[6:9]]
            )
            attitude_delta = response[6:9]
            combination_rpy = np.column_stack(
                (
                    np.unwrap(response_samples[:, 6] - baseline[6]),
                    np.unwrap(response_samples[:, 7] - baseline[7]),
                )
            )
            tilt = np.linalg.norm(combination_rpy, axis=1)
            peak_index = int(np.argmax(tilt))
            peak_tilt = float(tilt[peak_index])
            finite = bool(np.all(np.isfinite(response)) and math.isfinite(peak_tilt))
            expected_velocity = COMBINATION_EXPECTED_VELOCITY_SIGN[name]
            velocity_checks = {
                str(axis): bool(
                    sign * response[axis] >= self.min_linear_response
                )
                for axis, sign in expected_velocity.items()
            }
            passed = bool(
                finite
                and all(velocity_checks.values())
                and peak_tilt <= math.radians(self.max_combination_tilt_degrees)
            )
            overall_pass = overall_pass and passed
            combination_results[name] = {
                "passed": passed,
                "velocity_response": response[:6].tolist(),
                "attitude_response_rad": attitude_delta.tolist(),
                "peak_tilt_rad": peak_tilt,
                "peak_roll_pitch_rad": combination_rpy[peak_index].tolist(),
                "command_pattern": pattern,
                "velocity_sign_checks": velocity_checks,
            }
            status = "PASS" if passed else "FAIL"
            self.get_logger().info(
                f"{status} {name}: velocity="
                f"{np.array2string(response[:3], precision=4)}, "
                f"attitude={np.array2string(attitude_delta, precision=4)}, "
                f"peak_tilt={math.degrees(peak_tilt):.3f} deg"
            )
        self._publish_for([0.0] * 6, 0.2)
        report = {
            "passed": overall_pass,
            "amplitude": self.amplitude,
            "frame_convention": "world=NED, body=FRD, twist=body",
            "response_contract": (
                "linear axes use tail body velocity; angular axes use tail "
                "attitude displacement because restoring moments can reverse "
                "angular velocity before a pulse ends"
            ),
            "results": results,
            "combination_results": combination_results,
            "combination_hold_time": self.combination_hold_time,
            "max_combination_tilt_degrees": self.max_combination_tilt_degrees,
        }
        if self.report_path:
            path = Path(self.report_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            self.get_logger().info(f"Wrote dynamics report to {path}")
        return overall_pass

    def _reset_vehicle(self) -> bool:
        request = Respawn.Request()
        request.name = self.vehicle_name
        request.origin.position.z = self.spawn_depth
        request.origin.orientation.w = 1.0
        future = self.respawn.call_async(request)
        end = time.monotonic() + 5.0
        while rclpy.ok() and not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None:
            self.get_logger().error("Vehicle respawn timed out")
            return False
        if not future.result().success:
            self.get_logger().error(f"Vehicle respawn failed: {future.result().message}")
            return False
        return True

    def _publish_for(self, values: list[float], duration: float) -> np.ndarray:
        message = Float64MultiArray(data=values)
        samples = []
        end = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end:
            self.publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.015)
            if self.latest_state is not None:
                samples.append(self.latest_state.copy())
        if not samples:
            raise RuntimeError("No odometry samples received")
        return np.asarray(samples)

    def _tail_mean(self, samples: np.ndarray) -> np.ndarray:
        count = max(1, int(math.ceil(self.tail_time * 100.0)))
        return samples[-count:].mean(axis=0)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_to_rpy(
    x: float,
    y: float,
    z: float,
    w: float,
) -> tuple[float, float, float]:
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = DOFDynamicsTest()
    exit_code = 1
    try:
        exit_code = 0 if node.run() else 1
    except KeyboardInterrupt:
        pass
    except Exception as error:
        node.get_logger().error(f"Dynamics test failed: {error}")
    finally:
        node.publisher.publish(Float64MultiArray(data=[0.0] * 6))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
