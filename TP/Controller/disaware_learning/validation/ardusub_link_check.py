"""Check that ArduSub JSON output is advancing through the ROS bridge."""

from __future__ import annotations

import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, UInt64

from disaware_learning.control.contract import ACTUATOR_COUNT


class ArduSubLinkCheck(Node):
    """Validate frame counts, normalized motor output, and NED odometry."""

    def __init__(self) -> None:
        super().__init__("ardusub_link_check")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("minimum_frames", 20)
        vehicle = str(self.get_parameter("vehicle_name").value).strip("/")
        self.timeout = max(1.0, float(self.get_parameter("timeout").value))
        self.minimum_frames = max(2, int(self.get_parameter("minimum_frames").value))
        prefix = f"/{vehicle}"
        self.frames: list[int] = []
        self.pwm: list[float] | None = None
        self.odometry: Odometry | None = None
        self.create_subscription(
            UInt64,
            f"{prefix}/ardusub/motor_output/frame_count",
            self._frame_callback,
            20,
        )
        self.create_subscription(
            Float64MultiArray,
            f"{prefix}/ardusub/motor_output",
            self._pwm_callback,
            20,
        )
        self.create_subscription(
            Odometry,
            f"{prefix}/odometry",
            self._odometry_callback,
            qos_profile_sensor_data,
        )

    def _frame_callback(self, message: UInt64) -> None:
        frame = int(message.data)
        if not self.frames or frame != self.frames[-1]:
            self.frames.append(frame)

    def _pwm_callback(self, message: Float64MultiArray) -> None:
        self.pwm = [float(value) for value in message.data]

    def _odometry_callback(self, message: Odometry) -> None:
        self.odometry = message

    def run(self) -> bool:
        end = time.monotonic() + self.timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.frames) >= self.minimum_frames and self._messages_valid():
                self.get_logger().info(
                    "PASS ArduSub link: "
                    f"{len(self.frames)} advancing frames, last={self.frames[-1]}, "
                    f"{ACTUATOR_COUNT} bounded motor outputs, world_ned odometry"
                )
                return True
        self.get_logger().error(
            "FAIL ArduSub link: "
            f"frames={len(self.frames)}, pwm={self.pwm}, "
            f"odom_frame={self.odometry.header.frame_id if self.odometry else None}"
        )
        return False

    def _messages_valid(self) -> bool:
        if self.pwm is None or self.odometry is None:
            return False
        if len(self.pwm) != ACTUATOR_COUNT:
            return False
        if any(not math.isfinite(value) or abs(value) > 1.0001 for value in self.pwm):
            return False
        if self.odometry.header.frame_id != "world_ned":
            return False
        return all(new > old for old, new in zip(self.frames, self.frames[1:]))


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = ArduSubLinkCheck()
    try:
        return 0 if node.run() else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
