"""Adapt VINS-Fusion ROS2 output to the canonical deployable VIO contract.

The referenced VINS-Fusion fork publishes its IMU-propagated state with
``estimator.Vs`` (world-frame velocity) in
``nav_msgs/Odometry.twist.linear`` and leaves angular velocity unset.  The
learning pipeline instead requires a local NED pose and body-FRD
linear/angular velocity.  This node performs that conversion explicitly and
uses the latest non-future IMU sample for angular velocity.

Loop-fusion output is deliberately unsupported: pose-graph corrections can
jump and do not provide the smooth local velocity needed for dynamics data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger

from disaware_learning.estimation.vio_state import VioState


SOURCE_ID = "vins_fusion_ros2_imu_propagate"
DEFAULT_FLU_TO_FRD_QUATERNION_XYZW = (1.0, 0.0, 0.0, 0.0)


def _finite_vector(value: Any, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {width} finite values")
    return result


def _unit_quaternion(value: Any, name: str) -> np.ndarray:
    result = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} cannot be zero")
    return result / norm


def quaternion_to_matrix_xyzw(value: Any) -> np.ndarray:
    x, y, z, w = _unit_quaternion(value, "quaternion_xyzw")
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(matrix: Any) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix must be finite [3, 3]")
    # Eigenvalue extraction is stable for all traces and avoids separate
    # near-pi branches.  The matrix must already be a proper rotation.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("matrix is not a proper rotation")
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    symmetric = np.asarray(
        (
            (r00 - r11 - r22, r01 + r10, r02 + r20, r21 - r12),
            (r01 + r10, r11 - r00 - r22, r12 + r21, r02 - r20),
            (r02 + r20, r12 + r21, r22 - r00 - r11, r10 - r01),
            (r21 - r12, r02 - r20, r10 - r01, r00 + r11 + r22),
        ),
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion / np.linalg.norm(quaternion)


@dataclass(frozen=True)
class VinsFusionFrameConfig:
    """Static frame contract surrounding VINS-Fusion's local estimator frame.

    Defaults assume both VINS local-world and IMU/body axes use the common
    forward-left-up convention.  A 180 degree rotation around +x maps those
    axes to local NED and body FRD.  Calibrated installations may override
    either quaternion without changing the downstream schema.
    """

    ned_from_vins_world_xyzw: tuple[float, float, float, float] = (
        DEFAULT_FLU_TO_FRD_QUATERNION_XYZW
    )
    vins_body_from_body_frd_xyzw: tuple[float, float, float, float] = (
        DEFAULT_FLU_TO_FRD_QUATERNION_XYZW
    )
    anchor_first_position: bool = True

    def __post_init__(self) -> None:
        for name in (
            "ned_from_vins_world_xyzw",
            "vins_body_from_body_frd_xyzw",
        ):
            normalized = _unit_quaternion(getattr(self, name), name)
            object.__setattr__(self, name, tuple(float(value) for value in normalized))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class VinsFusionFrameAdapter:
    """Pure coordinate adapter used by the ROS node and unit tests."""

    def __init__(self, config: VinsFusionFrameConfig = VinsFusionFrameConfig()) -> None:
        self.config = config
        self._ned_from_world = quaternion_to_matrix_xyzw(
            config.ned_from_vins_world_xyzw
        )
        self._vins_body_from_frd = quaternion_to_matrix_xyzw(
            config.vins_body_from_body_frd_xyzw
        )
        self.reset()

    def reset(self) -> None:
        self._origin_world_m: np.ndarray | None = None

    def adapt(
        self,
        *,
        position_vins_world_m: Any,
        orientation_vins_body_to_world_xyzw: Any,
        linear_velocity_vins_world_m_s: Any,
        angular_velocity_vins_body_rad_s: Any,
        timestamp_seconds: float,
    ) -> VioState:
        position = _finite_vector(position_vins_world_m, 3, "position_vins_world_m")
        velocity_world = _finite_vector(
            linear_velocity_vins_world_m_s, 3, "linear_velocity_vins_world_m_s"
        )
        angular_vins_body = _finite_vector(
            angular_velocity_vins_body_rad_s,
            3,
            "angular_velocity_vins_body_rad_s",
        )
        rotation_world_from_vins_body = quaternion_to_matrix_xyzw(
            orientation_vins_body_to_world_xyzw
        )
        if self._origin_world_m is None:
            self._origin_world_m = position.copy()
        relative_position = (
            position - self._origin_world_m
            if self.config.anchor_first_position
            else position
        )
        rotation_ned_from_frd = (
            self._ned_from_world
            @ rotation_world_from_vins_body
            @ self._vins_body_from_frd
        )
        velocity_ned = self._ned_from_world @ velocity_world
        velocity_frd = rotation_ned_from_frd.T @ velocity_ned
        angular_velocity_frd = self._vins_body_from_frd.T @ angular_vins_body
        return VioState(
            position_ned_m=self._ned_from_world @ relative_position,
            orientation_body_to_ned_xyzw=matrix_to_quaternion_xyzw(
                rotation_ned_from_frd
            ),
            linear_velocity_body_frd_m_s=velocity_frd,
            angular_velocity_body_frd_rad_s=angular_velocity_frd,
            timestamp_seconds=float(timestamp_seconds),
            source=SOURCE_ID,
        )


def _message_stamp(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _set_message_stamp(message: Any, timestamp: float) -> None:
    seconds = math.floor(timestamp)
    nanoseconds = int(round((timestamp - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = nanoseconds


def select_causal_angular_velocity(
    samples: Any,
    timestamp_seconds: float,
    maximum_age_seconds: float,
) -> np.ndarray | None:
    """Select the newest IMU rate at or before an estimator timestamp."""
    timestamp = float(timestamp_seconds)
    maximum_age = float(maximum_age_seconds)
    if (
        not math.isfinite(timestamp)
        or not math.isfinite(maximum_age)
        or maximum_age <= 0.0
    ):
        raise ValueError("timestamp and maximum age are invalid")
    selected: tuple[float, np.ndarray] | None = None
    for sample_timestamp, angular_velocity in samples:
        sample_time = float(sample_timestamp)
        if sample_time <= timestamp + 1e-9:
            selected = (
                sample_time,
                _finite_vector(angular_velocity, 3, "angular_velocity"),
            )
        else:
            break
    if selected is None or timestamp - selected[0] > maximum_age:
        return None
    return selected[1].copy()


def publish_while_ros_is_running(publisher: object, message: object) -> bool:
    """Suppress only the rclpy publish race caused by an active shutdown."""
    try:
        publisher.publish(message)
    except RuntimeError:
        if rclpy.ok():
            raise
        return False
    return True


class VinsFusionAdapterNode(Node):
    """Publish canonical odometry from the real-time VINS propagated state."""

    def __init__(self) -> None:
        super().__init__("vins_fusion_adapter")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("vins_state_topic", "/vins_estimator/imu_propagate")
        self.declare_parameter("vins_imu_topic", "/imu0")
        self.declare_parameter("output_odometry_topic", "")
        self.declare_parameter("expected_vins_world_frame", "world")
        self.declare_parameter("expected_vins_body_frame", "body")
        self.declare_parameter(
            "ned_from_vins_world_xyzw",
            list(DEFAULT_FLU_TO_FRD_QUATERNION_XYZW),
        )
        self.declare_parameter(
            "vins_body_from_body_frd_xyzw",
            list(DEFAULT_FLU_TO_FRD_QUATERNION_XYZW),
        )
        self.declare_parameter("anchor_first_position", True)
        self.declare_parameter("maximum_imu_age_seconds", 0.03)
        self.declare_parameter("imu_buffer_seconds", 2.0)

        self.vehicle_name = str(self.get_parameter("vehicle_name").value).strip("/")
        if not self.vehicle_name:
            raise ValueError("vehicle_name cannot be empty")
        self.vins_state_topic = str(self.get_parameter("vins_state_topic").value).strip()
        self.vins_imu_topic = str(self.get_parameter("vins_imu_topic").value).strip()
        output = str(self.get_parameter("output_odometry_topic").value).strip()
        self.output_odometry_topic = output or f"/{self.vehicle_name}/vio/odometry"
        if not self.vins_state_topic or not self.vins_imu_topic:
            raise ValueError("VINS propagated-state and IMU topics are required")
        self.expected_world_frame = str(
            self.get_parameter("expected_vins_world_frame").value
        ).strip()
        self.expected_body_frame = str(
            self.get_parameter("expected_vins_body_frame").value
        ).strip()
        self.maximum_imu_age = float(
            self.get_parameter("maximum_imu_age_seconds").value
        )
        self.imu_buffer_seconds = float(self.get_parameter("imu_buffer_seconds").value)
        if (
            not math.isfinite(self.maximum_imu_age)
            or self.maximum_imu_age <= 0.0
            or not math.isfinite(self.imu_buffer_seconds)
            or self.imu_buffer_seconds <= self.maximum_imu_age
        ):
            raise ValueError("IMU age/buffer settings are invalid")
        self.adapter = VinsFusionFrameAdapter(
            VinsFusionFrameConfig(
                ned_from_vins_world_xyzw=tuple(
                    self.get_parameter("ned_from_vins_world_xyzw").value
                ),
                vins_body_from_body_frd_xyzw=tuple(
                    self.get_parameter("vins_body_from_body_frd_xyzw").value
                ),
                anchor_first_position=bool(
                    self.get_parameter("anchor_first_position").value
                ),
            )
        )
        self._imu_buffer: deque[tuple[float, np.ndarray]] = deque()
        self._last_odometry_stamp: float | None = None
        self._published_count = 0
        self._dropped_missing_imu = 0
        self._dropped_frame = 0
        self._last_warning_at = -math.inf
        self.publisher = self.create_publisher(
            Odometry, self.output_odometry_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu, self.vins_imu_topic, self._imu_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            self.vins_state_topic,
            self._odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_service(Trigger, "~/reset_origin", self._reset_origin_service)
        self.create_service(Trigger, "~/status", self._status_service)
        self.get_logger().info(
            f"VINS adapter: {self.vins_state_topic} + {self.vins_imu_topic} "
            f"-> {self.output_odometry_topic}"
        )

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_at >= 2.0:
            self.get_logger().warning(message)
            self._last_warning_at = now

    def _imu_callback(self, message: Imu) -> None:
        timestamp = _message_stamp(message)
        angular = np.asarray(
            (
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ),
            dtype=np.float64,
        )
        if not math.isfinite(timestamp) or not np.all(np.isfinite(angular)):
            return
        if self._imu_buffer and timestamp <= self._imu_buffer[-1][0] + 1e-9:
            return
        self._imu_buffer.append((timestamp, angular))
        cutoff = timestamp - self.imu_buffer_seconds
        while len(self._imu_buffer) > 1 and self._imu_buffer[1][0] < cutoff:
            self._imu_buffer.popleft()

    def _causal_angular_velocity(self, timestamp: float) -> np.ndarray | None:
        return select_causal_angular_velocity(
            self._imu_buffer, timestamp, self.maximum_imu_age
        )

    def _odometry_callback(self, message: Odometry) -> None:
        timestamp = _message_stamp(message)
        if (
            not math.isfinite(timestamp)
            or (
                self._last_odometry_stamp is not None
                and timestamp <= self._last_odometry_stamp + 1e-9
            )
        ):
            return
        if (
            self.expected_world_frame
            and message.header.frame_id != self.expected_world_frame
        ) or (
            self.expected_body_frame
            and message.child_frame_id != self.expected_body_frame
        ):
            self._dropped_frame += 1
            self._warn(
                "VINS frame mismatch: expected "
                f"{self.expected_world_frame}->{self.expected_body_frame}, got "
                f"{message.header.frame_id}->{message.child_frame_id}"
            )
            return
        angular_velocity = self._causal_angular_velocity(timestamp)
        if angular_velocity is None:
            self._dropped_missing_imu += 1
            self._warn("No causal IMU angular-rate sample for VINS odometry")
            return
        pose = message.pose.pose
        twist = message.twist.twist
        try:
            state = self.adapter.adapt(
                position_vins_world_m=(
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ),
                orientation_vins_body_to_world_xyzw=(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
                # zinuok/VINS-Fusion-ROS2 publishes estimator.Vs here, which
                # is expressed in the VINS world frame despite child_frame_id.
                linear_velocity_vins_world_m_s=(
                    twist.linear.x,
                    twist.linear.y,
                    twist.linear.z,
                ),
                angular_velocity_vins_body_rad_s=angular_velocity,
                timestamp_seconds=timestamp,
            )
        except ValueError as error:
            self._warn(f"Rejected invalid VINS odometry: {error}")
            return
        output = Odometry()
        _set_message_stamp(output, state.timestamp_seconds)
        output.header.frame_id = "world_ned"
        output.child_frame_id = f"{self.vehicle_name}/base_link_frd"
        output.pose.pose.position.x, output.pose.pose.position.y, output.pose.pose.position.z = (
            float(value) for value in state.position_ned_m
        )
        (
            output.pose.pose.orientation.x,
            output.pose.pose.orientation.y,
            output.pose.pose.orientation.z,
            output.pose.pose.orientation.w,
        ) = (float(value) for value in state.orientation_body_to_ned_xyzw)
        output.twist.twist.linear.x, output.twist.twist.linear.y, output.twist.twist.linear.z = (
            float(value) for value in state.linear_velocity_body_frd_m_s
        )
        (
            output.twist.twist.angular.x,
            output.twist.twist.angular.y,
            output.twist.twist.angular.z,
        ) = (float(value) for value in state.angular_velocity_body_frd_rad_s)
        if not publish_while_ros_is_running(self.publisher, output):
            return
        self._last_odometry_stamp = timestamp
        self._published_count += 1

    def _reset_origin_service(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self.adapter.reset()
        self._last_odometry_stamp = None
        response.success = True
        response.message = "VINS local-NED origin will anchor on the next estimate"
        return response

    def _status_service(self, request: Trigger.Request, response: Trigger.Response):
        del request
        response.success = self._published_count > 0
        response.message = str(
            {
                "source_id": SOURCE_ID,
                "published_count": self._published_count,
                "dropped_missing_imu": self._dropped_missing_imu,
                "dropped_frame": self._dropped_frame,
                "last_odometry_stamp": self._last_odometry_stamp,
                "frame_config": self.adapter.config.as_dict(),
            }
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VinsFusionAdapterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_FLU_TO_FRD_QUATERNION_XYZW",
    "SOURCE_ID",
    "VinsFusionAdapterNode",
    "VinsFusionFrameAdapter",
    "VinsFusionFrameConfig",
    "matrix_to_quaternion_xyzw",
    "publish_while_ros_is_running",
    "quaternion_to_matrix_xyzw",
    "select_causal_angular_velocity",
]
