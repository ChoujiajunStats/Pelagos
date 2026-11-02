"""Convert the simulator's body-FRD IMU stream to VINS body-FLU axes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from disaware_learning.estimation.vins_fusion_adapter import (
    matrix_to_quaternion_xyzw,
    publish_while_ros_is_running,
    quaternion_to_matrix_xyzw,
)


FRD_TO_FLU = np.diag((1.0, -1.0, -1.0))


def transform_frd_vector_to_flu(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("FRD vector must contain three finite values")
    return FRD_TO_FLU @ vector


def transform_frd_covariance_to_flu(values: Iterable[float]) -> np.ndarray:
    covariance = np.asarray(tuple(values), dtype=np.float64)
    if covariance.shape != (9,) or not np.all(np.isfinite(covariance)):
        raise ValueError("IMU covariance must contain nine finite values")
    matrix = covariance.reshape(3, 3)
    return (FRD_TO_FLU @ matrix @ FRD_TO_FLU.T).reshape(9)


def transform_ned_frd_orientation_to_flu(
    quaternion_xyzw: Iterable[float],
) -> np.ndarray:
    """Express the same attitude with FLU world and FLU body coordinates."""
    rotation_ned_from_frd = quaternion_to_matrix_xyzw(tuple(quaternion_xyzw))
    rotation_flu_world_from_flu_body = (
        FRD_TO_FLU @ rotation_ned_from_frd @ FRD_TO_FLU.T
    )
    return matrix_to_quaternion_xyzw(rotation_flu_world_from_flu_body)


def _vector(message: object) -> tuple[float, float, float]:
    return (float(message.x), float(message.y), float(message.z))


def adapt_imu_message(message: Imu, output_frame_id: str) -> Imu:
    output = Imu()
    output.header = message.header
    output.header.frame_id = output_frame_id

    orientation = transform_ned_frd_orientation_to_flu(
        _vector(message.orientation) + (float(message.orientation.w),)
    )
    output.orientation.x = float(orientation[0])
    output.orientation.y = float(orientation[1])
    output.orientation.z = float(orientation[2])
    output.orientation.w = float(orientation[3])

    angular_velocity = transform_frd_vector_to_flu(_vector(message.angular_velocity))
    output.angular_velocity.x = float(angular_velocity[0])
    output.angular_velocity.y = float(angular_velocity[1])
    output.angular_velocity.z = float(angular_velocity[2])

    acceleration = transform_frd_vector_to_flu(_vector(message.linear_acceleration))
    output.linear_acceleration.x = float(acceleration[0])
    output.linear_acceleration.y = float(acceleration[1])
    output.linear_acceleration.z = float(acceleration[2])

    output.orientation_covariance = transform_frd_covariance_to_flu(
        message.orientation_covariance
    ).tolist()
    output.angular_velocity_covariance = transform_frd_covariance_to_flu(
        message.angular_velocity_covariance
    ).tolist()
    output.linear_acceleration_covariance = transform_frd_covariance_to_flu(
        message.linear_acceleration_covariance
    ).tolist()
    return output


class VinsSensorAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("vins_sensor_adapter")
        self.declare_parameter("input_imu_topic", "/bluerov2/imu")
        self.declare_parameter("output_imu_topic", "/bluerov2/vins/imu")
        self.declare_parameter("output_frame_id", "bluerov2/imu_flu")
        input_topic = str(self.get_parameter("input_imu_topic").value).strip()
        output_topic = str(self.get_parameter("output_imu_topic").value).strip()
        self.output_frame_id = str(
            self.get_parameter("output_frame_id").value
        ).strip()
        if not input_topic or not output_topic or not self.output_frame_id:
            raise ValueError("VINS IMU topics and output frame must be non-empty")
        self.publisher = self.create_publisher(
            Imu, output_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            Imu, input_topic, self._callback, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"VINS IMU adapter: {input_topic} (FRD) -> {output_topic} (FLU)"
        )

    def _callback(self, message: Imu) -> None:
        try:
            publish_while_ros_is_running(
                self.publisher, adapt_imu_message(message, self.output_frame_id)
            )
        except ValueError as error:
            self.get_logger().warning(f"Rejected invalid IMU sample: {error}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VinsSensorAdapterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
