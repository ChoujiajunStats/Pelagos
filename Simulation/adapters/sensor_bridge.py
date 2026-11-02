#!/usr/bin/env python3
"""Adapt the existing Stonefish M2 sensors to Pelagos and OKVIS2 topics."""
import array
import copy

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu


class SensorBridge(Node):
    def __init__(self):
        super().__init__('pelagos_stonefish_sensors')
        # OKVIS2's subscriber uses the default reliable ROS QoS.
        reliable = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.color = {}
        self.gray = {}
        for index, side in enumerate(('left', 'right')):
            self.color[side] = self.create_publisher(
                Image, f'/zed/zed_node/{side}/color/raw/image', qos_profile_sensor_data)
            self.gray[side] = self.create_publisher(Image, f'/okvis/cam{index}/image_raw', reliable)
            self.create_subscription(Image, f'/rov01/sensors/stereo/{side}/image_raw',
                                     lambda m, s=side: self.image(m, s), qos_profile_sensor_data)
        self.imu = self.create_publisher(Imu, '/okvis/imu0', reliable)
        self.create_subscription(Imu, '/rov01/sensors/imu', self.imu.publish, qos_profile_sensor_data)

    def image(self, msg, side):
        if msg.encoding not in ('rgb8', 'bgr8') or msg.step < msg.width * 3:
            self.get_logger().error(f'Unsupported {side} camera image: {msg.encoding}')
            return
        try:
            rgb = np.ndarray((msg.height, msg.width, 3), dtype=np.uint8,
                             buffer=memoryview(msg.data), strides=(msg.step, 3, 1))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY)
        except (ValueError, TypeError, cv2.error) as exc:
            self.get_logger().error(f'Invalid {side} camera image: {exc}')
            return
        color = copy.copy(msg)
        color.header = copy.deepcopy(msg.header)
        # Pelagos LAS2's current adapter checks these labels; the calibration
        # file still describes the simulated camera, not the real ZED.
        color.header.frame_id = f'zed_{side}_camera_frame_optical'
        self.color[side].publish(color)
        mono = Image()
        mono.header = copy.deepcopy(msg.header)
        mono.height, mono.width = msg.height, msg.width
        mono.encoding, mono.is_bigendian, mono.step = 'mono8', 0, msg.width
        mono.data = array.array('B', np.ascontiguousarray(gray).tobytes())
        self.gray[side].publish(mono)


def main():
    rclpy.init()
    node = SensorBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
