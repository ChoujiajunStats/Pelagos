#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""将已完成外参变换的相机里程计发布到 nvblox 独立 TF 通道。"""
import math
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class CameraTransform(Node):
    def __init__(self):
        super().__init__('nvblox_camera_transform')
        self.broadcaster = TransformBroadcaster(self)
        self.last_stamp = -1
        self.subscription = self.create_subscription(
            Odometry, '/odometry/estimated', self.receive, qos_profile_sensor_data)

    def receive(self, msg):
        if msg.header.frame_id != 'world' or msg.child_frame_id != 'cam0_rect_optical':
            raise ValueError('相机里程计坐标系不匹配')
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        if not all(math.isfinite(x) for x in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            raise ValueError('相机位姿包含非有限值')
        stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if stamp <= self.last_stamp:
            raise ValueError('相机位姿时间戳未严格递增')
        transform = TransformStamped()
        transform.header = msg.header
        transform.child_frame_id = msg.child_frame_id
        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z
        transform.transform.rotation = q
        self.broadcaster.sendTransform(transform)
        self.last_stamp = stamp


def main():
    rclpy.init()
    node = CameraTransform()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
