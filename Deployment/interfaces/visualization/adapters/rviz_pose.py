#!/usr/bin/env python3
"""主机显示用：复制估计位姿供 RViz 查看，不消费或生成协方差。

在主机的 ROS 2 Humble 环境运行：python3 Deployment/interfaces/visualization/adapters/rviz_pose.py
RViz 添加 Pose，话题选择 /visualization/camera_pose。
"""
import copy
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class RvizPose(Node):
    def __init__(self):
        super().__init__('rviz_pose')
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(
            PoseStamped, '/visualization/camera_pose', output_qos)
        self.subscriber = self.create_subscription(
            Odometry, '/odometry/estimated', self.receive, input_qos)
        self.get_logger().info(
            '主机位姿显示已启动：/odometry/estimated → /visualization/camera_pose；'
            '保留原坐标系、时间戳和位姿数值，仅供 RViz 显示')

    def receive(self, source):
        pose = source.pose.pose
        position = (pose.position.x, pose.position.y, pose.position.z)
        quaternion = (pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w)
        if not all(math.isfinite(value) for value in position + quaternion):
            self.get_logger().warning(
                '忽略含非有限位置或四元数的显示帧', throttle_duration_sec=5.0)
            return
        if not any(value != 0.0 for value in quaternion):
            self.get_logger().warning(
                '忽略四元数全为零的显示帧', throttle_duration_sec=5.0)
            return
        output = PoseStamped()
        output.header = copy.deepcopy(source.header)
        output.pose = copy.deepcopy(pose)
        self.publisher.publish(output)


def main():
    rclpy.init()
    node = RvizPose()
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
