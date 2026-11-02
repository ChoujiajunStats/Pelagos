#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""将 OKVIS2 的 world→body 位姿变换到左校正相机光学坐标。"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


CAMERA_FRAME = 'cam0_rect_optical'


def rigid_transform(value):
    """接受米制齐次刚体变换，拒绝尺度、反射及损坏标定。"""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('外参必须是有限数值构成的 4×4 矩阵')
    if not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-9, rtol=0):
        raise ValueError('外参最后一行必须为 [0, 0, 0, 1]')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('外参旋转矩阵不正交')
    if not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0):
        raise ValueError('外参旋转矩阵必须是右手旋转，行列式为 1')
    return matrix.copy()


def quaternion_rotation(quaternion_xyzw):
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError('输入四元数必须是四个有限数值，顺序为 xyzw')
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('输入四元数长度无效')
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def rotation_quaternion(rotation):
    """返回 xyzw；分支覆盖接近 180° 的旋转，不依赖 scipy。"""
    trace = float(np.trace(rotation))
    if trace > 0.:
        scale = 2. * np.sqrt(trace + 1.)
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale, scale / 4.,
        ])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        other, last = (axis + 1) % 3, (axis + 2) % 3
        scale = 2. * np.sqrt(1. + rotation[axis, axis]
                               - rotation[other, other] - rotation[last, last])
        quaternion = np.zeros(4, dtype=np.float64)
        quaternion[axis] = scale / 4.
        quaternion[other] = (rotation[other, axis] + rotation[axis, other]) / scale
        quaternion[last] = (rotation[last, axis] + rotation[axis, last]) / scale
        quaternion[3] = (rotation[last, other] - rotation[other, last]) / scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.:
        quaternion = -quaternion
    return quaternion


def camera_pose_from_body(position, quaternion_xyzw, t_body_camera):
    """T_world_camera = T_world_body @ T_body_camera，含旋转后的杆臂。"""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('输入位移必须是三个有限数值')
    r_world_body = quaternion_rotation(quaternion_xyzw)
    camera_position = position + r_world_body @ t_body_camera[:3, 3]
    camera_rotation = r_world_body @ t_body_camera[:3, :3]
    if not np.isfinite(camera_position).all():
        raise ValueError('相机位置计算结果非有限')
    return camera_position, rotation_quaternion(camera_rotation)


def load_extrinsics(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('外参配置 schema_version 必须为 1')
    if config.get('body_is_imu') is not True:
        raise ValueError('当前配置要求 body=IMU，必须显式声明 body_is_imu=true')
    if config.get('translation_unit') != 'm':
        raise ValueError('外参平移单位必须显式声明 translation_unit=m')
    for name, default in [('body_frame', 'body'), ('world_frame', 'world'),
                          ('camera_frame', CAMERA_FRAME)]:
        frame = config.get(name, default)
        if (not isinstance(frame, str) or not frame or frame.startswith('/')
                or any(character.isspace() for character in frame)):
            raise ValueError(f'外参配置 {name} 必须是非空且无前导斜线的坐标名称')
        config[name] = frame
    return config, rigid_transform(config['T_body_left_optical'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extrinsics', required=True, help='真实相机外参 JSON 文件')
    parser.add_argument('--input-topic', default='/okvis/okvis_odometry')
    parser.add_argument('--output-topic', default='/odometry/estimated')
    args, ros_args = parser.parse_known_args()
    config, t_body_camera = load_extrinsics(args.extrinsics)

    # 保持几何函数可在没有 ROS 的主机上独立核验。
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    class OkvisPoseAdapter(Node):
        def __init__(self):
            super().__init__('okvis_pose_adapter')
            self.last_stamp_ns = None
            self.frames = 0
            self.publisher = self.create_publisher(
                Odometry, args.output_topic, qos_profile_sensor_data)
            self.subscriber = self.create_subscription(
                Odometry, args.input_topic, self.receive, qos_profile_sensor_data)
            self.create_timer(10., self.report)
            self.get_logger().info(
                f'使用米制外参将 {config["world_frame"]}→{config["body_frame"]} '
                f'转换到 {config["camera_frame"]}；twist 与协方差未知，以 NaN 标记')

        def receive(self, source):
            if source.header.frame_id != config['world_frame']:
                raise ValueError(f'输入世界坐标不匹配：{source.header.frame_id}')
            if source.child_frame_id != config['body_frame']:
                raise ValueError(f'输入机体坐标不匹配：{source.child_frame_id}')
            stamp = source.header.stamp
            if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                raise ValueError('输入时间戳无效')
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
                raise ValueError('OKVIS 位姿时间戳重复或倒退，结束当前地图运行')
            pose = source.pose.pose
            position, quaternion = camera_pose_from_body(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w], t_body_camera)
            output = Odometry()
            output.header = copy.deepcopy(source.header)
            output.child_frame_id = config['camera_frame']
            output.pose.pose.position.x = float(position[0])
            output.pose.pose.position.y = float(position[1])
            output.pose.pose.position.z = float(position[2])
            output.pose.pose.orientation.x = float(quaternion[0])
            output.pose.pose.orientation.y = float(quaternion[1])
            output.pose.pose.orientation.z = float(quaternion[2])
            output.pose.pose.orientation.w = float(quaternion[3])
            # Odometry 没有通用的“未知协方差”标志；本地契约明确使用 NaN。
            # Wavemap 仅消费上述 pose，不将未知量解释为零速度或确定估计。
            unknown = float('nan')
            output.pose.covariance = [unknown] * 36
            output.twist.covariance = [unknown] * 36
            for vector in [output.twist.twist.linear, output.twist.twist.angular]:
                vector.x = vector.y = vector.z = unknown
            self.publisher.publish(output)
            self.last_stamp_ns = stamp_ns
            self.frames += 1

        def report(self):
            self.get_logger().info(f'累计输出左校正相机位姿 {self.frames} 帧')

    rclpy.init(args=ros_args)
    node = OkvisPoseAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
