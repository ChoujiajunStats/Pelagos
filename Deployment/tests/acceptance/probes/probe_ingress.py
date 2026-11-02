#!/usr/bin/env python3
"""采集真实话题时序、标定与 TF，保存可复核的相机输入验收记录。"""
import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import CameraInfo, Image, Imu
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-dir', type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Node('zed_x_ingress_acceptance')
    records = {}
    subscriptions = []

    def subscribe(topic, kind, qos=qos_profile_sensor_data):
        record = records.setdefault(topic, dict(count=0, stamps_ns=[], first=None))

        def receive(msg):
            record['count'] += 1
            if hasattr(msg, 'header'):
                record['stamps_ns'].append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
            if record['first'] is None:
                if isinstance(msg, Image):
                    record['first'] = dict(width=msg.width, height=msg.height, encoding=msg.encoding,
                                           step=msg.step, frame_id=msg.header.frame_id)
                    if args.sample_dir and msg.encoding == 'mono8':
                        args.sample_dir.mkdir(parents=True, exist_ok=True)
                        pixels = b''.join(bytes(msg.data[row * msg.step:row * msg.step + msg.width])
                                          for row in range(msg.height))
                        side = 'left' if '/left/' in topic else 'right'
                        (args.sample_dir / (side + '-raw.pgm')).write_bytes(
                            f'P5\n{msg.width} {msg.height}\n255\n'.encode() + pixels)
                else:
                    record['first'] = message_to_ordereddict(msg)
        subscriptions.append(node.create_subscription(kind, topic, receive, qos))

    prefix = '/zed/zed_node'
    for side in ('left', 'right'):
        for color, mode in (('color', 'raw'), ('gray', 'raw')):
            base = f'{prefix}/{side}/{color}/{mode}'
            subscribe(base + '/image', Image)
            subscribe(base + '/camera_info', CameraInfo)
    # SDK 按图像周期批量发布约 13 条 IMU，默认 sensor_data 的深度 5 会丢样。
    imu_qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
    subscribe(prefix + '/imu/data_raw', Imu, imu_qos)
    subscribe(prefix + '/left_cam_imu_transform', TransformStamped, imu_qos)
    subscribe('/tf_static', TFMessage, QoSProfile(
        depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    for record in records.values():
        stamps = record['stamps_ns']
        record['strictly_increasing'] = all(a < b for a, b in zip(stamps, stamps[1:]))
        record['rate_hz'] = ((len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])
                             if len(stamps) > 1 and stamps[-1] > stamps[0] else None)
        if len(stamps) > 1:
            gaps = [(b-a)/1e6 for a,b in zip(stamps, stamps[1:])]
            record['stamp_gap_ms'] = dict(min=min(gaps), max=max(gaps))
    left = set(records[prefix + '/left/gray/raw/image']['stamps_ns'])
    right = set(records[prefix + '/right/gray/raw/image']['stamps_ns'])
    result = dict(duration_s=time.monotonic() - started, topics=records,
                  stereo_common_stamps=len(left & right),
                  stereo_unmatched_stamps=len(left ^ right))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({topic: {k: v for k, v in data.items() if k not in ('stamps_ns', 'first')}
                      for topic, data in records.items()}, indent=2))
    print('双目共有时间戳:', result['stereo_common_stamps'])
    print('记录:', args.output)
    node.destroy_node()
    rclpy.shutdown()
    if any(records[prefix + '/' + suffix]['count'] == 0 for suffix in (
            'left/color/raw/image', 'right/color/raw/image', 'left/gray/raw/image',
            'right/gray/raw/image', 'imu/data_raw')):
        raise SystemExit('输入验收失败：存在没有数据的话题')


if __name__ == '__main__':
    main()
