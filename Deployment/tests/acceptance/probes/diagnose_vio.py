#!/usr/bin/env python3
"""Read-only bounded VIO diagnosis; save one stereo pair and small sensor statistics."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('read_only_vio_diagnosis')
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    samples = {'imu': [], 'body': [], 'optimized': []}
    images = {}
    subs = {}

    def stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def image(side, m):
        if side in images:
            return
        if m.encoding not in ('mono8', '8UC1'):
            images[side] = {'encoding': m.encoding, 'error': 'unsupported snapshot encoding'}
            return
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)[:, :m.width]
        (args.output / (side + '.pgm')).write_bytes(f'P5\n{m.width} {m.height}\n255\n'.encode() + a.tobytes())
        images[side] = {'stamp': stamp(m), 'encoding': m.encoding,
                        'mean': float(a.mean()), 'std': float(a.std()),
                        'gradient_mean': float(np.abs(np.diff(a.astype(float), axis=1)).mean())}
        node.destroy_subscription(subs[side])

    for side in ('left', 'right'):
        subs[side] = node.create_subscription(Image, '/zed/zed_node/' + side + '/gray/raw/image',
                                              lambda m, s=side: image(s, m), qos)

    def imu(m):
        a, g = m.linear_acceleration, m.angular_velocity
        samples['imu'].append([stamp(m), a.x, a.y, a.z, g.x, g.y, g.z])

    def body(m):
        p, v = m.pose.pose.position, m.twist.twist.linear
        samples['body'].append([stamp(m), p.x, p.y, p.z, v.x, v.y, v.z])

    node.create_subscription(Imu, '/zed/zed_node/imu/data_raw', imu, qos)
    node.create_subscription(Odometry, '/okvis/okvis_odometry', body, qos)
    node.create_subscription(TransformStamped, '/okvis/okvis_transform',
                             lambda m: samples['optimized'].append([stamp(m)]), qos)
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            rclpy.spin_once(node, timeout_sec=.05)
        report = {'duration_s': time.monotonic() - start, 'images': images}
        for name, rows in samples.items():
            a = np.asarray(rows)
            report[name] = {'count': len(rows)}
            if len(rows) > 1:
                report[name].update(source_hz=(len(rows) - 1) / (a[-1, 0] - a[0, 0]),
                                    first=rows[0], last=rows[-1])
            if name == 'imu' and len(rows):
                report[name].update(mean=a[:, 1:].mean(axis=0).tolist(), std=a[:, 1:].std(axis=0).tolist())
            if name == 'body' and len(rows):
                report[name].update(span_m=np.ptp(a[:, 1:4], axis=0).tolist(),
                                    max_speed_mps=float(np.linalg.norm(a[:, 4:7], axis=1).max()),
                                    finite=bool(np.isfinite(a).all()))
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
