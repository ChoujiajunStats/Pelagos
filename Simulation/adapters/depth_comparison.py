#!/usr/bin/env python3
"""Compare LAS2 with Stonefish native depth at the same acquisition timestamp."""
import array
from collections import OrderedDict, deque
import copy
import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from depth_metrics import compare_depths
from depth_panel import comparison_panel, waiting_panel


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def depth_array(msg):
    if msg.encoding != '32FC1' or msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 4:
        raise ValueError('expected a 32FC1 depth image with valid dimensions and row stride')
    dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=memoryview(msg.data),
                      strides=(msg.step, 4)).astype(np.float32, copy=True)


def image_msg(values, header, encoding):
    values = np.ascontiguousarray(values)
    msg = Image()
    msg.header = copy.deepcopy(header)
    msg.header.frame_id = 'cam0_rect_optical'
    msg.height, msg.width = values.shape[:2]
    msg.encoding, msg.is_bigendian = encoding, 0
    msg.step = values.strides[0]
    msg.data = array.array('B', values.tobytes())
    return msg


class DepthComparison(Node):
    def __init__(self):
        super().__init__('pelagos_depth_comparison')
        self.minimum = float(self.declare_parameter('min_depth_m', 0.3).value)
        self.maximum = float(self.declare_parameter('max_depth_m', 10.0).value)
        self.flip = bool(self.declare_parameter('flip_gt_vertical', False).value)
        if not (0 < self.minimum < self.maximum and math.isfinite(self.maximum)):
            raise ValueError('invalid depth display range')
        self.gt, self.estimated = OrderedDict(), OrderedDict()
        self.gt_pub = self.create_publisher(Image, '/pelagos/depth/ground_truth', qos_profile_sensor_data)
        self.gt_color = self.create_publisher(Image, '/pelagos/depth/gt_color', qos_profile_sensor_data)
        self.est_color = self.create_publisher(Image, '/pelagos/depth/estimate_color', qos_profile_sensor_data)
        self.err_color = self.create_publisher(Image, '/pelagos/depth/error_color', qos_profile_sensor_data)
        self.metrics = self.create_publisher(String, '/pelagos/depth/metrics', 10)
        self.panel = self.create_publisher(Image, '/pelagos/depth/comparison', qos_profile_sensor_data)
        self.counts = dict(gt=0, estimate=0, pairs=0)
        self.pair_times = deque(maxlen=40)
        self.last_error = ''
        self.last_pair_wall = None
        self.last_panel = None
        self.create_timer(1.0, self.status_panel, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.create_subscription(Image, '/rov01/sim/raw/left_depth/image_depth',
                                 lambda m: self.receive(m, True), qos_profile_sensor_data)
        self.create_subscription(Image, '/stereo/depth',
                                 lambda m: self.receive(m, False), qos_profile_sensor_data)
        self.last_report = 0

    def receive(self, msg, is_gt):
        self.counts['gt' if is_gt else 'estimate'] += 1
        stamp = stamp_ns(msg.header)
        if stamp <= 0:
            return
        target = self.gt if is_gt else self.estimated
        target[stamp] = msg
        # LAS2 can finish several frames after capture; do not substitute a
        # neighboring depth image just to make a comparison appear.
        while len(target) > (120 if is_gt else 12):
            target.popitem(last=False)
        if stamp not in self.gt or stamp not in self.estimated:
            return
        gt_msg, est_msg = self.gt.pop(stamp), self.estimated.pop(stamp)
        try:
            gt, estimate = depth_array(gt_msg), depth_array(est_msg)
            if self.flip:
                gt = np.flipud(gt).copy()
            valid_gt, valid_est, valid, error, metrics = compare_depths(gt, estimate, self.minimum, self.maximum)
            row = dict(stamp_ns=stamp, **metrics,
                       gt='Stonefish DepthCamera optical Z', estimate='LAS2-S optical Z',
                       pairing='exact acquisition timestamp')
            self.metrics.publish(String(data=json.dumps(row, allow_nan=False)))
            self.gt_pub.publish(image_msg(gt, est_msg.header, '32FC1'))
            images = (self.color_array(gt, valid_gt, self.maximum),
                      self.color_array(estimate, valid_est, self.maximum),
                      self.color_array(error, valid, 2.0, minimum=0.0))
            for publisher, pixels in zip((self.gt_color, self.est_color, self.err_color), images):
                publisher.publish(image_msg(pixels, est_msg.header, 'rgb8'))
            self.counts['pairs'] += 1
            self.last_pair_wall = time.monotonic()
            self.pair_times.append(self.last_pair_wall)
            rate = ((len(self.pair_times)-1)/(self.pair_times[-1]-self.pair_times[0])
                    if len(self.pair_times) > 1 and self.pair_times[-1] > self.pair_times[0] else 0.)
            self.last_panel = image_msg(comparison_panel(images, row, stamp, rate), est_msg.header, 'rgb8')
            self.panel.publish(self.last_panel)
            self.last_error = ''
            if stamp - self.last_report >= 2_000_000_000:
                self.get_logger().info(f"depth MAE={row['mae_m']:.3f} m, RMSE={row['rmse_m']:.3f} m, valid={row['valid_fraction']:.1%}")
                self.last_report = stamp
        except (ValueError, TypeError, cv2.error) as exc:
            self.last_error = str(exc)
            self.get_logger().warn(f'depth comparison skipped: {exc}')

    def color_array(self, data, valid, maximum, minimum=None):
        low = self.minimum if minimum is None else minimum
        scaled = np.zeros(data.shape, dtype=np.uint8)
        scaled[valid] = np.clip((data[valid] - low) * (255 / (maximum - low)), 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        rgb[~valid] = 0
        return rgb

    def status_panel(self):
        age = None if self.last_pair_wall is None else time.monotonic() - self.last_pair_wall
        if age is not None and age < 2.0 and self.last_panel is not None:
            self.panel.publish(self.last_panel)
            return
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        self.panel.publish(image_msg(waiting_panel(self.counts, self.last_error, age), header, 'rgb8'))


def main():
    rclpy.init()
    node = DepthComparison()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
