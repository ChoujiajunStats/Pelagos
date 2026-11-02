#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Publish a three-dimensional planning goal; no motion-control interface."""
import argparse
import json
import math
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('x', type=float);parser.add_argument('y', type=float);parser.add_argument('z', type=float)
    args = parser.parse_args()
    if not all(math.isfinite(v) and abs(v) < 100000 for v in (args.x, args.y, args.z)):
        parser.error('目标必须是有限的 odom 坐标，单位米')
    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init();node = rclpy.create_node('navigation_goal_sender')
    publisher = node.create_publisher(PoseStamped, '/planning/goal', 10)
    began = time.monotonic()
    while publisher.get_subscription_count() == 0 and time.monotonic()-began < 5:
        rclpy.spin_once(node, timeout_sec=.1)
    if not publisher.get_subscription_count():
        raise SystemExit('没有规划节点订阅 /planning/goal；请先启动原生导航。')
    message = PoseStamped();message.header.frame_id = 'odom';message.header.stamp = node.get_clock().now().to_msg()
    message.pose.position.x = args.x;message.pose.position.y = args.y;message.pose.position.z = args.z
    message.pose.orientation.w = 1.;publisher.publish(message)
    deadline = time.monotonic()+.3
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    print(json.dumps({'topic':'/planning/goal','frame':'odom','position_m':[args.x,args.y,args.z]}, ensure_ascii=False))
    node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':
    main()
