#!/usr/bin/env python3
"""Read-only pose stream for a bounded operator-authorized motion check."""
import json
import math
import rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile,ReliabilityPolicy


def main():
    rclpy.init();node=rclpy.create_node('motion_check_pose_monitor');state={};visual=[0]
    qos=QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT)
    def body(m):
        stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec;p=m.pose.pose.position;q=m.pose.pose.orientation;v=m.twist.twist.linear
        state.update(stamp_ns=stamp,position=[p.x,p.y,p.z],quaternion=[q.x,q.y,q.z,q.w],velocity=[v.x,v.y,v.z])
    def optimized(m):visual[0]=m.header.stamp.sec*10**9+m.header.stamp.nanosec
    node.create_subscription(Odometry,'/okvis/okvis_odometry',body,qos)
    node.create_subscription(TransformStamped,'/okvis/okvis_transform',optimized,qos)
    def emit():
        now=node.get_clock().now().nanoseconds;s=dict(state)
        if s:
            s['pose_age_s']=(now-s['stamp_ns'])*1e-9;s['visual_age_s']=(now-visual[0])*1e-9 if visual[0] else 1e9
            s['finite']=all(math.isfinite(v) for k in ['position','quaternion','velocity'] for v in s[k])
        print(json.dumps(s),flush=True)
    node.create_timer(.1,emit)
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
