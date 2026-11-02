#!/usr/bin/env python3
"""Wait for three seconds of quiet, fresh IMU before a diagnostic VIO start."""
import collections
import math
import time
import rclpy
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy
rclpy.init();node=rclpy.create_node('vio_quiet_initialization_check');window=collections.deque()
def observe(m):
    a=m.linear_acceleration;g=m.angular_velocity
    age=(node.get_clock().now().nanoseconds-m.header.stamp.sec*10**9-m.header.stamp.nanosec)*1e-9
    if not -.1<age<.3 or math.hypot(g.x,g.y,g.z)>.035 or abs(math.hypot(a.x,a.y,a.z)-9.80665)>.25:
        window.clear();return
    now=time.monotonic();window.append((now,a.x,a.y,a.z))
    while window and now-window[0][0]>3.1:window.popleft()
node.create_subscription(Imu,'/zed/zed_node/imu/data_raw',observe,QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT))
start=time.monotonic();ok=False
try:
    while time.monotonic()-start<90:
        rclpy.spin_once(node,timeout_sec=.05)
        if len(window)>100 and window[-1][0]-window[0][0]>=3.:
            spread=max(max(s[i] for s in window)-min(s[i] for s in window) for i in (1,2,3))
            if spread<.3:ok=True;break
    print('IMU quiet for 3 seconds' if ok else 'No sustained quiet IMU within 90 seconds',flush=True)
finally:
    node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
raise SystemExit(0 if ok else 1)
