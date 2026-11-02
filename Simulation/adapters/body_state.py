#!/usr/bin/env python3
"""Use native OKVIS body odometry, never the camera-pose visualization adapter."""
import copy
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from native_vio_loss import NativeVioLoss


def ns(stamp): return stamp.sec*10**9 + stamp.nanosec


class BodyState(Node):
    def __init__(self):
        super().__init__('pelagos_okvis_body_state')
        self.out = Path(self.declare_parameter('run_dir','').value)
        self.state = None
        self.stamp = 0
        self.received = 0
        self.epoch = 1
        self.fault = None
        self.native_loss=NativeVioLoss(self.out/'logs/okvis2.log')
        self.ever_tracking=False;self.native_loss_evidence=None
        self.image_stamp = self.imu_stamp = self.visual_stamp = 0
        self.visual_wall = 0
        self.pub = self.create_publisher(Odometry,'/rov01/state/odometry',1)
        self.status = self.create_publisher(String,'/rov01/localization/status',1)
        self.create_subscription(Odometry,'/okvis/okvis_odometry',self.on_state,1)
        self.create_subscription(TransformStamped,'/okvis/okvis_transform',self.on_visual,1)
        self.create_subscription(Image,'/rov01/sensors/stereo/left/image_raw',
                                 lambda m: setattr(self,'image_stamp',ns(m.header.stamp)),qos_profile_sensor_data)
        self.create_subscription(Imu,'/rov01/sensors/imu',
                                 lambda m: setattr(self,'imu_stamp',ns(m.header.stamp)),qos_profile_sensor_data)
        self.static = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.frame_id, transform.child_frame_id = 'world','rov01/odom'
        transform.transform.rotation.w = 1.
        self.static.sendTransform(transform)
        self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_visual(self,m):
        stamp = ns(m.header.stamp)
        if stamp > self.visual_stamp:
            self.visual_stamp = stamp
            self.visual_wall = time.monotonic_ns()

    def on_state(self,m):
        if m.header.frame_id!='world' or m.child_frame_id!='body': return
        stamp=ns(m.header.stamp)
        if stamp < self.stamp:
            self.epoch += 1
            self.fault = 'estimator_time_rewind_restart_required'
        if stamp <= self.stamp or self.fault: return
        p,q,v=m.pose.pose.position,m.pose.pose.orientation,m.twist.twist
        values=[p.x,p.y,p.z,q.x,q.y,q.z,q.w,v.linear.x,v.linear.y,v.linear.z,
                v.angular.x,v.angular.y,v.angular.z]
        if not all(math.isfinite(x) for x in values) or abs(sum(x*x for x in values[3:7])-1.)>1e-4:
            self.fault='invalid_okvis_body_state'; return
        self.state=copy.deepcopy(m)
        self.state.header.frame_id,self.state.child_frame_id='rov01/odom','rov01/base_link'
        self.stamp,self.received=stamp,time.monotonic_ns()
        self.pub.publish(self.state)

    def tick(self):
        now=time.monotonic_ns()
        lost=self.native_loss.poll()
        if lost:
            self.fault='okvis_native_tracking_lost_restart_required';self.native_loss_evidence=lost
        clock=self.get_clock().now().nanoseconds
        healthy=(self.state is not None and not self.fault and now-self.received<250_000_000 and
                 now-self.visual_wall<500_000_000 and -.05e9<=clock-self.visual_stamp<.5e9 and
                 -.05e9<=clock-self.stamp<.25e9)
        self.ever_tracking=self.ever_tracking or healthy
        report=dict(run=self.out.name,source='OKVIS2_STEREO_IMU',epoch=self.epoch,
                    state='LOST' if self.native_loss_evidence else 'TRACKING' if healthy else 'WAITING',
                    reason=self.fault or ('' if healthy else 'awaiting_fresh_visual_and_body_updates'),
                    image_stamp=self.image_stamp*1e-9,imu_stamp=self.imu_stamp*1e-9,
                    visual_stamp=self.visual_stamp*1e-9,body_stamp=self.stamp*1e-9,
                    covariance_available=False,
                    covariance_note='Upstream OKVIS ROS2 publisher does not export estimator covariance')
        report['native_loss_evidence']=self.native_loss_evidence
        self.status.publish(String(data=json.dumps(report)))


def main():
    rclpy.init(); node=BodyState()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__=='__main__': main()
