#!/usr/bin/env python3
"""Expose native OKVIS evidence, raw poses and an observed ROS input graph."""
from collections import deque
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock,ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,PointCloud2
from std_msgs.msg import String


class VioEvidence(Node):
    def __init__(self):
        super().__init__('pelagos_vio_evidence')
        self.out=Path(self.declare_parameter('run_dir','').value)
        self.bridge=CvBridge();self.events={};self.error={};self.error_received=0
        self.landmarks=0;self.landmark_source_stamp=0;self.debug_source_stamp=None
        self.audit={};self.last_audit=0
        self.raw_file=(self.out/'vio-raw-body.jsonl').open('a',buffering=1)
        self.optimized_file=(self.out/'vio-optimized-body.jsonl').open('a',buffering=1)
        self.panel=self.create_publisher(Image,'/pelagos/vio/evidence',qos_profile_sensor_data)
        self.metrics=self.create_publisher(String,'/pelagos/vio/evidence_metrics',5)
        for index in range(2):
            self.create_subscription(Image,f'/okvis/cam{index}_matches',
                lambda m,key=f'cam{index}':self.on_image(key,m),qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'/okvis/okvis_points_matched',self.on_cloud,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/okvis/okvis_odometry',self.on_body,1)
        self.create_subscription(TransformStamped,'/okvis/okvis_transform',self.on_visual,1)
        self.create_subscription(String,'/pelagos/vio/pose_error',self.on_error,5)
        self.create_timer(.5,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def event(self,key):
        self.events.setdefault(key,deque(maxlen=500)).append(time.monotonic())

    def on_image(self,key,message):
        self.event(key)
        self.debug_source_stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec

    def on_cloud(self,message):
        self.event('landmarks');self.landmarks=message.width*message.height
        self.landmark_source_stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec

    def on_body(self,message):
        self.event('odometry')
        p,q,v=message.pose.pose.position,message.pose.pose.orientation,message.twist.twist
        record=dict(stamp_ns=message.header.stamp.sec*10**9+message.header.stamp.nanosec,
                    frame=message.header.frame_id,child=message.child_frame_id,
                    position=[p.x,p.y,p.z],quaternion_xyzw=[q.x,q.y,q.z,q.w],
                    body_velocity=[v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z])
        self.raw_file.write(json.dumps(record)+'\n')

    def on_visual(self,message):
        self.event('visual_update')
        p,q=message.transform.translation,message.transform.rotation
        self.optimized_file.write(json.dumps(dict(
            stamp_ns=message.header.stamp.sec*10**9+message.header.stamp.nanosec,
            frame=message.header.frame_id,child=message.child_frame_id,
            position=[p.x,p.y,p.z],quaternion_xyzw=[q.x,q.y,q.z,q.w]))+'\n')

    def on_error(self,message):
        self.error=json.loads(message.data);self.error_received=time.monotonic()

    def rate(self,key,now):
        events=self.events.get(key,[])
        recent=[value for value in events if now-value<3.]
        if len(recent)<2:return 0.
        return (len(recent)-1)/max(recent[-1]-recent[0],1e-6)

    def age(self,key,now):
        events=self.events.get(key,[])
        return now-events[-1] if events else None

    def audit_graph(self):
        topics=('/okvis/okvis_odometry','/odometry/estimated',
                '/okvis/cam0/image_raw','/okvis/cam1/image_raw','/okvis/imu0')
        publishers={topic:[dict(name=info.node_name,namespace=info.node_namespace)
                          for info in self.get_publishers_info_by_topic(topic)] for topic in topics}
        try:
            inputs=self.get_subscriber_names_and_types_by_node('okvis_node_subscriber','/okvis')
            bridge_inputs=self.get_subscriber_names_and_types_by_node('pelagos_stonefish_sensors','/')
            camera_inputs=self.get_subscriber_names_and_types_by_node('okvis_pose_adapter','/')
            expected={'/okvis/cam0/image_raw','/okvis/cam1/image_raw','/okvis/imu0'}
            names={name for name,_ in inputs}
            chain_names=names|{name for name,_ in bridge_inputs}|{name for name,_ in camera_inputs}
            unexpected=[name for name in chain_names if 'ground_truth' in name or 'odometry_ned' in name]
            native_publishers=publishers['/okvis/okvis_odometry']
            authority=(len(native_publishers)==1 and
                       native_publishers[0]==dict(name='okvis_node_subscriber',namespace='/okvis'))
            bridge_authority=all(publishers[topic]==[dict(name='pelagos_stonefish_sensors',namespace='/')]
                                 for topic in expected)
            camera_authority=publishers['/odometry/estimated']==[dict(name='okvis_pose_adapter',namespace='/')]
            checked=expected.issubset(names) and not unexpected and authority and bridge_authority and camera_authority
            detail=dict(native_inputs=inputs,sensor_bridge_inputs=bridge_inputs,
                        camera_adapter_inputs=camera_inputs,
                        unexpected_truth_inputs=unexpected,inputs_verified=checked)
        except Exception as error:
            detail=dict(inputs_verified=False,graph_error=str(error))
        self.audit=dict(publishers=publishers,**detail)
        (self.out/'vio-input-audit.json').write_text(json.dumps(self.audit,indent=2))

    def tick(self):
        now=time.monotonic()
        if now-self.last_audit>3:
            self.audit_graph();self.last_audit=now
        rates={key:self.rate(key,now) for key in ('cam0','cam1','odometry','visual_update','landmarks')}
        ages={key:self.age(key,now) for key in rates}
        error=self.error if now-self.error_received<1.5 else {}
        report=dict(native_rates_hz=rates,receive_ages_sec=ages,sparse_landmarks=self.landmarks,
                    debug_image_source_stamp_ns=self.debug_source_stamp,
                    landmark_source_stamp_ns=self.landmark_source_stamp,
                    inputs_verified=self.audit.get('inputs_verified',False),pose_error=error)
        self.metrics.publish(String(data=json.dumps(report)))
        temporary=self.out/'vio-status.json.tmp';temporary.write_text(json.dumps(report,indent=2))
        temporary.replace(self.out/'vio-status.json')
        panel=np.zeros((294,1100,3),np.uint8);panel[:]=(32,26,20)
        lines=[
            'OKVIS2 NATIVE EVIDENCE | estimator outputs; no separate feature detector',
            f'Match images: left {rates["cam0"]:.1f} Hz | right {rates["cam1"]:.1f} Hz | sparse landmarks {self.landmarks}',
            f'Body odometry {rates["odometry"]:.1f} Hz | visual optimization updates {rates["visual_update"]:.1f} Hz',
        ]
        if error.get('matched'):
            lines.append(f'Fixed-initial-SE3 error: {error["position_error_m"]:.4f} m | {error["rotation_error_deg"]:.3f} deg | RMSE {error["position_rmse_m"]:.4f} m')
        else:lines.append('GT comparison: waiting for a current timestamp-matched pose pair')
        lines.extend([
            'Native matches: green=initialized landmark, yellow=uninitialized, red=no landmark',
            'Landmark IDs, keyframe flag, tracking state and reprojection lines: native match images',
            'Observed input graph: '+('stereo + IMU chain verified; no GT subscriptions' if report['inputs_verified'] else 'awaiting graph verification; inspect vio-input-audit.json'),
            'Raw body + optimized body + camera/GT pose pairs saved; alignment fixed once at startup',
            'Debug rates use receive time; upstream debug image/cloud headers may have zero stamps',
        ])
        for index,line in enumerate(lines):
            color=(100,240,150) if index in (0,6) else (230,230,230)
            cv2.putText(panel,line,(12,25+index*30),cv2.FONT_HERSHEY_SIMPLEX,.53,color,1,cv2.LINE_AA)
        image=self.bridge.cv2_to_imgmsg(panel,encoding='bgr8')
        image.header.stamp=self.get_clock().now().to_msg();self.panel.publish(image)

    def close(self):
        self.raw_file.close();self.optimized_file.close()


def main():
    rclpy.init();node=VioEvidence()
    try:rclpy.spin(node)
    finally:node.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
