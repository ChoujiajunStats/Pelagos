#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""ROS 2 Humble sensor/reference gateway and RViz proposals. No actuator outputs."""
import argparse
from collections import deque
import json
import math
import queue
from pathlib import Path
import socket
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from nav_msgs.msg import Odometry,Path as RosPath
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped,TransformStamped
from visualization_msgs.msg import Marker,MarkerArray

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'contracts'))
from wire import receive_packet,send_packet


def ns(stamp):return stamp.sec*10**9+stamp.nanosec


class Gateway(Node):
    def __init__(self,port):
        super().__init__('jepa_control_gateway')
        self.body=None;self.imu=deque(maxlen=512);self.reference=None
        self.samples=deque(maxlen=64);self.sample_snapshot=[]
        self.visual_ns=0;self.depth_ns=0;self.packet=None
        self.last_output_ns=0
        self.incoming=queue.Queue(maxsize=2);self.stop=threading.Event()
        sensor=QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT)
        durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry,'/okvis/okvis_odometry',self.on_body,sensor)
        self.create_subscription(Imu,'/zed/zed_node/imu/data_raw',self.on_imu,sensor)
        self.create_subscription(String,'/planning/execution_reference',self.on_reference,durable)
        self.create_subscription(TransformStamped,'/okvis/okvis_transform',self.on_visual,sensor)
        self.create_subscription(Marker,'/okvis/okvis_path',self.on_visual,sensor)
        self.create_subscription(String,'/stereo/statistics',self.on_depth,sensor)
        self.status=self.create_publisher(String,'/control/status_json',durable)
        self.proposal=self.create_publisher(String,'/control/proposal',durable)
        self.predicted=self.create_publisher(RosPath,'/control/predicted_path',durable)
        self.reference_pub=self.create_publisher(RosPath,'/control/reference_path',durable)
        self.markers=self.create_publisher(MarkerArray,'/control/markers',durable)
        self.create_timer(.05,self.tick)
        self.listener=socket.socket();self.listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.listener.bind(('127.0.0.1',port));self.listener.listen(1);self.listener.settimeout(.5)
        self.thread=threading.Thread(target=self.serve,daemon=True);self.thread.start()

    def on_body(self,m):
        p=m.pose.pose.position;q=m.pose.pose.orientation;v=m.twist.twist.linear;w=m.twist.twist.angular
        values=[p.x,p.y,p.z,q.x,q.y,q.z,q.w,v.x,v.y,v.z,w.x,w.y,w.z]
        if m.header.frame_id!='world' or m.child_frame_id!='body' or not all(map(math.isfinite,values)):
            self.body=None;return
        self.body=dict(stamp_ns=ns(m.header.stamp),position=values[:3],quaternion=values[3:7],
                       velocity=values[7:10],omega=values[10:],frame='world',child='body')
        if self.imu:
            imu=min(self.imu,key=lambda x:abs(x['stamp_ns']-self.body['stamp_ns']))
            self.samples.append(dict(body=self.body,imu=imu))

    def on_imu(self,m):
        a=m.linear_acceleration;g=m.angular_velocity
        values=[a.x,a.y,a.z,g.x,g.y,g.z]
        if all(map(math.isfinite,values)):
            self.imu.append(dict(stamp_ns=ns(m.header.stamp),acceleration=values[:3],gyro=values[3:],frame=m.header.frame_id))

    def on_reference(self,m):
        try:self.reference=json.loads(m.data)
        except ValueError:self.reference=None

    def on_visual(self,m):self.visual_ns=ns(m.header.stamp)

    def on_depth(self,m):
        try:self.depth_ns=int(json.loads(m.data)['stamp_ns'])
        except (ValueError,KeyError):pass

    def path(self,points):
        m=RosPath();m.header.frame_id='world';m.header.stamp=self.get_clock().now().to_msg()
        for p in points:
            if len(p)!=3 or not all(map(math.isfinite,p)):continue
            pose=PoseStamped();pose.header=m.header
            pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=map(float,p);pose.pose.orientation.w=1.
            m.poses.append(pose)
        return m

    def tick(self):
        if self.body and self.imu:
            imu=min(self.imu,key=lambda x:abs(x['stamp_ns']-self.body['stamp_ns']))
            self.packet=dict(type='state',now_ns=self.get_clock().now().nanoseconds,
                body=self.body,imu=imu,visual_stamp_ns=self.visual_ns,depth_stamp_ns=self.depth_ns,
                reference=self.reference)
            self.sample_snapshot=list(self.samples)
        try:m=self.incoming.get_nowait()
        except queue.Empty:return
        if m.get('stamp_ns',0)==self.last_output_ns:return
        self.last_output_ns=m.get('stamp_ns',0)
        self.status.publish(String(data=json.dumps(m)))
        self.proposal.publish(String(data=json.dumps(m.get('proposal',{}))))
        self.predicted.publish(self.path(m.get('predicted_world',[])))
        self.reference_pub.publish(self.path(m.get('reference_world',[])))
        if self.body:
            marker=Marker();marker.header.frame_id='world';marker.header.stamp=self.get_clock().now().to_msg()
            marker.ns='controller_status';marker.id=0;marker.type=Marker.TEXT_VIEW_FACING
            marker.pose.position.x,marker.pose.position.y,marker.pose.position.z=self.body['position']
            marker.pose.position.z+=.65;marker.pose.orientation.w=1.;marker.scale.z=.11
            marker.color.r=1.;marker.color.g=.7;marker.color.b=.2;marker.color.a=1.
            p=m.get('proposal',{});detail=m.get('detail') or {}
            marker.text='JEPA-MPPI: '+m.get('state','waiting')
            if detail.get('solve_ms') is not None:marker.text+=' | %.0f ms'%detail['solve_ms']
            if p.get('blockers'):marker.text+='\n'+', '.join(p['blockers'][:2])
            self.markers.publish(MarkerArray(markers=[marker]))

    def serve(self):
        while not self.stop.is_set():
            try:c,_=self.listener.accept()
            except socket.timeout:continue
            except OSError:return
            try:
                with c:
                    c.settimeout(2)
                    echo={}
                    last_body=0
                    while not self.stop.is_set():
                        if self.packet is None:self.stop.wait(.05);continue
                        samples=[s for s in self.sample_snapshot if s['body']['stamp_ns']>last_body][:50]
                        send_packet(c,dict(self.packet,samples=samples,robot_send_ns=time.time_ns(),**echo))
                        if samples:last_body=samples[-1]['body']['stamp_ns']
                        m,payload=receive_packet(c,self.stop.is_set)
                        echo=dict(echo_host_send_ns=m.get('host_send_ns',0),robot_receive_ns=time.time_ns())
                        if payload or m.get('type')!='control_status':raise ValueError('Invalid controller reply')
                        try:self.incoming.put_nowait(m)
                        except queue.Full:pass
                        self.stop.wait(.05)
            except (OSError,EOFError,ValueError) as e:self.get_logger().info(str(e))

    def close(self):self.stop.set();self.listener.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18781);args=p.parse_args()
    rclpy.init();node=Gateway(args.port)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
