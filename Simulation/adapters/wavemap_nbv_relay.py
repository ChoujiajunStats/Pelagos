#!/usr/bin/env python3
"""Transport Wavemap's full OctoMap projection into NBV; no sensor integration."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import math
from pathlib import Path
import threading
import time
import zlib

import rclpy
from rclpy.clock import Clock,ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from rclpy.time import Time
from octomap_msgs.msg import Octomap
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import Buffer,TransformListener
from nbv_protocol import exchange


class MapRelay(Node):
    def __init__(self):
        super().__init__('pelagos_wavemap_nbv_relay')
        self.out=Path(self.declare_parameter('run_dir','').value)
        self.map_topic=self.declare_parameter('octomap_topic','/mapping/octomap').value
        self.state=None;self.pending=None;self.future=None;self.job=None
        self.pool=ThreadPoolExecutor(max_workers=1);self.stop=threading.Event()
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.report=dict(run=self.out.name,map_source='wavemap',depth_fusion_enabled=False,
            ready=False,reason='waiting_for_wavemap_projection',source_revision=0,map_revision=0,imports=0)
        self.status=self.create_publisher(String,'/pelagos/nbv/map_status',1)
        self.create_subscription(Odometry,'/rov01/state/odometry',self.on_state,1)
        self.create_subscription(Octomap,self.map_topic,self.on_map,
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.last_status=0.
        self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_state(self,message):
        if message.header.frame_id=='rov01/odom' and message.child_frame_id=='rov01/base_link':self.state=message

    def on_map(self,message):
        self.pending=message # Full snapshots: keep newest, never accumulate a queue.

    def request(self,message):
        if message.binary or message.id!='OcTree':raise ValueError('full_OcTree_projection_required')
        if self.count_publishers(self.map_topic)!=1:raise ValueError('ambiguous_wavemap_projection_publisher')
        if message.header.frame_id!='rov01/odom':
            transform=self.tf.lookup_transform('rov01/odom',message.header.frame_id,Time())
            t,q=transform.transform.translation,transform.transform.rotation
            if max(abs(t.x),abs(t.y),abs(t.z),abs(q.x),abs(q.y),abs(q.z),abs(abs(q.w)-1.))>1.e-6:
                raise ValueError('map_frame_requires_resampling_nonidentity_transform')
        revision=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        if revision<=0:raise ValueError('projection_has_no_acquisition_stamp')
        if revision<=self.report['source_revision']:return None
        raw=message.data.tobytes() if hasattr(message.data,'tobytes') else bytes(v%256 for v in message.data)
        if 4*((len(raw)+2)//3)>36*1024*1024-4096:raise ValueError('projection_exceeds_existing_socket_capacity')
        p,q=self.state.pose.pose.position,self.state.pose.pose.orientation
        body=[p.x,p.y,p.z,q.x,q.y,q.z,q.w]
        if not all(math.isfinite(v) for v in body):raise ValueError('invalid_body_pose')
        return dict(run=self.out.name,epoch=1,sequence=time.monotonic_ns(),stamp_ns=revision,
            body=body,operation='replace_map',map_frame='rov01/odom',map_binary=False,
            map_id=message.id,map_resolution=message.resolution,source_revision=revision,
            map_bytes=len(raw),map_crc32=zlib.crc32(raw),map_data_b64=base64.b64encode(raw).decode('ascii'))

    def tick(self):
        import_progressed=False
        if self.future is not None and self.future.done():
            try:
                answer=self.future.result()
                if not answer.get('valid'):raise ValueError(answer.get('reason','map_import_rejected'))
                if answer['depth_fusion_enabled']:raise ValueError('native_depth_fusion_was_not_disabled')
                import_progressed=answer['source_revision']>self.report['source_revision']
                self.report.update(ready=True,reason=answer['reason'],map_revision=answer['map_revision'],
                    source_revision=answer['source_revision'],depth_fusion_enabled=answer['depth_fusion_enabled'],
                    imports=self.report['imports']+1,bytes=self.job['map_bytes'],
                    import_seconds=time.monotonic()-self.started)
            except Exception as error:self.report.update(ready=False,reason=str(error))
            self.future=None;self.job=None
        if self.future is None and self.pending is not None and self.state is not None and (self.out/'nbv.sock').exists():
            try:
                request=self.request(self.pending)
                if request is not None:
                    self.job=request;self.started=time.monotonic()
                    self.future=self.pool.submit(exchange,self.out/'nbv.sock',request,self.stop)
                self.pending=None
            except Exception as error:
                # Keep the latest snapshot for a TF/startup retry.
                self.report.update(ready=False,reason=str(error))
        now=time.monotonic()
        # Announce a genuinely installed snapshot without waiting for the
        # periodic heartbeat. A repeated heartbeat never advances source time.
        if import_progressed or now-self.last_status>=.2:
            self.report['wall_ns']=time.monotonic_ns();self.last_status=now
            message=json.dumps(self.report);self.status.publish(String(data=message))
            temporary=self.out/'map-source-status.json.tmp';temporary.write_text(message)
            temporary.replace(self.out/'map-source-status.json')

    def close(self):
        self.stop.set();self.pool.shutdown(wait=True,cancel_futures=True)


def main():
    rclpy.init();node=MapRelay()
    try:rclpy.spin(node)
    finally:node.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
