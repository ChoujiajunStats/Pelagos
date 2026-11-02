#!/usr/bin/env python3
"""Measure live SLAM component rates; keep only counters, timing and small statistics."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import struct
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import String, UInt64

from probe_pipeline import TopicRecord, read_resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--skip-images',action='store_true',
                        help='Only observe small messages and producer counters; do not subscribe to images.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rclpy.init();node=Node('dapeng_component_frequency_probe')
    records={};subscriptions=[];measuring=False;stereo=[];integrated=[];poses={};map_points=[]
    header_errors={};logs={}
    runs=Path(__file__).resolve().parents[4]/'runs'
    manifests=list(runs.glob('*/manifest.json'))
    current=max(manifests,key=lambda p:p.stat().st_mtime) if manifests else None
    if current:
        for path in current.parent.glob('logs/*.log'):
            logs[str(path)]={'offset':path.stat().st_size}

    def record(topic,stamp=None):
        if not measuring:return
        records[topic].observe(time.monotonic_ns(),stamp,node.get_clock().now().nanoseconds)

    def stamped(topic,message):
        s=message.header.stamp
        record(topic,s.sec*1000000000+s.nanosec)

    sensor=QoSProfile(depth=20,reliability=ReliabilityPolicy.BEST_EFFORT)
    durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def subscribe(topic,kind,callback,qos=sensor,raw=False):
        records[topic]=TopicRecord()
        subscriptions.append(node.create_subscription(kind,topic,callback,qos,raw=raw))

    def image(topic,data):
        if not measuring:return
        try:
            sec,nanosec=struct.unpack_from('<iI' if data[1]&1 else '>iI',data,4)
            if not 0<=nanosec<1000000000:raise ValueError('invalid CDR stamp')
            record(topic,sec*1000000000+nanosec)
        except (ValueError,IndexError,struct.error):header_errors[topic]=header_errors.get(topic,0)+1

    for side in (() if args.skip_images else ('left','right')):
        topic=f'/zed/zed_node/{side}/gray/raw/image'
        subscribe(topic,Image,lambda data,t=topic:image(t,data),
                  QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT),raw=True)
        topic=f'/zed/zed_node/{side}/color/raw/camera_info'
        subscribe(topic,CameraInfo,lambda message,t=topic:stamped(t,message))
    topic='/zed/zed_node/imu/data_raw'
    subscribe(topic,Imu,lambda message:stamped('/zed/zed_node/imu/data_raw',message),
              QoSProfile(depth=1000,reliability=ReliabilityPolicy.BEST_EFFORT))

    def pose(topic,message):
        if not measuring:return
        stamped(topic,message)
        p=message.pose.pose.position
        position=[p.x,p.y,p.z]
        data=poses.setdefault(topic,{'first':position,'last':position,'nonfinite':0,'max_norm_m':0})
        data['last']=position;data['nonfinite']+=not all(math.isfinite(x) for x in position)
        if all(math.isfinite(x) for x in position):data['max_norm_m']=max(data['max_norm_m'],math.hypot(*position))

    for topic in ('/okvis/okvis_odometry','/odometry/estimated'):
        subscribe(topic,Odometry,lambda message,t=topic:pose(t,message))
    topic='/okvis/okvis_transform'
    subscribe(topic,TransformStamped,lambda message:stamped('/okvis/okvis_transform',message))

    def stats(message):
        if not measuring:return
        data=json.loads(message.data);record('/stereo/statistics',data.get('stamp_ns'))
        stereo.append({'arrival_s':time.monotonic(),'data':data})

    subscribe('/stereo/statistics',String,stats)
    if not args.skip_images:
        subscribe('/stereo/depth',Image,lambda data:image('/stereo/depth',data),raw=True)

    def integration(message):
        if not measuring:return
        record('/mapping/integrated_frames');integrated.append([time.monotonic(),int(message.data)])

    def occupied(message):
        if not measuring:return
        stamped('/mapping/occupied',message);map_points.append(int(message.width)*int(message.height))

    subscribe('/mapping/integrated_frames',UInt64,integration,durable)
    subscribe('/mapping/occupied',PointCloud2,occupied,durable)
    warmup=time.monotonic()+3
    while time.monotonic()<warmup:rclpy.spin_once(node,timeout_sec=.1)
    publishers={name:[{'node':p.node_name,'reliability':str(p.qos_profile.reliability),
                      'durability':str(p.qos_profile.durability)}
                     for p in node.get_publishers_info_by_topic(name)] for name in records}
    start=time.monotonic();began=dt.datetime.now().astimezone().isoformat()
    for path,info in logs.items():info['offset']=Path(path).stat().st_size
    resources=[read_resources(0.)];measuring=True
    print('Measurement started: '+began,flush=True)
    while time.monotonic()-start<args.seconds:
        rclpy.spin_once(node,timeout_sec=.05)
        elapsed=time.monotonic()-start
        if elapsed-resources[-1]['elapsed_s']>=10:
            resources.append(read_resources(elapsed))
            print(f'{elapsed:.0f}s: pose={records["/okvis/okvis_odometry"].count}, visual={records["/okvis/okvis_transform"].count}, depth_statistics={records["/stereo/statistics"].count}, map={records["/mapping/occupied"].count}',flush=True)
    duration=time.monotonic()-start;measuring=False
    result={'began_at':began,'ended_at':dt.datetime.now().astimezone().isoformat(),'duration_s':duration,
            'skip_images':args.skip_images,
            'run_manifest':str(current) if current else None,'topics':{n:r.result(duration) for n,r in records.items()},
            'publishers':publishers,'poses':poses,'header_errors':header_errors,'stereo_samples':stereo,
            'integration_samples':integrated,'resources':resources,'logs':logs}
    for key,field in [('inference_s','inference_s'),('processing_s','processing_s')]:
        values=[s['data'][field] for s in stereo if field in s['data']]
        if values:result[key]={'median':statistics.median(values),'mean':statistics.mean(values),'max':max(values)}
    if len(stereo)>1:
        a,b=stereo[0],stereo[-1]
        result['depth_producer_hz']=(b['data']['frames']-a['data']['frames'])/(b['arrival_s']-a['arrival_s'])
    if len(integrated)>1:
        a,b=integrated[0],integrated[-1]
        result['mapping_fusion_hz']=(b[1]-a[1])/(b[0]-a[0])
        result['mapping_net_frames']=b[1]-a[1]
    if map_points:result['map_points']={'first':map_points[0],'last':map_points[-1],'min':min(map_points),'max':max(map_points)}
    for path,info in logs.items():
        with Path(path).open('rb') as stream:
            stream.seek(info['offset']);text=stream.read().decode(errors='replace')
        info.update(bytes_added=len(text.encode()),counts={term:text.count(term) for term in
                     ['TRACKING FAILURE','RANSAC FAIL','frame drop','Frame not added','no odometry within']},
                    last_lines=text.splitlines()[-12:])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    node.destroy_node();rclpy.shutdown()
    print(json.dumps({n:{'count':r['count'],'received_hz':r['received_rate_hz'],'source_hz':r['source_rate_hz']}
                      for n,r in result['topics'].items()},indent=2))
    print('Report: '+str(args.output),flush=True)


if __name__=='__main__':main()
