#!/usr/bin/env python3
"""Read-only field visualization observer. Stores metadata and sampled geometry."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import struct
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from diagnostic_msgs.msg import DiagnosticStatus
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, UInt64
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from probe_pipeline import TopicRecord, read_resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=60);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rclpy.init();node=rclpy.create_node('incremental_visualization_observer')
    records={};subs=[];voxels=[];clouds=[];paths=[];metrics=[];fusion=[];depth=[];poses=[];resources=[]
    sensor=QoSProfile(depth=30,reliability=ReliabilityPolicy.BEST_EFFORT)
    durable=QoSProfile(depth=2,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    def sub(kind,topic,callback,qos=sensor):
        records[topic]=TopicRecord();subs.append(node.create_subscription(kind,topic,callback,qos))
    def observe(topic,stamp=None):records[topic].observe(time.monotonic_ns(),stamp,node.get_clock().now().nanoseconds)
    def stamped(topic,m):observe(topic,m.header.stamp.sec*10**9+m.header.stamp.nanosec)
    def markers(m):
        additions=[v for v in m.markers if v.action==Marker.ADD]
        stamp=m.markers[0].header.stamp if m.markers else None
        observe('/mapping/local_voxels',stamp.sec*10**9+stamp.nanosec if stamp else None)
        points=[p for v in additions for p in v.points]
        colors={(round(c.r,3),round(c.g,3),round(c.b,3)) for v in additions for c in v.colors}
        voxels.append(dict(t=time.monotonic(),count=len(points),batches=len(additions),colors=len(colors),
            frames=sorted({v.header.frame_id for v in m.markers}),types=sorted({int(v.type) for v in additions}),
            clears_first=bool(m.markers and m.markers[0].action==Marker.DELETEALL),
            sizes=sorted({round(v.scale.x,5) for v in additions}),
            z_range=[min(p.z for p in points),max(p.z for p in points)] if points else None,
            points_sample=[[p.x,p.y,p.z] for p in points[::max(1,len(points)//32)]][:32]))
    sub(MarkerArray,'/mapping/local_voxels',markers,durable)
    def cloud(m):
        stamped('/mapping/explored_cloud',m);sample=[]
        if m.point_step==16:
            for i in range(0,m.width*m.height,max(1,m.width*m.height//32)):
                sample.append(list(struct.unpack_from('<fffI',m.data,i*16)))
        clouds.append(dict(t=time.monotonic(),points=m.width*m.height,frame=m.header.frame_id,samples=sample[:32]))
    sub(PointCloud2,'/mapping/explored_cloud',cloud,durable)
    def path(m):
        stamped('/vio/trajectory',m)
        paths.append(dict(t=time.monotonic(),points=len(m.poses),frame=m.header.frame_id,
            first=[m.poses[0].pose.position.x,m.poses[0].pose.position.y,m.poses[0].pose.position.z] if m.poses else None,
            last=[m.poses[-1].pose.position.x,m.poses[-1].pose.position.y,m.poses[-1].pose.position.z] if m.poses else None))
    sub(RosPath,'/vio/trajectory',path,durable)
    def diagnostic(m):
        observe('/mapping/visualization_metrics');metrics.append(dict(t=time.monotonic(),message=m.message,values={v.key:v.value for v in m.values}))
    sub(DiagnosticStatus,'/mapping/visualization_metrics',diagnostic,durable)
    def body(m):
        stamped('/okvis/okvis_odometry',m);p=m.pose.pose.position
        poses.append(dict(t=time.monotonic(),position=[p.x,p.y,p.z]))
    sub(Odometry,'/okvis/okvis_odometry',body)
    sub(TransformStamped,'/okvis/okvis_transform',lambda m:stamped('/okvis/okvis_transform',m))
    sub(UInt64,'/mapping/integrated_frames',lambda m:(observe('/mapping/integrated_frames'),fusion.append([time.monotonic(),m.data])),durable)
    sub(String,'/stereo/statistics',lambda m:(observe('/stereo/statistics'),depth.append([time.monotonic(),json.loads(m.data)])))
    warmup=time.monotonic()+3
    while time.monotonic()<warmup:rclpy.spin_once(node,timeout_sec=.05)
    for t in records:records[t]=TopicRecord()
    for a in [voxels,clouds,paths,metrics,fusion,depth,poses]:a.clear()
    begin=time.monotonic();started=datetime.datetime.now().astimezone().isoformat()
    print('Live incremental visualization probe '+started,flush=True)
    while time.monotonic()-begin<args.seconds:
        rclpy.spin_once(node,timeout_sec=.03);elapsed=time.monotonic()-begin
        if not resources or elapsed-resources[-1]['elapsed_s']>=10:
            resources.append(read_resources(elapsed));print(f'{elapsed:.0f}s local={voxels[-1]["count"] if voxels else 0} blue={clouds[-1]["points"] if clouds else 0} path={paths[-1]["points"] if paths else 0}',flush=True)
    duration=time.monotonic()-begin
    checks=dict(local_cubes_observed=any(v['count'] for v in voxels),blue_cloud_observed=any(c['points'] for c in clouds),
        only_cube_list=bool(voxels) and all(v['types'] in [[],[Marker.CUBE_LIST]] for v in voxels),
        old_ids_cleared=bool(voxels) and all(v['clears_first'] for v in voxels),
        frames_odom=bool(voxels and clouds and paths) and all(v['frames']==['odom'] for v in voxels) and all(c['frame']=='odom' for c in clouds) and all(p['frame']=='odom' for p in paths),
        height_rainbow_observed=any(v['colors']>=3 for v in voxels),
        live_fusion=len(fusion)>1 and fusion[-1][1]>fusion[0][1],
        renderer_empty_start=any(m['values'].get('empty_start')=='true' for m in metrics),
        crop_policy_active_only=bool(metrics) and all(m['values'].get('history_policy','active_native_map_only').startswith('active_native_map_only') for m in metrics))
    span=max((math.dist(a['position'],b['position']) for a in poses[::max(1,len(poses)//100)] for b in poses[::max(1,len(poses)//100)]),default=0)
    timing={}
    for key in ['native_read_hold_ms','capture_ms','build_ms','total_ms']:
        values=[float(m['values'][key]) for m in metrics if key in m['values']]
        if values:timing[key]=dict(median=statistics.median(values),max=max(values))
    result=dict(started_at=started,duration_s=duration,checks=checks,passed=all(checks.values()),
        estimated_pose_span_m=span,trajectory_grew=bool(paths) and paths[-1]['points']>paths[0]['points'],
        topics={t:r.result(duration) for t,r in records.items()},timing=timing,voxels=voxels,clouds=clouds,paths=paths,
        metrics=metrics,fusion=fusion,depth=depth,poses=poses,resources=resources)
    if len(fusion)>1:result['fusion_hz']=(fusion[-1][1]-fusion[0][1])/(fusion[-1][0]-fusion[0][0])
    if len(depth)>1:result['depth_producer_hz']=(depth[-1][1]['frames']-depth[0][1]['frames'])/(depth[-1][0]-depth[0][0])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ['passed','checks','estimated_pose_span_m','trajectory_grew','fusion_hz','depth_producer_hz','timing']},ensure_ascii=False,indent=2),flush=True)
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
