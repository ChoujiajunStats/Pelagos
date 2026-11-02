#!/usr/bin/env python3
"""Read-only field probe; track mesh deltas, VIO quality, producer rates and resources."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from diagnostic_msgs.msg import DiagnosticStatus
from nav_msgs.msg import Odometry, Path as RosPath
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String, UInt64
from visualization_msgs.msg import MarkerArray, Marker
from probe_pipeline import TopicRecord, read_resources


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seconds',type=float,default=60)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stationary',action='store_true');args=ap.parse_args()
    rclpy.init();node=rclpy.create_node('read_only_surface_observer');records={};subs=[];metrics=[];poses=[];resources=[];packets=[];state={};fusion=[];depth=[];paths=[]
    sensor=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT);durable=QoSProfile(depth=2,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    def sub(kind,topic,cb,qos=sensor):
        records[topic]=TopicRecord();subs.append(node.create_subscription(kind,topic,cb,qos))
    def record(topic,stamp=None):records[topic].observe(time.monotonic_ns(),stamp,node.get_clock().now().nanoseconds)
    def mesh(a):
        record('/visualization/surface_mesh');estimated_bytes=0;adds=deletes=0
        for m in a.markers:
            estimated_bytes+=160+24*len(m.points)+16*len(m.colors)
            key=(m.ns,m.id)
            if m.action==Marker.DELETEALL:state.clear()
            elif m.action==Marker.DELETE:state.pop(key,None);deletes+=1
            else:
                adds+=1;state[key]=len(m.points)//3
                assert m.type==Marker.TRIANGLE_LIST and m.header.frame_id=='odom' and not m.colors
                assert len(m.points)%3==0 and all(math.isfinite(v) for p in m.points for v in (p.x,p.y,p.z))
        packets.append(dict(t=time.monotonic(),triangles=sum(state.values()),adds=adds,deletes=deletes,estimated_bytes=estimated_bytes))
    sub(MarkerArray,'/visualization/surface_mesh',mesh,durable)
    def body(m):
        s=m.header.stamp.sec*10**9+m.header.stamp.nanosec;record('/okvis/okvis_odometry',s);p=m.pose.pose.position;v=m.twist.twist.linear
        poses.append(dict(position=[p.x,p.y,p.z],speed=math.hypot(v.x,v.y,v.z),age=(node.get_clock().now().nanoseconds-s)*1e-9))
    sub(Odometry,'/okvis/okvis_odometry',body)
    sub(TransformStamped,'/okvis/okvis_transform',lambda m:record('/okvis/okvis_transform',m.header.stamp.sec*10**9+m.header.stamp.nanosec))
    sub(DiagnosticStatus,'/mapping/visualization_metrics',lambda m:(record('/mapping/visualization_metrics'),metrics.append(dict(message=m.message,values={x.key:x.value for x in m.values}))),durable)
    sub(RosPath,'/visualization/trajectory',lambda m:(record('/visualization/trajectory'),paths.append(len(m.poses))),durable)
    sub(UInt64,'/mapping/integrated_frames',lambda m:(record('/mapping/integrated_frames'),fusion.append([time.monotonic(),m.data])),durable)
    sub(String,'/stereo/statistics',lambda m:(record('/stereo/statistics'),depth.append([time.monotonic(),json.loads(m.data)])))
    logs={}
    for f in (Path(__file__).resolve().parents[4]/'runs').glob('*/manifest.json'):
        if '"running"' in f.read_text():
            for p in f.parent.glob('logs/okvis2.log'):logs[str(p)]=p.stat().st_size
    warmup=time.monotonic()+3
    while time.monotonic()<warmup:rclpy.spin_once(node,timeout_sec=.05)
    for t in records:records[t]=TopicRecord()
    for a in [metrics,poses,resources,packets,fusion,depth,paths]:a.clear()
    began=time.monotonic();started=datetime.datetime.now().astimezone().isoformat();print('Probe '+started,flush=True)
    while time.monotonic()-began<args.seconds:
        rclpy.spin_once(node,timeout_sec=.03);elapsed=time.monotonic()-began
        if not resources or elapsed-resources[-1]['elapsed_s']>=10:
            resources.append(read_resources(elapsed));print(f'{elapsed:.0f}s triangles={sum(state.values())} poses={len(poses)}',flush=True)
    dt=time.monotonic()-began
    span=math.sqrt(sum((max(s['position'][i] for s in poses)-min(s['position'][i] for s in poses))**2 for i in range(3))) if poses else 1e9
    log_counts={}
    for p,offset in logs.items():
        with open(p) as f:f.seek(offset);text=f.read()
        log_counts[p]={k:text.count(k) for k in ['TRACKING FAILURE','RANSAC FAIL','frame drop']}
    failures=sum(x['TRACKING FAILURE'] for x in log_counts.values())
    checks=dict(surface_observed=any(p['triangles']>0 for p in packets),trajectory_observed=bool(paths),live_fusion=len(fusion)>1 and fusion[-1][1]>fusion[0][1],
                no_tracking_failures=failures==0,finite_poses=bool(poses) and all(math.isfinite(x) for p in poses for x in p['position']),
                stationary_pose_within_15cm=(not args.stationary or span<.15),reasonable_speed=bool(poses) and max(p['speed'] for p in poses)<(.15 if args.stationary else .5))
    timing={}
    for k in ['native_read_hold_ms','build_ms','total_ms']:
        a=[float(m['values'][k]) for m in metrics if k in m['values']]
        if a:timing[k]=dict(median=statistics.median(a),p95=sorted(a)[int(.95*(len(a)-1))],max=max(a))
    report=dict(started=started,duration_s=dt,checks=checks,passed=all(checks.values()),estimated_pose_span_m=span,
        max_speed_mps=max((s['speed'] for s in poses),default=None),topics={k:v.result(dt) for k,v in records.items()},
        timing=timing,metrics=metrics,packets=packets,resources=resources,logs=log_counts,
        estimated_mesh_mbit_s=sum(p['estimated_bytes'] for p in packets)*8/dt/1e6)
    if len(fusion)>1:report['fusion_hz']=(fusion[-1][1]-fusion[0][1])/(fusion[-1][0]-fusion[0][0])
    if len(depth)>1:report['depth_producer_hz']=(depth[-1][1]['frames']-depth[0][1]['frames'])/(depth[-1][0]-depth[0][0])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','packets','resources','topics']},indent=2),flush=True)
    node.destroy_node()
    if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
