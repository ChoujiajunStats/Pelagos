#!/usr/bin/env python3
"""Observe live SLAM/navigation metadata; optional test goals only publish PoseStamped."""
import argparse
from collections import Counter
import datetime
import json
import math
from pathlib import Path
import statistics
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, UInt64
from tf2_msgs.msg import TFMessage
from probe_pipeline import TopicRecord, read_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exercise-goals', action='store_true')
    args = parser.parse_args()
    rclpy.init();node = Node('navigation_live_probe')
    began = time.monotonic();records = {};subscriptions = [];statuses = [];metrics = []
    versions = [];planning_versions = [];fusion = [];stereo = [];paths = [];clouds = [];tf = [];goals = [];resources = []
    sensor = QoSProfile(depth=100,reliability=ReliabilityPolicy.BEST_EFFORT)
    durable = QoSProfile(depth=20,durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def record(topic, stamp=None):
        records[topic].observe(time.monotonic_ns(),stamp,node.get_clock().now().nanoseconds)

    def sub(kind, topic, callback, qos=sensor):
        records[topic] = TopicRecord()
        subscriptions.append(node.create_subscription(kind,topic,callback,qos))

    def stamped(topic, message):
        s = message.header.stamp;record(topic,s.sec*10**9+s.nanosec)

    for topic in ('/okvis/okvis_odometry','/odometry/estimated'):
        sub(Odometry,topic,lambda m,t=topic:stamped(t,m))
    sub(TransformStamped,'/okvis/okvis_transform',lambda m:stamped('/okvis/okvis_transform',m))

    def counts(topic, values, message):
        record(topic);values.append([time.monotonic(),int(message.data)])

    for topic,values in [('/mapping/map_version',versions),('/mapping/planning_map_version',planning_versions),('/mapping/integrated_frames',fusion)]:
        sub(UInt64,topic,lambda m,t=topic,v=values:counts(t,v,m),durable)

    def diagnostic(topic, values, message):
        record(topic)
        level = int.from_bytes(message.level,'little') if isinstance(message.level,bytes) else int(message.level)
        values.append(dict(t=time.monotonic(),message=message.message,level=level,values={v.key:v.value for v in message.values}))

    sub(DiagnosticStatus,'/planning/status',lambda m:diagnostic('/planning/status',statuses,m),durable)
    sub(DiagnosticStatus,'/mapping/metrics',lambda m:diagnostic('/mapping/metrics',metrics,m),durable)

    def path(message):
        stamped('/planning/path',message)
        points = [[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in message.poses]
        paths.append(dict(t=time.monotonic(),frame=message.header.frame_id,points=points))

    sub(RosPath,'/planning/path',path,durable)

    def cloud(message):
        stamped('/mapping/occupied',message)
        clouds.append(dict(t=time.monotonic(),frame=message.header.frame_id,points=int(message.width)*int(message.height)))

    sub(PointCloud2,'/mapping/occupied',cloud,durable)

    def transforms(message):
        for transform in message.transforms:
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_link':
                s=transform.header.stamp;stamp=s.sec*10**9+s.nanosec
                record('/tf:odom->base_link',stamp)
                p=transform.transform.translation
                tf.append(dict(t=time.monotonic(),stamp_ns=stamp,position=[p.x,p.y,p.z]))

    records['/tf:odom->base_link']=TopicRecord()
    subscriptions.append(node.create_subscription(TFMessage,'/tf',transforms,sensor))

    def stereo_stat(message):
        data=json.loads(message.data);record('/stereo/statistics',data.get('stamp_ns'))
        stereo.append(dict(t=time.monotonic(),data=data))

    sub(String,'/stereo/statistics',stereo_stat)
    goal_pub = node.create_publisher(PoseStamped,'/planning/goal',10) if args.exercise_goals else None
    warmup=time.monotonic()+3
    while time.monotonic()<warmup:rclpy.spin_once(node,timeout_sec=.05)
    for topic in records:records[topic]=TopicRecord()
    for values in (statuses,metrics,versions,planning_versions,fusion,stereo,paths,clouds,tf):values.clear()
    began=time.monotonic();started = datetime.datetime.now().astimezone().isoformat()
    print('Live navigation observation started '+started,flush=True)
    while time.monotonic()-began < args.seconds:
        rclpy.spin_once(node,timeout_sec=.03)
        elapsed = time.monotonic()-began
        if not resources or elapsed-resources[-1]['elapsed_s'] >= 10:
            resources.append(read_resources(elapsed))
            print(f'{elapsed:.0f}s: fusion={fusion[-1][1] if fusion else 0}, map={versions[-1][1] if versions else 0}, planning={planning_versions[-1][1] if planning_versions else 0}',flush=True)
        if goal_pub and len(goals)<2 and elapsed >= (5 if not goals else args.seconds/2) and tf and planning_versions:
            last=tf[-1];age=(node.get_clock().now().nanoseconds-last['stamp_ns'])*1e-9
            if -.5 <= age <= 3:
                offsets=([.8,0,.3],[.4,.5,.2])[len(goals)]
                xyz=[p+d for p,d in zip(last['position'],offsets)]
                goal=PoseStamped();goal.header.frame_id='odom';goal.header.stamp=node.get_clock().now().to_msg()
                goal.pose.position.x,goal.pose.position.y,goal.pose.position.z=xyz;goal.pose.orientation.w=1.
                goal_pub.publish(goal);goals.append(dict(t=time.monotonic(),position=xyz))
                print('Published 3D planning test goal '+json.dumps(xyz),flush=True)
    duration = time.monotonic()-began
    def increasing(values):return bool(values) and all(a[1]<=b[1] for a,b in zip(values,values[1:]))
    checks=dict(native_fusion_grew=len(fusion)>1 and fusion[-1][1]>fusion[0][1],
        map_version_monotonic=increasing(versions),planning_version_monotonic=increasing(planning_versions),
        planning_map_generated=bool(planning_versions),odom_base_tf_available=bool(tf),
        frames_odom=bool(clouds) and all(c['frame']=='odom' for c in clouds) and all(p['frame']=='odom' for p in paths),
        paths_finite=all(math.isfinite(v) for p in paths for point in p['points'] for v in point))
    if args.exercise_goals:
        checks['two_3d_goals_sent']=len(goals)==2
        checks['two_goal_ids_processed']=len({s['values']['goal_id'] for s in statuses if 'goal_id' in s['values']})>=2
    result=dict(started_at=started,ended_at=datetime.datetime.now().astimezone().isoformat(),duration_s=duration,
        checks=checks,interfaces_passed=all(checks.values()),valid_field_path_observed=any(s['message']=='success' and s['values'].get('collision_free')=='true' for s in statuses),
        topics={t:r.result(duration) for t,r in records.items()},statuses=statuses,metrics=metrics,
        versions=versions,planning_versions=planning_versions,fusion=fusion,stereo=stereo,paths=paths,
        clouds=clouds,tf=tf,goals=goals,resources=resources,status_counts=dict(Counter(s['message'] for s in statuses)))
    if len(stereo)>1:result['depth_producer_hz']=(stereo[-1]['data']['frames']-stereo[0]['data']['frames'])/(stereo[-1]['t']-stereo[0]['t'])
    if len(fusion)>1:result['fusion_hz']=(fusion[-1][1]-fusion[0][1])/(fusion[-1][0]-fusion[0][0])
    result['metric_summary']={}
    for key in ('native_memory_bytes','maintenance_ms','capture_ms','map_read_hold_ms','roi_ms','sdf_ms','classification_ms'):
        values=[float(m['values'][key]) for m in metrics if key in m['values']]
        if values:result['metric_summary'][key]=dict(median=statistics.median(values),max=max(values))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ['checks','interfaces_passed','valid_field_path_observed','depth_producer_hz','fusion_hz','metric_summary','status_counts']},ensure_ascii=False,indent=2),flush=True)
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
