#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Measure the real ROS 2 preview. --goal only requests planning, never execution."""
import argparse
import collections
import json
import time
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, UInt64


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=25)
    parser.add_argument('--goal',type=float,nargs=3)
    args=parser.parse_args()
    rclpy.init();node=rclpy.create_node('omniplanner_preview_probe')
    counts=collections.Counter(); latest={}; statuses=[]
    sensor=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    def body(m):
        counts['body']+=1
        p=m.pose.pose.position
        latest['body']=[p.x,p.y,p.z]
        latest['body_age_s']=(node.get_clock().now().nanoseconds-m.header.stamp.sec*10**9-m.header.stamp.nanosec)*1e-9
    def frame(m):counts['fused']+=1;latest['integrated_frames']=m.data
    def stats(name,m):
        counts[name]+=1
        try:latest[name]=json.loads(m.data)
        except ValueError:latest[name]=m.data
        if name=='status':statuses.append(latest[name])
    def path(m):
        counts['path']+=1
        latest['path_points']=len(m.poses)
        if m.poses:
            latest['path']=[[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in m.poses]
    subscriptions=[
        node.create_subscription(Odometry,'/okvis/okvis_odometry',body,sensor),
        node.create_subscription(UInt64,'/mapping/integrated_frames',frame,durable),
        node.create_subscription(String,'/stereo/statistics',lambda m:stats('depth',m),sensor),
        node.create_subscription(String,'/omniplanner/status',lambda m:stats('status',m),durable),
        node.create_subscription(String,'/planning/map_stats',lambda m:stats('map_stats',m),durable),
        node.create_subscription(Path,'/planning/path',path,durable)]
    pub=node.create_publisher(PoseStamped,'/planning/goal',5)
    started=time.monotonic();sent=False
    try:
        while time.monotonic()-started<args.seconds:
            rclpy.spin_once(node,timeout_sec=.1)
            if args.goal and not sent and time.monotonic()-started>3 and 'body' in latest:
                goal=PoseStamped();goal.header.frame_id='world';goal.header.stamp=node.get_clock().now().to_msg()
                goal.pose.position.x,goal.pose.position.y,goal.pose.position.z=args.goal
                goal.pose.orientation.w=1.;pub.publish(goal);sent=True
        elapsed=time.monotonic()-started
        print(json.dumps(dict(seconds=elapsed,goal_sent=sent,requested_goal=args.goal,
            counts=dict(counts),rates_hz={k:v/elapsed for k,v in counts.items()},
            latest=latest,status_transitions=statuses),ensure_ascii=False,indent=2))
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
