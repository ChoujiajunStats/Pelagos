#!/usr/bin/env python3
"""Small CPU-only Wavemap fusion/save check in a separate ROS domain."""
import argparse
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt64
from std_srvs.srv import Trigger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--domain',type=int,default=89)
    args=parser.parse_args()
    if not 1<=args.domain<=232 or args.domain==int(os.environ.get('ROS_DOMAIN_ID','0')):
        parser.error('Use a separate, nonzero ROS domain')
    os.environ.update(ROS_DOMAIN_ID=str(args.domain),ROS_LOCALHOST_ONLY='1')
    root=Path(__file__).resolve().parents[4]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    binary=root/'Deployment/infrastructure/ros2_ws/install/underwater_wavemap_ros/lib/underwater_wavemap_ros/wavemap_node'
    command=[str(binary),'--ros-args','-r','__node:=wavemap_cpu_audit',
             '--params-file',str(root/'Deployment/config/profiles/orin-nano/wavemap.yaml')]
    for key,value in {'width':32,'height':24,'fx':20.,'fy':20.,'cx':16.,'cy':12.,
                      'max_range_m':3.,'depth_topic':'/audit_wavemap/depth',
                      'odom_topic':'/audit_wavemap/odom','map_output_path':str(output/'map.wvmp')}.items():
        command+=['-p',f'{key}:={value}']
    log=(output/'node.log').open('w')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    report={'domain':args.domain,'image_size':[32,24],'command':command,'passed':False}
    rclpy.init();node=Node('wavemap_cpu_audit_probe')
    sensor=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    count={'integrated':0,'occupied':0}
    node.create_subscription(UInt64,'/mapping/integrated_frames',
                             lambda m:count.update(integrated=int(m.data)),durable)
    node.create_subscription(PointCloud2,'/mapping/occupied',
                             lambda m:count.update(occupied=int(m.width)),durable)
    odom_pub=node.create_publisher(Odometry,'/audit_wavemap/odom',sensor)
    depth_pub=node.create_publisher(Image,'/audit_wavemap/depth',sensor)
    save=node.create_client(Trigger,'/save_map')

    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if process.poll() is not None:raise RuntimeError('Wavemap exited; see node.log')
            rclpy.spin_once(node,timeout_sec=.05)

    try:
        deadline=time.monotonic()+12
        while not (odom_pub.get_subscription_count() and depth_pub.get_subscription_count() and save.service_is_ready()):
            if time.monotonic()>deadline:raise TimeoutError('Wavemap discovery timed out')
            spin(.1)
        for _ in range(10):
            stamp=node.get_clock().now().to_msg()
            odom=Odometry();odom.header.frame_id='world';odom.header.stamp=stamp
            odom.child_frame_id='cam0_rect_optical';odom.pose.pose.orientation.w=1.
            odom_pub.publish(odom);spin(.05)
            depth=Image();depth.header.frame_id='cam0_rect_optical';depth.header.stamp=stamp
            depth.width=32;depth.height=24;depth.encoding='32FC1';depth.step=32*4
            depth.data=struct.pack('<768f',*([2.]*768))
            depth_pub.publish(depth);spin(.2)
        spin(.5)
        assert count['integrated']>=6 and count['occupied']>0,count
        maps=Path(f'/proc/{process.pid}/maps').read_text().splitlines()
        gpu=[line for line in maps if any(name in line.lower() for name in
             ['libcuda','libcudart','libcublas','libcudnn','libnvrtc'])]
        assert not gpu,gpu
        future=save.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node,future,timeout_sec=10)
        assert future.done() and future.result().success,'Map save/reload did not succeed'
        map_path=output/'map.wvmp'
        assert map_path.is_file() and map_path.stat().st_size>0
        report.update(count,loaded_cuda_libraries=gpu,map_bytes=map_path.stat().st_size,
                      save_success=True,passed=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        report['node_returncode']=process.returncode
        log.close();node.destroy_node();rclpy.shutdown()
        if process.returncode:report['passed']=False
        (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
