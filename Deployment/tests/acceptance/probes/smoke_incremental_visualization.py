#!/usr/bin/env python3
"""Isolated ROS acceptance: empty native map, sensor fusion, renderer, motion and crop.

Synthetic sensor fixtures are confined to a nonzero ROS domain, never sent to
the field pipeline. No map is preloaded and no physical control API is used.
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time
import traceback
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger


class Probe(Node):
    def __init__(self):
        super().__init__('incremental_render_acceptance')
        self.y = 0.;self.depth_enabled = False;self.body_enabled = True;self.last_depth = 0.
        self.clouds = [];self.markers = [];self.paths = [];self.metrics = [];self.poses = []
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        durable = QoSProfile(depth=20, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.body = self.create_publisher(Odometry, '/test/body', qos)
        self.camera = self.create_publisher(Odometry, '/test/camera', qos)
        self.depth = self.create_publisher(Image, '/test/depth', qos)
        self.create_subscription(PointCloud2, '/mapping/explored_cloud', self.clouds.append, durable)
        self.create_subscription(MarkerArray, '/mapping/local_voxels', self.markers.append, durable)
        self.create_subscription(RosPath, '/vio/trajectory', self.paths.append, durable)
        self.create_subscription(PoseStamped, '/vio/robot_pose', self.poses.append, durable)
        self.create_subscription(DiagnosticStatus, '/mapping/visualization_metrics', self.metrics.append, durable)
        self.timer = self.create_timer(.02, self.inputs)

    def inputs(self):
        now = self.get_clock().now().to_msg()
        body = Odometry();body.header.stamp = now;body.header.frame_id = 'world';body.child_frame_id = 'body'
        body.pose.pose.position.y = self.y;body.pose.pose.orientation.w = 1.
        if self.body_enabled:self.body.publish(body)
        body.child_frame_id = 'cam0_rect_optical'
        body.pose.pose.orientation.y = math.sqrt(.5);body.pose.pose.orientation.w = math.sqrt(.5)
        self.camera.publish(body)
        if self.depth_enabled and time.monotonic() - self.last_depth > .2:
            m = Image();m.header.stamp = now;m.header.frame_id = 'cam0_rect_optical'
            m.width = 32;m.height = 24;m.encoding = '32FC1';m.step = 128
            m.data = array.array('B', array.array('f', [1.94] * 768).tobytes())
            self.depth.publish(m);self.last_depth = time.monotonic()

    def wait(self, label, predicate, seconds=12):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.03)
            if predicate():print('PASS '+label, flush=True);return
        raise AssertionError(label+'; last metrics '+repr([(m.message, {v.key:v.value for v in m.values}) for m in self.metrics[-2:]]))

    def spin(self, seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:rclpy.spin_once(self,timeout_sec=.02)

    def cubes(self):
        return [m for m in self.markers[-1].markers if m.action==Marker.ADD] if self.markers else []

    def cloud_points(self):
        return {tuple(round(v,3) for v in p[:3]) for p in struct.iter_unpack('<fffI',bytes(self.clouds[-1].data))}

    def local_points(self):
        return {tuple(round(v,3) for v in (p.x,p.y,p.z)) for m in self.cubes() for p in m.points}

    def parameter(self, name, value):
        c=self.create_client(SetParameters,'/wavemap_navigation/set_parameters');assert c.wait_for_service(timeout_sec=3)
        kind=ParameterType.PARAMETER_DOUBLE if isinstance(value,float) else ParameterType.PARAMETER_INTEGER
        parameter_value=ParameterValue(type=kind)
        if kind==ParameterType.PARAMETER_DOUBLE:parameter_value.double_value=value
        else:parameter_value.integer_value=value
        request=SetParameters.Request(parameters=[Parameter(name=name,value=parameter_value)])
        future=c.call_async(request);self.wait('parameter service '+name,future.done)
        result=future.result().results[0];self.destroy_client(c);return result

    def save(self):
        c=self.create_client(Trigger,'/save_map');assert c.wait_for_service(timeout_sec=3)
        future=c.call_async(Trigger.Request());self.wait('save observed native map',future.done)
        result=future.result();self.destroy_client(c);assert result.success;return result.message


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if int(os.environ.get('ROS_DOMAIN_ID','0'))==0:parser.error('requires isolated nonzero ROS_DOMAIN_ID')
    args.output.mkdir(parents=True,exist_ok=True);os.environ['ROS_LOG_DIR']=str(args.output/'ros_logs')
    configuration=yaml.safe_load(args.config.read_text());p=configuration['/**']['ros__parameters']
    p['mapping'].update(width=32,height=24,fx=24.,fy=24.,cx=15.5,cy=11.5,min_cell_width_m=.1,
        tree_height=4,max_range_m=6.,map_input_path='',map_output_path=str(args.output/'online.wvmp'),calibration_path='')
    p['inputs'].update(body_odometry='/test/body',camera_odometry='/test/camera',depth='/test/depth')
    p['management'].update(crop_radius_m=4.,crop_period_s=.3,threshold_period_s=2.,prune_period_s=5.)
    p['visualization'].update(enabled=True,require_empty_start=True,update_hz=4.,local_radius_m=2.1,voxel_size_m=.1,pose_max_age_s=.6)
    config=args.output/'test.yaml';config.write_text(yaml.safe_dump(configuration,sort_keys=False))
    log=(args.output/'node.log').open('w');child=subprocess.Popen(['ros2','launch','underwater_navigation_ros',
        'incremental_mapping.launch.py','rviz:=false','config:='+str(config),'map_output:='+str(args.output/'online.wvmp')],stdout=log,stderr=subprocess.STDOUT)
    probe=None;error=None;checks=[];started=time.monotonic();report={}
    try:
        rclpy.init();probe=Probe()
        probe.wait('empty start publishes no cloud or fabricated cubes',lambda:probe.clouds and probe.markers and
            probe.metrics and probe.metrics[-1].message=='native_online_map' and probe.clouds[-1].width==0 and not probe.cubes())
        checks.append('initial native map empty; no preloaded or filled unknown geometry')
        probe.depth_enabled=True
        probe.wait('real integrator creates both local cubes and blue active cloud',lambda:probe.cubes() and probe.clouds[-1].width>0)
        assert all(m.type==Marker.CUBE_LIST and m.header.frame_id=='odom' and abs(m.pose.orientation.w-1)<1e-6 for m in probe.cubes())
        assert all(len(m.points)==len(m.colors) and abs(m.scale.x-.1)<1e-5 and abs(m.color.a-.95)<1e-5 for m in probe.cubes())
        colors={(round(c.r,2),round(c.g,2),round(c.b,2)) for m in probe.cubes() for c in m.colors}
        assert len(colors)>3
        rgba=next(struct.iter_unpack('<fffI',bytes(probe.clouds[-1].data)))[3]
        assert (rgba&0xffffff)==(45<<16|135<<8|255) and 80<=rgba>>24<=95
        checks.append('CUBE_LIST batching, native-aligned size, height rainbow and blue alpha')
        assert probe.parameter('visualization.local_radius_m',.5).successful
        probe.wait('live radius parameter repartitions local and blue',lambda:not probe.cubes() and probe.clouds[-1].width>0)
        assert not probe.parameter('visualization.voxel_size_m',.15).successful
        assert probe.parameter('visualization.local_radius_m',2.1).successful
        probe.wait('valid parameter restores cubes',lambda:bool(probe.cubes()))
        checks.append('validated live visualization changes and invalid-size rejection')
        native=probe.save();shutil.copy2(native,args.output/'observed-before-crop.wvmp')
        probe.depth_enabled=False;probe.spin(.6);before=probe.local_points()
        probe.y=.5;probe.spin(.6);probe.y=1.
        probe.wait('robot motion extends trajectory',lambda:probe.paths and len(probe.paths[-1].poses)>=3 and abs(probe.paths[-1].poses[-1].pose.position.y-1)<1e-6)
        probe.wait('leaving local region changes real occupied geometry to blue',lambda:bool((before-probe.local_points())&probe.cloud_points()))
        assert all(p.header.frame_id=='odom' and abs(p.pose.orientation.w-1)<1e-6 for p in probe.paths[-1].poses)
        checks.append('moving window, local-to-blue transition, real odometry path and orientation')
        probe.y=20.
        probe.wait('rolling crop removes old geometry from both active outputs',lambda:probe.metrics and
            probe.metrics[-1].message=='native_online_map' and any(v.key=='source_occupied_leaves' and v.value=='0' for v in probe.metrics[-1].values)
            and probe.clouds[-1].width==0 and not probe.cubes())
        assert probe.markers[-1].markers[0].action==Marker.DELETEALL
        checks.append('cropped cells disappear, and explicit DELETEALL removes stale RViz IDs')
        probe.body_enabled=False
        probe.wait('stale robot pose clears local renderer',lambda:probe.metrics[-1].message=='waiting_for_fresh_robot_pose')
        checks.append('stale robot pose handled without blocking native mapping')
        report.update(checks=checks,colors=len(colors),metrics=[{'message':m.message,'values':{v.key:v.value for v in m.values}} for m in probe.metrics],
            trajectory_samples=len(probe.paths[-1].poses))
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    finally:
        child.send_signal(signal.SIGINT)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()
        if probe:probe.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    if error is None:
        guard=subprocess.run([str(args.binary),'--ros-args','--params-file',str(config),'-p',
            'mapping.map_input_path:='+str(args.output/'observed-before-crop.wvmp')],capture_output=True,text=True,timeout=10)
        (args.output/'preload-rejection.log').write_text(guard.stdout+guard.stderr)
        if guard.returncode==0 or 'empty map_input' not in guard.stderr+guard.stdout:error='preloaded map was not rejected by online renderer'
        else:checks.append('online renderer explicitly rejects preloaded environment maps')
    report.update(passed=error is None and child.returncode==0,error=error,node_returncode=child.returncode,
        checks=checks,domain=os.environ.get('ROS_DOMAIN_ID'),duration_s=time.monotonic()-started,
        fixture_scope='synthetic ROS sensors in isolated domain only; zero physical actuator commands')
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','error','node_returncode','checks']},ensure_ascii=False,indent=2),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
