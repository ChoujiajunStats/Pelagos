#!/usr/bin/env python3
"""Isolated Humble acceptance: real native map, ROS depth integration and replanning."""
import argparse
import array
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import struct
import time
import traceback
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt64
from std_srvs.srv import Trigger
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from tf2_msgs.msg import TFMessage


class Probe(Node):
    def __init__(self):
        super().__init__('navigation_acceptance_probe')
        self.body_enabled = self.depth_enabled = True
        self.depth_value = 4.0
        self.last_depth = 0.0
        self.statuses = []
        self.paths = []
        self.visual_paths = []
        self.metrics = []
        self.versions = []
        self.planning_versions = []
        self.tf_samples = []
        self.integrated = 0
        self.occupied = []
        self.goal_id = 0
        sensor = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        durable = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.body = self.create_publisher(Odometry, '/test/body', sensor)
        self.camera = self.create_publisher(Odometry, '/test/camera', sensor)
        self.depth = self.create_publisher(Image, '/test/depth', sensor)
        self.goal = self.create_publisher(PoseStamped, '/planning/goal', 10)
        self.create_subscription(DiagnosticStatus, '/planning/status', self.observe_status, durable)
        self.create_subscription(DiagnosticStatus, '/mapping/metrics', self.observe_metrics, durable)
        self.create_subscription(RosPath, '/planning/path', self.observe_path, durable)
        self.create_subscription(RosPath, '/visualization/planned_path', lambda m: self.visual_paths.append(dict(frame=m.header.frame_id, points=[[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in m.poses])), durable)
        self.create_subscription(PointCloud2, '/mapping/occupied', self.observe_cloud, durable)
        self.create_subscription(UInt64, '/mapping/map_version', lambda m: self.versions.append(m.data), durable)
        self.create_subscription(UInt64, '/mapping/planning_map_version', lambda m: self.planning_versions.append(m.data), durable)
        self.create_subscription(UInt64, '/mapping/integrated_frames', lambda m: setattr(self, 'integrated', m.data), durable)
        self.create_subscription(TFMessage, '/tf', self.observe_tf, sensor)
        self.timer = self.create_timer(.02, self.publish_inputs)

    def observe_status(self, message):
        level = int.from_bytes(message.level, 'little') if isinstance(message.level, bytes) else int(message.level)
        self.statuses.append(dict(t=time.monotonic(), message=message.message, level=level,
                                  values={v.key: v.value for v in message.values}))

    def observe_metrics(self, message):
        self.metrics.append({v.key: v.value for v in message.values})

    def observe_path(self, message):
        self.paths.append(dict(t=time.monotonic(), frame=message.header.frame_id,
                               points=[[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                                       for p in message.poses]))

    def observe_tf(self, message):
        for tf in message.transforms:
            if tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_link':
                self.tf_samples.append(dict(t=time.monotonic(), body_enabled=self.body_enabled,
                    position=[tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
                    quaternion_xyzw=[tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w]))

    def observe_cloud(self, message):
        self.occupied = [list(p[:3]) for p in struct.iter_unpack('<ffff', bytes(message.data))]

    def publish_inputs(self):
        stamp = self.get_clock().now().to_msg()
        body = Odometry();body.header.frame_id = 'world';body.child_frame_id = 'body'
        body.header.stamp = stamp;body.pose.pose.position.x = -1.5;body.pose.pose.orientation.w = 1.
        if self.body_enabled:
            self.body.publish(body)
        camera = copy.deepcopy(body);camera.child_frame_id = 'cam0_rect_optical'
        camera.pose.pose.orientation.y = math.sqrt(.5);camera.pose.pose.orientation.w = math.sqrt(.5)
        self.camera.publish(camera)
        if self.depth_enabled and time.monotonic() - self.last_depth >= .25:
            image = Image();image.header.stamp = stamp;image.header.frame_id = 'cam0_rect_optical'
            image.width = 32;image.height = 24;image.encoding = '32FC1';image.step = 128
            image.data = array.array('B', array.array('f', [self.depth_value] * 768).tobytes())
            self.depth.publish(image);self.last_depth = time.monotonic()

    def wait(self, label, predicate, seconds=15):
        began = time.monotonic()
        while time.monotonic() - began < seconds:
            rclpy.spin_once(self, timeout_sec=.03)
            if predicate():
                print('PASS ' + label, flush=True)
                return
        raise AssertionError(label + ' timed out; last statuses: ' + repr(self.statuses[-4:]))

    def goal_message(self, xyz, frame='odom'):
        message = PoseStamped();message.header.frame_id = frame
        message.header.stamp = self.get_clock().now().to_msg();message.pose.orientation.w = 1.
        message.pose.position.x, message.pose.position.y, message.pose.position.z = xyz
        return message

    def send_goal(self, xyz, frame='odom'):
        self.goal_id += 1
        began = time.monotonic();self.goal.publish(self.goal_message(xyz, frame))
        wanted = str(self.goal_id)
        self.wait('goal %s succeeds with collision validation' % wanted,
            lambda: any(s['t'] >= began and s['message'] == 'success'
                        and s['values'].get('goal_id') == wanted
                        and s['values'].get('collision_free') == 'true' for s in self.statuses))
        self.wait('goal %s exact 3D endpoint in odom' % wanted,
            lambda: any(p['t'] >= began and p['frame'] == 'odom' and p['points']
                        and max(abs(a-b) for a,b in zip(p['points'][-1], xyz)) < 1e-4 for p in self.paths))
        self.wait('visual planning path preserves exact 3D endpoint', lambda: any(p['frame']=='odom' and p['points'] and max(abs(a-b) for a,b in zip(p['points'][-1],xyz))<1e-4 for p in self.visual_paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--fixture-binary', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--use-launch', action='store_true', help='Exercise the installed ROS 2 launch entrypoint.')
    args = parser.parse_args()
    if int(os.environ.get('ROS_DOMAIN_ID', '0')) == 0:
        parser.error('must use an isolated nonzero ROS_DOMAIN_ID')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_LOG_DIR'] = str(args.output / 'ros_logs')
    fixture = args.output / 'room.wvmp'
    subprocess.run([str(args.fixture_binary), str(fixture)], check=True)
    configuration = yaml.safe_load(args.config.read_text())
    parameters = configuration['/**']['ros__parameters']
    parameters.setdefault('visualization', {}).update(enabled=False, voxel_size_m=.2, mesh_resolution_m=.2)
    parameters['mapping'].update(width=32, height=24, fx=24., fy=24., cx=15.5, cy=11.5,
        min_cell_width_m=.2, tree_height=4, max_range_m=6., map_input_path=str(fixture),
        map_output_path=str(args.output / 'online.wvmp'), calibration_path='')
    parameters['inputs'].update(body_odometry='/test/body', camera_odometry='/test/camera', depth='/test/depth')
    parameters['planner'].update(max_pose_age_s=.7, max_depth_age_s=.7, max_snapshot_age_s=3.)
    parameters['management'].update(threshold_period_s=2., prune_period_s=5.)
    parameters['planning_map']['update_period_s'] = .1
    config_path = args.output / 'test.yaml'
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False))
    log = (args.output / 'node.log').open('w')
    command = ['ros2', 'launch', 'underwater_navigation_ros', 'navigation.launch.py',
               'config:='+str(config_path), 'map_input:='+str(fixture),
               'map_output:='+str(args.output / 'online.wvmp')] if args.use_launch else [str(args.binary), '--ros-args', '--params-file', str(config_path)]
    child = subprocess.Popen(command,
                              stdout=log, stderr=subprocess.STDOUT)
    probe = None;error = None;checks = [];started = time.monotonic()
    try:
        rclpy.init();probe = Probe()
        probe.wait('ROS discovery and real depth fusion', lambda: probe.integrated >= 3 and len(probe.tf_samples) >= 20)
        assert all(abs(p['position'][0] + 1.5) < 1e-6 for p in probe.tf_samples)
        assert all(max(abs(a-b) for a,b in zip(p['quaternion_xyzw'],[0,0,0,1])) < 1e-6 for p in probe.tf_samples)
        checks.append('body TF uses robot pose, distinct from optical camera rotation')
        probe.send_goal([-.5, .6, .4])
        parameter_client = probe.create_client(SetParameters, '/wavemap_navigation/set_parameters')
        assert parameter_client.wait_for_service(timeout_sec=3)
        request = SetParameters.Request()
        request.parameters = [Parameter(name='planner.algorithm', value=ParameterValue(
            type=ParameterType.PARAMETER_STRING, string_value='a_star'))]
        future = parameter_client.call_async(request)
        probe.wait('startup-frozen parameters reject misleading live changes', future.done)
        assert not future.result().results[0].successful
        checks.append('configuration changes rejected explicitly; YAML/launch algorithm overrides remain available')
        probe.send_goal([-.2, -.6, .3], frame='world')
        checks.append('3D goal updates and world-to-odom TF transform')
        previous_version = max(probe.versions)
        probe.wait('map change triggers new-version replanning',
            lambda: any(s['message'] == 'success' and s['values'].get('goal_id') == str(probe.goal_id)
                        and int(s['values'].get('map_version', '0')) > previous_version for s in probe.statuses))
        checks.append('incremental map versions trigger replanning')
        probe.send_goal([1.1, 0., 0.])
        initial_length = next(float(s['values']['length_m']) for s in reversed(probe.statuses)
                              if s['message'] == 'success' and s['values'].get('goal_id') == str(probe.goal_id))
        # Avoid a degenerate plane exactly on all voxel centers: the official
        # continuous ray model has zero update at the measured surface itself.
        probe.depth_value = 1.94;began = time.monotonic()
        probe.wait('real depth obstacle changes the planning result', lambda: any(s['t'] >= began
            and s['values'].get('goal_id') == str(probe.goal_id)
            and (s['message'] in ('goal_not_free', 'unreachable', 'path_collision_or_unknown') or
                 (s['message'] == 'success' and float(s['values'].get('length_m', '0')) > initial_length + .05
                  and s['values'].get('collision_free') == 'true')) for s in probe.statuses), seconds=30)
        probe.wait('new depth obstacle appears in occupied map',
                   lambda: any(0 < p[0] < 1 and abs(p[1]) < .5 and abs(p[2]) < .5 for p in probe.occupied))
        obstacle = next(p for p in probe.occupied if 0 < p[0] < 1 and abs(p[1]) < .5 and abs(p[2]) < .5)
        probe.goal_id += 1;began = time.monotonic();probe.goal.publish(probe.goal_message(obstacle))
        probe.wait('goal in measured obstacle fails explicitly', lambda: any(s['t'] >= began
                   and s['values'].get('goal_id') == str(probe.goal_id) and s['message'] == 'goal_not_free' for s in probe.statuses))
        probe.wait('unreachable path is cleared', lambda: bool(probe.paths) and not probe.paths[-1]['points'])
        probe.wait('visual planning path also clears on failure',lambda:probe.visual_paths and not probe.visual_paths[-1]['points'])
        checks.append('depth integration creates obstacle; planning outcome changes; occupied goal rejected with empty path')
        probe.send_goal([-1., .5, .5])
        probe.goal_id += 1;began = time.monotonic()
        probe.goal.publish(probe.goal_message([0., 0., 0.], frame='missing_frame'))
        probe.wait('invalid-frame goal rejected', lambda: any(s['t'] >= began and s['message'].startswith('invalid_goal') for s in probe.statuses))
        probe.wait('invalid goal clears path', lambda: not probe.paths[-1]['points'])
        checks.append('invalid goal frame rejected and old path cleared')
        probe.send_goal([-1., -.5, .4])
        def rejected_count():
            return max((int(float(m.get('rejected_inputs', '0'))) for m in probe.metrics), default=0)
        for case in ('wrong_frame', 'wrong_encoding', 'truncated', 'negative_stamp', 'all_nan'):
            before = rejected_count()
            bad = Image();bad.header.stamp = probe.get_clock().now().to_msg();bad.header.frame_id = 'cam0_rect_optical'
            bad.width = 32;bad.height = 24;bad.encoding = '32FC1';bad.step = 128
            bad.data = array.array('B', array.array('f', [1.94] * 768).tobytes())
            if case == 'wrong_frame': bad.header.frame_id = 'body'
            elif case == 'wrong_encoding': bad.encoding = '16UC1'
            elif case == 'truncated': bad.data = array.array('B', [0]*4)
            elif case == 'negative_stamp': bad.header.stamp.sec = -1
            else: bad.data = array.array('B', array.array('f', [float('nan')] * 768).tobytes())
            probe.depth.publish(bad)
            probe.wait('malformed depth rejected: '+case, lambda: rejected_count() > before)
        checks.append('malformed depth frame/encoding/size/time and all-NaN data rejected without stopping fusion')
        probe.body_enabled = False;began = time.monotonic()
        probe.wait('stale body TF blocks planning', lambda: any(s['t'] >= began and s['message'].startswith('stale_') for s in probe.statuses))
        probe.wait('stale TF clears path', lambda: not probe.paths[-1]['points'])
        probe.body_enabled = True;began = time.monotonic()
        probe.wait('fresh body pose recovers planning', lambda: any(s['t'] >= began and s['message'] == 'success' for s in probe.statuses))
        checks.append('stale TF rejection and recovery')
        probe.depth_enabled = False;began = time.monotonic()
        probe.wait('stale depth blocks planning', lambda: any(s['t'] >= began and s['message'].startswith('stale_') for s in probe.statuses))
        probe.wait('stale depth clears path', lambda: not probe.paths[-1]['points'])
        probe.depth_enabled = True;began = time.monotonic()
        probe.wait('fresh depth recovers planning', lambda: any(s['t'] >= began and s['message'] == 'success' for s in probe.statuses))
        checks.append('stale depth rejection and recovery despite maintenance heartbeats')
        client = probe.create_client(Trigger, '/save_map')
        assert client.wait_for_service(timeout_sec=3)
        future = client.call_async(Trigger.Request())
        probe.wait('native map service saves online map', future.done)
        assert future.result().success and (args.output / 'online.wvmp').stat().st_size > 0
        checks.append('online map saved')
        assert all(a <= b for a,b in zip(probe.versions, probe.versions[1:])), 'map versions regressed'
        assert probe.planning_versions and probe.metrics
        assert all(all(math.isfinite(v) for point in p['points'] for v in point) for p in probe.paths)
        assert child.poll() is None, 'node exited during acceptance'
    except Exception:
        error = traceback.format_exc();print(error, flush=True)
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill();child.wait();error = (error or '') + '\nnode required SIGKILL'
        log.close()
        recording = {}
        if error is None:
            try:
                record_path = args.output / 'navigation-metrics.jsonl'
                records = [json.loads(line) for line in record_path.read_text().splitlines()]
                assert any('native_memory_bytes' in r['values'] for r in records)
                assert any('sdf_ms' in r['values'] for r in records)
                assert any('planning_latency_ms' in r['values'] for r in records)
                assert any(r['values'].get('collision_free') == 'true' for r in records)
                recording = dict(path=str(record_path), lines=len(records), bytes=record_path.stat().st_size)
                checks.append('bounded asynchronous JSONL metrics include maintenance, planning-map, latency and collision results')
            except Exception:
                error = traceback.format_exc();print(error, flush=True)
        result = dict(passed=error is None and child.returncode == 0, checks=checks, error=error,
                      node_returncode=child.returncode, duration_s=time.monotonic()-started,
                      domain=os.environ['ROS_DOMAIN_ID'], radius_m=parameters['planning_map']['robot_radius_m'])
        result['uses_ros_launch'] = args.use_launch
        result['recording'] = recording
        if probe:
            result.update(statuses=probe.statuses, paths=probe.paths, metrics=probe.metrics,
                          map_versions=probe.versions, planning_versions=probe.planning_versions,
                          tf_samples=probe.tf_samples, integrated_frames=probe.integrated,
                          occupied_points=len(probe.occupied),occupied_sample=probe.occupied[:20])
            probe.destroy_node();rclpy.shutdown()
        (args.output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
