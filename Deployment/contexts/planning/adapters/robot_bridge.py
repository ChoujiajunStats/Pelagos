#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Expose bounded Wavemap/nvblox snapshots and RViz goals to the isolated planner."""
import argparse
import json
import math
import queue
import socket
import threading
import time
import uuid
from pathlib import Path as FilePath

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, UInt64
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import sys
from pathlib import Path as DeploymentPath
sys.path.insert(0, str(DeploymentPath(__file__).resolve().parents[3] / 'contracts'))
from wire import receive_packet, send_packet
from planning_grid import path_status, sphere_path_status


class RobotBridge(Node):
    def __init__(self, port, backend='nvblox', planner_name='omniplanner'):
        super().__init__(planner_name + '_map_bridge')
        self.planner_name = planner_name
        self.session_id = uuid.uuid4().hex
        self.stop_event = threading.Event()
        self.connection = None
        self.connection_lock = threading.Lock()
        self.incoming = queue.Queue(maxsize=16)
        self.odom = None
        self.odom_arrival = 0.0
        self.depth_arrival = 0.0
        self.depth_stamp = 0
        self.map_sequence = 0
        self.goal_sequence = 0
        self.last_map = None
        self.map_arrival = 0.0
        self.future = None
        self.goal = None
        self.last_status = None
        self.last_status_goal_sequence = -1
        self.path_is_empty = False
        self.backend = backend
        self.path_poses = []
        import yaml
        self.robot_size = yaml.safe_load((FilePath(__file__).resolve().parent.parent/'config/planner.yaml').read_text())['RobotParams']['size']
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        durable = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/omniplanner/path', durable)
        self.status_pub = self.create_publisher(String, '/omniplanner/status', durable)
        self.marker_pub = self.create_publisher(MarkerArray, '/omniplanner/markers', durable)
        self.planning_path_pub = self.create_publisher(Path, '/planning/path', durable)
        self.execution_reference_pub = self.create_publisher(String, '/planning/execution_reference', durable)
        self.planning_status_pub = self.create_publisher(DiagnosticStatus, '/planning/status', durable)
        self.map_version_pub = self.create_publisher(UInt64, '/mapping/planning_map_version', durable)
        self.map_stats_pub = self.create_publisher(String, '/planning/map_stats', durable)
        self.create_subscription(Odometry, '/okvis/okvis_odometry', self.on_odometry, sensor_qos)
        self.create_subscription(String, '/stereo/statistics', self.on_depth, sensor_qos)
        self.create_subscription(PoseStamped, '/omniplanner/goal', self.on_goal, 5)
        self.create_subscription(PoseStamped, '/planning/goal', lambda m: self.on_goal(m, hold_depth=False), 5)
        self.create_service(Trigger, '/omniplanner/clear', self.on_clear)
        self.wavemap = None
        if backend == 'wavemap':
            from wavemap_source import WavemapSource
            self.wavemap = WavemapSource(self, durable)
        else:
            from nvblox_msgs.srv import EsdfAndGradients
            self.esdf_type = EsdfAndGradients
            self.esdf = self.create_client(EsdfAndGradients, '/nvblox_node/get_esdf_and_gradient')
        self.create_timer(.1, self.tick)
        self.create_timer(2.0 if self.wavemap else 1.0, self.request_map)
        self.create_timer(.1, self.send_odometry)
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen(1)
        self.listener.settimeout(.5)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()
        delete = Marker(); delete.action = Marker.DELETEALL; delete.header.frame_id = 'world'
        self.marker_pub.publish(MarkerArray(markers=[delete]))
        self.status('waiting_for_map')

    def fresh(self):
        if self.odom is None:
            return False
        now_ns = self.get_clock().now().nanoseconds
        stamp = self.odom.header.stamp
        pose_age = (now_ns - stamp.sec*1000000000 - stamp.nanosec)*1e-9
        depth_age = (now_ns - self.depth_stamp)*1e-9
        return (time.monotonic() - self.odom_arrival < 1.5
                and time.monotonic() - self.depth_arrival < 2.5
                and -1 < pose_age < 3 and -1 < depth_age < 3)

    def status(self, state, detail=None):
        if state == self.last_status and self.goal_sequence == self.last_status_goal_sequence:
            return
        self.last_status = state
        self.last_status_goal_sequence = self.goal_sequence
        self.status_pub.publish(String(data=json.dumps({'state': state, 'detail': detail,
            'backend':self.backend, 'planner':self.planner_name, 'robot_size_m':self.robot_size, 'execution_enabled':False,
            'map_sequence': self.map_sequence, 'goal_sequence': self.goal_sequence})))
        diagnostic = DiagnosticStatus(name=self.planner_name, message=state,
            level=DiagnosticStatus.OK if state in ('path_ready','ready_for_goal','goal_cleared') else DiagnosticStatus.WARN)
        diagnostic.values = [KeyValue(key='map_sequence', value=str(self.map_sequence)),
            KeyValue(key='detail', value=json.dumps(detail, ensure_ascii=False))]
        self.planning_status_pub.publish(diagnostic)
        self.get_logger().info(f'Planner state: {state}' + (f' ({detail})' if detail else ''))
        self.show_goal(state)

    def show_goal(self, state):
        if self.goal is None:
            return
        colors = {'path_ready': (.2, 1., .2), 'partial_path': (1., .6, .1),
                  'planning': (1., 1., .2)}
        color = colors.get(state, (1., .2, .2))
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'omniplanner_goal'; marker.id = 0
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = self.goal['pose'][:3]
        marker.pose.orientation.w = 1.
        marker.scale.x = marker.scale.y = marker.scale.z = .16
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.
        label = Marker()
        label.header = marker.header; label.ns = marker.ns; label.id = 1
        label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
        label.pose.position.x, label.pose.position.y, label.pose.position.z = self.goal['pose'][:3]
        label.pose.position.z += .3
        label.pose.orientation.w = 1.; label.scale.z = .14
        label.color = marker.color
        radius = max(self.robot_size)/2
        label.text = f'{self.planner_name}: {state} (preview r={radius:.2f}m)'
        self.marker_pub.publish(MarkerArray(markers=[marker, label]))

    def on_clear(self, request, response):
        self.goal_sequence += 1
        self.goal = None
        self.transmit({'type': 'clear', 'sequence': self.goal_sequence})
        self.clear_path()
        marker = Marker(); marker.action = Marker.DELETEALL
        marker.header.frame_id = 'world'
        self.marker_pub.publish(MarkerArray(markers=[marker]))
        self.status('goal_cleared')
        response.success = True
        response.message = 'Preview goal and path cleared'
        return response

    def clear_path(self):
        self.path_poses = []
        if self.path_is_empty:
            return
        message = Path()
        message.header.frame_id = 'world'
        message.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(message)
        self.planning_path_pub.publish(message)
        self.execution_reference_pub.publish(String(data=json.dumps(dict(frame='world',poses=[],
            session_id=self.session_id,goal_sequence=self.goal_sequence,map_sequence=self.map_sequence,stamp_ns=self.get_clock().now().nanoseconds))))
        self.path_is_empty = True

    def on_odometry(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        norm = sum(v*v for v in values[3:])
        if (message.header.frame_id.lstrip('/') != 'world' or message.child_frame_id != 'body'
                or not all(math.isfinite(v) for v in values)
                or max(abs(v) for v in values[:3]) > 30 or not .9 < norm < 1.1):
            self.odom = None
            self.clear_path()
            self.status('invalid_pose')
            return
        if self.odom is not None:
            previous = self.odom.pose.pose.position
            jump = math.dist([p.x,p.y,p.z], [previous.x,previous.y,previous.z])
            if jump > .5:
                self.odom = None
                self.clear_path()
                self.status('pose_jump')
                return
        self.odom = message
        self.odom_arrival = time.monotonic()

    def on_depth(self, message):
        try:
            self.depth_stamp = int(json.loads(message.data)['stamp_ns'])
            self.depth_arrival = time.monotonic()
        except (ValueError, KeyError, TypeError):
            pass

    def transmit(self, metadata, payload=b''):
        with self.connection_lock:
            connection = self.connection
            if connection is None:
                return False
            try:
                send_packet(connection, metadata, payload)
                return True
            except (OSError, ValueError):
                self.connection = None
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return False

    def serve(self):
        while not self.stop_event.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            connection.settimeout(1)
            with self.connection_lock:
                if self.connection:
                    self.connection.close()
                self.connection = connection
            try:
                if self.last_map and time.monotonic() - self.map_arrival < 3:
                    self.transmit(*self.last_map)
                if self.goal:
                    self.transmit(self.goal)
                while not self.stop_event.is_set():
                    metadata, payload = receive_packet(connection, self.stop_event.is_set)
                    if payload or metadata['type'] not in ('path', 'status'):
                        raise ValueError('Unexpected planner reply')
                    try:
                        self.incoming.put_nowait(metadata)
                    except queue.Full:
                        pass
            except (EOFError, OSError, ValueError) as error:
                self.get_logger().info(f'Planner transport closed: {error}')
            finally:
                with self.connection_lock:
                    if self.connection is connection:
                        self.connection = None
                connection.close()

    def send_odometry(self):
        if not self.fresh():
            return
        m = self.odom
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.transmit({'type':'odometry', 'stamp':[m.header.stamp.sec,m.header.stamp.nanosec],
            'frame':'world', 'child_frame':m.child_frame_id,
            'pose':[p.x,p.y,p.z,q.x,q.y,q.z,q.w]})

    def request_map(self):
        if self.wavemap:
            if self.connection is not None:
                self.wavemap.request()
            return
        if self.connection is None or not self.fresh() or not self.esdf.service_is_ready() or self.future is not None:
            return
        request = self.esdf_type.Request()
        request.update_esdf = False
        request.visualize_esdf = False
        request.use_aabb = True
        request.frame_id = 'world'
        p = self.odom.pose.pose.position
        request.aabb_min_m.x, request.aabb_min_m.y, request.aabb_min_m.z = p.x-4, p.y-4, p.z-1.5
        request.aabb_size_m.x, request.aabb_size_m.y, request.aabb_size_m.z = 8., 8., 3.
        self.future = self.esdf.call_async(request)
        self.future.add_done_callback(self.on_map)

    def on_map(self, future):
        self.future = None
        try:
            result = future.result()
            if not result.success or not self.fresh():
                return
            dims = result.esdf_and_gradients.layout.dim
            if len(dims)!=3 or [d.label for d in dims] != ['x','y','z']:
                raise ValueError('Unsupported nvblox ESDF layout')
            shape = [int(d.size) for d in dims]
            count = math.prod(shape)
            if not 0 < count <= 1000000:
                raise ValueError('ESDF snapshot exceeds configured cell bound')
            data = np.asarray(result.esdf_and_gradients.data, dtype='<f4')
            if data.size != count:
                raise ValueError('ESDF array size does not match dimensions')
            data[(data <= -999) | ~np.isfinite(data)] = np.nan
            stamp = self.get_clock().now().to_msg()
            metadata = {
                'origin':[result.origin_m.x,result.origin_m.y,result.origin_m.z],
                'resolution':result.voxel_size_m, 'shape':shape, 'frame':'world',
                'observed_cells':int(np.isfinite(data).sum()), 'source_depth_stamp_ns':self.depth_stamp}
            self.accept_map(metadata, data, stamp)
        except Exception as error:
            self.get_logger().error(f'ESDF snapshot rejected: {error}')

    def accept_map(self, metadata, data, stamp, native=b''):
        if self.path_poses:
            grid = np.asarray(data).reshape(metadata['shape'])
            checked = (sphere_path_status(grid,metadata,self.path_poses,max(self.robot_size)/2)
                if self.planner_name=='wavestar' else path_status(grid,metadata,self.path_poses,self.robot_size))
            if checked != 'free':
                self.clear_path()
                self.status('path_invalidated_by_map')
        self.map_sequence += 1
        metadata.update(type='map',session_id=self.session_id, sequence=self.map_sequence, stamp=[stamp.sec,stamp.nanosec], frame='world')
        self.last_map = (metadata, data.tobytes()+native)
        self.map_arrival = time.monotonic()
        version = UInt64(data=int(metadata.get('source_map_version', self.map_sequence)))
        self.map_version_pub.publish(version)
        self.map_stats_pub.publish(String(data=json.dumps(metadata)))
        if self.transmit(*self.last_map) and self.goal is None:
            self.status('ready_for_goal', metadata)

    def on_goal(self, message, hold_depth=True):
        if message.header.frame_id != 'world' or not self.fresh():
            self.status('goal_rejected_stale_pose')
            return
        p, q = message.pose.position, message.pose.orientation
        target = [p.x,p.y,self.odom.pose.pose.position.z if hold_depth else p.z,q.x,q.y,q.z,q.w]
        if not all(math.isfinite(v) for v in target):
            return
        if not .9 < sum(v*v for v in target[3:]) < 1.1:
            self.status('goal_rejected_invalid_orientation')
            return
        self.goal_sequence += 1
        self.goal = {'type':'goal', 'sequence':self.goal_sequence, 'frame':'world', 'pose':target}
        self.clear_path()
        self.status('planning')
        self.transmit(self.goal)

    def tick(self):
        if self.connection is None:
            self.clear_path()
            self.status('planner_disconnected')
        elif not self.fresh():
            self.clear_path()
            self.status('waiting_for_fresh_pose_and_depth')
        elif self.map_arrival and time.monotonic() - self.map_arrival > 5:
            self.clear_path()
            self.status('waiting_for_fresh_map')
        for _ in range(8):
            try:
                metadata = self.incoming.get_nowait()
            except queue.Empty:
                break
            if self.goal is None or metadata.get('goal_sequence') != self.goal_sequence:
                continue
            if metadata['type'] == 'status':
                if not self.fresh():
                    continue
                self.clear_path()
                self.status(metadata.get('state','planning_failed'), metadata.get('detail'))
                continue
            if metadata.get('map_sequence') != self.map_sequence:
                self.clear_path()
                self.status('planning_retry_map_changed')
                continue
            if not self.fresh() or not self.map_arrival or time.monotonic()-self.map_arrival > 5:
                continue
            poses = metadata.get('poses', [])
            if len(poses)>2000 or any(len(p)!=7 or not all(math.isfinite(v) for v in p) for p in poses):
                continue
            if poses:
                stamp = metadata.get('map_stamp', [0,0])
                map_age = (self.get_clock().now().nanoseconds-stamp[0]*1000000000-stamp[1])*1e-9
                if not -1 < map_age < 5:
                    self.clear_path()
                    self.status('planning_retry_stale_result')
                    continue
                p = self.odom.pose.pose.position
                if math.dist(poses[0][:3], [p.x, p.y, p.z]) > .5:
                    self.clear_path()
                    self.status('planning_retry_pose_moved')
                    continue
            path = Path()
            path.header.frame_id='world'
            path.header.stamp=self.get_clock().now().to_msg()
            for p in poses:
                m=PoseStamped();m.header=path.header
                m.pose.position.x,m.pose.position.y,m.pose.position.z=p[:3]
                m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w=p[3:]
                path.poses.append(m)
            self.path_pub.publish(path)
            self.planning_path_pub.publish(path)
            self.execution_reference_pub.publish(String(data=json.dumps(dict(frame='world',poses=poses,
                session_id=self.session_id,goal_sequence=self.goal_sequence,map_sequence=self.map_sequence,
                stamp_ns=self.get_clock().now().nanoseconds,preview_radius_m=max(self.robot_size)/2,
                map_stamp=metadata.get('map_stamp'),planner=self.planner_name))))
            self.path_poses = poses
            self.path_is_empty = not path.poses
            self.status(metadata.get('state','path_ready'), {'poses':len(poses),
                'planning_ms':metadata.get('planning_ms'), 'collision_check':'passed' if poses else 'no_path',
                'remaining_distance_m':metadata.get('remaining_distance_m')})

    def close(self):
        self.stop_event.set()
        self.listener.close()
        with self.connection_lock:
            if self.connection:
                try:self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                self.connection.close()
        if self.wavemap:
            self.wavemap.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18771)
    parser.add_argument('--backend', choices=('nvblox','wavemap'), default='nvblox')
    parser.add_argument('--planner', choices=('omniplanner','wavestar'), default='omniplanner')
    args=parser.parse_args()
    if args.planner=='wavestar' and args.backend!='wavemap':
        parser.error('Wavestar preview requires native Wavemap')
    rclpy.init();node=RobotBridge(args.port,args.backend,args.planner)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
