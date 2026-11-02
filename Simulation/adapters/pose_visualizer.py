#!/usr/bin/env python3
"""Display GT and OKVIS camera trajectories and camera frustums in RViz."""
from collections import deque
import copy
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker

from geometry import compose, inverse, optical_transform
from trajectory_geometry import DistanceSampler,frustum_edges,transformed_frustum,pose_error
import yaml


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pose_of(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    value = ((float(p.x), float(p.y), float(p.z)), (float(q.x), float(q.y), float(q.z), float(q.w)))
    if not all(math.isfinite(x) for part in value for x in part):
        raise ValueError('pose contains non-finite values')
    return value


def fill_pose(pose, transform):
    p, q = transform
    pose.position.x, pose.position.y, pose.position.z = p
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q


class PoseVisualizer(Node):
    def __init__(self):
        super().__init__('pelagos_pose_visualizer')
        path = Path(self.declare_parameter('robot_profile_path', '').value)
        profile = yaml.safe_load(path.read_text())
        camera = profile['cameras']['left']
        self.body_camera = optical_transform(camera)
        self.hfov = math.radians(float(camera['horizontal_fov_deg']))
        self.aspect = float(camera['height']) / float(camera['width'])
        self.frame = 'pelagos_view'
        self.gt_history = deque(maxlen=2000)
        # Retain the session's start and ascent. A rolling 2000 x 0.1 s buffer
        # discarded all initial movement after ~200 s when the vehicle settled.
        self.gt_path, self.est_path = [], []
        self.gt_start = None
        self.gt_distance = 0.0
        self.gt_previous = None
        self.guard_state = 'unknown'
        self.joystick_status = ''
        self.mapping_status = 'Wavemap: waiting for depth + camera pose'
        self.status_file = Path(self.declare_parameter('run_dir', '').value)
        self.alignment = None
        self.last_gt_path_stamp = 0
        self.last_est_path_stamp = 0
        spacing=float(self.declare_parameter('camera_pose_spacing_m',.75).value)
        self.samplers={name:DistanceSampler(spacing) for name in ('gt','estimated')}
        self.frustum_points={name:[] for name in self.samplers}
        self.frustum_counts={name:0 for name in self.samplers}
        durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.history_pub={name:self.create_publisher(Marker,f'/pelagos/pose/{name}_camera_history',durable)
                          for name in self.samplers}
        self.error_pub=self.create_publisher(String,'/pelagos/vio/pose_error',5)
        self.error_count=0;self.error_squared_sum=0.
        self.anchor_pub = self.create_publisher(String, '/pelagos/pose/alignment', 10)
        self.gt_marker = self.create_publisher(Marker, '/pelagos/pose/gt_camera', 10)
        self.est_marker = self.create_publisher(Marker, '/pelagos/pose/estimated_camera', 10)
        self.gt_path_pub = self.create_publisher(RosPath, '/pelagos/pose/gt_path', 10)
        self.est_path_pub = self.create_publisher(RosPath, '/pelagos/pose/estimated_path', 10)
        self.motion_pub = self.create_publisher(Marker, '/pelagos/pose/motion', 5)
        self.start_pub = self.create_publisher(Marker, '/pelagos/pose/gt_start', 5)
        self.create_subscription(String, '/rov01/control/status', self.on_guard, 1)
        self.create_subscription(String, '/pelagos/joystick/status', self.on_joystick, 1)
        self.create_subscription(String, '/mapping/status', self.on_mapping, 1)
        self.create_subscription(Odometry, '/rov01/sim/ground_truth/odometry', self.on_gt,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odometry/estimated', self.on_est,
                                 qos_profile_sensor_data)
        # A small static child registers the visualization-only fixed frame in TF.
        self.static_tf = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id, transform.child_frame_id = self.frame, 'pelagos_view_axes'
        transform.transform.rotation.w = 1.0
        self.static_tf.sendTransform(transform)
        self.focus_tf = TransformBroadcaster(self)
        cave = self.declare_parameter('cave_asset_path', '').value
        if cave:
            self.show_cave(Path(cave))
        else:
            center=TransformStamped()
            center.header.frame_id,center.child_frame_id=self.frame,'pelagos_map_center'
            center.transform.rotation.w=1.
            self.static_tf.sendTransform(center)

    def show_cave(self, directory):
        from uw_robot.frames import Q_ENU_NED, qmul, rpy_quaternion, ned_to_enu
        asset = json.loads((directory / 'asset.json').read_text())
        self.cave_pub = self.create_publisher(Marker, '/pelagos/scene/porth',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'PORTH_SCENE_GROUND_TRUTH'
        marker.type, marker.action = Marker.MESH_RESOURCE, Marker.ADD
        # This existing mesh covers the entire cave. The previous cutaway
        # asset was a radius-8 crop around the spawn and hid the rest of Porth.
        marker.mesh_resource = (directory / 'collision.obj').as_uri()
        marker.mesh_use_embedded_materials = False
        pose = (ned_to_enu(asset['cave_offset_ned']),
                qmul(Q_ENU_NED, rpy_quaternion(0., 0., asset['cave_yaw_ned'])))
        fill_pose(marker.pose, pose)
        marker.scale.x = marker.scale.y = marker.scale.z = float(asset['scale'])
        marker.color.r,marker.color.g,marker.color.b=.6,.7,.8
        marker.color.a = .12
        self.cave_pub.publish(marker)
        center_source=tuple((lo+hi)*.5*float(asset['scale'])
                            for lo,hi in zip(*asset['original_bounds_ned']))
        center_pose=compose(pose,(center_source,(0.,0.,0.,1.)))
        center=TransformStamped()
        center.header.frame_id,center.child_frame_id=self.frame,'pelagos_map_center'
        center.transform.translation.x,center.transform.translation.y,center.transform.translation.z=center_pose[0]
        center.transform.rotation.w=1.
        self.static_tf.sendTransform(center)

    def marker(self, transform, header, name, color):
        msg = Marker()
        msg.header = copy.deepcopy(header)
        msg.header.frame_id = self.frame
        msg.ns, msg.id, msg.type, msg.action = name, 0, Marker.LINE_LIST, Marker.ADD
        fill_pose(msg.pose, transform)
        msg.scale.x = .006 if name=='ground_truth' else .003
        msg.color.r, msg.color.g, msg.color.b, msg.color.a = (*color, 1.0)
        depth = 0.4
        for start, end in frustum_edges(self.hfov,self.aspect,depth):
            msg.points.extend([Point(x=start[0], y=start[1], z=start[2]),
                               Point(x=end[0], y=end[1], z=end[2])])
        return msg

    def retain_camera(self,name,transform,header,color):
        if not self.samplers[name].accept(transform[0]):return
        self.frustum_counts[name]+=1
        self.frustum_points[name].extend(Point(x=p[0],y=p[1],z=p[2])
            for p in transformed_frustum(transform,self.hfov,self.aspect))
        marker=Marker()
        marker.header=copy.deepcopy(header);marker.header.frame_id=self.frame
        marker.ns=name+'_distance_sampled_cameras'
        marker.type,marker.action=Marker.LINE_LIST,Marker.ADD
        marker.pose.orientation.w=1.;marker.scale.x=.009 if name=='gt' else .0045
        marker.color.r,marker.color.g,marker.color.b,marker.color.a=(*color,.9)
        marker.points=self.frustum_points[name]
        self.history_pub[name].publish(marker)

    def publish_path(self, publisher, history, header):
        msg = RosPath()
        msg.header = copy.deepcopy(header)
        msg.header.frame_id = self.frame
        msg.poses = list(history)
        publisher.publish(msg)

    def point(self, history, transform, header):
        if history:
            previous = history[-1].pose.position
            if math.dist((previous.x, previous.y, previous.z), transform[0]) < .01:
                return
        if len(history) >= 20000:
            history[:] = history[::2]  # Preserve the start and full temporal span.
        pose = PoseStamped()
        pose.header = copy.deepcopy(header)
        pose.header.frame_id = self.frame
        fill_pose(pose.pose, transform)
        history.append(pose)

    def on_guard(self, msg):
        try:
            self.guard_state = json.loads(msg.data).get('state', 'unknown')
        except (ValueError, TypeError):
            pass

    def on_joystick(self, msg):
        try:
            report = json.loads(msg.data)
            self.joystick_status = 'Gamepad: ' + report.get('reason', 'unknown')
        except (ValueError, TypeError):
            pass

    def on_mapping(self,msg):
        try:
            report=json.loads(msg.data)
            self.mapping_status=(f'Wavemap {report["integrated_frames"]} frames | '
                f'{report["occupied_leaves"]} cells | {report["memory_bytes"]/1024**2:.1f} MB | '
                f'{report["integration_hz"]:.1f} Hz')
        except (ValueError,TypeError,KeyError):pass

    def motion(self, transform, msg):
        if self.gt_start is None:
            self.gt_start = transform[0]
        if self.gt_previous is not None:
            self.gt_distance += math.dist(transform[0], self.gt_previous)
        self.gt_previous = transform[0]
        displacement = math.dist(transform[0], self.gt_start)
        rise = transform[0][2] - self.gt_start[2]
        report = dict(gt_distance_m=self.gt_distance, gt_displacement_m=displacement,
                      gt_rise_m=rise, guard=self.guard_state,
                      gt_camera_frustums=self.frustum_counts['gt'],
                      estimated_camera_frustums=self.frustum_counts['estimated'],
                      gt_path_points=len(self.gt_path), stamp_ns=stamp_ns(msg.header.stamp))
        text = Marker()
        text.header = copy.deepcopy(msg.header)
        text.header.frame_id = self.frame
        text.ns = 'motion_status'
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        fill_pose(text.pose, (tuple(p + d for p,d in zip(transform[0], (0.,0.,.5))), (0.,0.,0.,1.)))
        text.scale.z = .10
        text.color.r = text.color.g = text.color.b = text.color.a = 1.
        text.text = (f'GT displacement {displacement:.2f} m | rise {rise:+.2f} m\n'
                     f'Travel {self.gt_distance:.2f} m | {self.guard_state}\n{self.joystick_status}\n{self.mapping_status}')
        self.motion_pub.publish(text)
        start = Marker()
        start.header = text.header
        start.ns = 'gt_start'
        start.type, start.action = Marker.SPHERE, Marker.ADD
        fill_pose(start.pose, (self.gt_start, (0.,0.,0.,1.)))
        start.scale.x = start.scale.y = start.scale.z = .08
        start.color.r = start.color.a = 1.
        self.start_pub.publish(start)
        if str(self.status_file) != '.':
            temporary = self.status_file / 'motion-status.json.tmp'
            temporary.write_text(json.dumps(report))
            temporary.replace(self.status_file / 'motion-status.json')

    def on_gt(self, msg):
        if msg.header.frame_id != 'rov01/truth/odom' or msg.child_frame_id != 'rov01/truth/base_link':
            return
        try:
            transform = compose(pose_of(msg), self.body_camera)
        except ValueError:
            return
        stamp = stamp_ns(msg.header.stamp)
        if self.gt_history and stamp <= self.gt_history[-1][0]:
            return
        self.gt_history.append((stamp, transform))
        focus = TransformStamped()
        focus.header = copy.deepcopy(msg.header)
        focus.header.frame_id, focus.child_frame_id = self.frame, 'pelagos_focus'
        focus.transform.translation.x, focus.transform.translation.y, focus.transform.translation.z = transform[0]
        focus.transform.rotation.w = 1.0
        self.focus_tf.sendTransform(focus)
        self.gt_marker.publish(self.marker(transform, msg.header, 'ground_truth', (1.0, 0.0, 0.0)))
        if stamp - self.last_gt_path_stamp >= 100_000_000:
            self.retain_camera('gt',transform,msg.header,(1.,0.,0.))
            self.point(self.gt_path, transform, msg.header)
            self.publish_path(self.gt_path_pub, self.gt_path, msg.header)
            self.last_gt_path_stamp = stamp
            self.motion(transform, msg)

    def on_est(self, msg):
        if msg.header.frame_id != 'world' or msg.child_frame_id != 'cam0_rect_optical':
            return
        stamp = stamp_ns(msg.header.stamp)
        matched=None
        if self.gt_history:
            nearest_stamp,gt_camera=min(self.gt_history,key=lambda entry:abs(entry[0]-stamp))
            if abs(nearest_stamp-stamp)<=50_000_000:matched=(nearest_stamp,gt_camera)
        try:
            estimated = pose_of(msg)
            if self.alignment is None:
                if matched is None:return
                nearest_stamp,gt_camera=matched
                self.alignment = compose(gt_camera, inverse(estimated))
                view = TransformStamped()
                view.header = copy.deepcopy(msg.header)
                view.header.frame_id, view.child_frame_id = self.frame, 'world'
                view.transform.translation.x, view.transform.translation.y, view.transform.translation.z = self.alignment[0]
                view.transform.rotation.x, view.transform.rotation.y, view.transform.rotation.z, view.transform.rotation.w = self.alignment[1]
                self.static_tf.sendTransform(view)
                alignment_report={
                    'method': 'first timestamp-matched camera pose, fixed SE(3)',
                    'estimated_stamp_ns': stamp, 'gt_stamp_ns': nearest_stamp,
                    'translation_m': self.alignment[0], 'quaternion_xyzw': self.alignment[1]}
                self.anchor_pub.publish(String(data=json.dumps(alignment_report)))
                if str(self.status_file)!='.':
                    (self.status_file/'vio-alignment.json').write_text(json.dumps(alignment_report,indent=2))
            aligned = compose(self.alignment, estimated)
        except ValueError:
            return
        self.est_marker.publish(self.marker(aligned, msg.header, 'estimated', (0.0, 1.0, 0.0)))
        if stamp - self.last_est_path_stamp >= 100_000_000:
            self.retain_camera('estimated',aligned,msg.header,(0.,1.,0.))
            self.point(self.est_path, aligned, msg.header)
            self.publish_path(self.est_path_pub, self.est_path, msg.header)
            self.last_est_path_stamp = stamp
            report=dict(estimated_stamp_ns=stamp,matched=matched is not None,
                        alignment='one fixed initial SE3, no subsequent fit',
                        estimated_raw_camera_pose=estimated,estimated_aligned_camera_pose=aligned)
            if matched is not None:
                error=pose_error(matched[1],aligned)
                self.error_count+=1;self.error_squared_sum+=error['position_error_m']**2
                report.update(error,gt_stamp_ns=matched[0],gt_camera_pose=matched[1],
                              position_rmse_m=math.sqrt(self.error_squared_sum/self.error_count),
                              error_pairs=self.error_count,stamp_difference_ms=(stamp-matched[0])*1e-6)
            self.error_pub.publish(String(data=json.dumps(report)))
            if str(self.status_file)!='.':
                with (self.status_file/'vio-pose-comparison.jsonl').open('a') as stream:
                    stream.write(json.dumps(report)+'\n')


def main():
    rclpy.init()
    node = PoseVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
