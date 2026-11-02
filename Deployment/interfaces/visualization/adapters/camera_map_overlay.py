#!/usr/bin/env python3
"""Display-only comparison: underwater-rectified camera and projected native mesh edges.

No mapping, pose correction, TF publication, recording or robot control. All caches are
bounded; the right image is a wireframe diagnostic, not an occlusion-correct renderer.
"""
import argparse
import array
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'contexts/depth/core'))
from estimation.okvis_pose_adapter import quaternion_rotation


def stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def interpolate_pose(poses, target, max_gap_ns=100_000_000):
    """Interpolate only inside a fresh bracketing pair; never extrapolate."""
    if not poses:
        return None
    i = bisect.bisect_left([p[0] for p in poses], target)
    if i < len(poses) and poses[i][0] == target:
        return poses[i][1], poses[i][2], 0
    if i == 0 or i == len(poses):
        return None
    a, b = poses[i-1], poses[i]
    gap = b[0] - a[0]
    if gap <= 0 or gap > max_gap_ns:
        return None
    t = (target-a[0]) / gap
    qa, qb = a[2], b[2].copy()
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    dot = min(1., max(-1., dot))
    if dot > .9995:
        q = (1-t)*qa + t*qb
    else:
        angle = math.acos(dot)
        q = (math.sin((1-t)*angle)*qa + math.sin(t*angle)*qb) / math.sin(angle)
    return (1-t)*a[1]+t*b[1], q/np.linalg.norm(q), gap


def project_triangles(vertices, position, quaternion, intrinsics, width, height):
    """World/odom vertices -> rectified optical pixels. Reject near-plane crossings."""
    if not len(vertices):
        return np.empty((0, 3, 2), dtype=np.int32)
    camera = ((vertices-position) @ quaternion_rotation(quaternion)).reshape(-1, 3, 3)
    keep = np.all((camera[:, :, 2] > .3) & (camera[:, :, 2] < 12.), axis=1)
    camera = camera[keep]
    if not len(camera):
        return np.empty((0, 3, 2), dtype=np.int32)
    pixels = camera[:, :, :2] / camera[:, :, 2:3]
    pixels = pixels * [intrinsics[0, 0], intrinsics[1, 1]] + [intrinsics[0, 2], intrinsics[1, 2]]
    keep = ((pixels[:, :, 0].max(axis=1) >= 0) & (pixels[:, :, 0].min(axis=1) < width)
            & (pixels[:, :, 1].max(axis=1) >= 0) & (pixels[:, :, 1].min(axis=1) < height))
    return np.rint(np.clip(pixels[keep], -20000, 20000)).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--hz', type=float, default=2.)
    args, ros_args = parser.parse_known_args()
    if not 0 < args.hz <= 5:
        parser.error('--hz must be in (0, 5]')
    calibration = json.loads(args.calibration.read_text())
    import cv2
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    cv2.setNumThreads(1)
    sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    mesh_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    width, height = calibration['width'], calibration['height']
    k = np.asarray(calibration['P_left'], dtype=float)[:, :3] * .5
    k[2, 2] = 1.
    maps = cv2.initUndistortRectifyMap(np.asarray(calibration['K_left']),
        np.asarray(calibration['D_left']), np.asarray(calibration['R_left']), k,
        (width//2, height//2), cv2.CV_32FC1)
    offset = round(calibration['image_time_offset_s']*1e9)

    class Overlay(Node):
        def __init__(self):
            super().__init__('wavemap_camera_comparison')
            self.image = None
            self.poses = []
            self.mesh = {}
            self.last_output = 0.
            self.last_stamp = -1
            self.frames = 0
            self.publisher = self.create_publisher(Image, '/visualization/camera_comparison/image', sensor)
            self.stats = self.create_publisher(String, '/visualization/camera_comparison/status', 1)
            self.create_subscription(Image, '/zed/zed_node/left/color/raw/image', self.on_image, sensor)
            self.create_subscription(Odometry, '/odometry/estimated', self.on_pose, sensor)
            self.create_subscription(MarkerArray, '/visualization/surface_mesh', self.on_mesh, mesh_qos)
            self.create_timer(.05, self.render)
            self.get_logger().info('Display-only underwater camera comparison, 2 Hz default, no recording/control')

        def on_image(self, message):
            if (message.width, message.height) == (width, height) and message.header.frame_id == 'zed_left_camera_frame_optical':
                self.image = message

        def on_pose(self, message):
            if message.header.frame_id != 'world' or message.child_frame_id != calibration['camera_frame']:
                return
            p, q = message.pose.pose.position, message.pose.pose.orientation
            p, q = np.array([p.x,p.y,p.z]), np.array([q.x,q.y,q.z,q.w])
            if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
                return
            stamp = stamp_ns(message.header.stamp)
            if self.poses and stamp <= self.poses[-1][0]:
                self.poses.clear()
            self.poses.append((stamp, p, q/np.linalg.norm(q)))
            self.poses = self.poses[-300:]

        def on_mesh(self, message):
            for m in message.markers:
                key = m.ns, m.id
                if m.action == Marker.DELETEALL:
                    self.mesh.clear()
                elif m.action == Marker.DELETE:
                    self.mesh.pop(key, None)
                elif (m.type == Marker.TRIANGLE_LIST and m.header.frame_id == 'odom'
                      and len(m.points) % 3 == 0 and m.color.a > 0):
                    points = np.array([[p.x,p.y,p.z] for p in m.points], dtype=np.float32).reshape(-1,3)
                    q, p = m.pose.orientation, m.pose.position
                    if not np.isfinite(points).all():
                        continue
                    points = points @ quaternion_rotation([q.x,q.y,q.z,q.w]).T + [p.x,p.y,p.z]
                    old = self.mesh.pop(key, None)
                    if (len(self.mesh) < 128 and len(points) + sum(len(v) for v in self.mesh.values()) <= 360000):
                        self.mesh[key] = points
                    elif old is not None:
                        self.mesh[key] = old

        def render(self):
            began = time.monotonic()
            if self.image is None or began-self.last_output < 1/args.hz:
                return
            source = self.image
            stamp = stamp_ns(source.header.stamp)+offset
            if stamp <= self.last_stamp:
                return
            age = (self.get_clock().now().nanoseconds-stamp)/1e9
            if not -.2 < age < 1.:
                return
            pose = interpolate_pose(self.poses, stamp)
            if pose is None:
                return
            channels = {'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(source.encoding)
            if channels is None or source.step < width*channels or len(source.data) < source.step*height:
                return
            raw = np.ndarray((height,width,channels), dtype=np.uint8, buffer=source.data,
                             strides=(source.step,channels,1))[:, :, :3]
            if source.encoding.startswith('bgr'):
                raw = raw[:, :, ::-1]
            rectified = cv2.remap(raw,maps[0],maps[1],cv2.INTER_LINEAR)
            points = np.concatenate(list(self.mesh.values())) if self.mesh else np.empty((0,3))
            triangles = project_triangles(points,pose[0],pose[1],k,width//2,height//2)
            overlay = rectified.copy()
            if len(triangles):
                cv2.polylines(overlay,list(triangles),True,(30,245,220),1,cv2.LINE_AA)
                overlay = cv2.addWeighted(overlay,.7,rectified,.3,0.)
            output = np.zeros((height//2+28,width,3),dtype=np.uint8)
            output[28:,:width//2] = rectified
            output[28:,width//2:] = overlay
            cv2.putText(output,'RECTIFIED CAMERA',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,240,248),1,cv2.LINE_AA)
            cv2.putText(output,f'WAVEMAP EDGES | {len(triangles)} triangles',(width//2+8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,245,220),1,cv2.LINE_AA)
            message = Image()
            message.header = copy.deepcopy(source.header)
            message.header.frame_id = 'visualization_comparison_panel'
            message.header.stamp.sec,message.header.stamp.nanosec = divmod(stamp,10**9)
            message.height,message.width = output.shape[:2]
            message.encoding = 'rgb8'
            message.step = output.shape[1]*3
            message.data = array.array('B',output.tobytes())
            self.publisher.publish(message)
            self.frames += 1
            self.last_stamp,self.last_output = stamp,began
            self.stats.publish(String(data=json.dumps(dict(frames=self.frames,stamp_ns=stamp,
                age_s=age,pose_bracket_ms=pose[2]/1e6,processing_ms=(time.monotonic()-began)*1e3,
                mesh_blocks=len(self.mesh),cached_triangles=len(points)//3,visible_triangles=len(triangles),
                calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                wireframe_only=True,occlusion_correct=False,recording=False))))

    rclpy.init(args=ros_args)
    node = Overlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
