#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""以冻结水下标定校正原始双目，复用 LAS2 原生推理和米制深度转换。"""
import argparse
import array
import copy
import hashlib
import json
import math
import threading
import time
from pathlib import Path
import sys

import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from message_filters import Subscriber, TimeSynchronizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric_depth import mapping_depth, metric_depth
from core.models import build_model, load_model_weights
from core.utils.utils import InputPadder


def rgb_image(msg, maps):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if (channels is None or msg.width <= 0 or msg.height <= 0
            or msg.step < msg.width * channels or len(msg.data) < msg.step * msg.height):
        raise ValueError(f'不支持的彩色输入编码或行长度：{msg.encoding}')
    pixels = np.ndarray((msg.height, msg.width, channels), dtype=np.uint8,
                        buffer=msg.data, strides=(msg.step, channels, 1))
    rgb = pixels[:, :, :3]
    if msg.encoding.startswith('bgr'):
        rgb = rgb[:, :, ::-1]
    rgb = cv2.remap(rgb, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def calibration_parameters(calibration):
    """运行前核对校正投影与米制基线，避免错误文件静默改变尺度。"""
    for key in ('width', 'height'):
        if type(calibration[key]) is not int or calibration[key] <= 0:
            raise ValueError('标定图像尺寸必须为正整数')
    left = np.asarray(calibration['P_left'], dtype=np.float64)
    right = np.asarray(calibration['P_right'], dtype=np.float64)
    if (left.shape != (3, 4) or right.shape != (3, 4)
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or left[0, 0] <= 0 or left[1, 1] <= 0
            or not np.allclose(left[:, :3], right[:, :3], rtol=0, atol=1e-8)
            or not np.allclose(left[:, 3], 0, rtol=0, atol=1e-8)
            or not np.allclose(right[1:, 3], 0, rtol=0, atol=1e-8)):
        raise ValueError('需要具有共同内参和主点的水平校正双目投影')
    baseline = float(calibration['baseline_m'])
    offset = float(calibration['image_time_offset_s'])
    if (not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(offset)
            or not math.isclose(baseline, -right[0, 3] / right[0, 0], rel_tol=1e-7, abs_tol=1e-9)):
        raise ValueError('米制基线与校正投影不一致，或时间补偿无效')
    if not isinstance(calibration['camera_frame'], str) or not calibration['camera_frame'].strip():
        raise ValueError('校正相机坐标帧不能为空')
    for side in ('left', 'right'):
        for prefix in ('K_', 'D_', 'R_'):
            if not np.all(np.isfinite(np.asarray(calibration[prefix + side], dtype=np.float64))):
                raise ValueError('原始标定或校正旋转包含非有限值')
    return float(left[0, 0]), baseline, round(offset * 1e9)


class Las2Node(Node):
    def __init__(self, args):
        super().__init__('las2_stereo')
        if not torch.cuda.is_available():
            raise RuntimeError('LAS2 在线基线需要板载 CUDA')
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        self.calibration = json.loads(args.calibration.read_text())
        self.calibration_sha256 = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
        self.fx, self.baseline, self.offset_ns = calibration_parameters(self.calibration)
        self.maps = []
        for side in ('left', 'right'):
            self.maps.append(cv2.initUndistortRectifyMap(
                np.asarray(self.calibration['K_' + side], dtype=np.float64),
                np.asarray(self.calibration['D_' + side], dtype=np.float64),
                np.asarray(self.calibration['R_' + side], dtype=np.float64),
                np.asarray(self.calibration['P_' + side], dtype=np.float64),
                (self.calibration['width'], self.calibration['height']), cv2.CV_32FC1))
        self.model = build_model('las2', fnet_pretrained=False, model_size=args.variant, max_disp=192)
        load_model_weights(self.model, torch.load(args.checkpoint, map_location='cpu'), strict=True)
        self.model.to('cuda').eval()
        self.lock = threading.Condition()
        self.pending = None
        self.stopping = False
        self.frames = 0
        self.dropped = 0
        self.error = None
        self.max_rate = args.max_rate
        self.period = 1.0 / args.max_rate if args.max_rate is not None else 0.0
        self.depth_pub = self.create_publisher(Image, '/stereo/depth', qos_profile_sensor_data)
        # Approved 2026-11-01: same measured disparity, uncapped finite far Z.
        self.mapping_depth_pub = self.create_publisher(Image, '/stereo/mapping_depth', qos_profile_sensor_data)
        self.disparity_pub = self.create_publisher(Image, '/stereo/disparity', qos_profile_sensor_data)
        self.valid_pub = self.create_publisher(Image, '/stereo/valid', qos_profile_sensor_data)
        self.stats_pub = self.create_publisher(String, '/stereo/statistics', 10)
        self.camera_pub = self.create_publisher(CameraInfo, '/stereo/camera_info', qos_profile_sensor_data)
        self.image_subs = []
        for side in ('left', 'right'):
            base = f'/zed/zed_node/{side}/color/raw'
            self.image_subs.append(Subscriber(self, Image, base + '/image', qos_profile=qos_profile_sensor_data))
        self.sync = TimeSynchronizer(self.image_subs, 5)
        self.sync.registerCallback(self.receive)
        self.worker = threading.Thread(target=self.infer, daemon=True)
        self.worker.start()
        self.create_timer(10.0, self.report)
        rate_description = f'最大处理频率 {args.max_rate} Hz' if args.max_rate is not None else '未设置处理频率上限'
        self.get_logger().info(f'LAS2-{args.variant.upper()} 模型已就绪，{rate_description}')

    def receive(self, left, right):
        with self.lock:
            if self.pending is not None:
                self.dropped += 1
            self.pending = left, right
            self.lock.notify()

    def image_message(self, values, header, encoding):
        msg = Image()
        msg.header = copy.deepcopy(header)
        msg.height, msg.width = values.shape
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = values.shape[1] * values.dtype.itemsize
        # Humble 对 bytes 逐元素做两遍断言；array('B') 使用生成消息的直接赋值路径。
        msg.data = array.array('B', values.tobytes())
        return msg

    def publish_depths(self, disparity, header):
        """Publish display and mapping depths with the same acquisition header."""
        depth, valid = metric_depth(disparity, self.fx, self.baseline)
        map_depth, _ = mapping_depth(disparity, self.fx, self.baseline)
        self.depth_pub.publish(self.image_message(depth, header, '32FC1'))
        self.mapping_depth_pub.publish(self.image_message(map_depth, header, '32FC1'))
        return valid

    def infer(self):
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self.pending is not None or self.stopping)
                if self.stopping:
                    return
                left, right = self.pending
                self.pending = None
            started = time.monotonic()
            try:
                expected = self.calibration['width'], self.calibration['height']
                if (left.width, left.height) != expected or (right.width, right.height) != expected:
                    raise ValueError('双目原图尺寸与冻结水下标定不一致')
                if (left.header.frame_id != 'zed_left_camera_frame_optical' or
                        right.header.frame_id != 'zed_right_camera_frame_optical'):
                    raise ValueError('双目原始光学相机坐标帧不匹配')
                source_stamp_ns = left.header.stamp.sec * 10**9 + left.header.stamp.nanosec
                if source_stamp_ns <= getattr(self, 'last_source_stamp_ns', -1):
                    raise ValueError('原图时间戳重复或倒退，结束本次深度运行')
                lt = torch.from_numpy(rgb_image(left, self.maps[0])).float()[None].to('cuda')
                rt = torch.from_numpy(rgb_image(right, self.maps[1])).float()[None].to('cuda')
                padder = InputPadder(lt.shape, divis_by=32)
                lt, rt = padder.pad(lt, rt)
                torch.cuda.synchronize()
                inference_started = time.monotonic()
                with torch.inference_mode():
                    disparity = self.model(lt, rt, max_disp=192, test_mode=True)
                torch.cuda.synchronize()
                inference_s = time.monotonic() - inference_started
                disparity = padder.unpad(disparity.float()).detach().cpu().numpy().squeeze().astype(np.float32)
                if disparity.shape != (expected[1], expected[0]):
                    raise ValueError('模型去填充后的视差尺寸与冻结标定不一致')
                header = copy.deepcopy(left.header)
                header.frame_id = self.calibration['camera_frame']
                header.stamp.sec, header.stamp.nanosec = divmod(source_stamp_ns + self.offset_ns, 10**9)
                valid = self.publish_depths(disparity, header)
                if self.disparity_pub.get_subscription_count():
                    self.disparity_pub.publish(self.image_message(disparity, header, '32FC1'))
                if self.valid_pub.get_subscription_count():
                    self.valid_pub.publish(self.image_message(valid.astype(np.uint8) * 255, header, 'mono8'))
                camera_info = CameraInfo()
                camera_info.header = copy.deepcopy(header)
                camera_info.width, camera_info.height = expected
                camera_info.distortion_model = 'plumb_bob'
                camera_info.d = [0.] * 5
                camera_info.k = np.asarray(self.calibration['P_left'])[:, :3].reshape(-1).tolist()
                camera_info.r = np.eye(3).reshape(-1).tolist()
                camera_info.p = np.asarray(self.calibration['P_left']).reshape(-1).tolist()
                self.camera_pub.publish(camera_info)
                self.frames += 1
                self.last_source_stamp_ns = source_stamp_ns
                statistics = dict(frames=self.frames, superseded_pairs=self.dropped, inference_s=inference_s,
                                  max_rate_hz=self.max_rate,
                                  processing_s=time.monotonic() - started, valid_fraction=float(valid.mean()),
                                  stamp_ns=source_stamp_ns + self.offset_ns, source_stamp_ns=source_stamp_ns,
                                  image_time_offset_s=self.offset_ns / 1e9,
                                  fx=self.fx, baseline_m=self.baseline, frame_id=header.frame_id,
                                  calibration_sha256=self.calibration_sha256,
                                  depth_definition='CAMERA_OPTICAL_Z', confidence_available=False)
                self.stats_pub.publish(String(data=json.dumps(statistics)))
                self.last_stats = statistics
            except Exception as exc:
                self.error = str(exc)
                self.get_logger().error(self.error)
                return
            if self.period > 0:
                with self.lock:
                    self.lock.wait_for(lambda: self.stopping,
                                       timeout=max(0, self.period - (time.monotonic() - started)))

    def report(self):
        if self.error:
            raise RuntimeError(self.error)
        self.get_logger().info(json.dumps(getattr(self, 'last_stats', dict(frames=0))))

    def close(self):
        with self.lock:
            self.stopping = True
            self.lock.notify_all()
        self.worker.join(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--variant', choices=('s', 'm'), default='s')
    parser.add_argument('--max-rate', type=float, default=None,
                        help='可选最大处理频率（Hz）；省略时不限速')
    args, ros_args = parser.parse_known_args()
    if args.max_rate is not None and (not math.isfinite(args.max_rate) or args.max_rate <= 0):
        parser.error('--max-rate 必须为有限正数')
    rclpy.init(args=ros_args)
    node = Las2Node(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
