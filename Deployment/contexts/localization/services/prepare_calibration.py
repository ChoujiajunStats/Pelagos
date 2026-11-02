#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""恢复已冻结的 ZED X 水下标定，并生成共用校正几何与运行配置。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from okvis_pose_adapter import CAMERA_FRAME, rigid_transform


ROOT = Path(__file__).resolve().parents[4]
PROFILE = ROOT / 'Deployment/config/profiles/orin-nano'
BASELINE = ROOT / 'Deployment/config/templates/okvis2-template.yaml'
FROZEN_SHA256 = {
    'cam0_underwater.yaml': '40b18994421ee2a4506ff6fe61e1f35e3db5c320fa6e660d85f3c721e0049c6a',
    'cam1_underwater.yaml': '800266419637d301a67a8c3dd441a217be494438349caf03ed8daf42e01f381b',
    'zedx_wrapper_vins.yaml': 'abaa10888080ee9e638fc892675aa2b36d732ba92b8a14f9fda6bf044483f33c',
    'zedx_stereo_imu_config.yaml': 'fe91cb2b6f3a61944163a0c9cadb53b749c2147256ff8e56529a42b276230c90',
}


class OpenCvYamlDumper(yaml.SafeDumper):
    """OpenCV FileStorage 要求序列缩进，数值数组采用其兼容的行内写法。"""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, indentless=False)


def represent_sequence(dumper, sequence):
    scalar = all(isinstance(item, (str, int, float, bool)) or item is None
                 for item in sequence)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', sequence, flow_style=scalar)


OpenCvYamlDumper.add_representer(list, represent_sequence)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_storage(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f'无法读取 OpenCV 标定文件：{path}')
    return storage


def number(node, key):
    value = node.getNode(key)
    if value.empty():
        raise ValueError(f'标定缺少字段：{key}')
    result = float(value.real())
    if not np.isfinite(result):
        raise ValueError(f'标定字段不是有限值：{key}')
    return result


def read_camera(path):
    storage = file_storage(path)
    try:
        if storage.getNode('model_type').string() != 'PINHOLE':
            raise ValueError('首轮恢复只支持已冻结的 PINHOLE 四畸变模型')
        root = storage.root()
        width, height = int(number(root, 'image_width')), int(number(root, 'image_height'))
        projection = storage.getNode('projection_parameters')
        fx, fy, cx, cy = [number(projection, key) for key in ['fx', 'fy', 'cx', 'cy']]
        distortion = storage.getNode('distortion_parameters')
        coefficients = np.array([number(distortion, key) for key in ['k1', 'k2', 'p1', 'p2']])
        intrinsics = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        if min(width, height, fx, fy) <= 0:
            raise ValueError('图像尺寸与焦距必须为正')
        return width, height, intrinsics, coefficients
    finally:
        storage.release()


def load_frozen_calibration(directory):
    for name, expected in FROZEN_SHA256.items():
        actual = sha256(directory / name)
        if actual != expected:
            raise ValueError(f'冻结标定摘要不匹配：{name}，拒绝生成配置')
    left = read_camera(directory / 'cam0_underwater.yaml')
    right = read_camera(directory / 'cam1_underwater.yaml')
    if left[:2] != right[:2] or left[:2] != (960, 600):
        raise ValueError('左右标定必须对应同一 960×600 原始图像')
    storage = file_storage(directory / 'zedx_stereo_imu_config.yaml')
    try:
        root = storage.root()
        if (int(number(root, 'image_width')), int(number(root, 'image_height'))) != left[:2]:
            raise ValueError('惯性配置与相机标定尺寸不一致')
        if int(number(root, 'estimate_extrinsic')) != 0 or int(number(root, 'estimate_td')) != 0:
            raise ValueError('旧部署必须使用固定外参与固定时间补偿')
        t_body_left = rigid_transform(storage.getNode('body_T_cam0').mat())
        t_body_right = rigid_transform(storage.getNode('body_T_cam1').mat())
        offset = number(root, 'td')
        noise = {key: number(root, key) for key in ['acc_n', 'gyr_n', 'acc_w', 'gyr_w', 'g_norm']}
    finally:
        storage.release()
    wrapper = yaml.safe_load((directory / 'zedx_wrapper_vins.yaml').read_text(encoding='utf-8'))
    settings = wrapper['/**']['ros__parameters']['general']
    if settings['serial_number'] != 42312885 or settings['grab_resolution'] != 'HD1200':
        raise ValueError('恢复文件不属于本台 ZED X 的 HD1200 采集配置')
    return left, right, t_body_left, t_body_right, offset, noise


def rectified_geometry(left, right, t_body_left, t_body_right, offset):
    width, height, k_left, d_left = left
    _, _, k_right, d_right = right
    # 原始文件给的是 IMU<-相机；OpenCV 要的是右相机<-左相机。
    t_right_left = np.linalg.inv(t_body_right) @ t_body_left
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, (width, height),
        t_right_left[:3, :3], t_right_left[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    t_left_raw_rectified = np.eye(4)
    t_left_raw_rectified[:3, :3] = r_left.T
    t_body_rectified = rigid_transform(t_body_left @ t_left_raw_rectified)
    baseline = -float(p_right[0, 3] / p_right[0, 0])
    if baseline <= 0 or not np.isclose(p_right[1, 3], 0., atol=1e-10):
        raise ValueError('首轮要求右相机位于校正后左相机 +X，且采用水平双目')
    geometry = dict(
        schema_version=1, body_is_imu=True, translation_unit='m',
        body_frame='body', world_frame='world', camera_frame=CAMERA_FRAME,
        width=width, height=height, K_left=k_left.tolist(), D_left=d_left.tolist(),
        K_right=k_right.tolist(), D_right=d_right.tolist(),
        R_left=r_left.tolist(), R_right=r_right.tolist(),
        P_left=p_left.tolist(), P_right=p_right.tolist(), baseline_m=baseline,
        image_time_offset_s=offset, T_body_left_optical=t_body_rectified.tolist(),
        T_body_left_raw_optical=t_body_left.tolist(),
        T_body_right_raw_optical=t_body_right.tolist(),
        T_right_raw_left_raw=t_right_left.tolist(), Q=q.tolist(),
        roi_left=list(roi_left), roi_right=list(roi_right),
        camera_serial_number=42312885,
        calibration_sha256=copy.deepcopy(FROZEN_SHA256),
        rectification=dict(method='cv2.stereoRectify', flags='CALIB_ZERO_DISPARITY',
                           alpha=0.0, interpolation='cv2.remap_LINEAR',
                           opencv_version=cv2.__version__),
        time_convention='t_imu = t_raw_image + image_time_offset_s',
        source_calibration='ZED X underwater; frozen real-device VINS configuration',
        imu_noise_origin='旧部署保守暂定值，非 Allan 方差实测标定',
    )
    return geometry


def verify_geometry(geometry):
    """通过原始畸变投影检查校正、深度、外参及时间方向。"""
    arrays = {key: np.asarray(geometry[key], dtype=np.float64) for key in
              ['K_left', 'D_left', 'K_right', 'D_right', 'R_left', 'R_right',
               'P_left', 'P_right', 'T_body_left_optical', 'T_body_left_raw_optical',
               'T_body_right_raw_optical', 'T_right_raw_left_raw']}
    for key in ['R_left', 'R_right']:
        rigid = np.eye(4)
        rigid[:3, :3] = arrays[key]
        rigid_transform(rigid)
    relative = arrays['T_right_raw_left_raw']
    np.testing.assert_allclose(arrays['R_right'] @ relative[:3, :3], arrays['R_left'],
                               atol=1e-10, rtol=0)
    np.testing.assert_allclose(geometry['baseline_m'], np.linalg.norm(relative[:3, 3]),
                               atol=1e-10, rtol=0)
    np.testing.assert_allclose(arrays['P_left'][:3, :3], arrays['P_right'][:3, :3],
                               atol=1e-10, rtol=0)
    random = np.random.default_rng(20261019)
    points = np.column_stack([random.uniform(-.4, .4, 200),
                              random.uniform(-.3, .3, 200), random.uniform(.8, 8., 200)])
    right_points = points @ relative[:3, :3].T + relative[:3, 3]
    left_raw, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                  arrays['K_left'], arrays['D_left'])
    right_raw, _ = cv2.projectPoints(right_points, np.zeros(3), np.zeros(3),
                                   arrays['K_right'], arrays['D_right'])
    left_pixels = cv2.undistortPointsIter(
        left_raw, arrays['K_left'], arrays['D_left'], arrays['R_left'], arrays['P_left'],
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-14)).reshape(-1, 2)
    right_pixels = cv2.undistortPointsIter(
        right_raw, arrays['K_right'], arrays['D_right'], arrays['R_right'], arrays['P_right'],
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-14)).reshape(-1, 2)
    epipolar_error = float(np.max(np.abs(left_pixels[:, 1] - right_pixels[:, 1])))
    if epipolar_error > 1e-8:
        raise ValueError(f'双目校正后极线误差过大：{epipolar_error}')
    disparity = left_pixels[:, 0] - right_pixels[:, 0]
    if not np.all(disparity > 0):
        raise ValueError('投影验证出现非正视差，需复核相机顺序')
    depth = arrays['P_left'][0, 0] * geometry['baseline_m'] / disparity
    rectified_points = points @ arrays['R_left'].T
    depth_error = float(np.max(np.abs(depth - rectified_points[:, 2])))
    np.testing.assert_allclose(depth, rectified_points[:, 2], atol=1e-9, rtol=0)
    source = arrays['T_body_left_raw_optical']
    target = rigid_transform(arrays['T_body_left_optical'])
    body_from_raw = points @ source[:3, :3].T + source[:3, 3]
    body_from_rectified = rectified_points @ target[:3, :3].T + target[:3, 3]
    np.testing.assert_allclose(body_from_raw, body_from_rectified, atol=1e-12, rtol=0)
    original_stamp = 100.
    image_delay = -geometry['image_time_offset_s']
    np.testing.assert_allclose(original_stamp - image_delay,
                               original_stamp + geometry['image_time_offset_s'], atol=1e-14)
    return dict(projected_points=len(points), maximum_epipolar_error_px=epipolar_error,
                maximum_depth_error_m=depth_error,
                maximum_body_point_error_m=float(np.max(np.abs(body_from_raw - body_from_rectified))),
                baseline_matches_raw_extrinsics=True, time_offset_sign_checked=True)


def make_okvis_config(baseline_path, left, right, t_body_left, t_body_right, offset, noise):
    text = baseline_path.read_text(encoding='utf-8')
    config = yaml.safe_load(text.removeprefix('%YAML:1.0\n'))
    for index, (camera, transform) in enumerate([(left, t_body_left), (right, t_body_right)]):
        width, height, k, distortion = camera
        item = copy.deepcopy(config['cameras'][index])
        item.update(T_SC=transform.reshape(-1).tolist(), image_dimension=[width, height],
                    distortion_coefficients=distortion.tolist(), distortion_type='radialtangential',
                    focal_length=[float(k[0, 0]), float(k[1, 1])],
                    principal_point=[float(k[0, 2]), float(k[1, 2])])
        config['cameras'][index] = item
    config['camera_parameters']['image_delay'] = -offset
    config['camera_parameters']['online_calibration']['do_extrinsics'] = False
    config['camera_parameters']['online_calibration']['do_extrinsics_final_ba'] = False
    config['imu_parameters'].update(
        T_BS=np.eye(4).reshape(-1).tolist(), sigma_g_c=noise['gyr_n'],
        sigma_a_c=noise['acc_n'], sigma_gw_c=noise['gyr_w'], sigma_aw_c=noise['acc_w'],
        g=noise['g_norm'])
    return config


def make_wavemap_config(geometry):
    p = geometry['P_left']
    return {'/**': {'ros__parameters': dict(
        width=geometry['width'], height=geometry['height'],
        fx=p[0][0], fy=p[1][1], cx=p[0][2], cy=p[1][2],
        min_range_m=.30, max_range_m=12., max_pose_gap_s=.10, min_cell_width_m=.10,
        tree_height=6, publish_every_n_frames=1, depth_topic='/stereo/depth',
        odom_topic='/odometry/estimated', world_frame=geometry['world_frame'], map_output_path='')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-dir', type=Path, default=PROFILE / 'calibration')
    parser.add_argument('--baseline-config', type=Path, default=BASELINE)
    parser.add_argument('--output-dir', type=Path, default=PROFILE)
    args = parser.parse_args()
    calibration = load_frozen_calibration(args.calibration_dir)
    left, right, t_body_left, t_body_right, offset, noise = calibration
    geometry = rectified_geometry(left, right, t_body_left, t_body_right, offset)
    geometry['verification'] = verify_geometry(geometry)
    geometry['okvis_baseline_sha256'] = sha256(args.baseline_config)
    try:
        baseline_name = args.baseline_config.resolve().relative_to(ROOT)
    except ValueError:
        baseline_name = args.baseline_config.resolve()
    geometry['okvis_baseline_path'] = str(baseline_name)
    upstream = ROOT / 'TP/Localization/okvis2/okvis_multisensor_processing/src/ThreadedSlam.cpp'
    if 'stamp - Duration(parameters_.camera.image_delay)' not in upstream.read_text(encoding='utf-8'):
        raise ValueError('上游 image_delay 应用方式发生改变，需人工复核时间补偿方向')
    geometry['okvis_time_expression'] = 'stamp_corrected = stamp - image_delay'
    okvis = make_okvis_config(args.baseline_config, *calibration)
    wavemap = make_wavemap_config(geometry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'stereo-geometry.json').write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    okvis_comment = ('%YAML:1.0\n---\n'
                     '# 原始水下双目输入：960×600 RAW，不能改接 SDK 校正图。\n'
                     '# 固定时间关系：图像时间 + 0.0087 s = IMU时间；image_delay = -0.0087。\n'
                     '# IMU 噪声沿用旧部署保守暂定值，非 Allan 方差实测标定。\n')
    (args.output_dir / 'okvis2.yaml').write_text(
        okvis_comment + yaml.dump(okvis, Dumper=OpenCvYamlDumper, sort_keys=False,
                                 allow_unicode=True), encoding='utf-8')
    (args.output_dir / 'wavemap.yaml').write_text(
        '# 相机参数与 stereo-geometry.json 的 P_left 一致。\n'
        + yaml.safe_dump(wavemap, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(json.dumps(dict(output_dir=str(args.output_dir), baseline_m=geometry['baseline_m'],
                          rectified_fx=geometry['P_left'][0][0],
                          image_time_offset_s=offset, verification=geometry['verification']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
