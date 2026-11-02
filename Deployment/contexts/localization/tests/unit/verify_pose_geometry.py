#!/usr/bin/env python3
import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
"""独立核验相机杆臂、旋转组合及外参拒绝条件，不需要 ROS。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from okvis_pose_adapter import (camera_pose_from_body, load_extrinsics,
                                quaternion_rotation, rigid_transform,
                                rotation_quaternion)


def rodrigues(axis, angle):
    """用 Rodrigues 公式构造参考旋转，与四元数转换实现独立。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (np.eye(3) * np.cos(angle)
            + (1. - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross)


class PoseGeometryVerification(unittest.TestCase):
    def test_noncommuting_rotation_and_camera_lever_arm(self):
        # body 绕 world 的 Z 轴转 90°；相机在 body 中绕 X 轴转 90°。
        # 杆臂 [1,2,3] 必须转为 [-2,1,3]，不能直接加到 world 位移。
        t_body_camera = np.array([[1., 0., 0., 1.],
                                  [0., 0., -1., 2.],
                                  [0., 1., 0., 3.],
                                  [0., 0., 0., 1.]])
        position, quaternion = camera_pose_from_body(
            [10., 20., 30.], [0., 0., np.sqrt(.5), np.sqrt(.5)], t_body_camera)
        np.testing.assert_allclose(position, [8., 21., 33.], atol=1e-12, rtol=0)
        np.testing.assert_allclose(quaternion, [.5, .5, .5, .5], atol=1e-12, rtol=0)
        expected_rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        np.testing.assert_allclose(quaternion_rotation(quaternion), expected_rotation,
                                   atol=1e-12, rtol=0)

    def test_point_action_for_200_arbitrary_poses(self):
        random = np.random.default_rng(20261019)
        for _ in range(200):
            axis = random.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = random.uniform(-np.pi, np.pi)
            q_world_body = np.r_[axis * np.sin(angle / 2.), np.cos(angle / 2.)]
            r_world_body = rodrigues(axis, angle)
            r_body_camera = rodrigues(random.normal(size=3), random.uniform(-np.pi, np.pi))
            p_world_body = random.normal(size=3)
            lever_arm = random.normal(size=3)
            t_body_camera = np.eye(4)
            t_body_camera[:3, :3] = r_body_camera
            t_body_camera[:3, 3] = lever_arm
            position, quaternion = camera_pose_from_body(
                p_world_body, q_world_body, rigid_transform(t_body_camera))
            # 检查相机点经过 body、world 两次坐标变换后的实际落点。
            point = random.normal(size=3)
            reference = r_world_body @ (r_body_camera @ point + lever_arm) + p_world_body
            actual = quaternion_rotation(quaternion) @ point + position
            np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=0)

    def test_half_turns_and_near_half_turns(self):
        for axis in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., -3.]]:
            for angle in [np.pi, np.pi - 1e-10, np.pi + 1e-10]:
                reference = rodrigues(axis, angle)
                quaternion = rotation_quaternion(reference)
                self.assertTrue(np.isfinite(quaternion).all())
                np.testing.assert_allclose(quaternion_rotation(quaternion), reference,
                                           atol=1e-12, rtol=0)

    def test_bad_geometry_is_rejected(self):
        bad_transforms = [np.eye(3), np.full((4, 4), np.nan)]
        scaled = np.eye(4)
        scaled[0, 0] = 2.
        bad_transforms.append(scaled)
        reflected = np.eye(4)
        reflected[0, 0] = -1.
        bad_transforms.append(reflected)
        bad_row = np.eye(4)
        bad_row[3, 0] = .1
        bad_transforms.append(bad_row)
        for transform in bad_transforms:
            with self.assertRaises(ValueError):
                rigid_transform(transform)
        for quaternion in [[0., 0., 0., 0.], [np.nan, 0., 0., 1.]]:
            with self.assertRaises(ValueError):
                quaternion_rotation(quaternion)
        with self.assertRaises(ValueError):
            camera_pose_from_body([np.inf, 0., 0.], [0., 0., 0., 1.], np.eye(4))

    def test_configuration_requires_imu_body_and_metre_units(self):
        config = dict(schema_version=1, body_is_imu=True, translation_unit='m',
                      T_body_left_optical=np.eye(4).tolist())
        with tempfile.TemporaryDirectory(prefix='okvis-pose-') as directory:
            path = Path(directory) / 'extrinsics.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            parsed, transform = load_extrinsics(path)
            self.assertEqual(parsed['camera_frame'], 'cam0_rect_optical')
            np.testing.assert_array_equal(transform, np.eye(4))
            for key, value in [('body_is_imu', False), ('translation_unit', 'mm'),
                               ('camera_frame', '/cam0_rect_optical'),
                               ('world_frame', '/world'), ('schema_version', 2)]:
                invalid = dict(config, **{key: value})
                path.write_text(json.dumps(invalid), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_extrinsics(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
