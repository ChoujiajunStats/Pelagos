#!/usr/bin/env python3
import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
"""离线核对真实输入布局、冻结几何、深度数值和原生去填充，不启动 ROS。"""
import ast
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'TP/Stereo/LiteAnyStereoV2'))
sys.path.insert(0, str(ROOT / 'Deployment/contexts/depth/core'))
# Load the actual upstream utility without initializing the unrelated model package.
import importlib.util
_spec = importlib.util.spec_from_file_location('las2_input_utils', ROOT / 'TP/Stereo/LiteAnyStereoV2/core/utils/utils.py')
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
InputPadder = _utils.InputPadder
from metric_depth import metric_depth


def node_helpers():
    # 直接执行节点中的纯函数，避免导入 ROS 或初始化 CUDA 模型。
    path = ROOT / 'Deployment/contexts/depth/adapters/las2_node.py'
    parsed = ast.parse(path.read_text())
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('rgb_image', 'calibration_parameters')]
    namespace = dict(np=np, cv2=cv2, math=math)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['rgb_image'], namespace['calibration_parameters']


rgb_image, calibration_parameters = node_helpers()


class StereoInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.geometry = json.loads((ROOT / 'Deployment/config/profiles/orin-nano/stereo-geometry.json').read_text())

    def test_frozen_projection_scale_and_time_sign(self):
        fx, baseline, offset_ns = calibration_parameters(self.geometry)
        self.assertAlmostEqual(fx, self.geometry['P_left'][0][0])
        self.assertAlmostEqual(fx * baseline, -self.geometry['P_right'][0][3])
        self.assertEqual(offset_ns, 8_700_000)
        original = 1_797_000_000_999_000_000
        seconds, nanoseconds = divmod(original + offset_ns, 10**9)
        self.assertEqual(seconds * 10**9 + nanoseconds - original, 8_700_000)
        self.assertLess(nanoseconds, 10**9)

    def test_bad_baseline_or_unequal_principal_points_are_rejected(self):
        wrong_baseline = copy.deepcopy(self.geometry)
        wrong_baseline['baseline_m'] *= 2
        wrong_principal = copy.deepcopy(self.geometry)
        wrong_principal['P_right'][0][2] += 1
        for geometry in (wrong_baseline, wrong_principal):
            with self.subTest(geometry=geometry['baseline_m']), self.assertRaises(ValueError):
                calibration_parameters(geometry)

    def test_nonfinite_calibration_is_rejected(self):
        for key in ('baseline_m', 'image_time_offset_s'):
            geometry = copy.deepcopy(self.geometry)
            geometry[key] = float('nan')
            with self.subTest(key=key), self.assertRaises(ValueError):
                calibration_parameters(geometry)

    def test_rgb_and_bgr_with_alpha_and_row_padding(self):
        expected = np.array([[[1, 21, 201], [2, 22, 202]],
                             [[3, 23, 203], [4, 24, 204]]], dtype=np.uint8)
        maps = np.meshgrid(np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32))
        for encoding in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
            pixels = expected[..., ::-1] if encoding.startswith('bgr') else expected.copy()
            if 'a' in encoding:
                pixels = np.concatenate([pixels, np.full((2, 2, 1), 99, dtype=np.uint8)], axis=2)
            row_bytes = pixels.shape[1] * pixels.shape[2]
            message = SimpleNamespace(width=2, height=2, encoding=encoding, step=row_bytes + 3,
                                      data=b''.join(row.tobytes() + b'xxx' for row in pixels))
            with self.subTest(encoding=encoding):
                output = rgb_image(message, maps)
                np.testing.assert_array_equal(output, expected.transpose(2, 0, 1))
                self.assertTrue(output.flags.c_contiguous)
                self.assertEqual(output.dtype, np.uint8)
                self.assertEqual(int(output.max()), 204)  # 输入保持 0 至 255；归一化由模型完成。

    def test_truncated_color_is_rejected(self):
        message = SimpleNamespace(width=2, height=2, encoding='rgb8', step=6, data=b'0' * 11)
        with self.assertRaises(ValueError):
            rgb_image(message, None)

    def test_actual_calibration_rectification_keeps_native_dimensions(self):
        geometry = self.geometry
        size = geometry['width'], geometry['height']
        for side in ('left', 'right'):
            maps = cv2.initUndistortRectifyMap(
                np.asarray(geometry['K_' + side]), np.asarray(geometry['D_' + side]),
                np.asarray(geometry['R_' + side]), np.asarray(geometry['P_' + side]), size, cv2.CV_32FC1)
            self.assertEqual(maps[0].shape, (size[1], size[0]))
            self.assertTrue(np.isfinite(maps).all())
            # 常量图只在边界可能受零填充影响；中心应保持 RGB 值和原尺寸。
            raw = np.full((size[1], size[0], 3), (23, 47, 191), dtype=np.uint8)
            message = SimpleNamespace(width=size[0], height=size[1], encoding='rgb8',
                                      step=size[0] * 3, data=raw.tobytes())
            output = rgb_image(message, maps)
            self.assertEqual(output.shape, (3, size[1], size[0]))
            np.testing.assert_array_equal(output[:, size[1] // 2, size[0] // 2], [23, 47, 191])

    def test_native_padding_and_unpadding_preserve_pixels(self):
        image = torch.arange(3 * 600 * 960, dtype=torch.float32).reshape(1, 3, 600, 960)
        padder = InputPadder(image.shape, divis_by=32)
        left, right = padder.pad(image, image.clone())
        self.assertEqual(left.shape, (1, 3, 608, 960))
        self.assertTrue(torch.equal(padder.unpad(left), image))
        self.assertTrue(torch.equal(left, right))

    def test_metric_depth_recovers_known_optical_z_and_invalid_mask(self):
        fx, baseline, _ = calibration_parameters(self.geometry)
        depths = np.array([0.5, 2., 8.], dtype=np.float32)
        disparities = fx * baseline / depths
        disparities = np.asarray(np.concatenate([disparities, [0., -1., np.nan, np.inf, 1e-40]]),
                                 dtype=np.float32).reshape(2, 4)
        with np.errstate(all='raise'):
            depth, valid = metric_depth(disparities, fx, baseline)
        np.testing.assert_allclose(depth.flat[:3], depths, rtol=1e-6)
        self.assertEqual(valid.sum(), 3)
        self.assertTrue(np.isnan(depth.flat[3:]).all())
        self.assertEqual(depth.dtype, np.float32)

    def test_nonfinite_depth_parameters_fail_before_publishing(self):
        for fx, baseline, maximum in ((np.nan, 0.12, 12.), (655., np.inf, 12.), (655., 0.12, np.inf)):
            with self.subTest(fx=fx, baseline=baseline, maximum=maximum), self.assertRaises(ValueError):
                metric_depth(np.ones((2, 2)), fx, baseline, max_depth_m=maximum)


if __name__ == '__main__':
    unittest.main(verbosity=2)
