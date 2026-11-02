"""Verify uncapped mapping measurements and the existing display contract offline."""
import array
import ast
import copy
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

DEPTH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPTH_ROOT/'core'))
from metric_depth import mapping_depth, metric_depth


class MappingDepthTests(unittest.TestCase):
    fx = 417.0321193091858
    baseline = .145

    def test_far_measurements_retain_original_depth_without_clamping(self):
        expected = np.array([.3, 1., 8., 8.5, 12., 13., 30., 100.], dtype=np.float32)
        disparity = self.fx*self.baseline/expected
        mapping, mapping_valid = mapping_depth(disparity, self.fx, self.baseline)
        display, display_valid = metric_depth(disparity, self.fx, self.baseline)
        np.testing.assert_allclose(mapping, expected, rtol=1e-6)
        self.assertTrue(mapping_valid.all())
        self.assertEqual(mapping.dtype, np.float32)
        np.testing.assert_allclose(display[:5], expected[:5], rtol=1e-6)
        self.assertTrue(np.isnan(display[5:]).all())
        np.testing.assert_array_equal(display_valid, [True]*5+[False]*3)

    def test_invalid_subminimum_and_nonfinite_derived_depth_remain_nan(self):
        disparity = np.array([0., -1., np.nan, np.inf, -np.inf,
                              self.fx*self.baseline/.2, np.finfo(np.float32).smallest_subnormal],
                             dtype=np.float32)
        original = disparity.copy()
        with np.errstate(all='raise'):
            depth, valid = mapping_depth(disparity, self.fx, self.baseline)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(depth).all())
        np.testing.assert_array_equal(disparity, original)

    def test_mapping_minimum_is_explicit_and_has_no_upper_limit(self):
        expected = np.array([.5, 1., 20.], dtype=np.float32)
        depth, valid = mapping_depth(self.fx*self.baseline/expected,
                                     self.fx, self.baseline, min_depth_m=1.)
        np.testing.assert_array_equal(valid, [False, True, True])
        self.assertTrue(np.isnan(depth[0]))
        np.testing.assert_allclose(depth[1:], expected[1:], rtol=1e-6)

    def test_existing_metric_depth_custom_bounds_still_apply(self):
        expected = np.array([.5, 1., 20., 30.], dtype=np.float32)
        depth, valid = metric_depth(self.fx*self.baseline/expected,
                                    self.fx, self.baseline, min_depth_m=1., max_depth_m=20.)
        np.testing.assert_array_equal(valid, [False, True, True, False])
        self.assertTrue(np.isnan(depth[[0, 3]]).all())
        np.testing.assert_allclose(depth[1:3], expected[1:3], rtol=1e-6)

    def test_mapping_rejects_invalid_geometry_and_minimum(self):
        cases = [(0., .145, .3), (-1., .145, .3), (np.nan, .145, .3),
                 (417., 0., .3), (417., np.inf, .3),
                 (417., .145, 0.), (417., .145, -1.), (417., .145, np.nan)]
        for fx, baseline, minimum in cases:
            with self.subTest(fx=fx, baseline=baseline, minimum=minimum), self.assertRaises(ValueError):
                mapping_depth(np.ones((2, 2)), fx, baseline, min_depth_m=minimum)

    def test_actual_publication_keeps_same_capture_header_and_display_mask(self):
        # Execute the actual publishing/packing methods without importing ROS,
        # Torch or CUDA. This checks both payloads and the retained legacy mask.
        source = DEPTH_ROOT/'adapters/las2_node.py'
        tree = ast.parse(source.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'Las2Node')
        methods = [item for item in node.body if isinstance(item, ast.FunctionDef)
                   and item.name in ('image_message', 'publish_depths')]
        namespace = dict(array=array, copy=copy, Image=SimpleNamespace,
                         metric_depth=metric_depth, mapping_depth=mapping_depth)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
        display_messages, mapping_messages = [], []
        fake = SimpleNamespace(fx=self.fx, baseline=self.baseline,
                               depth_pub=SimpleNamespace(publish=display_messages.append),
                               mapping_depth_pub=SimpleNamespace(publish=mapping_messages.append))
        fake.image_message = MethodType(namespace['image_message'], fake)
        header = SimpleNamespace(stamp=SimpleNamespace(sec=1793536000, nanosec=123456789),
                                 frame_id='cam0_rect_optical')
        expected = np.array([[1., 9.], [13., np.nan]], dtype=np.float32)
        disparity = self.fx*self.baseline/expected
        valid = namespace['publish_depths'](fake, disparity, header)
        np.testing.assert_array_equal(valid, [[True, True], [False, False]])
        self.assertEqual(len(display_messages), 1)
        self.assertEqual(len(mapping_messages), 1)
        for message in display_messages+mapping_messages:
            self.assertEqual(message.header, header)
            self.assertIsNot(message.header, header)
            self.assertEqual((message.height, message.width, message.encoding, message.step),
                             (2, 2, '32FC1', 8))
        display = np.frombuffer(display_messages[0].data, dtype=np.float32).reshape(2, 2)
        mapping = np.frombuffer(mapping_messages[0].data, dtype=np.float32).reshape(2, 2)
        np.testing.assert_allclose(mapping, expected, rtol=1e-6, equal_nan=True)
        self.assertTrue(np.isnan(display[1]).all())
        np.testing.assert_allclose(display[0], expected[0], rtol=1e-6)
        self.assertIsNot(display_messages[0].header, mapping_messages[0].header)


if __name__ == '__main__':
    unittest.main(verbosity=2)
