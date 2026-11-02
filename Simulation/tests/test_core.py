"""Focused checks for the two calculations that affect the displayed comparisons."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from depth_metrics import compare_depths
from geometry import compose, inverse, optical_transform, rotate


class ComparisonChecks(unittest.TestCase):
    def test_camera_axes_and_gauge_alignment(self):
        camera = dict(xyz_frd=[0.16, -0.0725, 0.15],
                      rpy_frd=[math.pi / 2, 0.0, math.pi / 2])
        body_camera = optical_transform(camera)
        np.testing.assert_allclose(rotate(body_camera[1], (0, 0, 1)), (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(rotate(body_camera[1], (1, 0, 0)), (0, -1, 0), atol=1e-12)
        np.testing.assert_allclose(rotate(body_camera[1], (0, 1, 0)), (0, 0, -1), atol=1e-12)
        gt = ((3., -2., 5.), (0., 0., math.sqrt(.5), math.sqrt(.5)))
        estimate = ((2., 1., .5), (0., 0., 0., 1.))
        alignment = compose(gt, inverse(estimate))
        aligned = compose(alignment, estimate)
        np.testing.assert_allclose(aligned[0], gt[0], atol=1e-12)
        np.testing.assert_allclose(aligned[1], gt[1], atol=1e-12)

    def test_depth_error_excludes_invalid_pixels(self):
        gt = np.array([[1., 2., 0., np.nan]], dtype=np.float32)
        prediction = np.array([[1.5, 1., 3., 4.]], dtype=np.float32)
        _, _, valid, _, result = compare_depths(gt, prediction, .3, 10.)
        self.assertEqual(valid.tolist(), [[True, True, False, False]])
        self.assertEqual(result['samples'], 2)
        self.assertAlmostEqual(result['mae_m'], .75)
        self.assertAlmostEqual(result['rmse_m'], math.sqrt(.625), places=6)


if __name__ == '__main__':
    unittest.main()
