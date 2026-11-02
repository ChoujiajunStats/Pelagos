"""Check actual retained SensorWindow behavior using recorded alignment timing."""
import ast
from collections import deque
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'adapters'))
from sensor_alignment import aligned_sensor_window


def load_source_window():
    paths = [Path('/stonefish-workspace/controller/uw_controller/checkpoint.py'),
             Path('/home/hong/Desktop/MY_Stonefish_ws/stonefish_slam_ws/controller/uw_controller/checkpoint.py')]
    source = next(path for path in paths if path.is_file())
    parsed = ast.parse(source.read_text())
    cls = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == 'SensorWindow')
    namespace = dict(np=np, deque=deque, time=time)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['SensorWindow'], source


SourceWindow, SOURCE_PATH = load_source_window()
AlignedWindow = aligned_sensor_window(SourceWindow)


class SensorAlignmentChecks(unittest.TestCase):
    def test_recorded_pressure_alignment_retains_old_causal_value(self):
        now = 280_187_780_246_807
        original, aligned = SourceWindow(), AlignedWindow()
        for window in (original, aligned):
            window.append(531.45, [64848.1105468668], now-191_759_775)
            window.append(531.60, [64850.], now-40_624_351)
        self.assertIsNone(original.at(531.49, now))
        np.testing.assert_array_equal(aligned.at(531.49, now), [64848.1105468668])
        diagnostic = aligned.alignment_diagnostic(531.49, now)
        self.assertTrue(diagnostic['ready'])
        self.assertAlmostEqual(diagnostic['causal_source_gap_s'], .04)
        self.assertAlmostEqual(diagnostic['selected_receive_age_s'], .191759775)
        self.assertAlmostEqual(diagnostic['latest_receive_age_s'], .040624351)
        self.assertEqual(diagnostic['causal_stamp_s'], 531.45)
        self.assertEqual(diagnostic['latest_stamp_s'], 531.60)
        self.assertNotIn('value', diagnostic)
        json.dumps(diagnostic, allow_nan=False)

    def test_stale_entire_stream_rejects(self):
        window = AlignedWindow()
        window.append(10., [1.], 1_000_000_000)
        self.assertIsNone(window.at(10., 1_150_000_001))
        self.assertEqual(window.alignment_diagnostic(10., 1_150_000_001)['reason'], 'latest_stream_stale')

    def test_future_samples_prove_liveness_but_never_supply_values(self):
        window = AlignedWindow()
        window.append(10.01, [99.], 1_000_000_000)
        self.assertIsNone(window.at(10., 1_010_000_000))
        self.assertEqual(window.alignment_diagnostic(10., 1_010_000_000)['reason'], 'no_causal_sample')
        # Even a sub-nanosecond future stamp is not selected.
        self.assertIsNone(window.at(10.01-1e-10, 1_010_000_000))

    def test_acquisition_gap_remains_limited_even_when_stream_is_live(self):
        window = AlignedWindow()
        window.append(10., [1.], 900_000_000)
        window.append(10.20, [2.], 1_000_000_000)
        self.assertIsNone(window.at(10.16, 1_010_000_000))
        self.assertEqual(window.alignment_diagnostic(10.16, 1_010_000_000)['reason'], 'causal_source_gap_exceeded')

    def test_replayed_or_reversed_append_does_not_renew_stream(self):
        window = AlignedWindow()
        self.assertTrue(window.append(10., [1.], 1_000_000_000))
        self.assertFalse(window.append(10., [99.], 1_200_000_000))
        self.assertFalse(window.append(9., [99.], 1_200_000_000))
        self.assertEqual(len(window.samples), 1)
        self.assertIsNone(window.at(10., 1_200_000_000))
        self.assertAlmostEqual(window.alignment_diagnostic(10., 1_200_000_000)['latest_receive_age_s'], .2)

    def test_negative_receive_clock_is_rejected(self):
        latest_future = AlignedWindow()
        latest_future.append(10., [1.], 1_000_000_001)
        self.assertIsNone(latest_future.at(10., 1_000_000_000))
        self.assertEqual(latest_future.alignment_diagnostic(10., 1_000_000_000)['reason'], 'latest_receive_clock_negative')
        selected_future = AlignedWindow()
        selected_future.append(10., [1.], 1_000_000_001)
        selected_future.append(10.1, [2.], 999_000_000)
        self.assertIsNone(selected_future.at(10., 1_000_000_000))
        self.assertEqual(selected_future.alignment_diagnostic(10., 1_000_000_000)['reason'], 'selected_receive_clock_negative')

    def test_source_default_limit_is_derived_and_explicit_limit_is_respected(self):
        original_default = inspect.signature(SourceWindow.at).parameters['max_age'].default
        self.assertEqual(inspect.signature(AlignedWindow.at).parameters['max_age'].default, original_default)
        self.assertEqual(inspect.signature(AlignedWindow.alignment_diagnostic).parameters['max_age'].default, original_default)
        window = AlignedWindow()
        window.append(1., [1.], 1_000_000_000)
        self.assertIsNotNone(window.at(1., 1_150_000_000))
        self.assertIsNone(window.at(1., 1_100_000_001, max_age=.1))

    def test_segments_remain_source_piecewise_causal_values(self):
        window = AlignedWindow()
        window.append(10., [1.], 700_000_000)
        window.append(10.05, [2.], 800_000_000)
        window.append(10.10, [3.], 1_000_000_000)
        segments = window.segments(10., 10.10, 1_010_000_000)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0][0], .05)
        self.assertAlmostEqual(segments[1][0], .05)
        np.testing.assert_array_equal(segments[0][1], [1.])
        np.testing.assert_array_equal(segments[1][1], [2.])
        self.assertIsNone(window.segments(9.9, 10., 1_010_000_000))

    def test_empty_invalid_request_and_invalid_limit_are_not_ready(self):
        window = AlignedWindow()
        self.assertEqual(window.alignment_diagnostic(1., 1)['reason'], 'no_samples')
        self.assertIsNone(window.at(float('nan'), 1))
        for maximum in [0., -1., float('nan'), float('inf')]:
            self.assertIsNone(window.at(1., 1, max_age=maximum))
        json.dumps(window.alignment_diagnostic(float('nan'), 1), allow_nan=False)


if __name__ == '__main__':
    print('Retained source SHA256:', hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest())
    unittest.main(verbosity=2)
