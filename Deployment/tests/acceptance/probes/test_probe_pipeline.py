"""无需 ROS 的边界检查：避免旧地图计数、坏时间戳和深度编码产生误判。"""
import math
import struct
import unittest

from probe_pipeline import (TopicRecord, assess_observation, json_safe, parse_meminfo,
                            summarize_depth, summarize_integration, summarize_pose)


class PipelineProbeTests(unittest.TestCase):
    def test_absent_or_single_stamp_cannot_establish_monotonicity(self):
        record = TopicRecord()
        self.assertIsNone(record.result(1)['strictly_increasing'])
        record.observe(10, 100, 90)
        result = record.result(1)
        self.assertIsNone(result['strictly_increasing'])
        self.assertIsNone(result['source_rate_hz'])
        self.assertEqual(result['latency_s']['negative_count'], 1)

    def test_duplicate_and_backward_stamps_do_not_produce_source_rate(self):
        record = TopicRecord()
        for arrival, stamp in enumerate((1_000_000_000, 1_000_000_000, 900_000_000, 2_000_000_000)):
            record.observe(arrival * 100_000_000, stamp, 3_000_000_000)
        result = record.result(1)
        self.assertFalse(result['strictly_increasing'])
        self.assertEqual(result['duplicate_stamp_count'], 1)
        self.assertEqual(result['backward_stamp_count'], 1)
        self.assertIsNone(result['source_rate_hz'])
        self.assertAlmostEqual(result['arrival_rate_hz'], 10.0)

    def test_float_depth_preserves_padding_and_big_endian(self):
        raw = struct.pack('>ff', 1.25, math.nan) + b'xxxx' + struct.pack('>ff', math.inf, -2.0) + b'yyyy'
        summary = summarize_depth(2, 2, '32FC1', 12, True, raw)
        self.assertEqual(summary['sampled_pixels'], 4)
        self.assertEqual(summary['nan_fraction'], 0.25)
        self.assertEqual(summary['infinity_count'], 1)
        self.assertEqual(summary['positive_finite_fraction'], 0.25)
        self.assertEqual(summary['finite_min_m'], -2.0)
        self.assertEqual(summary['finite_max_m'], 1.25)

    def test_uint_depth_converts_millimetres_and_does_not_call_zero_valid(self):
        summary = summarize_depth(3, 1, '16UC1', 6, False, struct.pack('<HHH', 0, 1000, 65535))
        self.assertEqual(summary['finite_fraction'], 1.0)
        self.assertAlmostEqual(summary['positive_finite_fraction'], 2 / 3)
        self.assertAlmostEqual(summary['within_0_30_to_12_m_fraction'], 1 / 3)
        self.assertAlmostEqual(summary['finite_max_m'], 65.535)

    def test_depth_truncation_empty_size_and_bad_encoding_are_rejected(self):
        for args in ((1, 1, '32FC1', 4, False, b'123'),
                     (2, 1, '32FC1', 4, False, b'12345678'),
                     (0, 1, '32FC1', 0, False, b''),
                     (1, 1, 'mono8', 1, False, b'0')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                summarize_depth(*args)

    def test_bounded_sampling_includes_first_and_last_pixel(self):
        raw = struct.pack('<fffff', 1, 2, 3, 4, 9)
        summary = summarize_depth(5, 1, '32FC1', 20, False, raw, max_samples=2)
        self.assertEqual(summary['sampled_pixels'], 2)
        self.assertFalse(summary['full_image_sampled'])
        self.assertEqual(summary['finite_min_m'], 1)
        self.assertEqual(summary['finite_max_m'], 9)

    def test_invalid_and_zero_quaternion_are_unusable(self):
        self.assertFalse(summarize_pose((0, 0, math.nan), (0, 0, 0, 1))['usable'])
        self.assertFalse(summarize_pose((0, 0, 0), (0, 0, 0, 0))['usable'])
        self.assertTrue(summarize_pose((1, 2, 3), (0, 0, 0, 1))['usable'])

    def test_retained_map_count_alone_does_not_prove_live_pipeline(self):
        topics = {name: dict(count=1) for name in (
            '/okvis/okvis_odometry', '/odometry/estimated', '/stereo/depth',
            '/stereo/statistics', '/mapping/integrated_frames', '/mapping/occupied')}
        integration = summarize_integration([dict(elapsed_s=0.2, value=100)])
        observation = assess_observation(topics, integration)
        self.assertTrue(observation['message_and_mapping_total_condition'])
        self.assertFalse(observation['end_to_end_observed'])
        integration = summarize_integration([dict(elapsed_s=0.2, value=100), dict(elapsed_s=1.0, value=101)])
        observation = assess_observation(topics, integration)
        self.assertTrue(observation['end_to_end_observed'])
        self.assertFalse(observation['qualified'])
        topics['/odometry/estimated']['count'] = 0
        self.assertFalse(assess_observation(topics, integration)['end_to_end_observed'])

    def test_counter_restart_is_recorded_instead_of_hidden(self):
        integration = summarize_integration([dict(value=value) for value in (4, 5, 0, 1)])
        self.assertEqual(integration['decreases'], 1)
        self.assertEqual(integration['increases'], 2)
        self.assertEqual(integration['net_change'], -3)

    def test_meminfo_unit_conversion_and_strict_json_values(self):
        self.assertEqual(parse_meminfo('MemTotal: 12 kB\nHugePages_Total: 0\ninvalid\nBad: z kB'),
                         dict(MemTotal=12288, HugePages_Total=0))
        self.assertEqual(json_safe(dict(a=[math.inf, math.nan, 2.0])), dict(a=[None, None, 2.0]))


if __name__ == '__main__':
    unittest.main()
