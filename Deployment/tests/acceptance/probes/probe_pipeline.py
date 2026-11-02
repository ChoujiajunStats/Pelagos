#!/usr/bin/env python3
"""观察板载估计、深度和地图链路；消息连通不等于运动质量验收。"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import socket
import struct
import time


class NumberSummary:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None
        self.negative_count = 0

    def add(self, value):
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.negative_count += value < 0

    def result(self):
        return dict(count=self.count, min=self.minimum, max=self.maximum,
                    mean=self.total / self.count if self.count else None,
                    negative_count=self.negative_count)


class TopicRecord:
    """原始消息时间与实际接收时间分别统计，单帧不声称时间单调。"""
    def __init__(self):
        self.count = 0
        self.first_arrival_ns = None
        self.last_arrival_ns = None
        self.stamped_count = 0
        self.first_stamp_ns = None
        self.last_stamp_ns = None
        self.zero_stamp_count = 0
        self.duplicate_stamp_count = 0
        self.backward_stamp_count = 0
        self.latency_s = NumberSummary()
        self.frame_ids = set()

    def observe(self, arrival_ns, stamp_ns=None, clock_ns=None, frame_id=None):
        self.count += 1
        if self.first_arrival_ns is None:
            self.first_arrival_ns = arrival_ns
        self.last_arrival_ns = arrival_ns
        if frame_id is not None:
            self.frame_ids.add(frame_id)
        if stamp_ns is None:
            return
        if self.last_stamp_ns is not None:
            self.duplicate_stamp_count += stamp_ns == self.last_stamp_ns
            self.backward_stamp_count += stamp_ns < self.last_stamp_ns
        if self.first_stamp_ns is None:
            self.first_stamp_ns = stamp_ns
        self.last_stamp_ns = stamp_ns
        self.stamped_count += 1
        self.zero_stamp_count += stamp_ns == 0
        if clock_ns is not None:
            self.latency_s.add((clock_ns - stamp_ns) / 1e9)

    def result(self, duration_s):
        arrival_span = (self.last_arrival_ns - self.first_arrival_ns
                        if self.count > 1 else 0)
        monotonic = (self.duplicate_stamp_count == 0 and self.backward_stamp_count == 0
                     if self.stamped_count > 1 else None)
        source_span = (self.last_stamp_ns - self.first_stamp_ns
                       if self.stamped_count > 1 else 0)
        return dict(count=self.count, received_rate_hz=self.count / duration_s if duration_s > 0 else None,
                    arrival_rate_hz=(self.count - 1) * 1e9 / arrival_span if arrival_span > 0 else None,
                    stamped_count=self.stamped_count, first_stamp_ns=self.first_stamp_ns,
                    last_stamp_ns=self.last_stamp_ns, strictly_increasing=monotonic,
                    duplicate_stamp_count=self.duplicate_stamp_count,
                    backward_stamp_count=self.backward_stamp_count, zero_stamp_count=self.zero_stamp_count,
                    source_rate_hz=((self.stamped_count - 1) * 1e9 / source_span
                                    if monotonic and source_span > 0 else None),
                    latency_s=self.latency_s.result(), frame_ids=sorted(self.frame_ids))


def summarize_depth(width, height, encoding, step, is_bigendian, data, max_samples=4096):
    """按均匀栅格抽样；尊重行填充和字节序，16UC1 按毫米转为米。"""
    if width <= 0 or height <= 0 or max_samples <= 0:
        raise ValueError('图像尺寸和抽样数必须为正')
    formats = {'32FC1': ('f', 4, 1.0), '16UC1': ('H', 2, 0.001)}
    if encoding not in formats:
        raise ValueError('不支持的深度编码：' + encoding)
    code, element_bytes, scale = formats[encoding]
    if step < width * element_bytes or len(data) < height * step:
        raise ValueError('深度行步长不足或数据被截断')
    pixel_count = width * height
    count = min(pixel_count, max_samples)
    unpack = struct.Struct(('>' if is_bigendian else '<') + code).unpack_from
    finite = positive = in_range = nan_count = infinity_count = 0
    minimum = maximum = None
    for sample in range(count):
        index = sample * (pixel_count - 1) // (count - 1) if count > 1 else 0
        row, column = divmod(index, width)
        value = unpack(data, row * step + column * element_bytes)[0] * scale
        if math.isnan(value):
            nan_count += 1
        elif not math.isfinite(value):
            infinity_count += 1
        else:
            finite += 1
            positive += value > 0
            in_range += 0.30 <= value <= 12.0
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)
    return dict(pixel_count=pixel_count, sampled_pixels=count, full_image_sampled=count == pixel_count,
                finite_count=finite, finite_fraction=finite / count, nan_count=nan_count,
                nan_fraction=nan_count / count, infinity_count=infinity_count,
                positive_finite_count=positive, positive_finite_fraction=positive / count,
                within_0_30_to_12_m_count=in_range, within_0_30_to_12_m_fraction=in_range / count,
                finite_min_m=minimum, finite_max_m=maximum,
                validity_definition_zh='分别统计有限正深度及 0.30 至 12 米范围；不代表置信度')


def summarize_pose(position, quaternion):
    values = list(position) + list(quaternion)
    finite = all(math.isfinite(value) for value in values)
    norm = math.hypot(*quaternion) if all(math.isfinite(value) for value in quaternion) else None
    return dict(position_m=list(position), quaternion_xyzw=list(quaternion),
                finite=finite, quaternion_norm=norm, usable=finite and norm > 1e-12)


def summarize_integration(samples):
    values = [item['value'] for item in samples]
    increases = sum(right > left for left, right in zip(values, values[1:]))
    decreases = sum(right < left for left, right in zip(values, values[1:]))
    return dict(first=values[0] if values else None, last=values[-1] if values else None,
                maximum=max(values) if values else None, increases=increases, decreases=decreases,
                observed_live_progress=increases > 0,
                net_change=values[-1] - values[0] if values else None,
                first_value_may_be_retained=True, samples=samples)


def assess_observation(topics, integration):
    received = {name: value['count'] > 0 for name, value in topics.items()}
    base = (received['/okvis/okvis_odometry'] and received['/odometry/estimated']
            and received['/stereo/depth'] and (integration['maximum'] or 0) >= 2)
    live = bool(base and integration['observed_live_progress'])
    return dict(message_and_mapping_total_condition=bool(base), end_to_end_observed=live,
                qualified=False,
                qualification_reason_zh='消息与融合计数仅用于连通验收；仍需现场运动、轨迹、尺度、外参与地图质量测试',
                end_to_end_definition_zh='两级位姿及深度有消息、融合总数至少为 2，且观察窗口内融合计数实际增长',
                missing_topics=[name for name, present in received.items() if not present])


def parse_meminfo(text):
    result = {}
    for line in text.splitlines():
        name, separator, raw = line.partition(':')
        parts = raw.split()
        if not separator or not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        result[name] = value * 1024 if len(parts) > 1 and parts[1] == 'kB' else value
    return result


def read_resources(elapsed_s):
    result = dict(elapsed_s=elapsed_s, wall_time=datetime.datetime.now().astimezone().isoformat())
    for name, path in (('meminfo', '/proc/meminfo'), ('loadavg', '/proc/loadavg'),
                       ('uptime', '/proc/uptime'), ('cpu_stat', '/proc/stat')):
        try:
            raw = Path(path).read_text()
            if name == 'meminfo':
                result[name] = parse_meminfo(raw)
            elif name == 'cpu_stat':
                result[name] = [int(value) for value in raw.splitlines()[0].split()[1:]]
            else:
                result[name] = raw.strip()
        except (OSError, ValueError, IndexError) as error:
            result[name + '_error'] = str(error)
    processes = []
    names = ('okvis_node', 'wavemap_node', 'wavemap_navigation', 'rviz2', 'las2_node.py', 'zed_camera', 'component_container')
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = directory.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            matched = any(Path(arg).name.startswith(names) for arg in argv[:3])
            if int(directory.name) != os.getpid() and not matched:
                continue
            status = {}
            for line in directory.joinpath('status').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('Name', 'State', 'PPid', 'VmRSS', 'VmHWM', 'VmSwap', 'Threads'):
                    status[key] = value.strip()
            fields=directory.joinpath('stat').read_text().rsplit(')',1)[1].split()
            processes.append(dict(pid=int(directory.name), argv=argv, status=status,
                                  cpu_ticks=int(fields[11])+int(fields[12]),clock_ticks_per_second=os.sysconf('SC_CLK_TCK')))
        except (OSError, ValueError):
            continue  # 进程可在读取期间退出。
    result['processes'] = processes
    result['meminfo_units_zh'] = '标记 kB 的容量已转为字节，其余字段保留原计数；进程 status 保留内核原文单位'
    return result


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser._optionals.title = '参数'
    parser.add_argument('-h', '--help', action='help', help='显示帮助并退出')
    parser.add_argument('--seconds', type=float, default=30.0, help='观察秒数，必须大于零')
    parser.add_argument('--output', type=Path, required=True, help='验收 JSON 输出路径')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds 必须是有限正数')
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image, PointCloud2
        from std_msgs.msg import String, UInt64
    except ImportError as error:
        parser.exit(2, '请先加载 ROS 2 Humble 环境，再用系统 Python 运行：' + str(error) + '\n')

    rclpy.init(args=[])
    node = Node('dapeng_pipeline_acceptance')
    topic_names = ('/okvis/okvis_odometry', '/odometry/estimated', '/stereo/depth',
                   '/stereo/statistics', '/mapping/integrated_frames', '/mapping/occupied')
    records = {name: TopicRecord() for name in topic_names}
    poses = {name: dict(child_frame_ids=set(), finite_count=0, usable_count=0, first=None, last=None)
             for name in topic_names[:2]}
    depth = dict(metadata_first=None, metadata_last=None, encodings=set(), dimensions=set(),
                 sampled_frame_count=0, samples=[], decode_error_count=0, last_sample_elapsed_s=None)
    statistics = []
    integrations = []
    occupied = dict(first=None, last=None, maximum_points=0, nonempty_count=0)
    subscriptions = []
    started = time.monotonic_ns()
    resources = [read_resources(0.0)]
    interrupted = False

    def timing(topic, message, source_stamp=None):
        arrival = time.monotonic_ns()
        header = getattr(message, 'header', None)
        if header is not None:
            source_stamp = header.stamp.sec * 10**9 + header.stamp.nanosec
        records[topic].observe(arrival, source_stamp, node.get_clock().now().nanoseconds,
                               header.frame_id if header is not None else None)
        return (arrival - started) / 1e9

    def receive_pose(topic, message):
        timing(topic, message)
        p, q = message.pose.pose.position, message.pose.pose.orientation
        sample = summarize_pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        summary = poses[topic]
        summary['child_frame_ids'].add(message.child_frame_id)
        summary['finite_count'] += sample['finite']
        summary['usable_count'] += sample['usable']
        summary['first'] = sample if summary['first'] is None else summary['first']
        summary['last'] = sample

    def receive_depth(message):
        elapsed = timing('/stereo/depth', message)
        metadata = dict(width=message.width, height=message.height, encoding=message.encoding,
                        step=message.step, frame_id=message.header.frame_id,
                        is_bigendian=bool(message.is_bigendian), data_bytes=len(message.data))
        depth['metadata_first'] = metadata if depth['metadata_first'] is None else depth['metadata_first']
        depth['metadata_last'] = metadata
        depth['encodings'].add(message.encoding)
        depth['dimensions'].add((message.width, message.height))
        if depth['last_sample_elapsed_s'] is not None and elapsed - depth['last_sample_elapsed_s'] < 2.0:
            return
        depth['last_sample_elapsed_s'] = elapsed
        depth['sampled_frame_count'] += 1
        sample = dict(elapsed_s=elapsed, stamp_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                      metadata=metadata)
        try:
            sample.update(summarize_depth(message.width, message.height, message.encoding, message.step,
                                          message.is_bigendian, message.data))
        except (ValueError, struct.error, TypeError) as error:
            sample['decode_error'] = str(error)
            depth['decode_error_count'] += 1
        depth['samples'].append(sample)
        if len(depth['samples']) > 16:
            del depth['samples'][1]  # 保留首帧和最近 15 个抽样帧。

    def receive_statistics(message):
        sample = dict(raw=message.data, parsed=None, parse_error=None)
        stamp_ns = None
        try:
            sample['parsed'] = json.loads(message.data)
            parsed = sample['parsed']
            if isinstance(parsed, dict) and type(parsed.get('stamp_ns')) is int:
                stamp_ns = parsed['stamp_ns']
        except (ValueError, TypeError) as error:
            sample['parse_error'] = str(error)
        sample['elapsed_s'] = timing('/stereo/statistics', message, stamp_ns)
        statistics.append(sample)

    def receive_integration(message):
        integrations.append(dict(elapsed_s=timing('/mapping/integrated_frames', message), value=int(message.data)))

    def receive_occupied(message):
        timing('/mapping/occupied', message)
        points = int(message.width) * int(message.height)
        sample = dict(width=message.width, height=message.height, points=points,
                      frame_id=message.header.frame_id, data_bytes=len(message.data),
                      point_step=message.point_step, row_step=message.row_step)
        occupied['first'] = sample if occupied['first'] is None else occupied['first']
        occupied['last'] = sample
        occupied['maximum_points'] = max(occupied['maximum_points'], points)
        occupied['nonempty_count'] += points > 0

    transient_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for topic in topic_names[:2]:
        subscriptions.append(node.create_subscription(
            Odometry, topic, lambda message, topic=topic: receive_pose(topic, message), qos_profile_sensor_data))
    for kind, topic, callback, qos in (
            (Image, '/stereo/depth', receive_depth, qos_profile_sensor_data),
            (String, '/stereo/statistics', receive_statistics, 10),
            (UInt64, '/mapping/integrated_frames', receive_integration, transient_qos),
            (PointCloud2, '/mapping/occupied', receive_occupied, transient_qos)):
        subscriptions.append(node.create_subscription(kind, topic, callback, qos))

    publishers = {}
    try:
        while (time.monotonic_ns() - started) / 1e9 < args.seconds and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = (time.monotonic_ns() - started) / 1e9
            if elapsed - resources[-1]['elapsed_s'] >= 5.0:
                resources.append(read_resources(elapsed))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        duration = (time.monotonic_ns() - started) / 1e9
        resources.append(read_resources(duration))
        for topic in topic_names:
            try:
                publishers[topic] = [dict(node_name=info.node_name, node_namespace=info.node_namespace,
                    reliability=str(info.qos_profile.reliability), durability=str(info.qos_profile.durability),
                    depth=info.qos_profile.depth) for info in node.get_publishers_info_by_topic(topic)]
            except Exception as error:
                publishers[topic] = dict(discovery_error=str(error))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    topics = {topic: record.result(duration) for topic, record in records.items()}
    integration_summary = summarize_integration(integrations)
    for summary in poses.values():
        summary['child_frame_ids'] = sorted(summary['child_frame_ids'])
    depth['encodings'] = sorted(depth['encodings'])
    depth['dimensions'] = sorted(depth['dimensions'])
    source_frames = topics['/stereo/depth']['frame_ids']
    pose_frames = poses['/odometry/estimated']['child_frame_ids']
    warnings = []
    if any(record['latency_s']['negative_count'] for record in topics.values()):
        warnings.append('存在负延迟：请核对源时间戳与 ROS 时钟；本报告不据此计算端到端处理耗时')
    if integration_summary['decreases']:
        warnings.append('融合计数在观察期间回退，可能存在建图进程重启或多个发布者')
    if source_frames and pose_frames and source_frames != pose_frames:
        warnings.append('深度 frame_id 与估计位姿 child_frame_id 不一致，需核对相机外参变换')
    if any(value['usable_count'] != topics[topic]['count'] for topic, value in poses.items()):
        warnings.append('位姿包含非有限值或零范数四元数')
    if depth['decode_error_count']:
        warnings.append('部分抽样深度帧无法解码')
    result = dict(schema_version=1, hostname=socket.gethostname(), duration_s=duration, interrupted=interrupted,
                  completed_at=datetime.datetime.now().astimezone().isoformat(), topics=topics,
                  observation=assess_observation(topics, integration_summary), poses=poses, depth=depth,
                  stereo_statistics=statistics, integration=integration_summary, occupied=occupied,
                  publishers=publishers, resources=resources, warnings_zh=warnings,
                  notes_zh=['延迟为接收时 ROS 时钟减原始采集时间戳，包含计算、调度和传输',
                            '深度每约两秒抽样最多 4096 个像素，仅保存首帧与最近 15 个抽样摘要',
                            '统计 JSON 完整保存；未解析地图点坐标；位姿有限不代表定位正确',
                            '地图计数使用可靠持久订阅；单个历史计数不能证明观察窗口内继续建图'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(端到端连通已观测=result['observation']['end_to_end_observed'],
                          运动质量已验收=False, 融合计数=integration_summary['last'],
                          观察期间融合增长=integration_summary['observed_live_progress'],
                          消息数={name: record['count'] for name, record in topics.items()},
                          提示=warnings, 记录=str(args.output)), ensure_ascii=False, indent=2))
    return 130 if interrupted else (0 if result['observation']['end_to_end_observed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
