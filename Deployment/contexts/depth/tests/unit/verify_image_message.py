#!/usr/bin/env python3
import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
"""在真实 Humble 消息类型上比较优化前后序列化字节与打包耗时。"""
import array
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def optimized_function():
    path = TaskPath(__file__).resolve().parents[2] / 'adapters/las2_node.py'
    source = ast.parse(path.read_text())
    node = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == 'Las2Node')
    method = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == 'image_message')
    namespace = dict(array=array, copy=copy, Image=Image)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return lambda values, header, encoding: namespace['image_message'](None, values, header, encoding)


def original_function(values, header, encoding):
    message = Image()
    message.header = copy.deepcopy(header)
    message.height, message.width = values.shape
    message.encoding = encoding
    message.is_bigendian = False
    message.step = values.shape[1] * values.dtype.itemsize
    message.data = values.tobytes()
    return message


def main():
    optimized = optimized_function()
    header = Header()
    header.stamp.sec = 1_797_000_000
    header.stamp.nanosec = 8_700_000
    header.frame_id = 'cam0_rect_optical'
    cases = [
        ('32FC1', np.array([[0., -0., np.nan], [np.inf, -np.inf, 4.25]], dtype=np.float32)),
        ('mono8', np.array([[0, 255, 1], [2, 3, 128]], dtype=np.uint8)),
        ('32FC1', np.linspace(0.30, 12.0, 960 * 600, dtype=np.float32).reshape(600, 960)),
        ('mono8', np.full((600, 960), 255, dtype=np.uint8)),
    ]
    results = []
    for encoding, values in cases:
        reference = original_function(values, header, encoding)
        changed = optimized(values, header, encoding)
        reference_wire = serialize_message(reference)
        changed_wire = serialize_message(changed)
        assert reference.data.tobytes() == changed.data.tobytes() == values.tobytes()
        assert reference_wire == changed_wire
        assert changed.header is not header
        elapsed = {}
        for label, function in (('原字节赋值', original_function), ('字节数组赋值', optimized)):
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                function(values, header, encoding)
                samples.append(time.perf_counter() - started)
            elapsed[label] = statistics.median(samples)
        results.append(dict(编码=encoding, 尺寸=list(values.shape), 图像字节数=values.nbytes,
                            载荷字节一致=True, 完整ROS序列化字节一致=True,
                            序列化SHA256=hashlib.sha256(reference_wire).hexdigest(),
                            打包耗时秒中位数=elapsed,
                            打包加速比=elapsed['原字节赋值'] / elapsed['字节数组赋值']))
    print(json.dumps(dict(检查通过=True, 说明='本机消息打包微测；板载完整帧耗时需重新实测', 结果=results),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
