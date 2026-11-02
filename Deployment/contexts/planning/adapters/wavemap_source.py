import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Low-rate immutable Wavemap occupancy snapshots for the existing planner bridge."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zlib

import numpy as np
from scipy.ndimage import distance_transform_edt
from std_msgs.msg import UInt64
from std_srvs.srv import Trigger


class WavemapSource:
    def __init__(self, bridge, qos):
        self.bridge = bridge
        self.frames = 0
        self.last_exported = -1
        self.fusion_arrival = 0.
        self.future = None
        self.job = None
        self.request_time = 0.
        self.epoch = 0
        self.temp = tempfile.TemporaryDirectory(prefix='omniplanner-wavemap-')
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.client = bridge.create_client(Trigger, '/save_map')
        bridge.create_subscription(UInt64, '/mapping/integrated_frames', self.on_frame, qos)
        bridge.create_timer(.1, self.poll)

    def on_frame(self, message):
        if message.data < self.frames:
            self.epoch += 1
            self.bridge.last_map = None
            self.bridge.map_arrival = 0.
            self.bridge.clear_path()
            self.bridge.on_clear(None, Trigger.Response())
            self.last_exported = -1
        self.frames = message.data
        self.fusion_arrival = time.monotonic()

    def request(self):
        b = self.bridge
        if (self.future is not None or self.job is not None or not b.fresh()
                or not self.client.service_is_ready() or self.frames == self.last_exported
                or not self.frames or time.monotonic()-self.fusion_arrival > 4):
            return
        self.request_time = time.monotonic()
        self.request_stamp = b.get_clock().now().to_msg()
        self.request_frames = self.frames
        self.request_epoch = self.epoch
        p = b.odom.pose.pose.position
        self.center = [p.x, p.y, p.z]
        self.future = self.client.call_async(Trigger.Request())
        self.future.add_done_callback(self.saved)

    def saved(self, future):
        if future is not self.future:
            return
        self.future = None
        try:
            if self.request_epoch != self.epoch:
                return
            result = future.result()
            if not result.success:
                raise ValueError(result.message)
            path = Path(result.message)
            if not path.is_file() or path.stat().st_size > 32*1024*1024:
                raise ValueError('native map snapshot missing or exceeds 32 MiB bound')
            self.save_ms = (time.monotonic()-self.request_time)*1000
            self.job = self.worker.submit(self.export, path, self.center)
        except Exception as error:
            self.bridge.get_logger().error(f'Wavemap save failed: {error}')

    def export(self, path, center):
        output = Path(self.temp.name)/'roi.f32'
        # Keep the native file immutable for the C++ reader as well. Reject a
        # concurrent external save instead of reading a partially rewritten map.
        before = path.stat()
        contents = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError('native snapshot changed during read')
        snapshot = Path(self.temp.name)/'snapshot.wvmp'
        snapshot.write_bytes(contents)
        executable = Path(__file__).resolve().parents[4] / 'build/wavemap_export/bin/wavemap_export'
        command = [str(executable), str(snapshot), str(output), *map(str, center)]
        if self.bridge.planner_name=='wavestar':command.append('official')
        result = subprocess.run(command,
                                capture_output=True, text=True, timeout=3, check=True)
        metadata = json.loads(result.stdout)
        metadata['occupancy_policy']='wavemap_official' if self.bridge.planner_name=='wavestar' else 'confidence_030_070'
        data = np.fromfile(output, dtype='<f4').reshape(metadata['shape'])
        # Distance to OCCUPIED or UNKNOWN in this bounded native-resolution ROI.
        # UNKNOWN stays NaN: this is a planning snapshot, never a second mapper.
        free = data == 1
        padded = np.pad(free, 1, constant_values=False)
        distance = distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]*metadata['resolution']
        distance[data == 0] = 0
        distance[~np.isfinite(data)] = np.nan
        native = b''
        if self.bridge.planner_name == 'wavestar':
            native = zlib.compress(contents, 1)
            metadata.update(native_map_bytes=len(native), native_map_raw_bytes=len(contents), native_map_codec='zlib')
        return metadata, distance.astype('<f4').ravel(), native

    def poll(self):
        if self.future is not None and time.monotonic()-self.request_time > 4:
            self.client.remove_pending_request(self.future)
            self.future = None
            self.bridge.clear_path()
            self.bridge.status('wavemap_snapshot_timeout')
        if self.job is None or not self.job.done():
            return
        job, self.job = self.job, None
        try:
            metadata, data, native = job.result()
            if (self.request_epoch != self.epoch or not self.bridge.fresh()
                    or time.monotonic()-self.request_time > 4):
                return
            self.last_exported = self.request_frames
            metadata.update(source='wavemap', source_map_version=self.request_frames,
                            save_service_ms=self.save_ms,
                            snapshot_ms=(time.monotonic()-self.request_time)*1000)
            self.bridge.accept_map(metadata, data, self.request_stamp, native)
        except Exception as error:
            self.bridge.clear_path()
            self.bridge.status('wavemap_snapshot_failed', str(error))

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.temp.cleanup()
