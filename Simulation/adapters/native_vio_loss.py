"""Expose explicit native OKVIS Lost events that its ROS2 pose topics omit."""
from pathlib import Path


class NativeVioLoss:
    # ViSlamBackend::applyStrategy emits these below .01, matching
    # ThreadedSlam's TrackingQuality::Lost; no new estimator threshold.
    markers=('TRACKING FAILURE: quality=','Tracking quality weak: quality=')
    def __init__(self,path):
        self.path=Path(path);self.offset=0;self.partial=''
    def poll(self):
        try:
            with self.path.open() as stream:
                if self.path.stat().st_size<self.offset:self.offset=0;self.partial=''
                stream.seek(self.offset);data=stream.read();self.offset=stream.tell()
        except OSError:return None
        lines=(self.partial+data).split('\n');self.partial=lines.pop()
        return next((line for line in reversed(lines) if any(marker in line for marker in self.markers)),None)
