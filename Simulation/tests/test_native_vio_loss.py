import ast
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'adapters'))
from native_vio_loss import NativeVioLoss


class NativeVioLossChecks(unittest.TestCase):
    def tick_node(self, *, lost=None, ever_tracking=False, visual_stamp=1_000_000_000):
        tree=ast.parse((root/'adapters/body_state.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BodyState')
        tick=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='tick')
        namespace=dict(time=time,json=json,String=lambda **kw:SimpleNamespace(**kw))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[tick],type_ignores=[])),'body_state.py','exec'),namespace)
        sent=[];now=time.monotonic_ns()
        node=SimpleNamespace(native_loss=SimpleNamespace(poll=lambda:lost),
            ever_tracking=ever_tracking,fault=None,native_loss_evidence=None,state=object(),received=now,
            visual_wall=now,stamp=1_000_000_000,get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_000_000_000)),
            out=Path('run'),epoch=1,image_stamp=1_000_000_000,imu_stamp=1_000_000_000,visual_stamp=visual_stamp,
            status=SimpleNamespace(publish=lambda m:sent.append(json.loads(m.data))))
        return node,sent,namespace['tick']

    def test_explicit_native_lost_and_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'okvis.log';path.write_text('ordinary odometry\n')
            monitor=NativeVioLoss(path);self.assertIsNone(monitor.poll())
            with path.open('a') as f:f.write('ViSlamBackend.cpp: TRACKING FAILURE: quality=0')
            self.assertIsNone(monitor.poll())
            with path.open('a') as f:f.write('\n')
            self.assertIn('TRACKING FAILURE',monitor.poll());self.assertIsNone(monitor.poll())
            with path.open('a') as f:f.write('ViSlamBackend.cpp: Tracking quality weak: quality=0.001\n')
            self.assertIn('quality=0.001',monitor.poll())

    def test_fresh_pose_does_not_override_native_lost(self):
        node,sent,tick=self.tick_node(lost='TRACKING FAILURE: quality=0',ever_tracking=True)
        tick(node)
        self.assertEqual(sent[-1]['state'],'LOST')
        self.assertEqual(sent[-1]['reason'],'okvis_native_tracking_lost_restart_required')

    def test_native_lost_before_first_tracking_remains_latched(self):
        node,sent,tick=self.tick_node(lost='TRACKING FAILURE: quality=0',ever_tracking=False)
        tick(node)
        self.assertEqual(sent[-1]['state'],'LOST')
        self.assertEqual(sent[-1]['reason'],'okvis_native_tracking_lost_restart_required')
        node.native_loss.poll=lambda:None
        tick(node)
        self.assertEqual(sent[-1]['state'],'LOST')
        self.assertEqual(sent[-1]['reason'],'okvis_native_tracking_lost_restart_required')

    def test_fresh_arrival_of_stale_visual_acquisition_is_not_tracking(self):
        node,sent,tick=self.tick_node(visual_stamp=400_000_000)
        tick(node)
        self.assertEqual(sent[-1]['state'],'WAITING')

    def test_fresh_visual_and_body_acquisitions_are_tracking(self):
        node,sent,tick=self.tick_node()
        tick(node)
        self.assertEqual(sent[-1]['state'],'TRACKING')


if __name__=='__main__':unittest.main()
