"""Exercise installed-map progress and bounded relay scheduling without ROS."""
import ast
import base64
import json
import math
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest
import zlib


ROOT=Path(__file__).resolve().parents[1]


class MapRelayProgressChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT/'adapters/wavemap_nbv_relay.py'
        relay=next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n,ast.ClassDef) and n.name=='MapRelay')
        methods=[n for n in relay.body if isinstance(n,ast.FunctionDef) and
                 n.name in ('tick','request','on_map')]
        cls.clock=NS(now=100.)
        cls.namespace=dict(time=NS(monotonic=lambda:cls.clock.now,
                                  monotonic_ns=lambda:round(cls.clock.now*1.e9)),
                           json=json,math=math,zlib=zlib,base64=base64,
                           String=lambda **kw:NS(**kw),exchange=object())
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),cls.namespace)

    def node(self):
        self.clock.now=100.
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out=Path(temporary.name)
        (out/'nbv.sock').touch()
        published,submitted=[],[]
        node=NS(out=out,map_topic='/mapping/octomap',last_status=99.95,
                report=dict(run=out.name,ready=True,reason='wavemap_snapshot_installed',
                            source_revision=10,map_revision=1,imports=1,depth_fusion_enabled=False),
                state=NS(pose=NS(pose=NS(position=NS(x=1.,y=2.,z=3.),
                                         orientation=NS(x=0.,y=0.,z=0.,w=1.)))),
                pending=None,future=None,job=None,started=99.98,stop=object(),
                status=NS(publish=lambda message:published.append(json.loads(message.data))),
                count_publishers=lambda topic:1)
        def submit(function,address,request,stop):
            submitted.append(request)
            return NS(done=lambda:False)
        node.pool=NS(submit=submit)
        for name in ('tick','request','on_map'):
            setattr(node,name,MethodType(self.namespace[name],node))
        node.published,node.submitted=published,submitted
        return node

    def completed(self,node,source=20,**changes):
        answer=dict(valid=True,reason='wavemap_snapshot_installed',map_revision=2,
                    source_revision=source,depth_fusion_enabled=False)
        answer.update(changes)
        node.future=NS(done=lambda:True,result=lambda:answer)
        node.job=dict(map_bytes=3)

    def message(self,stamp):
        return NS(binary=False,id='OcTree',resolution=.1,data=bytearray([0,127,255]),
                  header=NS(frame_id='rov01/odom',stamp=NS(sec=0,nanosec=stamp)))

    def test_newly_installed_source_is_published_inside_heartbeat_interval(self):
        node=self.node()
        self.completed(node)
        node.tick()
        self.assertEqual(len(node.published),1)
        report=node.published[0]
        self.assertTrue(report['ready'])
        self.assertEqual((report['source_revision'],report['map_revision'],report['imports']),(20,2,2))
        self.assertEqual(report['wall_ns'],100_000_000_000)
        self.assertEqual(json.loads((node.out/'map-source-status.json').read_text()),report)
        self.assertEqual(node.last_status,100.)

    def test_periodic_heartbeat_keeps_original_acquisition_and_import_identity(self):
        node=self.node()
        self.completed(node)
        node.tick()
        self.clock.now=100.1
        node.tick()
        self.assertEqual(len(node.published),1)
        self.clock.now=100.21
        node.tick()
        self.assertEqual(len(node.published),2)
        old,new=node.published
        for key in ('source_revision','map_revision','imports'):
            self.assertEqual(old[key],new[key])
        self.assertGreater(new['wall_ns'],old['wall_ns'])

    def test_duplicate_source_ack_does_not_trigger_extra_progress_publication(self):
        node=self.node()
        self.completed(node,source=10)
        node.tick()
        self.assertEqual(node.published,[])
        self.assertEqual(node.report['source_revision'],10)

    def test_rejected_import_is_not_ready_or_new_source_progress(self):
        node=self.node()
        self.completed(node,valid=False,reason='bad_crc')
        node.last_status=99.7
        node.tick()
        self.assertFalse(node.published[0]['ready'])
        self.assertEqual(node.published[0]['reason'],'bad_crc')
        self.assertEqual((node.report['source_revision'],node.report['imports']),(10,1))

    def test_wrong_native_fusion_contract_cannot_publish_ready_progress(self):
        node=self.node()
        self.completed(node,depth_fusion_enabled=True)
        node.last_status=99.7
        node.tick()
        self.assertFalse(node.published[0]['ready'])
        self.assertEqual(node.report['reason'],'native_depth_fusion_was_not_disabled')
        self.assertEqual((node.report['source_revision'],node.report['imports']),(10,1))

    def test_transport_error_does_not_invent_source_progress(self):
        node=self.node()
        def fail():raise ValueError('NBV response session or request sequence mismatch')
        node.future=NS(done=lambda:True,result=fail)
        node.job=dict(map_bytes=3)
        node.last_status=99.7
        node.tick()
        self.assertFalse(node.published[0]['ready'])
        self.assertIn('sequence mismatch',node.report['reason'])
        self.assertEqual(node.report['source_revision'],10)

    def test_only_latest_pending_snapshot_is_submitted_after_current_import(self):
        node=self.node()
        node.future=NS(done=lambda:False)
        node.on_map(self.message(21))
        node.on_map(self.message(30))
        node.tick()
        self.assertEqual(node.submitted,[])
        self.assertEqual(node.pending.header.stamp.nanosec,30)
        self.completed(node,source=20)
        node.tick()
        self.assertEqual(len(node.submitted),1)
        request=node.submitted[0]
        self.assertEqual(request['source_revision'],30)
        self.assertEqual(request['stamp_ns'],30)
        self.assertEqual(request['sequence'],100_000_000_000)
        self.assertEqual(request['run'],node.out.name)
        self.assertEqual(request['body'],[1.,2.,3.,0.,0.,0.,1.])
        self.assertEqual(base64.b64decode(request['map_data_b64']),bytes([0,127,255]))
        self.assertEqual(request['map_crc32'],zlib.crc32(bytes([0,127,255])))
        self.assertIsNone(node.pending)
        self.assertIsNotNone(node.future)
        self.assertEqual(node.published[0]['source_revision'],20)

    def test_old_pending_snapshot_does_not_resubmit_or_relabel_source(self):
        node=self.node()
        node.on_map(self.message(9))
        node.tick()
        self.assertEqual(node.submitted,[])
        self.assertIsNone(node.pending)
        self.assertEqual(node.published,[])
        self.assertEqual(node.report['source_revision'],10)


if __name__=='__main__':unittest.main(verbosity=2)
