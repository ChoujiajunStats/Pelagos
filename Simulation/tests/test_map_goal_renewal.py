"""Exercise the bridge tick's behavior while a newer map awaits validation."""
import ast
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest


root=Path(__file__).resolve().parents[1]


class MapGoalRenewalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((root/'adapters/nbv_bridge.py').read_text())
        bridge=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NbvBridge')
        tick=next(n for n in bridge.body if isinstance(n,ast.FunctionDef) and n.name=='tick')
        namespace=dict(time=time,json=json)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[tick],type_ignores=[])),
                     'nbv_bridge.py','exec'),namespace)
        cls.tick=staticmethod(namespace['tick'])

    def node(self, *, validated_revision=1, reached=False):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        goals=[]
        target=[1.,0.,0.,0.,0.,0.,1.]
        hold=[0.,0.,0.,0.,0.,0.,1.]
        node=SimpleNamespace(
            out=Path(directory.name),backend_state=None,enabled=True,
            authority_future=None,pending_control_mode=None,
            token='active-lease',generation=1,guard=dict(state='ARMED',generation=1),
            initialized=True,startup_profile='source_scan',forward_start_active=False,cruise_active=False,obstacle_phase='inactive',
            target=target,path=[hold[:],target[:]],hold_target=hold,hold_index=11,
            waypoint_index=12,map_revision=2,validated_revision=validated_revision,
            navigation=dict(reached=reached,generation=1,waypoint_index=12),
            mode='explore',route_complete=False,reason='',
            # An in-flight validation prevents transport from starting another job.
            future=object(),
            poll=lambda now:None,healthy=lambda now:True,state_healthy=lambda now:True,
            publish_goal=lambda target,*args:goals.append(target[:]),
            publish_status=lambda now,healthy:None)
        return node,goals

    def test_newer_map_waits_without_renewing_target_or_old_hold(self):
        node,goals=self.node()
        self.tick(node)
        self.assertEqual(goals,[])
        self.assertEqual(node.target,[1.,0.,0.,0.,0.,0.,1.])
        self.assertEqual(node.reason,'waiting_for_current_map_validation')

    def test_current_map_validation_renews_target(self):
        node,goals=self.node(validated_revision=2)
        self.tick(node)
        self.assertEqual(goals,[node.target])
        self.assertEqual(node.reason,'tracking_nbv_edge')

    def test_reached_on_unvalidated_map_does_not_advance_target(self):
        node,goals=self.node(reached=True)
        target=node.target[:]
        hold=node.hold_target[:]
        path=[pose[:] for pose in node.path]
        self.tick(node)
        self.assertEqual(node.target,target)
        self.assertEqual(node.hold_target,hold)
        self.assertEqual(node.path,path)
        self.assertEqual(node.validated_revision,1)
        self.assertEqual(goals,[])


if __name__=='__main__':unittest.main()
