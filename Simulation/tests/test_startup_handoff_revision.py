"""An observed startup corridor must belong to the currently installed map."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


root=Path(__file__).resolve().parents[1]


class StartupHandoffRevisionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((root/'adapters/nbv_bridge.py').read_text())
        bridge=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NbvBridge')
        poll=next(n for n in bridge.body if isinstance(n,ast.FunctionDef) and n.name=='poll')
        namespace=dict(pose_values=lambda pose:pose[:])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[poll],type_ignores=[])),
                     'nbv_bridge.py','exec'),namespace)
        cls.poll=staticmethod(namespace['poll'])

    def node(self, *, result_revision):
        events=[]
        target=[5.,0.,0.,0.,0.,0.,1.]
        body=[1.,0.,0.,0.,0.,0.,1.]
        result=dict(valid=True,map_revision=result_revision,
                    diagnostics=dict(startup_body_known='true',startup_can_plan_forward='false'))
        node=SimpleNamespace(
            future=SimpleNamespace(done=lambda:True,result=lambda:result),
            job=dict(operation='validate_forward_start',task_revision=1),
            task_revision=1,map_revision=2,validated_revision=-1,
            startup_map_status={},startup_profile='forward_until_observed',
            forward_start_active=True,forward_start_required=False,initialized=False,
            target=target,path=[body[:],target[:]],hold_target=None,waypoint_index=12,
            state=SimpleNamespace(pose=SimpleNamespace(pose=body)),
            event=lambda name,**kwargs:events.append(name))
        return node,events

    def test_stale_valid_response_keeps_startup_target_until_current_validation(self):
        node,events=self.node(result_revision=1)
        target=node.target[:]
        self.poll(node,1_000_000_000)
        self.assertEqual(node.map_revision,2)
        self.assertEqual(node.target,target)
        self.assertFalse(node.initialized)
        self.assertTrue(node.forward_start_active)
        self.assertIsNone(node.hold_target)
        self.assertEqual(events,[])

    def test_current_valid_response_hands_off_to_nbv(self):
        node,events=self.node(result_revision=2)
        self.poll(node,1_000_000_000)
        self.assertTrue(node.initialized)
        self.assertFalse(node.forward_start_active)
        self.assertFalse(node.forward_start_required)
        self.assertIsNone(node.target)
        self.assertEqual(node.path,[])
        self.assertEqual(node.hold_target,node.state.pose.pose)
        self.assertEqual(events,['startup_observed_corridor_switch_to_nbv'])


if __name__=='__main__':unittest.main()
