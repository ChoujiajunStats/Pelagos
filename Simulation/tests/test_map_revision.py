"""Exercise the real response handler without requiring a ROS installation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source=Path(__file__).resolve().parents[1]/'adapters/nbv_bridge.py'
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NbvBridge')
poll=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='poll')
namespace={}
exec(compile(ast.fix_missing_locations(ast.Module(body=[poll],type_ignores=[])),str(source),'exec'),namespace)


class MapRevisionChecks(unittest.TestCase):
    def validate_response(self,installed,queried):
        result=dict(valid=True,map_revision=queried)
        bridge=SimpleNamespace(future=SimpleNamespace(done=lambda:True,result=lambda:result),
            job=dict(operation='validate',task_revision=0),map_revision=installed,
            task_revision=0,validated_revision=-1)
        namespace['poll'](bridge,0)
        return bridge

    def test_old_query_does_not_validate_newer_map(self):
        bridge=self.validate_response(8,7)
        self.assertEqual(bridge.map_revision,8)
        self.assertEqual(bridge.validated_revision,7)
        self.assertNotEqual(bridge.validated_revision,bridge.map_revision)

    def test_response_can_advance_version_before_status_arrives(self):
        bridge=self.validate_response(7,8)
        self.assertEqual(bridge.map_revision,8)
        self.assertEqual(bridge.validated_revision,8)


if __name__=='__main__':unittest.main()
