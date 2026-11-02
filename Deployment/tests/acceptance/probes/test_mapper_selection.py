#!/usr/bin/env python3
"""Regression checks for mapper identity, exclusivity and frozen configuration."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[4]


def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend=load('mapper_backend','Deployment/application/services/mapper_backend.py')
pipeline=load('run_pipeline','Deployment/application/services/run_pipeline.py')


class MapperSelectionTests(unittest.TestCase):
    def inspect(self,states,arguments):
        return backend.inspect_backend(ROOT,
            read_state=lambda unit:{'ActiveState':states.get(unit,'inactive'),'MainPID':'123'},
            read_command=lambda pid:arguments)

    def command(self,*arguments):
        return ['python3',str(ROOT/'Deployment/application/services/run_pipeline.py'),*arguments]

    def test_identity_comes_from_process_not_unit_name(self):
        info=self.inspect({'dapeng-nvblox.service':'active'},self.command('--mapper','wavemap'))
        self.assertEqual(info['mapper'],'wavemap')

    def test_legacy_service_defaults_to_wavemap(self):
        info=self.inspect({'algorithm-pipeline.service':'active'},self.command())
        self.assertEqual(info['mapper'],'wavemap')

    def test_explicit_nvblox(self):
        info=self.inspect({'dapeng-nvblox.service':'active'},self.command('--mapper=nvblox'))
        self.assertEqual(info['mapper'],'nvblox')

    def test_parallel_services_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'多个'):
            self.inspect({'dapeng-nvblox.service':'active','dapeng-wavemap.service':'active'},self.command())

    def test_stopping_service_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'正在停止'):
            self.inspect({'dapeng-nvblox.service':'deactivating'},self.command('--mapper','nvblox'))

    def test_foreign_process_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'不属于'):
            self.inspect({'dapeng-nvblox.service':'active'},['python3','/other/run_pipeline.py'])

    def test_unknown_mapper_is_rejected(self):
        with self.assertRaises(RuntimeError):
            backend.mapper_from_command(self.command('--mapper','other'),ROOT)
        with self.assertRaises(ValueError):
            pipeline.process_commands(ROOT,ROOT/'config',ROOT/'weights',ROOT/'run','other')

    def test_navigation_is_explicit_and_only_uses_wavemap(self):
        info=self.inspect({'dapeng-wavemap.service':'active'},self.command('--mapper','wavemap','--navigation'))
        self.assertTrue(info['navigation'])
        self.assertFalse(self.inspect({'dapeng-wavemap.service':'active'},self.command())['navigation'])
        commands=pipeline.process_commands(ROOT,ROOT/'config',ROOT/'weights',ROOT/'run','wavemap',True)
        self.assertEqual([name for name,_ in commands],['okvis2','pose_adapter','las2','wavemap'])
        self.assertIn(str(ROOT/'Deployment/contexts/navigation/scripts/start_navigation.sh'),commands[-1][1])
        with self.assertRaisesRegex(ValueError,'Wavemap'):
            pipeline.process_commands(ROOT,ROOT/'config',ROOT/'weights',ROOT/'run','nvblox',True)

    def test_navigation_requires_its_frozen_yaml(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);profile=root/'profile';profile.mkdir();run=root/'run';run.mkdir()
            for name in ['okvis2.yaml','stereo-geometry.json','wavemap.yaml']:
                (profile/name).write_text('configuration\n')
            with self.assertRaisesRegex(FileNotFoundError,'navigation.yaml'):
                pipeline.snapshot_inputs(root,profile,root/'weight.pth',run,'wavemap',True)

    def test_wavemap_never_launches_nvblox_or_its_tf(self):
        commands=pipeline.process_commands(ROOT,ROOT/'config',ROOT/'weights',ROOT/'run','wavemap')
        self.assertEqual([name for name,_ in commands],['okvis2','pose_adapter','las2','wavemap'])
        self.assertNotIn('nvblox',repr(commands))
        gpu_commands=pipeline.process_commands(ROOT,ROOT/'config',ROOT/'weights',ROOT/'run','nvblox')
        self.assertEqual([name for name,_ in gpu_commands],['okvis2','pose_adapter','las2','nvblox_pose_tf','nvblox'])
        self.assertEqual(commands[:3],gpu_commands[:3])

    def test_only_selected_configuration_is_required(self):
        for selected,other in [('wavemap','nvblox'),('nvblox','wavemap')]:
            with self.subTest(selected=selected),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);profile=root/'profile';profile.mkdir()
                for name in ['okvis2.yaml','stereo-geometry.json',selected+'.yaml']:
                    (profile/name).write_text('frozen configuration\n')
                checkpoint=root/'weight.pth';checkpoint.write_bytes(b'fixture weight')
                run=root/'run';run.mkdir()
                configuration,_,_=pipeline.snapshot_inputs(root,profile,checkpoint,run,selected)
                self.assertTrue((configuration/(selected+'.yaml')).is_file())
                self.assertFalse((configuration/(other+'.yaml')).exists())

    def test_missing_selected_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);profile=root/'profile';profile.mkdir();run=root/'run';run.mkdir()
            for name in ['okvis2.yaml','stereo-geometry.json','wavemap.yaml']:
                (profile/name).write_text('configuration\n')
            with self.assertRaisesRegex(FileNotFoundError,'nvblox.yaml'):
                pipeline.snapshot_inputs(root,profile,root/'weight.pth',run,'nvblox')


if __name__=='__main__':
    unittest.main()
