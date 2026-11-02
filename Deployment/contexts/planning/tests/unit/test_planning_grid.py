import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
import unittest
import numpy as np
from planning_grid import box_status, path_status, sphere_path_status


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.data = np.ones((60, 60, 30), dtype=np.float32)
        self.meta = dict(resolution=.1, origin=[-3., -3., -1.5])
        self.path = [[-1.,0.,0.,0.,0.,0.,1.], [1.,0.,0.,0.,0.,0.,1.]]
        self.size = [.7]*3

    def test_free_path(self):
        self.assertEqual(path_status(self.data,self.meta,self.path,self.size),'free')

    def test_occupied_wall(self):
        self.data[30,:,:] = 0
        self.assertEqual(path_status(self.data,self.meta,self.path,self.size),'in_collision')

    def test_unknown_wall(self):
        self.data[30,:,:] = np.nan
        self.assertEqual(path_status(self.data,self.meta,self.path,self.size),'in_unknown_space')

    def test_body_and_boundary(self):
        self.data[22,33,15] = 0
        self.assertEqual(path_status(self.data,self.meta,self.path,self.size),'in_collision')
        self.assertEqual(box_status(self.data,self.meta,[-3.,0.,0.],self.size),'outside_map')

    def test_invalid_path(self):
        self.path[1][0] = float('nan')
        self.assertEqual(path_status(self.data,self.meta,self.path,self.size),'invalid_path')

    def test_sphere_excludes_cube_corner(self):
        self.data[30,31,16]=0
        self.assertEqual(sphere_path_status(self.data,self.meta,self.path,.10),'free')
        self.assertEqual(path_status(self.data,self.meta,self.path,[.2]*3),'in_collision')
        self.assertEqual(sphere_path_status(self.data,self.meta,self.path,.15),'in_collision')

    def test_sphere_grazing_unknown(self):
        self.data[30,31,15]=float('nan')
        self.assertEqual(sphere_path_status(self.data,self.meta,self.path,.10),'in_unknown_space')

    def test_sphere_wall(self):
        self.data[30,:,:]=0
        self.assertEqual(sphere_path_status(self.data,self.meta,self.path,.10),'in_collision')


if __name__ == '__main__':
    unittest.main()
