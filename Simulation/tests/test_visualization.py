"""Geometry regression for retained cameras and the reported VIO pose error."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from trajectory_geometry import DistanceSampler,transformed_frustum,pose_error


class VisualizationChecks(unittest.TestCase):
    def test_distance_sampled_cameras_and_visible_vio_error(self):
        sampler=DistanceSampler(.75)
        # Accumulated distance, rather than displacement from the initial pose.
        samples=[sampler.accept(p) for p in [(0.,0.,0.),(.3,0.,0.),(.3,.3,0.),(0.,.3,0.),(0.,.3,0.)]]
        self.assertEqual(samples,[True,False,False,True,False])
        transform=((2.,3.,4.),(0.,0.,0.,1.))
        points=transformed_frustum(transform,math.radians(75.),.75)
        self.assertEqual(len(points),16)
        for index in (0,2,4,6):self.assertEqual(points[index],transform[0])
        error=pose_error(transform,((2.3,3.4,4.),(0.,0.,0.,-1.)))
        self.assertAlmostEqual(error['position_error_m'],.5)
        self.assertEqual(error['rotation_error_deg'],0.)


if __name__=='__main__':unittest.main()
