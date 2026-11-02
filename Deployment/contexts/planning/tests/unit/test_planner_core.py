#!/usr/bin/env python3
import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
"""Exercise the real planner on an isolated ROS master, never on the live map."""
import json
import math
import os
import threading
import time

import numpy as np
import rospy
from dapeng_planning_msgs.msg import EsdfGrid
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from dapeng_planning_msgs.srv import Plan, PlanRequest


def main():
    if os.environ.get('ROS_MASTER_URI') != 'http://127.0.0.1:11332':
        raise RuntimeError('This fixture requires its own test master on port 11332')
    rospy.init_node('isolated_planner_verification')
    grid_pub = rospy.Publisher('/omniplanner/esdf', EsdfGrid, queue_size=1, latch=True)
    odom_pub = rospy.Publisher('/omniplanner/odometry', Odometry, queue_size=1)
    grid = EsdfGrid()
    grid.header.frame_id = 'world'; grid.sequence = 1
    grid.origin.x = -5.; grid.origin.y = -5.; grid.origin.z = -1.
    grid.voxel_size = .1; grid.size_x = grid.size_y = 101; grid.size_z = 21
    grid.distance = np.full((101,101,21), 2., dtype=np.float32).ravel().tolist()
    odom = Odometry(); odom.header.frame_id = 'world'; odom.child_frame_id = 'body'
    odom.pose.pose.position.x = -1.; odom.pose.pose.orientation.w = 1.
    goal = PoseStamped(); goal.header.frame_id = 'world'
    goal.pose.position.x = 1.; goal.pose.orientation.w = 1.
    stop = threading.Event()
    stale = False

    def publish():
        while not stop.is_set() and not rospy.is_shutdown():
            grid.header.stamp = rospy.Time.now() - rospy.Duration(10 if stale else 0)
            odom.header.stamp = rospy.Time.now()
            odom_pub.publish(odom); grid_pub.publish(grid)
            stop.wait(.3)

    worker = threading.Thread(target=publish, daemon=True); worker.start()
    service = rospy.ServiceProxy('/omniplanner/plan', Plan)
    service.wait_for_service(timeout=15)
    time.sleep(2)
    report = {}

    def plan(sequence=None):
        request = PlanRequest()
        request.header.frame_id = 'world'
        request.map_sequence = grid.sequence if sequence is None else sequence
        request.header.stamp = rospy.Time.now()
        request.root_pose = odom.pose.pose
        request.goal_pose = goal.pose
        return service(request)

    try:
        try:
            plan(sequence=grid.sequence+100)
        except rospy.ServiceException:
            report['wrong_map_version'] = 'rejected'
        else:
            raise AssertionError('Mismatched map version was accepted')
        best = None
        for _ in range(5):
            response = plan()
            if len(response.path) >= 2:
                gap = math.dist([response.path[-1].position.x,response.path[-1].position.y,
                                 response.path[-1].position.z], [1.,0.,0.])
                if best is None or gap < best[0]: best = (gap,response)
                if gap <= .25: break
            time.sleep(.5)
        assert best is not None and best[0] < .6, 'No usable path in fully known free space'
        path = best[1].path
        assert all(math.isfinite(v) for p in path for v in
                   [p.position.x,p.position.y,p.position.z,p.orientation.x,
                    p.orientation.y,p.orientation.z,p.orientation.w])
        report['free_space'] = {'poses':len(path),'endpoint_gap_m':best[0],
                               'path':[[p.position.x,p.position.y,p.position.z] for p in path]}

        # A solid wall spans the whole query volume; the returned path must stay on this side.
        data = np.full((101,101,21), 2., dtype=np.float32)
        data[54:57,:,:] = -.1
        grid.distance = data.ravel().tolist(); grid.sequence += 1
        time.sleep(.8)
        response = plan()
        assert all(p.position.x < .1 for p in response.path), 'Path crosses occupied wall'
        report['occupied_wall'] = {'poses':len(response.path),'crossed':False}

        grid.distance = np.full((101,101,21), np.nan, dtype=np.float32).ravel().tolist()
        grid.sequence += 1; time.sleep(.8)
        response = plan()
        assert len(response.path) < 2, 'Unknown map yielded a traversable path'
        report['unknown_space'] = {'poses':len(response.path)}

        stale = True; time.sleep(.8)
        try:
            plan()
        except rospy.ServiceException:
            report['stale_map'] = 'rejected'
        else:
            raise AssertionError('Stale map was accepted')
        print(json.dumps(report, indent=2))
    finally:
        stop.set(); worker.join(timeout=2)


if __name__ == '__main__':
    main()
