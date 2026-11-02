import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
"""Replay measured native occupancy ROI on an isolated ROS 1 master."""
import json
import math
import os
from pathlib import Path
import threading
import time
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from dapeng_planning_msgs.msg import EsdfGrid
from dapeng_planning_msgs.srv import Plan, PlanRequest
from planning_grid import box_status, path_status


def main():
    if os.environ.get('ROS_MASTER_URI') != 'http://127.0.0.1:11333':
        raise RuntimeError('Saved-map replay requires isolated master 11333')
    root = Path('/evidence')
    meta = json.loads((root/'online-roi.json').read_text())
    data = np.fromfile(root/'online-roi.f32',dtype='<f4').reshape(meta['shape'])
    # The exporter encodes confirmed FREE as 1, OCCUPIED as 0, UNKNOWN as NaN.
    # These distances only need to preserve the core's voxel status for this replay.
    pair = json.loads((root/'offline-pair.json').read_text())
    rospy.init_node('isolated_saved_wavemap_replay')
    pub = rospy.Publisher('/omniplanner/esdf',EsdfGrid,queue_size=1,latch=True)
    odom_pub = rospy.Publisher('/omniplanner/odometry',Odometry,queue_size=1)
    grid = EsdfGrid();grid.header.frame_id='world';grid.sequence=1
    grid.origin.x,grid.origin.y,grid.origin.z=meta['origin']
    grid.size_x,grid.size_y,grid.size_z=meta['shape'];grid.voxel_size=meta['resolution']
    grid.distance=data.ravel().tolist()
    odom=Odometry();odom.header.frame_id='world';odom.child_frame_id='body'
    odom.pose.pose.orientation.w=1.
    odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z=pair[0]
    stop=threading.Event()
    def publish():
        while not stop.is_set():
            grid.header.stamp=rospy.Time.now();odom.header.stamp=grid.header.stamp
            pub.publish(grid);odom_pub.publish(odom);stop.wait(.25)
    thread=threading.Thread(target=publish,daemon=True);thread.start()
    service=rospy.ServiceProxy('/omniplanner/plan',Plan);service.wait_for_service(timeout=20)
    time.sleep(1)
    try:
        request=PlanRequest();request.header.frame_id='world';request.map_sequence=1
        request.root_pose=odom.pose.pose;request.goal_pose.orientation.w=1.
        request.goal_pose.position.x,request.goal_pose.position.y,request.goal_pose.position.z=pair[1]
        assert box_status(data,meta,pair[0],[.7]*3)=='free'
        assert box_status(data,meta,pair[1],[.7]*3)=='free'
        report={'mode':'isolated saved Wavemap; start is a selected FREE map location, not live robot',
                'start':pair[0], 'goal':pair[1], 'native_map':meta}
        for attempt in range(8):
            request.header.stamp=rospy.Time.now();begin=time.monotonic();response=service(request)
            poses=[[p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,
                    p.orientation.z,p.orientation.w] for p in response.path]
            if len(poses)>=2 and math.dist(poses[-1][:3],pair[1])<=.25:
                assert path_status(data,meta,poses,[.7]*3)=='free'
                report.update(poses=poses,endpoint_gap_m=math.dist(poses[-1][:3],pair[1]),
                              planning_ms=(time.monotonic()-begin)*1000,collision_check='passed')
                print(json.dumps(report,indent=2));return
            time.sleep(.3)
        raise AssertionError('No complete collision-free path in selected measured FREE region')
    finally:
        stop.set();thread.join(timeout=2)


if __name__=='__main__':main()
