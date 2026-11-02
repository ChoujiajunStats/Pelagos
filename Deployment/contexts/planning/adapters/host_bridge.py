#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Connect the upstream OmniPlanner goal service to the robot's ROS 2 adapter."""
import argparse
import math
import socket
import threading
import time

import numpy as np
import rospy
from dapeng_planning_msgs.msg import EsdfGrid
from nav_msgs.msg import Odometry
from dapeng_planning_msgs.srv import Plan, PlanRequest

import sys
from pathlib import Path as DeploymentPath
sys.path.insert(0, str(DeploymentPath(__file__).resolve().parents[3] / 'contracts'))
from wire import receive_packet, send_packet
from planning_grid import box_status, path_status, body_summary


class HostBridge:
    def __init__(self, port):
        self.port = port
        self.connection = None
        self.lock = threading.Lock()
        self.latest_map = None
        self.latest_odometry = None
        self.latest_goal = None
        self.map_arrival = 0.0
        self.odom_arrival = 0.0
        self.published_map_sequence = -1
        self.previous_plan_time = 0.0
        self.robot_size = np.asarray(rospy.get_param('/dapeng_omniplanner/RobotParams/size'), dtype=float)
        self.collision_radius = float(max(self.robot_size)/2)
        self.map_pub = rospy.Publisher('/omniplanner/esdf', EsdfGrid, queue_size=1, latch=True)
        self.odom_pub = rospy.Publisher('/omniplanner/odometry', Odometry, queue_size=1)
        self.planner = rospy.ServiceProxy('/omniplanner/plan', Plan)
        threading.Thread(target=self.network, daemon=True).start()

    def transmit(self, data):
        with self.lock:
            if self.connection:
                try:
                    send_packet(self.connection, data)
                except (OSError, ValueError):
                    pass

    def network(self):
        while not rospy.is_shutdown():
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=3) as connection:
                    connection.settimeout(1)
                    with self.lock:
                        self.connection = connection
                    while not rospy.is_shutdown():
                        message, payload = receive_packet(connection, rospy.is_shutdown)
                        kind = message['type']
                        if kind == 'map':
                            shape = message.get('shape', [])
                            if (len(shape)!=3 or any(not isinstance(v,int) or not 0<v<=1000 for v in shape)
                                    or math.prod(shape)*4 != len(payload)):
                                raise ValueError('Invalid ESDF payload')
                            self.latest_map = (message, payload)
                            self.map_arrival = time.monotonic()
                        elif kind == 'odometry':
                            self.latest_odometry = message
                            self.odom_arrival = time.monotonic()
                        elif kind == 'goal':
                            self.latest_goal = message
                            rospy.loginfo('Received RViz goal %s: %s', message['sequence'], message['pose'][:3])
                        elif kind == 'clear':
                            self.latest_goal = None
            except (EOFError, OSError, ValueError) as error:
                rospy.logwarn_throttle(10, 'Robot planner link: %s', error)
            finally:
                with self.lock:
                    self.connection = None
                self.latest_map = None
                self.latest_odometry = None
                self.latest_goal = None
                self.published_map_sequence = -1
            time.sleep(1)

    @staticmethod
    def set_pose(pose, values):
        pose.position.x, pose.position.y, pose.position.z = values[:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = values[3:]

    def publish_inputs(self):
        odometry = self.latest_odometry
        if odometry:
            message = Odometry()
            message.header.frame_id = 'world'
            message.header.stamp = rospy.Time(odometry['stamp'][0], odometry['stamp'][1])
            message.child_frame_id = odometry['child_frame']
            self.set_pose(message.pose.pose, odometry['pose'])
            self.odom_pub.publish(message)
        snapshot = self.latest_map
        if snapshot and snapshot[0]['sequence'] != self.published_map_sequence:
            metadata, payload = snapshot
            message = EsdfGrid()
            message.header.frame_id = 'world'
            message.sequence = metadata['sequence']
            message.header.stamp = rospy.Time(metadata['stamp'][0],metadata['stamp'][1])
            message.origin.x,message.origin.y,message.origin.z = metadata['origin']
            message.voxel_size = metadata['resolution']
            message.size_x,message.size_y,message.size_z = metadata['shape']
            message.distance = np.frombuffer(payload, dtype='<f4').tolist()
            self.map_pub.publish(message)
            self.published_map_sequence = metadata['sequence']

    def plan(self):
        goal, odometry, snapshot = self.latest_goal, self.latest_odometry, self.latest_map
        if not goal or not odometry or not snapshot:
            return
        now = time.monotonic()
        if now-self.odom_arrival>1.5 or now-self.map_arrival>4 or now-self.previous_plan_time<2:
            return
        metadata, payload = snapshot
        self.previous_plan_time = now
        data = np.frombuffer(payload, dtype='<f4').reshape(metadata['shape'])
        for name, pose in [('root', odometry['pose']), ('goal', goal['pose'])]:
            result = box_status(data, metadata, pose[:3], self.robot_size)
            if result == 'free':
                continue
            state = name + '_' + result
            self.transmit({'type':'status','goal_sequence':goal['sequence'], 'state':state,
                           'detail':{'map_sequence':metadata['sequence'], 'robot_size_m':self.robot_size.tolist(),
                               'position_m':pose[:3], 'collision_radius_m':self.collision_radius,
                               'collision_sphere':body_summary(data,metadata,pose[:3],self.collision_radius)}})
            return
        self.publish_inputs()
        rospy.sleep(.15)
        request = PlanRequest()
        request.header.frame_id = 'world'
        request.map_sequence = metadata['sequence']
        request.header.stamp = rospy.Time.now()
        self.set_pose(request.root_pose, odometry['pose'])
        self.set_pose(request.goal_pose, goal['pose'])
        try:
            planning_start = time.monotonic()
            self.planner.wait_for_service(timeout=2)
            response = self.planner(request)
            if not self.latest_goal or self.latest_goal['sequence'] != goal['sequence']:
                return
            poses = [[p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,
                      p.orientation.z,p.orientation.w] for p in response.path]
            if len(poses)<2:
                self.transmit({'type':'path','goal_sequence':goal['sequence'], 'poses':[], 'state':'no_path',
                               'map_sequence':metadata['sequence'], 'map_stamp':metadata['stamp']})
                return
            if response.map_sequence != metadata['sequence']:
                raise rospy.ServiceException('map sequence mismatch')
            checked = path_status(data, metadata, poses, self.robot_size)
            if checked != 'free':
                self.transmit({'type':'status','goal_sequence':goal['sequence'],
                    'state':'path_validation_failed', 'detail':checked})
                return
            remaining = math.dist(poses[-1][:3], goal['pose'][:3])
            state = 'path_ready' if remaining<=.25 else 'partial_path'
            self.transmit({'type':'path','goal_sequence':goal['sequence'], 'map_sequence':metadata['sequence'],
                           'map_stamp':metadata['stamp'], 'poses':poses, 'state':state,
                           'planning_ms':(time.monotonic()-planning_start)*1000,
                           'remaining_distance_m':remaining})
            rospy.loginfo('OmniPlanner goal %s: %s poses, endpoint gap %.3f m', goal['sequence'],len(poses),remaining)
        except (rospy.ROSException, rospy.ServiceException) as error:
            self.transmit({'type':'status','goal_sequence':goal['sequence'],'state':'planning_retry'})
            rospy.logwarn_throttle(5,'OmniPlanner request failed: %s',error)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_inputs()
            self.plan()
            rate.sleep()


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18771)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('omniplanner_transport')
    HostBridge(args.port).run()


if __name__=='__main__':main()
