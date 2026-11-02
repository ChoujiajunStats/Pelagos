#!/usr/bin/env python3
"""Display real NBV diagnostics with GBPlanner's tree/viewpoint/workspace layers."""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String


class PlannerVisualizer(Node):
    def __init__(self):
        super().__init__('pelagos_planner_visualizer')
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tree=self.create_publisher(MarkerArray,'/pelagos/planner/tree',qos)
        self.views=self.create_publisher(PoseArray,'/pelagos/planner/viewpoints',qos)
        self.workspace=self.create_publisher(MarkerArray,'/pelagos/planner/workspace',qos)
        self.checks=self.create_publisher(MarkerArray,'/pelagos/planner/checks',qos)
        self.create_subscription(String,'/pelagos/planner/debug',self.update,qos)

    def marker(self,namespace,kind,identifier=0,color=(.094,.365,.737,1.)):
        m=Marker();m.header.frame_id='rov01/odom';m.header.stamp=self.get_clock().now().to_msg()
        m.ns=namespace;m.id=identifier;m.type=kind;m.action=Marker.ADD;m.pose.orientation.w=1.
        m.color.r,m.color.g,m.color.b,m.color.a=color
        return m

    def clear(self):
        m=self.marker('',Marker.LINE_LIST);m.action=Marker.DELETEALL
        return MarkerArray(markers=[m])

    @staticmethod
    def point(xyz):
        return Point(x=float(xyz[0]),y=float(xyz[1]),z=float(xyz[2]))

    def bounds(self,namespace,low,high,color):
        m=self.marker(namespace,Marker.LINE_LIST,color=color);m.scale.x=.025
        corners=[[float(high[i] if mask&(1<<i) else low[i]) for i in range(3)] for mask in range(8)]
        for mask in range(8):
            for axis in range(3):
                if not mask&(1<<axis):m.points.extend([self.point(corners[mask]),self.point(corners[mask|(1<<axis)])])
        return m

    def update(self,message):
        try:
            data=json.loads(message.data);d=data.get('diagnostics',{})
            root=list(map(float,d['body_position']))
            nodes=d.get('tree_nodes') or []
            tree=self.clear();views=PoseArray();views.header.frame_id='rov01/odom'
            views.header.stamp=self.get_clock().now().to_msg()
            edges=self.marker('native_rrt_edges',Marker.LINE_LIST);edges.scale.x=.025
            for i,node in enumerate(nodes):
                position=list(map(float,node['position']));parent=int(node['parent'])
                if 0<=parent<i:edges.points.extend([self.point(nodes[parent]['position']),self.point(position)])
                if parent>=0:
                    pose=Pose();pose.position=self.point(position);angle=float(node['yaw'])/2
                    pose.orientation.z=math.sin(angle);pose.orientation.w=math.cos(angle);views.poses.append(pose)
            tree.markers.append(edges);self.tree.publish(tree);self.views.publish(views)
            workspace=self.clear()
            workspace.markers.append(self.bounds('observed_map_bounds',d['map_min'],d['map_max'],(.1,.5,.8,.9)))
            if 'sampling_min' in d:
                workspace.markers.append(self.bounds('sampling_window',d['sampling_min'],d['sampling_max'],(.35,.45,.6,.5)))
            label=self.marker('planning_result',Marker.TEXT_VIEW_FACING)
            label.pose.position=self.point([root[0],root[1],root[2]+1.])
            label.scale.z=.13
            virtual=data.get('origin','actual_estimated_body')!='actual_estimated_body'
            label.text=('VIRTUAL ROOT | ' if virtual else 'ESTIMATED BODY | ')+f"NBV nodes {d.get('accepted_tree_nodes','-')} | path {d.get('native_path_points',0)}\n"+str(data.get('label',data.get('reason','')))
            workspace.markers.append(label)
            body=self.marker('planning_root_footprint',Marker.CUBE,color=(.1,.4,.8,.15))
            body.pose.position=self.point(root);body.scale.x,body.scale.y,body.scale.z=map(float,d['robot_box'])
            workspace.markers.append(body);self.workspace.publish(workspace)
            checks=self.clear();colors={'free':(.1,.65,.3,1.),'unknown':(1.,.55,.05,1.),'occupied':(.8,.08,.13,1.)}
            for i,sample in enumerate(d.get('root_footprint_samples') or []):
                m=self.marker('root_footprint_samples',Marker.SPHERE,i,colors[sample['status']])
                m.pose.position=self.point(sample['point']);m.scale.x=m.scale.y=m.scale.z=.065;checks.markers.append(m)
            reach=float(d['extension_m'])+float(d['overshoot_m'])
            for i,check in enumerate(d.get('native_extension_checks') or []):
                m=self.marker('native_extension_checks',Marker.LINE_LIST,i,colors[check['status']]);m.scale.x=.015
                m.points=[self.point(root),self.point([root[k]+reach*float(check['direction'][k]) for k in range(3)])]
                checks.markers.append(m)
            self.checks.publish(checks)
        except (KeyError,TypeError,ValueError,IndexError) as error:
            self.get_logger().warning(f'Invalid NBV diagnostics: {error}')


def main():
    rclpy.init();node=PlannerVisualizer()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
