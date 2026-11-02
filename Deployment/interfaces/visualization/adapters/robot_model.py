#!/usr/bin/env python3
"""Robot-side display-only six-thruster BlueROV2 asset publisher. No control I/O."""
import json
import math
from pathlib import Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def quaternion(r,p,y):
    cr,sr=math.cos(r/2),math.sin(r/2);cp,sp=math.cos(p/2),math.sin(p/2);cy,sy=math.cos(y/2),math.sin(y/2)
    return sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy


class VehicleVisual(Node):
    def __init__(self):
        super().__init__('bluerov2_six_visual')
        root=Path(__file__).resolve().parents[4]
        default=root/'Deployment/infrastructure/ros2_ws/src/underwater_navigation_ros/meshes/bluerov2_six'
        self.assets=Path(self.declare_parameter('asset_directory',str(default)).value).resolve()
        self.frame=self.declare_parameter('base_frame','base_link').value
        self.assembly=json.loads((self.assets/'assembly.json').read_text())
        if len(self.assembly['thrusters'])!=6 or sum(t['kind']=='horizontal' for t in self.assembly['thrusters'])!=4:
            raise ValueError('six-thruster assembly requires four horizontal and two vertical propellers')
        self.parts=[dict(name='hull',mesh=self.assembly['hull'],xyz=[0,0,0],rpy=[0,0,0])]+self.assembly['thrusters']
        for part in self.parts:
            if not (self.assets/part['mesh']).is_file():raise ValueError('missing visual asset '+part['mesh'])
        self.pub=self.create_publisher(MarkerArray,'/visualization/robot_model',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.timer=self.create_timer(1.,self.publish_model)
        self.publish_model()
        self.get_logger().info('BlueROV2 asset visual: hull + 4 horizontal + 2 vertical propellers; frame='+self.frame)

    def publish_model(self):
        a=MarkerArray()
        for i,part in enumerate(self.parts):
            m=Marker();m.header.frame_id=self.frame;m.header.stamp=self.get_clock().now().to_msg()
            m.ns='bluerov2_six_asset';m.id=i;m.type=Marker.MESH_RESOURCE;m.action=Marker.ADD;m.frame_locked=True
            m.mesh_resource=(self.assets/part['mesh']).as_uri();m.mesh_use_embedded_materials=True
            m.scale.x=m.scale.y=m.scale.z=float(self.assembly['scale_m'])
            # All-zero color selects the embedded material without RViz tinting.
            m.pose.position.x,m.pose.position.y,m.pose.position.z=[float(x) for x in part['xyz']]
            m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w=quaternion(*part['rpy'])
            a.markers.append(m)
        self.pub.publish(a)


def main():
    rclpy.init();node=VehicleVisual()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
