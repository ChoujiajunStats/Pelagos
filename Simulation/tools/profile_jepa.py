#!/usr/bin/env python3
"""Measure the frozen JEPA config on live causal history, without control publishers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure
from std_msgs.msg import String
from estimated_control import install


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir
    if json.loads((out/'control-status.json').read_text())['state']!='DISARMED':
        raise RuntimeError('Pause the mission before profiling')
    install()
    from uw_controller.checkpoint import CheckpointController
    from uw_robot.thrusters import Allocation,load_profile
    settings=yaml.safe_load((out/'jepa-settings.yaml').read_text())
    diagnostic=out/'jepa-profile';diagnostic.mkdir(exist_ok=True)
    settings['run_dir']=str(diagnostic)
    allocation=Allocation(load_profile(out/'thrusters.yaml'),
                          json.loads((out/'allocation-model.json').read_text())['model'])
    control=CheckpointController(yaml.safe_load((out/'controller-parameters.yaml').read_text()),
                                 allocation,settings)
    rclpy.init();node=Node('pelagos_read_only_jepa_profile')
    def imu(m):
        a,g=m.linear_acceleration,m.angular_velocity
        control.on_imu(stamp(m),[a.x,a.y,a.z],[g.x,g.y,g.z])
    def terminal(m):
        s=json.loads(m.data)
        control.on_applied(int(s['clock_ns'])*1e-9,s['applied_setpoint'])
    def state(m):
        p,q,v=m.pose.pose.position,m.pose.pose.orientation,m.twist.twist
        control.observe(stamp(m),[p.x,p.y,p.z],[q.x,q.y,q.z,q.w],
                        [v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z])
    node.create_subscription(Imu,'/rov01/sensors/imu',imu,qos_profile_sensor_data)
    node.create_subscription(FluidPressure,'/rov01/sensors/pressure',
                             lambda m:control.on_pressure(stamp(m),m.fluid_pressure),qos_profile_sensor_data)
    node.create_subscription(String,'/rov01/sim/terminal/status',terminal,1)
    node.create_subscription(Odometry,'/rov01/state/odometry',state,1)
    try:
        deadline=time.monotonic()+30
        while not control.ready and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
        if not control.ready:raise RuntimeError('Could not collect and prewarm real causal history')
        observation,action=control.history.arrays();current=control.current_modal.copy()
        target=np.array([0.,0.,-.04,-.12])
        for mode in ('direct','worker_with_ros_callbacks'):
            for _ in range(8):
                values=(control.epoch,observation,action,target,current[:4],
                        control.position.copy(),current[4])
                if mode=='direct':result=control._solve(*values)
                else:
                    future=control.pool.submit(control._solve,*values)
                    while not future.done():rclpy.spin_once(node,timeout_sec=.002)
                    result=future.result()
                print(json.dumps(dict(mode=mode,reason=result.reason,**{k:float(v) for k,v in
                    result.diagnostics.items() if k.endswith('_seconds') and isinstance(v,(int,float))})),flush=True)
    finally:
        node.destroy_node();rclpy.try_shutdown();control.close()


if __name__=='__main__':main()
