#!/usr/bin/env python3
"""Read-only timing trace of the existing control chain; never publishes."""
import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from uw_interfaces.msg import AuthorizedCommand, ActuatorOutput


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=120.)
    args=parser.parse_args();rclpy.init();node=Node('pelagos_control_timing_trace')
    output=args.run_dir/'control-timing-trace.jsonl'
    stream=output.open('a',buffering=65536);armed=False;fault=False
    latest_clock=None

    def write(stage,values):
        stream.write(json.dumps(dict(received_ns=time.monotonic_ns(),stage=stage,**values))+'\n')

    def authorization(message):
        r=message.request
        return dict(generation=r.generation,request_seq=r.sequence,issued_ns=r.issued_steady_ns,
            deadline_ns=message.deadline_steady_ns)

    def approved(message):write('approved',authorization(message))

    def controller(message):
        write('controller_output',dict(authorization(message.authorization),computed_ns=message.computed_steady_ns))

    def status(stage,message):
        nonlocal armed,fault,latest_clock
        value=json.loads(message.data)
        keys=('run','state','reason','health','generation','wall_ns','fault_ns','neutral_ns','applied_ns','physics_ns','request_seq','output_seq',
            'issued_ns','deadline_ns','computed_ns','clock_ns','ready','last_reject','request_age_sec',
            'period_sec','feedback_stamp_sec','feedback_age_sec','backend','failure_reason','enabled','requests_sent')
        keys+=('mission_id','waypoint_index','source_waypoint_index','execution_phase','phase_reached',
               'segment_yaw_rad','reached','distance_m','yaw_error_rad','target_velocity')
        write(stage,{key:value[key] for key in keys if key in value})
        if stage=='terminal':
            if 'clock_ns' in value:latest_clock=int(value['clock_ns'])
            if value.get('state')=='ARMED':armed=True
            if armed and value.get('state')=='FAULT':fault=True

    def body(stage,message):
        stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        # A receipt-time diagnostic paired with the latest received terminal
        # clock, not a replacement stamp or an estimator input.
        write(stage,dict(stamp_ns=stamp,clock_ns=latest_clock,
            clock_minus_stamp_ns=latest_clock-stamp if latest_clock is not None else None,
            frame=message.header.frame_id,child=message.child_frame_id))

    node.create_subscription(AuthorizedCommand,'/rov01/control/approved',approved,10)
    node.create_subscription(ActuatorOutput,'/rov01/control/output',controller,10)
    node.create_subscription(Odometry,'/okvis/okvis_odometry',lambda m:body('vio_raw_body',m),1)
    node.create_subscription(Odometry,'/rov01/state/odometry',lambda m:body('estimated_body',m),1)
    for stage,topic in (
        ('native_input','/rov01/sim/terminal/input'),('terminal','/rov01/sim/terminal/status'),
        ('guard','/rov01/control/status'),('controller_status','/rov01/control/controller_status'),
        ('navigation','/rov01/navigation/status'),
        ('gamepad','/pelagos/joystick/status')):
        node.create_subscription(String,topic,lambda m,s=stage:status(s,m),10)
    print(f'Read-only trace: {output}',flush=True)
    deadline=time.monotonic()+args.seconds
    try:
        while rclpy.ok() and time.monotonic()<deadline and not fault:rclpy.spin_once(node,timeout_sec=.05)
    finally:
        stream.close();node.destroy_node();rclpy.try_shutdown()
    print('Stopped: terminal fault captured' if fault else 'Stopped: observation interval finished',flush=True)


if __name__=='__main__':main()
