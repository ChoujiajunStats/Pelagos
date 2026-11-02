#!/usr/bin/env python3
"""Turn fresh desktop gamepad snapshots into the existing Guard's CLI requests."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import yaml
import rclpy
from rclpy.clock import Clock,ClockType
from rclpy.node import Node
from rosgraph_msgs.msg import Clock as ClockMsg
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from uw_interfaces.msg import ControlRequest
from uw_interfaces.srv import Control
from joystick_core import packet_error
from manual_envelope import state_diagnostic


class JoystickReceiver(Node):
    def __init__(self,out,config):
        super().__init__('pelagos_joystick',namespace=config['namespace'])
        self.out=out;self.config=config
        self.input_lock=threading.Lock();self.latest=None;self.input_closed=False
        self.packet=None;self.session=None;self.sequence=0
        self.released=False;self.pressed=False;self.blocked=False
        self.reset_previous=False;self.token='';self.generation=0
        self.future=None;self.action=None;self.action_started=0
        self.reason='release_LB_then_hold_to_enable'
        self.clock=None;self.clock_wall=0;self.clock_ns=0
        self.guard={};self.guard_wall=0;self.last_print=0;self.closed_since=None
        self.envelope=yaml.safe_load((out/'guard-parameters.yaml').read_text())['safety_envelope']
        self.body_diagnostic={};self.terminal={};self.controller={};self.requests_sent=0
        self.mode='autonomy' if config.get('planner_profile')=='ethz_nbvplanner' else 'manual'
        self.mode_pending=False;self.mode_wall=0
        self.pub=self.create_publisher(ControlRequest,'control/request',1)
        self.status=self.create_publisher(String,'/pelagos/joystick/status',1)
        self.client=self.create_client(Control,'control/authority')
        self.create_subscription(ClockMsg,'/clock',self.on_clock,1)
        self.create_subscription(String,'control/status',self.on_guard,1)
        state_topic='sim/ground_truth/odometry' if config.get('state_source')=='ground_truth_debug' else 'state/odometry'
        self.create_subscription(Odometry,state_topic,self.on_body,1)
        self.create_subscription(String,'sim/terminal/status',self.on_terminal,1)
        self.create_subscription(String,'control/controller_status',self.on_controller,1)
        if config.get('planner_profile')=='ethz_nbvplanner':
            self.create_subscription(String,'/pelagos/planner/status',self.on_mode,1)
        self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.thread=threading.Thread(target=self.read_input,daemon=True);self.thread.start()

    def read_input(self):
        try:
            while True:
                line=sys.stdin.buffer.readline(16384)
                if not line:break
                try:packet=json.loads(line)
                except (ValueError,UnicodeDecodeError):packet=None
                with self.input_lock:self.latest=packet
        finally:
            with self.input_lock:self.input_closed=True

    def on_clock(self,m):
        stamp=m.clock.sec*10**9+m.clock.nanosec
        if stamp!=self.clock_ns:self.clock_wall=time.monotonic_ns()
        self.clock_ns=stamp;self.clock=m.clock

    def on_guard(self,m):
        try:report=json.loads(m.data)
        except (ValueError,TypeError):return
        if report.get('run')==self.out.name:
            self.guard=report;self.guard_wall=time.monotonic_ns()

    def on_body(self,m):
        p,q,v=m.pose.pose.position,m.pose.pose.orientation,m.twist.twist
        self.body_diagnostic=state_diagnostic([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],
            [v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z],self.envelope)

    def on_terminal(self,m):
        try:report=json.loads(m.data)
        except (ValueError,TypeError):return
        self.terminal={key:report.get(key) for key in ('state','reason','applied_setpoint','clock_ns')}

    def on_controller(self,m):
        try:report=json.loads(m.data)
        except (ValueError,TypeError):return
        self.controller={key:report.get(key) for key in ('state','backend','dt','failure_reason')}

    def on_mode(self,message):
        value=json.loads(message.data)
        if value.get('run')==self.out.name:
            self.mode=value.get('control_mode','autonomy')
            self.mode_pending=bool(value.get('control_mode_pending'));self.mode_wall=time.monotonic_ns()

    def service(self,action):
        if self.future is not None:return
        if not self.client.service_is_ready():
            self.reason='waiting_for_control_guard';return
        self.future=self.client.call_async(Control.Request(run_id=self.out.name,action=action,source='cli'))
        self.action=action;self.action_started=time.monotonic_ns()

    def stop(self,reason):
        owned=bool(self.token)
        self.token='';self.reason=reason
        if (owned and self.guard.get('source')=='cli' and self.guard.get('generation')==self.generation
                and self.guard.get('state')=='ARMED'):self.service('DISARM')

    def tick(self):
        now=time.monotonic_ns()
        with self.input_lock:
            latest=self.latest;self.latest=None;closed=self.input_closed
        if latest is not None:
            error=packet_error(latest,now,self.session,self.sequence)
            if error:
                self.stop(error);self.packet=None;self.blocked=True
            else:
                self.packet=latest;self.session=latest['session'];self.sequence=latest['sequence']
        packet=self.packet
        input_valid=(packet is not None and packet['connected'] and not closed and
                     0<=now-packet['issued_ns']<=250_000_000)
        deadman=bool(input_valid and packet['deadman'])
        manual=(self.mode=='manual' and not self.mode_pending and
                (self.config.get('planner_profile')!='ethz_nbvplanner' or now-self.mode_wall<250_000_000))
        neutral=bool(input_valid and all(abs(v)<1e-8 for v in packet['velocity']))
        if not input_valid:
            self.released=False;self.pressed=False;self.blocked=True
        elif not deadman:
            self.released=True;self.blocked=False;self.pressed=False
        if self.future is not None and self.future.done():
            action=self.action
            try:response=self.future.result()
            except Exception as error:response=None;self.reason='service_error: '+str(error)
            self.future=None
            if action=='ARM' and response and response.accepted:
                if manual and deadman and not packet['stop'] and not self.blocked:
                    self.token=response.token;self.generation=response.generation;self.reason='enabled'
                else:
                    # LB may have been released while ARM was in flight.
                    self.service('DISARM');self.reason='enable_cancelled'
            elif response and not response.accepted:
                self.reason=f'{action}: {response.reason}'
                if action=='ARM':self.blocked=True
            elif response and action=='CLEAR_FAULT':self.reason='fault_cleared_hold_LB_again'
        if self.future is not None and now-self.action_started>2_000_000_000:
            self.stop('control_service_timeout');self.blocked=True
        clock_valid=self.clock is not None and now-self.clock_wall<250_000_000
        guard_valid=now-self.guard_wall<250_000_000
        if self.token and (not manual or not input_valid or not deadman or packet['stop'] or not clock_valid or not guard_valid):
            self.stop('LB_released' if input_valid and not deadman else 'stop_or_input_stale')
        if self.token and (self.guard.get('state')!='ARMED' or self.guard.get('source')!='cli' or
                           self.guard.get('generation')!=self.generation):
            # Allow the next guard heartbeat after a successful ARM service.
            if self.guard.get('generation',0)>=self.generation or self.guard.get('state')=='FAULT':
                self.stop('guard_'+str(self.guard.get('state'))+': '+str(self.guard.get('health')))
                self.blocked=True
        if input_valid and packet['stop']:
            self.stop('B_stop_release_LB_to_resume');self.blocked=True
        reset=bool(input_valid and packet['reset'])
        if reset and not self.reset_previous and not deadman and neutral and not self.token:
            self.service('CLEAR_FAULT')
        self.reset_previous=reset
        if deadman and not self.pressed:
            self.pressed=True
            if not manual:
                self.reason='select_manual_mode';self.blocked=True
            elif self.released and neutral and not packet['stop'] and clock_valid and guard_valid:
                if self.guard.get('state')=='DISARMED':self.service('ARM')
                else:self.reason='guard_not_disarmed: '+str(self.guard.get('health') or self.guard.get('state'));self.blocked=True
            else:
                self.reason='center_sticks_then_release_and_hold_LB';self.blocked=True
        if self.token and input_valid and deadman and not self.blocked:
            request=ControlRequest(run_id=self.out.name,source='cli',token=self.token,
                generation=self.generation,sequence=now,issued_steady_ns=packet['issued_ns'],
                valid_until_steady_ns=packet['issued_ns']+200_000_000)
            request.command.header.stamp=self.clock
            request.command.header.frame_id=self.config['namespace']+'/base_link'
            (request.command.twist.linear.x,request.command.twist.linear.y,
             request.command.twist.linear.z,request.command.twist.angular.z)=map(float,packet['velocity'])
            self.pub.publish(request);self.reason='enabled';self.requests_sent+=1
        report=dict(run=self.out.name,input='gamepad',guard_source='cli',enabled=bool(self.token),
                    control_mode=self.mode,
                    connected=bool(input_valid),reason=self.reason,guard=self.guard.get('state','WAITING'),
                    guard_health=self.guard.get('health'),velocity=packet['velocity'] if input_valid else [0.]*4,
                    body=self.body_diagnostic,terminal=self.terminal,controller=self.controller,
                    requests_sent=self.requests_sent,guard_request_seq=self.guard.get('request_seq'),
                    guard_last_reject=self.guard.get('last_reject'),
                    wall_ns=now)
        self.status.publish(String(data=json.dumps(report)))
        if now-self.last_print>1_000_000_000:
            message=json.dumps(report,ensure_ascii=False)
            print(message,flush=True)
            if not self.token and self.guard.get('health'):
                details='; '.join(self.body_diagnostic.get('violations',[])) or self.guard['health']
                print('控制未使能：'+details,flush=True)
            with (self.out/'logs/joystick.log').open('a') as log:log.write(message+'\n')
            (self.out/'joystick-status.json').write_text(message)
            self.last_print=now
        if closed and self.closed_since is None:self.closed_since=now

    def close(self):
        self.stop('joystick_exit')
        deadline=time.monotonic()+1.5
        while self.future is not None and not self.future.done() and time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.05)
        # Handle an ARM response arriving during shutdown, then revoke it.
        if self.future is not None and self.future.done() and self.action=='ARM':
            try:accepted=self.future.result().accepted
            except Exception:accepted=False
            self.future=None
            if accepted:
                self.service('DISARM')
                if self.future is not None:rclpy.spin_until_future_complete(self,self.future,timeout_sec=1.)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    config=yaml.safe_load((args.run_dir/'resolved_config.yaml').read_text())
    if config.get('mode')!='simulation_research' or config.get('backend')!='stonefish':
        raise ValueError('手柄入口只允许当前 Stonefish 仿真')
    lock=(args.run_dir/'joystick.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('当前仿真已有手柄控制进程')
    os.environ['ROS_DOMAIN_ID']=str(config['ros_domain_id'])
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=JoystickReceiver(args.run_dir,config);stopping=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    try:
        while rclpy.ok() and not stopping[0]:
            rclpy.spin_once(node,timeout_sec=.05)
            if node.closed_since is not None and time.monotonic_ns()-node.closed_since>500_000_000:break
    finally:
        node.close();node.destroy_node();rclpy.try_shutdown();lock.close()


if __name__=='__main__':main()
