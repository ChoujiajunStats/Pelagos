"""Regression: held VIO feedback must not freeze a fresh command's output lease."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from nav_msgs.msg import Odometry
from uw_interfaces.msg import AuthorizedCommand, ControlRequest
from clock_control_tick import tick_with_clock


class Sink:
    def __init__(self):self.messages=[]
    def publish(self,message):self.messages.append(message)


class Bank:
    def __init__(self):
        self.allocation=NS(names=['a']*6);self.integral=np.zeros(4);self.attitude_integral=np.zeros(2)
        self.calls=[]
    def update(self,target,velocity,angles,dt):
        if not 0<dt<=.1:raise ValueError('Invalid state or simulation dt')
        self.calls.append((list(target),dt));return np.zeros(6),np.zeros(6),np.zeros(6),False


def fixture():
    now=10_000_000_000;s=Odometry();s.pose.pose.orientation.w=1.
    r=ControlRequest(run_id='run',generation=1,sequence=20,issued_steady_ns=now-30_000_000)
    r.command.twist.linear.x=.12
    a=AuthorizedCommand(request=r,deadline_steady_ns=now+170_000_000)
    n=NS(control=Bank(),command=a,state=s,state_valid=True,state_wall=now-80_000_000,
        guard={'state':'ARMED'},terminal={'state':'ARMED','generation':1},folder=Path('run'),
        backend='body_pi',clock_ns=5_100_000_000,clock_wall=now-5_000_000,stamp=5.02,
        last_tick=now-20_000_000,last_stamp=5.08,generation=1,last_update_error=None,
        last_feedback_stamp=5.02,output=Sink(),status=Sink(),stabilization=None)
    def reset():n.last_stamp=None;n.generation=None
    n.reset=reset
    return now,n


class ControlClockRegression(unittest.TestCase):
    def test_held_pose_recomputes_using_latest_authorization(self):
        now,n=fixture()
        with patch('clock_control_tick.time.monotonic_ns',return_value=now):tick_with_clock(n)
        self.assertEqual(len(n.output.messages),1)
        self.assertEqual(n.output.messages[0].authorization.request.sequence,20)
        self.assertEqual(n.output.messages[0].authorization.deadline_steady_ns,n.command.deadline_steady_ns)
        status=json.loads(n.status.messages[-1].data)
        self.assertTrue(status['held_feedback'])
        self.assertEqual(status['feedback_stamp_sec'],5.02)
        self.assertAlmostEqual(status['integration_stamp_sec'],5.1)
        self.assertAlmostEqual(n.control.calls[0][1],.02)

    def test_expired_request_is_not_renewed(self):
        now,n=fixture();n.command.deadline_steady_ns=now-1
        with patch('clock_control_tick.time.monotonic_ns',return_value=now):tick_with_clock(n)
        self.assertFalse(n.output.messages);self.assertFalse(n.control.calls)

    def test_stale_feedback_or_clock_stops_output(self):
        for field in ('state_wall','clock_wall'):
            now,n=fixture();setattr(n,field,now-250_000_001)
            with patch('clock_control_tick.time.monotonic_ns',return_value=now):tick_with_clock(n)
            self.assertFalse(n.output.messages);self.assertFalse(n.control.calls)

    def test_reversed_integration_time_is_not_executed(self):
        now,n=fixture();n.last_stamp=5.12
        with patch('clock_control_tick.time.monotonic_ns',return_value=now):tick_with_clock(n)
        self.assertFalse(n.output.messages);self.assertIsNotNone(n.last_update_error)


if __name__=='__main__':unittest.main()
