"""Source controller tick with an explicit simulation integration clock.

Derived from stonefish_slam_ws/controller/uw_controller/node.py::Controller.tick.
As in that source's TinyMPC branch, a fresh held estimator sample may be used
at each simulation tick. Feedback stamps and all leases retain their real age.
"""
import json
import time

from std_msgs.msg import String
from uw_controller.core import rpy
from uw_interfaces.msg import ActuatorOutput


def status_message(value):return String(data=json.dumps(value))


def tick_with_clock(self):
    now=time.monotonic_ns();period=(now-self.last_tick)*1e-9 if self.last_tick else 0.;self.last_tick=now
    active=(self.control and self.command and self.state and self.state_valid and
        self.guard.get('state')=='ARMED' and self.terminal.get('state')=='ARMED' and
        self.command.request.run_id==self.folder.name and
        self.command.request.generation==int(self.terminal.get('generation',0)) and
        now<self.command.deadline_steady_ns and now-self.state_wall<250_000_000 and
        self.clock_ns is not None and now-self.clock_wall<250_000_000 and
        -.05<=self.clock_ns*1e-9-self.stamp<=.25)
    if active and self.backend=='jepa_mppi':active=self.estimated_feedback_healthy(now)
    if not active:
        self.reset()
        self.status.publish(status_message(dict(run=self.folder.name,wall_ns=now,
            state='FAULT' if self.last_update_error else 'IDLE',failure_reason=self.last_update_error,
            dt=0.,period_sec=period,generation=0,integral=[0.]*4,attitude_integral=[0.]*2,
            request_age_sec=0.,backend=self.backend,
            ready_for_arm=bool(self.backend=='jepa_mppi' and self.control and self.state and self.state_valid
                and self.control.ready and self.estimated_feedback_healthy(now)),
            **(self.control.report() if self.backend=='jepa_mppi' and self.control else {}))))
        return
    gen=self.command.request.generation
    if gen!=self.generation:self.reset();self.generation=gen
    integration_stamp=self.clock_ns*1e-9
    if self.last_stamp is None:self.last_stamp=integration_stamp;return
    dt=integration_stamp-self.last_stamp
    if dt==0:return
    self.last_stamp=integration_stamp
    v=self.state.twist.twist;q=self.state.pose.pose.orientation;t=self.command.request.command.twist
    velocity=[v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z]
    angles=rpy([q.x,q.y,q.z,q.w]);target=[t.linear.x,t.linear.y,t.linear.z,t.angular.z]
    if self.backend=='jepa_mppi' and self.stabilization is not None:
        angles[:2]=self.stabilization[0][:2];velocity[3:5]=self.stabilization[1][:2]
    try:command,requested,predicted,saturated=self.control.update(target,velocity,angles,dt)
    except ValueError as error:
        self.last_update_error=str(error);self.reset();return
    now=time.monotonic_ns()
    if now>=self.command.deadline_steady_ns or now-self.state_wall>=250_000_000:self.reset();return
    if self.backend=='jepa_mppi' and not self.estimated_feedback_healthy(now):self.reset();return
    output=ActuatorOutput(authorization=self.command,names=self.control.allocation.names,
        setpoint=command.tolist(),saturated=saturated,computed_steady_ns=now)
    for wrench,values in ((output.requested_wrench,requested),(output.predicted_wrench,predicted)):
        wrench.force.x,wrench.force.y,wrench.force.z=map(float,values[:3])
        wrench.torque.x,wrench.torque.y,wrench.torque.z=map(float,values[3:])
    self.output.publish(output);self.last_update_error=None
    if self.backend=='jepa_mppi':
        self.control.last_output_authorization=dict(generation=gen,
            request_seq=self.command.request.sequence,
            issued_ns=self.command.request.issued_steady_ns,
            deadline_ns=self.command.deadline_steady_ns,computed_ns=now)
    held_feedback=self.last_feedback_stamp==self.stamp;self.last_feedback_stamp=self.stamp
    self.status.publish(status_message(dict(run=self.folder.name,wall_ns=now,dt=dt,period_sec=period,
        feedback_stamp_sec=self.stamp,integration_stamp_sec=integration_stamp,held_feedback=held_feedback,
        feedback_age_sec=(now-self.state_wall)*1e-9,generation=gen,integral=self.control.integral.tolist(),
        attitude_integral=self.control.attitude_integral.tolist(),
        request_age_sec=(now-self.command.request.issued_steady_ns)*1e-9,backend=self.backend,
        **(self.control.report() if self.backend=='jepa_mppi' else {}))))
