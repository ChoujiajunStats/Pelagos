"""Explicit OKVIS2 extension of the installed Stonefish control adapters.

The guard, authorization protocol, JEPA history/solver and actuator watchdog
remain the installed implementations. No estimator is relabelled as OpenVINS.
"""
import importlib
import json
import math
from pathlib import Path
import sys
import time
from feedback_sampling import interpolate_feedback
from actuator_contract import model_motor_polarity


def install():
    import uw_runtime.localization_policy as policy
    import uw_controller.checkpoint as checkpoint
    original_source = policy.estimated_state_source
    original_validate = checkpoint.validate_feedback
    original_vendor = checkpoint.load_vendor
    original_controller=checkpoint.CheckpointController

    def source(config):
        if config.get('state_source') == 'okvis2_stereo_imu':
            return 'OKVIS2_STEREO_IMU'
        return original_source(config)

    def validate(config, frame_prefix):
        if config.get('state_source') != 'okvis2_stereo_imu':
            return original_validate(config, frame_prefix)
        if not (config.get('schema_version') == 4 and
                config.get('mode') == 'simulation_research' and
                config.get('control_state_source') == 'okvis2_stereo_imu' and
                config.get('controller_backend') == 'jepa_mppi' and
                config.get('planner_profile') == 'ethz_nbvplanner' and
                frame_prefix == config['namespace'] and
                not config.get('state_fusion_profile')):
            raise ValueError('Invalid explicit OKVIS2 / NBV / JEPA simulation configuration')

    def vendor(repository):
        root = Path(repository).resolve()
        if root != Path('/pelagos/TP/Controller'):
            return original_vendor(repository)
        sys.path.insert(0, str(root))
        module = importlib.import_module('disaware_learning.control.jepa_mppi')
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Unexpected JEPA source')
        return module

    policy.estimated_state_source = source
    checkpoint.validate_feedback = validate
    checkpoint.load_vendor = vendor
    class SampledCheckpointController(original_controller):
        def __init__(self,*args,**kwargs):
            self.feedback_previous=None;self.feedback_next=None
            super().__init__(*args,**kwargs)
            from sensor_alignment import aligned_sensor_window
            aligned_window=aligned_sensor_window(checkpoint.SensorWindow)
            for name in ('imu','pressure','applied'):
                original_window=getattr(self,name);window=aligned_window()
                window.samples=original_window.samples
                setattr(self,name,window)
            from jepa_timing import configured_plan_age
            self.max_plan_age=configured_plan_age(self.max_plan_age,self.settings,self.optimizer.config)
            import numpy as np
            from disaware_learning.control.body_twist_pid import BodyTwistPidConfig
            self.roll_config=BodyTwistPidConfig()
            self.motor_polarity=model_motor_polarity(self.allocation,self.mix_command)
            self.prewarm=None;self.prewarm_error=None;self.audit_stamp=None
            self.latest_feedback=None;self.latest_execution={};self.source_roll=0.

        def clear_history(self):
            self.feedback_previous=None;self.feedback_next=None
            super().clear_history()

        def observe(self,stamp,position,quaternion,velocity,now_ns=None):
            now=time.monotonic_ns() if now_ns is None else now_ns
            current=(stamp,list(position),list(quaternion),list(velocity))
            self.latest_feedback=dict(stamp=stamp,position=list(position),quaternion=list(quaternion),velocity=list(velocity))
            if not self.inputs_ready(stamp,now):
                self.clear_history();self.reason='sensor_stale';return False
            if (self.feedback_previous is None or stamp<=self.feedback_previous[0] or
                    stamp-self.feedback_previous[0]>self.model_dt+1e-9):
                self.clear_history()
                result=super().observe(stamp,position,quaternion,velocity,now)
                self.feedback_previous=current;self.feedback_next=stamp+self.model_dt
                return result
            while self.feedback_next<=stamp+1e-9:
                target=min(self.feedback_next,stamp)
                p,q,v=interpolate_feedback(self.feedback_previous,current,target)
                if not super().observe(target,p,q,v,now):
                    self.feedback_previous=current;self.feedback_next=stamp+self.model_dt
                    return False
                self.feedback_next+=self.model_dt
            # Also consume every native feedback sample. This preserves the
            # source adapter's short causal actuator intervals; only history
            # snapshots, not sensor/action processing, run on the model grid.
            if self.last_observation_stamp is None or stamp>self.last_observation_stamp+1e-9:
                if not super().observe(stamp,position,quaternion,velocity,now):
                    self.feedback_previous=current;self.feedback_next=stamp+self.model_dt
                    return False
            self.feedback_previous=current
            if original_controller.ready.fget(self) and self.prewarm is None:
                import numpy as np
                observation,action=self.history.arrays()
                self.prewarm=self.pool.submit(self._solve,self.epoch,observation,action,
                    np.zeros(4),self.current_modal[:4].copy(),self.position.copy(),self.current_modal[4])
            return True

        @property
        def ready(self):
            if not original_controller.ready.fget(self) or self.prewarm is None or not self.prewarm.done():return False
            try:self.prewarm.result()
            except Exception as error:self.prewarm_error=str(error);return False
            return True

        def applied_modal(self,setpoint):
            # Invert exactly the same shaft-polarity conversion on feedback.
            polarity=getattr(self,'motor_polarity',1.)
            return super().applied_modal(polarity*setpoint)

        def _solve(self,epoch,observation,action,target,previous,position,roll):
            # Record native optimizer timing and a single real input window.
            # This is diagnostic only; it neither changes the solve nor sends output.
            result=super()._solve(epoch,observation,action,target,previous,position,roll)
            if self.settings.get('run_dir'):
                import numpy as np
                out=Path(self.settings['run_dir'])
                snapshot=out/'jepa-real-history.npz'
                if not snapshot.exists():
                    np.savez(snapshot,observation=observation,action=action,target=target,
                             previous=previous,position=position,roll=roll)
                timing={key:float(value) for key,value in result.diagnostics.items()
                        if key.endswith('_seconds') and isinstance(value,(int,float))}
                with (out/'solver-evidence.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(wall_ns=time.monotonic_ns(),epoch=epoch,
                        target=list(map(float,target)),reason=result.reason,
                        command=result.command.tolist(),**timing),allow_nan=False)+'\n')
            return result

        def update(self,target,velocity,angles,dt):
            import numpy as np
            previous=self.previous.copy()
            # Reuse source checkpoint history, asynchronous solve and expiry.
            # Its generic force-PD output is replaced by the JEPA source mixer.
            super().update(target,velocity,angles,dt)
            cfg=self.roll_config
            self.source_roll=float(np.clip(-cfg.roll_kp*angles[0]-cfg.roll_rate_kd*velocity[3],
                                           -cfg.command_limit,cfg.command_limit))
            command=np.zeros(4) if self.plan is None else self.plan.copy()
            motors,_=self.mix_command(command,roll=self.source_roll)
            desired=self.motor_polarity*motors
            limit=float(self.allocation.profile['setpoint_limit'])
            bounded=np.clip(desired,-limit,limit)
            step=self.p['setpoint_rate_per_sec']*dt
            applied=np.clip(bounded,previous-step,previous+step)
            self.previous=applied;self.attitude_integral.fill(0.)
            requested=self.allocation.predict(desired);predicted=self.allocation.predict(applied)
            self.latest_execution=dict(target=list(map(float,target)),model_command=command.tolist(),
                roll_command=self.source_roll,native_command=applied.tolist(),
                actual_motor=self.motor_polarity*self.applied.samples[-1][1],plan_available=self.plan is not None)
            self.latest_execution['actual_motor']=self.latest_execution['actual_motor'].tolist()
            return applied,requested,predicted,bool(np.any(applied!=desired))

        def report(self):
            report=super().report()
            report.update(motor_polarity=self.motor_polarity,roll_source='BodyTwistPidConfig + JepaMppiRollout',
                          solve_deadline_seconds=self.optimizer.config.deadline_seconds,
                          maximum_plan_age_seconds=self.max_plan_age,
                          planner_latency_compensation_seconds=self.optimizer.config.planner_latency_compensation_seconds,
                          gpu_warmed=self.prewarm is not None and self.prewarm.done(),
                          warmup_error=self.prewarm_error)
            if self.latest_feedback is not None:
                now=time.monotonic_ns();stamp=self.latest_feedback['stamp']
                report['sensor_alignment']={name:getattr(self,name).alignment_diagnostic(stamp,now)
                    for name in ('imu','pressure','applied')}
            report['last_output_authorization']=getattr(self,'last_output_authorization',None)
            if self.prewarm is not None and not self.prewarm.done():report['reason']='gpu_warmup'
            elif self.ready and report['reason']=='history_warmup':report['reason']='ready_for_task'
            if self.sample_stamp!=self.audit_stamp and self.settings.get('run_dir'):
                self.audit_stamp=self.sample_stamp
                evidence=dict(wall_ns=time.monotonic_ns(),feedback=self.latest_feedback,
                    execution=self.latest_execution,**report)
                with (Path(self.settings['run_dir'])/'controller-evidence.jsonl').open('a') as stream:
                    stream.write(json.dumps(evidence,allow_nan=False)+'\n')
            return report
    checkpoint.CheckpointController=SampledCheckpointController

    # These packages are mounted as source modules in the simulation image.
    # Resolve their existing config directories without requiring a second
    # colcon installation of the same control algorithms.
    import ament_index_python.packages as packages
    original_share=packages.get_package_share_directory
    def share(name,*args,**kwargs):
        if name=='uw_controller':return '/stonefish-workspace/controller'
        if name=='uw_robot':return '/stonefish-workspace/robot'
        return original_share(name,*args,**kwargs)
    packages.get_package_share_directory=share


def main():
    role = sys.argv.pop(1)
    install()
    if role == 'guard':
        import uw_guard.node as guard
        original = guard.Guard
        class NbvGuard(original):
            def __init__(self):
                self.operator_mode='autonomy'
                super().__init__()
                validate = self.config.get('planner_profile') == 'ethz_nbvplanner'
                if not validate: raise ValueError('NBV guard requires its explicit planner profile')
                # No second spatial domain is placed around the source planner.
                # Keep the original guard's state, lease and actuator checks.
                self.estimated_horizontal_limit = float('inf')
                self.estimated_vertical_limit = float('inf')
                self.motion_evidence=None

            def on_state(self,message):
                q=message.pose.pose.orientation;v=message.twist.twist
                self.motion_evidence=dict(stamp_ns=message.header.stamp.sec*10**9+message.header.stamp.nanosec,
                    roll_deg=math.degrees(math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y))),
                    pitch_deg=math.degrees(math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))))),
                    linear_velocity=[v.linear.x,v.linear.y,v.linear.z],
                    angular_velocity=[v.angular.x,v.angular.y,v.angular.z])
                super().on_state(message)
                if self.operator_mode=='manual':
                    from joystick_core import COMMAND_LIMITS
                    # Reuse the already approved manual operating profile on
                    # the same estimated body state, with no XYZ geofence.
                    e=self.defaults['safety_envelope']
                    self.envelope_valid=(
                        max(abs(self.motion_evidence['roll_deg']),abs(self.motion_evidence['pitch_deg']))<e['roll_pitch_peak_deg'] and
                        max(abs(x) for x in self.motion_evidence['linear_velocity'])<e['linear_speed_peak_m_s'] and
                        max(abs(x) for x in self.motion_evidence['angular_velocity'])<2*COMMAND_LIMITS[3])

            def service(self,request,response):
                expected='cli' if self.operator_mode=='manual' else 'navigation'
                stale_disarm=(request.action=='DISARM' and self.state=='ARMED' and request.source!=self.source)
                if stale_disarm or (request.action=='ARM' and request.source!=expected):
                    response.accepted=False;response.reason='select_manual_mode' if request.source=='cli' else 'select_autonomy_mode'
                    if stale_disarm:response.reason='control_source_changed'
                    response.state=self.state;response.source=self.source;response.generation=self.generation
                    return response
                response=super().service(request,response)
                if request.action=='SELECT_SOURCE' and response.accepted:
                    self.operator_mode='manual' if request.source=='cli' else 'autonomy'
                    self.ready_since=None;self.ready_for_arm=False;self.envelope_valid=False
                return response

            def on_request(self,message):
                from joystick_core import COMMAND_LIMITS
                previous=self.defaults['velocity_limits']
                if self.operator_mode=='manual':self.defaults['velocity_limits']=list(COMMAND_LIMITS)
                try:super().on_request(message)
                finally:self.defaults['velocity_limits']=previous

            def fault(self,reason):
                armed=self.state=='ARMED'
                super().fault(reason)
                if armed and getattr(self,'motion_evidence',None):
                    self.fault_context['motion']=self.motion_evidence
                    with (self.run_dir/'control-fault-evidence.jsonl').open('a') as stream:
                        stream.write(json.dumps(self.fault_context)+'\n')
        guard.Guard = NbvGuard
        run = guard.main
    elif role == 'controller':
        import uw_controller.node as controller
        from uw_controller.core import BodyController
        from clock_control_tick import tick_with_clock
        from rosgraph_msgs.msg import Clock as ClockMsg
        original=controller.Controller
        class SelectableController(original):
            def __init__(self):
                self.manual_control=None;self.manual_selected=False
                super().__init__()
                self.create_subscription(ClockMsg,'/clock',self.on_clock,1)

            def on_terminal(self,message):
                super().on_terminal(message)
                if self.control is not None and self.manual_control is None:
                    self.manual_control=BodyController(self.p,self.control.allocation)

            def tick(self):
                manual=self.guard.get('source')=='cli'
                if manual!=self.manual_selected:
                    self.reset()
                    if self.manual_control:self.manual_control.reset()
                    self.manual_selected=manual
                if not manual or self.manual_control is None:return tick_with_clock(self)
                # One node retains the only output publisher. Sensor callbacks
                # always feed the JEPA history; only this tick chooses a bank.
                autonomous,self.control=self.control,self.manual_control
                backend,self.backend=self.backend,'body_pi'
                try:return tick_with_clock(self)
                finally:self.control=autonomous;self.backend=backend
        controller.Controller=SelectableController
        run=controller.main
    else:
        raise ValueError(role)
    run()


if __name__ == '__main__': main()
