"""OpenVINS/pressure adapter for the retained dev-jepa-water checkpoint."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
import sys
import time

import numpy as np
from uw_robot.frames import Q_ENU_NED,Q_FRD_FLU,qmul,normalized,frd_to_flu,ned_to_enu
from uw_controller.core import attitude_wrench,integrate_attitude


CHECKPOINT_SHA256='e4e09e92f5ae00a6183c6f698ad32d9bcf671705d93d6b81e44e14d43f67974d'


def validate_feedback(config,frame_prefix):
    if config.get('state_fusion_profile') not in (None,'rtabmap_ekf'):
        raise ValueError('Unsupported JEPA-MPPI estimated feedback fusion')
    source=config.get('state_source')
    vins=(source=='vins_fusion_stereo_imu' and config.get('estimator_profile')=='vins_fusion_m3'
          and config.get('controller_backend')=='tinympc' and config.get('state_fusion_profile')=='rtabmap_ekf'
          and bool(config.get('planner_restore_map')))
    if (config.get('schema_version')!=4 or not (source=='openvins_stereo_imu' or vins)
            or config.get('control_state_source',source)!=source
            or config.get('survey_profile')=='known_route_capture_v1'
            or frame_prefix!=config['namespace']):
        raise ValueError('MPC requires an explicit supported M3 estimator without truth remapping')


def load_vendor(repository):
    source=Path(repository).expanduser().resolve()/'src/disaware_learning'
    if not (source/'disaware_learning/control/jepa_mppi.py').is_file():
        raise ValueError(f'dev-jepa-water source missing: {source}')
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    module=importlib.import_module('disaware_learning.control.jepa_mppi')
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError('A different disaware_learning installation is already imported')
    return module


class SensorWindow:
    """Select the latest causal sample; duplicate/reversed stamps never refresh it."""
    def __init__(self):self.samples=deque(maxlen=64)

    def append(self,stamp,value,received_ns=None):
        value=np.asarray(value,dtype=float)
        if not np.isfinite(stamp) or stamp<=0 or not np.isfinite(value).all():return False
        if self.samples and stamp<=self.samples[-1][0]:return False
        self.samples.append((stamp,value.copy(),time.monotonic_ns() if received_ns is None else received_ns))
        return True

    def at(self,stamp,now_ns,max_age=.15):
        for sample_stamp,value,received in reversed(self.samples):
            if sample_stamp<=stamp+1e-9:
                if stamp-sample_stamp<=max_age and 0<=now_ns-received<=int(max_age*1e9):return value
                break
        return None

    def segments(self,start,stop,now_ns):
        """Piecewise constant values executed over [start, stop)."""
        value=self.at(start,now_ns)
        if value is None or not start<stop:return None
        cursor=start;segments=[]
        for stamp,next_value,_ in self.samples:
            if start<stamp<stop:
                segments.append((stamp-cursor,value));cursor=stamp;value=next_value
        segments.append((stop-cursor,value))
        return segments


class CheckpointController:
    def __init__(self,parameters,allocation,settings,reference_allocation=None):
        self.p=parameters;self.allocation=allocation;self.settings=settings
        self.vendor=load_vendor(settings['repository_root'])
        root=Path(settings['repository_root']).expanduser().resolve()
        checkpoint=Path(settings['checkpoint']).expanduser().resolve()
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
            raise ValueError('Checkpoint SHA-256 differs from the requested full4096 four-modal model')
        import torch
        torch.set_num_threads(int(settings['torch_threads']))
        device=torch.device(settings['device'])
        if device.type=='cuda' and not torch.cuda.is_available():
            raise ValueError('JEPA-MPPI requested CUDA, but the ROS Python environment has no usable CUDA device')
        config=self.vendor.load_jepa_mppi_config(root/settings['mppi_config'])
        config=replace(config,**settings.get('mppi_overrides',{}))
        self.optimizer=self.vendor.ReferenceConditionedJepaMppi.from_checkpoint(checkpoint,config=config,device=device)
        from disaware_learning.deployment.history import CausalHistoryBuffer
        from disaware_learning.learning.motor_jepa_data import modal_to_motor_output,motor_output_to_modal,quaternion_to_rotation6d,gravity_compensated_body_acceleration
        from disaware_learning.deployment.protocol import command_to_six_motor_normalized
        from disaware_learning.estimation.pressure_depth import pressure_to_depth_down_m
        self.history=CausalHistoryBuffer(self.optimizer.history_steps)
        self.modal_to_motor=modal_to_motor_output
        self.motor_to_modal=motor_output_to_modal;self.mix_command=command_to_six_motor_normalized
        self.rotation6d=quaternion_to_rotation6d;self.dynamic_acceleration=gravity_compensated_body_acceleration
        self.pressure_to_depth=pressure_to_depth_down_m
        self.reference_allocation=reference_allocation or allocation
        if self.reference_allocation.profile.get('variant')!='bluerov2_six_disaware':
            raise ValueError('Checkpoint action transfer requires the retained six-thruster allocation')
        self.model_dt=1/self.optimizer.sample_rate_hz
        self.target_tolerance=np.asarray(settings['target_change_tolerance'],dtype=float)
        if self.target_tolerance.shape!=(4,) or not np.isfinite(self.target_tolerance).all() or np.any(self.target_tolerance<0):
            raise ValueError('Invalid reference change tolerance')
        self.max_plan_age=min(float(config.maximum_plan_age_seconds),.2)
        if not 0<self.max_plan_age<=.2:raise ValueError('Invalid plan expiry')
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='jepa-mppi')
        self.future=None;self.worker_epoch=None;self.epoch=0
        self.imu=SensorWindow();self.pressure=SensorWindow();self.applied=SensorWindow()
        self.last_observation_stamp=None;self.sample_stamp=None;self.sample_count=0
        self.action_sum=np.zeros(5);self.action_duration=0.;self.position=None;self.depth_anchor=None
        self.current_modal=np.zeros(5)
        self.reason='history_warmup';self.diagnostics={};self.last_submission=None
        self.reset()

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

    def reset(self):
        self.epoch+=1;self.previous=np.zeros(len(self.allocation.names));self.integral=np.zeros(4);self.attitude_integral=np.zeros(2)
        self.plan=None;self.plan_target=None;self.plan_started_ns=0;self.last_submission=None

    def clear_history(self):
        self.reset();self.history.reset();self.sample_count=0;self.sample_stamp=None
        self.last_observation_stamp=None;self.action_sum.fill(0);self.action_duration=0.;self.depth_anchor=None
        self.reason='history_warmup'

    def on_imu(self,stamp,acceleration,gyro,received_ns=None):
        return self.imu.append(stamp,np.r_[acceleration,gyro],received_ns)

    def on_pressure(self,stamp,pressure,received_ns=None):
        depth=self.pressure_to_depth(pressure,fluid_density_kg_m3=float(self.settings['fluid_density_kg_m3']),
                                     surface_pressure_pa=float(self.settings['surface_pressure_pa']))
        return self.pressure.append(stamp,[depth],received_ns)

    def on_applied(self,stamp,setpoint,received_ns=None):
        values=np.asarray(setpoint,dtype=float)
        if values.shape!=(len(self.allocation.names),) or np.any(np.abs(values)>.6+1e-9):return False
        return self.applied.append(stamp,values,received_ns)

    def applied_modal(self,setpoint):
        if self.reference_allocation is self.allocation:motor=np.asarray(setpoint,dtype=float)
        else:motor,_,_=self.reference_allocation.allocate(self.allocation.predict(setpoint))
        return self.motor_to_modal(motor)

    def modal_to_wrench(self,modal):
        motor,_=self.mix_command(modal[:4],roll=float(modal[4]))
        return self.reference_allocation.predict(motor)

    def inputs_ready(self,stamp,now_ns):
        return all(window.at(stamp,now_ns) is not None for window in (self.imu,self.pressure,self.applied))

    def observe(self,stamp,position,quaternion,velocity,now_ns=None):
        now=time.monotonic_ns() if now_ns is None else now_ns
        imu=self.imu.at(stamp,now);depth=self.pressure.at(stamp,now);applied=self.applied.at(stamp,now)
        if imu is None or depth is None or applied is None:
            self.clear_history();self.reason='sensor_stale';return False
        if self.last_observation_stamp is not None:
            dt=stamp-self.last_observation_stamp
            if dt==0:return True
            if not 0<dt<=.1+1e-9:self.clear_history()
        q=normalized(qmul(qmul(Q_ENU_NED,normalized(quaternion)),Q_FRD_FLU))
        velocity=np.asarray(velocity,dtype=float);position=np.asarray(position,dtype=float)
        if velocity.shape!=(6,) or position.shape!=(3,) or not np.isfinite(np.r_[velocity,position]).all():
            self.clear_history();raise ValueError('Invalid estimated state')
        acceleration=np.asarray(frd_to_flu(imu[:3]));gyro=np.asarray(frd_to_flu(imu[3:]))
        dynamic=self.dynamic_acceleration(acceleration.reshape(1,3),np.asarray(q).reshape(1,4))[0]
        if self.depth_anchor is None:self.depth_anchor=float(depth[0])
        observation=np.r_[self.rotation6d(np.asarray(q).reshape(1,4))[0],frd_to_flu(velocity[:3]),gyro,dynamic,depth[0]-self.depth_anchor].astype(np.float32)
        self.position=np.asarray(ned_to_enu(position));self.position[2]=depth[0]
        modal=self.applied_modal(applied)
        if self.last_observation_stamp is not None:
            segments=self.applied.segments(self.last_observation_stamp,stamp,now)
            if segments is None:
                self.clear_history();self.reason='sensor_stale';return False
            for duration,setpoint in segments:
                self.action_sum+=duration*self.applied_modal(setpoint)
                self.action_duration+=duration
        self.current_modal=modal
        self.last_observation_stamp=stamp
        if self.sample_stamp is None:
            self.history.seed(observation,self.modal_to_motor(modal));self.sample_count=1;self.sample_stamp=stamp
        elif stamp-self.sample_stamp>=self.model_dt-1e-9:
            mean=self.action_sum/max(self.action_duration,1e-9)
            self.history.append_transition(observation,self.modal_to_motor(mean))
            self.sample_count+=1;self.sample_stamp=stamp;self.action_sum.fill(0);self.action_duration=0.
        return True

    @property
    def ready(self):return self.sample_count>=self.optimizer.history_steps

    def _solve(self,epoch,observation,action,target,previous,position,roll):
        if self.worker_epoch!=epoch:self.optimizer.reset(previous);self.worker_epoch=epoch
        reference=self.vendor.ReferenceTrajectory.body_twist(np.asarray(target)*[1,-1,-1,-1])
        return self.optimizer.optimize(observation,action,reference,previous,current_position_ned_m=position,roll_command=float(roll))

    def target_matches(self,target,planned_target):
        return bool(np.all(np.abs(target-planned_target)<=self.target_tolerance))

    def update(self,target,velocity,angles,dt):
        target=np.asarray(target,dtype=float)
        if target.shape!=(4,) or not 0<dt<=.1 or not np.isfinite(np.r_[target,velocity,angles,dt]).all():
            self.reset();raise ValueError('Invalid state or simulation dt')
        now=time.monotonic_ns()
        if self.plan_target is not None and not self.target_matches(target,self.plan_target):self.plan=None
        if self.future is not None and self.future.done():
            future,epoch,started,planned_target=self.future,self.job_epoch,self.job_started,self.job_target
            self.future=None
            try:
                result=future.result()
                self.diagnostics={key:result.diagnostics[key] for key in
                    ('timing_seconds','selected_cost','nominal_cost','supported_candidate_count') if key in result.diagnostics}
                self.diagnostics['solver_reason']=result.reason
                if (epoch==self.epoch and now-started<=self.max_plan_age*1e9
                        and self.target_matches(target,planned_target) and result.reason in ('accepted','no predicted improvement')):
                    command=np.asarray(result.command,dtype=float)
                    if command.shape!=(4,) or not np.isfinite(command).all():raise ValueError('Invalid MPPI command')
                    self.plan=command;self.plan_target=planned_target;self.plan_started_ns=started
                    self.reason=result.reason
                else:self.reason='discarded_plan'
            except Exception as error:
                self.plan=None;self.reason='solver_error';self.diagnostics={'error':str(error)}
        if self.plan is not None and now-self.plan_started_ns>self.max_plan_age*1e9:self.plan=None;self.reason='plan_expired'
        if self.ready and self.future is None and self.sample_stamp!=self.last_submission:
            observation,action=self.history.arrays();current=self.current_modal.copy()
            self.job_epoch=self.epoch;self.job_started=now;self.job_target=target.copy();self.last_submission=self.sample_stamp
            self.future=self.pool.submit(self._solve,self.epoch,observation,action,target.copy(),current[:4].astype(np.float32),self.position.copy(),current[4])
        modal=np.zeros(5) if self.plan is None else np.r_[self.plan,0.]
        desired=self.modal_to_wrench(modal)
        attitude_mask=np.array([3 in self.allocation.controlled_axes,4 in self.allocation.controlled_axes])
        desired[3:5]=attitude_wrench(self.p,velocity,angles,self.attitude_integral)*attitude_mask
        clipped=np.clip(desired,-np.asarray(self.p['wrench_limits']),self.p['wrench_limits'])
        command,predicted,saturated=self.allocation.allocate(clipped)
        step=self.p['setpoint_rate_per_sec']*dt
        limited=np.clip(command,self.previous-step,self.previous+step)
        saturated=saturated or bool(np.any(limited!=command) or np.any(clipped!=desired))
        self.previous=limited;predicted=self.allocation.predict(limited)
        self.attitude_integral=integrate_attitude(self.p,self.attitude_integral,angles,desired,predicted,dt)
        self.attitude_integral*=attitude_mask
        return limited,desired,predicted,saturated

    def report(self):
        return dict(model='stage2_zhat_response_potential_full4096_four_modal_e5',checkpoint_sha256=CHECKPOINT_SHA256,
                    device=str(self.optimizer.device),sample_count=self.optimizer.config.sample_count,
                    state_source=self.settings.get('state_source','OPENVINS_STEREO_IMU'),depth_source='PRESSURE_SENSOR',history_samples=min(self.sample_count,self.optimizer.history_steps),
                    model_rate_hz=self.optimizer.sample_rate_hz,reason=self.reason,**self.diagnostics)
