#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Checkpoint + JEPA-MPPI for Wavestar paths, with physical-clearance and manual-priority HMU gates."""
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import time
import urllib.request

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'TP/Controller'))
sys.path.insert(0,str(ROOT/'Deployment/contracts'))
from disaware_learning.control.jepa_mppi import ReferenceConditionedJepaMppi,ReferenceTrajectory,load_jepa_mppi_config
from wire import send_packet,receive_packet
from collision import swept_sphere_status
from geometry import LocalFrame,timed_reference,minimum_centered_sphere_radius
from history import RealHistory
from clock_sync import ClockAlignment
from motor_contract import CONTRACT_ID,MODEL_ORDER


def api(path, payload=None):
    request=urllib.request.Request('http://127.0.0.1:8765/'+path,
        data=None if payload is None else json.dumps(payload,allow_nan=False).encode(),
        headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(request,timeout=.4))


class Runtime:
    def __init__(self,args):
        self.args=args;self.config=json.loads(Path(args.config).read_text())
        self.max_pose_age_s=float(self.config['max_pose_age_s'])
        if not math.isfinite(self.max_pose_age_s) or self.max_pose_age_s<=0:
            raise ValueError('max_pose_age_s must be finite and positive')
        self.minimum_body_radius=minimum_centered_sphere_radius(self.config['body_dimensions_m'])
        self.states=queue.Queue(maxsize=256);self.telemetry=queue.Queue(maxsize=512)
        self.latest_state=None;self.health=None;self.latest_status={'type':'control_status','state':'starting'}
        self.frame=None;self.last_body=None;self.last_goal=None;self.yaw_anchor=0.;self.previous_command=np.zeros(4)
        self.pose_jump_fault=False
        self.proof_saved=False
        self.tx_generation=None
        self.motor_frame_contract_verified=False
        self.clock=ClockAlignment();self.clock_fault=None
        self.pressure=deque(maxlen=256);self.world=deque(maxlen=256)
        self.history=RealHistory();self.map_stamp=0;self.map_data=None;self.map_metadata=None
        self.map_cache=Path(os.environ.get('XDG_RUNTIME_DIR','/tmp'))/'dapeng-wavestar/planning_roi.npz'
        self.log_path=ROOT/'runs/controller/runtime-metrics.jsonl'
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.checkpoint=ROOT/'TP/Controller/disaware_learning/checkpoints/stage2_zhat_response_potential_full4096_four_modal_e5.pt'
        expected='e4e09e92f5ae00a6183c6f698ad32d9bcf671705d93d6b81e44e14d43f67974d'
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint hash mismatch')
        torch.set_num_threads(1)
        config=load_jepa_mppi_config(ROOT/'TP/Controller/disaware_learning/config/jepa_mppi_feedback_search.json')
        config=replace(config,manual_limit=tuple(self.config['command_limits']),execution_hold_steps=2)
        if self.config['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no silent CPU fallback')
        self.controller=ReferenceConditionedJepaMppi.from_checkpoint(self.checkpoint,config=config,device=self.config['device'])
        threading.Thread(target=self.network,daemon=True).start()
        threading.Thread(target=self.sensors,daemon=True).start()
        print(json.dumps(dict(event='model_loaded',checkpoint_sha256=expected,device=self.config['device'],
                             history_steps=self.controller.history_steps,mode='shadow',
                             max_pose_age_s=self.max_pose_age_s)),flush=True)

    def pose_age_valid(self,age):
        return math.isfinite(age) and -.1<age<self.max_pose_age_s

    def network(self):
        while True:
            try:
                with socket.create_connection(('127.0.0.1',self.args.port),timeout=3) as c:
                    c.settimeout(2)
                    while True:
                        s,payload=receive_packet(c)
                        if payload or s.get('type')!='state':raise ValueError('Invalid ROS gateway data')
                        try:s=self.clock.align(s,time.time_ns())
                        except ValueError as e:
                            self.clock_fault=str(e);raise
                        if s is not None:
                            self.latest_state=s
                            for sample in s.get('samples',[]):
                                try:self.states.put_nowait(sample)
                                except queue.Full:pass
                        send_packet(c,dict(self.latest_status,host_send_ns=time.time_ns()))
            except (OSError,EOFError,ValueError) as e:
                self.latest_state=None
                print('ROS gateway:',str(e),flush=True);time.sleep(1)

    def sensors(self):
        cursor=0;generation=None
        while True:
            started=time.monotonic()
            try:
                health=api('health');ledger=api('api/autonomy/tx?after='+str(cursor))
                if generation is not None and generation!=ledger['generation']:
                    cursor=0;generation=ledger['generation'];raise ValueError('HMU bridge restarted')
                generation=ledger['generation']
                if ledger['last_tx_sequence']<cursor:cursor=0;raise ValueError('HMU TX generation reset')
                if ledger['events']:cursor=ledger['events'][-1]['sequence']
                self.health=health
                self.telemetry.put((health,ledger),timeout=.05)
            except (OSError,ValueError,queue.Full) as e:
                self.health=None
            time.sleep(max(.01,.1-(time.monotonic()-started)))

    def publish(self,state, *, proposal=None,predicted=None,reference=None,detail=None):
        p=proposal or dict(command=[0.]*4,eligible=False,collision_radius_m=self.config['collision_radius_m'],
                           map_age_s=99.,source_age_s=99.,solve_age_s=99.,
                           max_pose_age_s=self.max_pose_age_s,
                           actuation_contract=CONTRACT_ID,motor_frame_contract_verified=False,action_history_verified=False)
        if detail is None:
            s=self.latest_state or {};now=time.time_ns()
            detail=dict(clock_offset_ms=None if self.clock.offset_ns is None else self.clock.offset_ns*1e-6,
                clock_rtt_ms=None if self.clock.round_trip_ns is None else self.clock.round_trip_ns*1e-6,
                body_age_s=(now-s['body']['stamp_ns'])*1e-9 if s.get('body') else None,
                visual_age_s=(now-s.get('visual_stamp_ns',0))*1e-9,
                imu_skew_s=abs(s['imu']['stamp_ns']-s['body']['stamp_ns'])*1e-9 if s.get('body') else None,
                history_span_s=self.history.observations[-1][0]-self.history.observations[0][0] if self.history.observations else 0,
                observations=len(self.history.observations),tx_events=len(self.history.tx),
                pressure_samples=len(self.pressure))
        self.latest_status=dict(type='control_status',state=state,mode='shadow' if not self.health else 'HMU_mux',
            stamp_ns=time.time_ns(),proposal=p,predicted_world=[] if predicted is None else predicted,
            reference_world=[] if reference is None else reference,detail=detail)
        try:api('api/autonomy/proposal',p)
        except (OSError,ValueError):pass
        # Bound the scalar log; no sensor images, bags or unbounded tensor traces.
        if self.log_path.exists() and self.log_path.stat().st_size>16*1024*1024:self.log_path.write_text('')
        with self.log_path.open('a') as f:
            f.write(json.dumps({k:v for k,v in self.latest_status.items() if k not in ('predicted_world','reference_world')},allow_nan=False)+'\n')

    def ingest(self):
        while True:
            try:h,ledger=self.telemetry.get_nowait()
            except queue.Empty:break
            if ledger['generation']!=self.tx_generation:
                self.tx_generation=ledger['generation'];self.history=RealHistory();self.pressure.clear()
            self.motor_frame_contract_verified=bool(ledger.get('actuation_contract')==CONTRACT_ID
                and ledger.get('autonomy_protocol')=='motor'
                and ledger.get('motor_channels')==self.config['motor_channels']==list(MODEL_ORDER)
                and ledger.get('motor_signs')==self.config['motor_signs']
                and ledger.get('unused_motor_pwm')=={'L4':1500,'R4':1500})
            self.history.append_tx(ledger['events'])
            if ledger.get('pressure_stamp_ns') and ledger.get('pressure_depth_m') is not None:
                stamp=ledger['pressure_stamp_ns']*1e-9
                if not self.pressure or stamp>self.pressure[-1][0]:self.pressure.append((stamp,ledger['pressure_depth_m']))
            if ledger['events']:
                e=ledger['events'][-1];self.previous_command=np.asarray(e['command'] if e['armed'] else [0.]*4,dtype=np.float32)
        while True:
            try:s=self.states.get_nowait()
            except queue.Empty:break
            if not self.pressure:continue
            b=s['body'];imu=s['imu'];stamp=b['stamp_ns']*1e-9
            if self.last_body and b['stamp_ns']<=self.last_body['stamp_ns']:continue
            if abs(imu['stamp_ns']-b['stamp_ns'])*1e-9>self.config['max_imu_skew_s']:continue
            ts=np.asarray([x[0] for x in self.pressure]);depths=np.asarray([x[1] for x in self.pressure])
            if stamp<ts[0] or stamp>ts[-1]+.15:continue
            depth=float(np.interp(stamp,ts,depths))
            if self.last_body and math.dist(b['position'],self.last_body['position'])>self.config['max_pose_jump_m']:
                self.history.observations.clear();self.world.clear();self.last_goal=None
                self.pose_jump_fault=True
            self.last_body=b
            if self.frame is None:self.frame=LocalFrame(b['position'],b['quaternion'],depth,self.config['body_to_frd'])
            imu=dict(imu);rotation=np.asarray(self.config['imu_to_body'])
            imu['acceleration']=(rotation@imu['acceleration']).tolist();imu['gyro']=(rotation@imu['gyro']).tolist()
            obs,pos,_=self.frame.state(b,imu,depth)
            self.history.append(stamp,obs,pos);self.world.append((stamp,np.asarray(b['position'])))

    def clearance(self,poses):
        if not self.map_cache.is_file():return 'planning_map_unavailable',99.
        stamp=self.map_cache.stat().st_mtime_ns
        if stamp!=self.map_stamp:
            with np.load(self.map_cache,allow_pickle=False) as z:
                self.map_data=z['data'].copy();self.map_metadata=json.loads(str(z['metadata']))
            self.map_stamp=stamp
        m=self.map_metadata;t=m['stamp'][0]+m['stamp'][1]*1e-9
        age=time.time()-t-(self.clock.offset_ns or 0)*1e-9
        if not -1<age<self.config['max_map_age_s']:return 'planning_map_stale',age
        checked=swept_sphere_status(self.map_data,m,poses,self.config['collision_radius_m'])
        return checked,age

    def tick(self):
        self.ingest();s=self.latest_state
        if self.clock_fault:raise ValueError(self.clock_fault)
        if self.pose_jump_fault:raise ValueError('VIO_pose_jump_latched_restart_required')
        if s is None:raise ValueError('waiting_for_robot_state')
        h=self.health
        if h is None or not h['rov']['connected'] or not h.get('main'):raise ValueError('waiting_for_HMU_telemetry_and_successful_TX')
        if h['main'].get('leak_detected'):raise ValueError('leak_detected')
        b=s['body'];now=time.time();pose_age=now-b['stamp_ns']*1e-9
        if not self.pose_age_valid(pose_age):raise ValueError('VIO_state_stale')
        if now-s.get('visual_stamp_ns',0)*1e-9>2.5:raise ValueError('VIO_visual_update_stale')
        if now-s.get('depth_stamp_ns',0)*1e-9>3.:raise ValueError('mapping_depth_stale')
        if not self.pressure or now-self.pressure[-1][0]>self.config['max_pressure_age_s']:raise ValueError('pressure_depth_stale')
        obs,actions,position,endpoint=self.history.arrays()
        ref=s.get('reference')
        if not ref or ref.get('frame')!='world' or len(ref.get('poses',[]))<2:raise ValueError('waiting_for_planner_path')
        if now-ref['stamp_ns']*1e-9>self.config['max_path_age_s']:raise ValueError('planner_path_stale')
        rotation=np.column_stack((obs[-1,:3],obs[-1,3:6],np.cross(obs[-1,:3],obs[-1,3:6])))
        goal_key=(ref.get('session_id'),ref['goal_sequence'])
        if goal_key!=self.last_goal:
            self.last_goal=goal_key;self.yaw_anchor=math.atan2(rotation[1,0],rotation[0,0])
            self.controller.reset(self.previous_command)
        world_times=np.asarray([x[0] for x in self.world]);world_positions=np.asarray([x[1] for x in self.world])
        anchor=np.asarray([np.interp(endpoint,world_times,world_positions[:,i]) for i in range(3)])
        path=np.asarray([p[:3] for p in ref['poses']],dtype=float)
        points=self.frame.path_to_ned(path,anchor,position)
        times=np.arange(self.controller.model_horizon_steps+1,dtype=np.float32)/self.controller.sample_rate_hz
        target,velocity=timed_reference(points,position,rotation@obs[-1,6:9],times,
            speed=self.config['reference_speed_m_s'],acceleration=self.config['reference_acceleration_m_s2'])
        reference=ReferenceTrajectory.planner(time_seconds=times,position_ned_m=target,
            linear_velocity_ned_m_s=velocity,yaw_rad=self.yaw_anchor,yaw_rate_rad_s=0.)
        started=time.monotonic()
        result=self.controller.optimize(obs,actions,reference,self.previous_command,current_position_ned_m=position)
        elapsed=time.monotonic()-started
        if result.accepted and not self.proof_saved:
            np.savez(self.log_path.with_name('real-input-proof.npz'),observations=obs,actions=actions,
                position_ned=position,world_anchor=anchor,history_endpoint=endpoint,
                previous_successful_TX_command=self.previous_command,planner_reference_world=path,
                checkpoint_sha256='e4e09e92f5ae00a6183c6f698ad32d9bcf671705d93d6b81e44e14d43f67974d')
            self.proof_saved=True
        delta=result.predicted_position_delta_ned_m
        if delta is None:raise ValueError('model_did_not_return_position_rollout')
        prediction=self.frame.predicted_to_world(delta,anchor)
        predicted=np.vstack((anchor,prediction))
        if len(predicted)>1:
            direction=predicted[-1]-predicted[-2];norm=np.linalg.norm(direction)
            if norm>1e-6:predicted=np.vstack((predicted,predicted[-1]+direction/norm*self.config['braking_distance_m']))
        checked,map_age=self.clearance(predicted)
        path_checked,_=self.clearance(path)
        current=self.latest_state.get('reference') if self.latest_state else None
        goal_current=current and (current.get('session_id'),current.get('goal_sequence'))==goal_key and len(current.get('poses',[]))>=2
        session_current=bool(ref.get('session_id') and ref['session_id']==(self.map_metadata or {}).get('session_id'))
        source_age=time.time()-endpoint
        eligible=bool(result.accepted and goal_current and session_current and elapsed<.16 and self.pose_age_valid(source_age)
            and checked=='free' and path_checked=='free' and self.motor_frame_contract_verified
            and self.history.action_contract_verified and self.config['collision_radius_m']>=self.minimum_body_radius)
        command=[float(x) for x in result.command]
        proposal=dict(command=command,eligible=eligible,collision_radius_m=self.config['collision_radius_m'],
            body_dimensions_m=self.config['body_dimensions_m'],minimum_centered_radius_m=self.minimum_body_radius,
            map_age_s=map_age,source_age_s=source_age,solve_age_s=elapsed,
            max_pose_age_s=self.max_pose_age_s,
            actuation_contract=CONTRACT_ID,motor_frame_contract_verified=self.motor_frame_contract_verified,
            action_history_verified=self.history.action_contract_verified,
            hardware_response_validated=self.config['hardware_response_validated'],goal_sequence=ref['goal_sequence'],
            session_id=ref.get('session_id'),
            map_sequence=(self.map_metadata or {}).get('sequence',0),model_accepted=bool(result.accepted),model_reason=result.reason,
            prediction_check=checked,reference_check=path_checked,
            action_source=('successful_motor_TX_or_disarmed_zero' if self.history.action_contract_verified
                           else 'channel_TX_reconstructed_for_shadow_only'),
            action_is_measured_thrust=False)
        state=('goal_hold' if np.linalg.norm(path[-1]-anchor)<self.config['position_tolerance_m'] else 'controller_ready') if eligible else 'shadow_only'
        blockers=[]
        if not result.accepted:blockers.append('model_'+result.reason)
        if not goal_current:blockers.append('goal_changed_during_solve')
        if not session_current:blockers.append('map_reference_session_mismatch')
        if elapsed>=.16:blockers.append('solve_deadline_exceeded')
        if not self.pose_age_valid(source_age):blockers.append('state_age_after_solve')
        if checked!='free':blockers.append('prediction_'+checked)
        if path_checked!='free':blockers.append('reference_'+path_checked)
        if not self.motor_frame_contract_verified:blockers.append('vendor_motor_contract_mismatch')
        if not self.history.action_contract_verified:blockers.append('manual_channel_history_not_motor_verified')
        if self.config['collision_radius_m']<self.minimum_body_radius:blockers.append('collision_sphere_does_not_enclose_body')
        proposal['blockers']=blockers
        reference_world=(target-position)@self.frame.world_to_ned+anchor
        self.publish(state,proposal=proposal,predicted=predicted.tolist(),reference=reference_world.tolist(),
            detail=dict(solve_ms=elapsed*1000,history_shape=list(obs.shape),action_shape=list(actions.shape),
                        model_accepted=bool(result.accepted),model_reason=result.reason,
                        clock_offset_ms=self.clock.offset_ns*1e-6,clock_rtt_ms=self.clock.round_trip_ns*1e-6))

    def run(self):
        while True:
            started=time.monotonic()
            try:self.tick()
            except (ValueError,OSError,KeyError,RuntimeError) as e:self.publish(str(e))
            time.sleep(max(.005,1/self.config['solve_hz']-(time.monotonic()-started)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str((Path(__file__).resolve().parent.parent / 'config/config.json')))
    p.add_argument('--port',type=int,default=18781);args=p.parse_args();Runtime(args).run()


if __name__=='__main__':main()
