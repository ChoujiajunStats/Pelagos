#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Load the existing HMU bridge with a manual-priority controller input and TX ledger."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import uuid
from aiohttp import web
from hmu_bridge import server
from hmu_bridge.config import load_config
from hmu_bridge.protocol import ControlCommand, pack_channel_command,pack_motor_command
from motor_contract import CONTRACT_ID,MODEL_ORDER,decode_motor_frame


class MotorControlCommand(ControlCommand):
    """Explicit per-command transport tag; manual commands keep their AA frame."""


class ControlMux:
    def __init__(self, runtime, clock=time.monotonic, control_config=None):
        self.runtime=runtime;self.clock=clock;self.manual=runtime.rov.command_provider
        if control_config is None:
            control_config=json.loads((Path(__file__).resolve().parent.parent / 'config/config.json').read_text())
        self.max_pose_age_s=float(control_config['max_pose_age_s'])
        if not math.isfinite(self.max_pose_age_s) or self.max_pose_age_s<=0:
            raise ValueError('max_pose_age_s must be finite and positive')
        self.mode='shadow';self.proposal=None;self.received=0.;self.fault=None
        self.events=deque(maxlen=512);self.sequence=0
        self.motor_output_active=False
        self.pending_motor_neutral=True
        self.generation=uuid.uuid4().hex;self.pressure_stamp_ns=0
        if tuple(runtime.config.rov.motor_channels)!=MODEL_ORDER:
            raise ValueError('Vendor/model motor order must be R1,L1,R3,L3,R2,L2')

    def manual_or_motor_neutral(self,manual):
        if self.motor_output_active or self.pending_motor_neutral:
            self.pending_motor_neutral=True
            return MotorControlCommand.neutral()
        return manual

    def propose(self, value):
        command=value.get('command',[])
        if len(command)!=4 or not all(math.isfinite(x) and abs(x)<=.20 for x in command):
            raise ValueError('Expected four normalized axes limited to 0.20')
        self.proposal=dict(value);self.received=self.clock()

    def eligible(self):
        p=self.proposal or {}
        proposal_age=self.clock()-self.received
        source_age=p.get('source_age_s',99)+proposal_age
        return (proposal_age<.30 and p.get('eligible') is True
                and p.get('collision_radius_m',0)>=.35 and p.get('map_age_s',99)<3.
                and -.1<source_age<self.max_pose_age_s and p.get('solve_age_s',99)<.20
                and p.get('motor_frame_contract_verified') is True
                and p.get('action_history_verified') is True
                and p.get('actuation_contract')==CONTRACT_ID)

    def set_mode(self, mode):
        if mode not in ('manual','shadow','auto'):raise ValueError('Unknown control mode')
        if mode=='auto':
            if self.runtime.safety.snapshot()['armed']:raise ValueError('DISARM before selecting auto')
            if not self.eligible():raise ValueError('Controller inputs / physical clearance are not ready')
        self.mode=mode;self.fault=None

    def output(self):
        manual=self.manual()
        if self.pending_motor_neutral:return MotorControlCommand.neutral()
        if self.mode!='auto':return self.manual_or_motor_neutral(manual)
        if any(abs(getattr(manual,k))>.08 for k in ('surge','sway','heave','yaw')):
            self.mode='manual';self.fault='manual_override';return self.manual_or_motor_neutral(manual)
        if manual.gear!=1 or manual.heading_hold or manual.depth_hold:
            self.mode='manual';self.fault='manual_auxiliary_mode_changed';return self.manual_or_motor_neutral(manual)
        if not manual.armed:return self.manual_or_motor_neutral(manual)
        if not self.runtime.rov.state.connected or not self.eligible():
            self.mode='shadow';self.fault='controller_input_expired_or_rejected';return self.manual_or_motor_neutral(manual)
        if (self.runtime.main_telemetry or {}).get('leak_detected'):
            self.mode='shadow';self.fault='leak_detected';return self.manual_or_motor_neutral(ControlCommand.neutral())
        surge,sway,heave,yaw=self.proposal['command']
        return MotorControlCommand(**dict(asdict(manual),surge=surge,sway=sway,heave=heave,yaw=yaw,
                       heading_hold=False,depth_hold=False,gear=1))

    def pack(self,command):
        if isinstance(command,MotorControlCommand):
            return pack_motor_command(command,self.runtime.config.rov.motor_channels,self.runtime.config.rov.motor_signs)
        return pack_channel_command(command)

    def transmitted(self, frame):
        if len(frame)!=24 or frame[0]!=255:return
        if frame[23]==187:
            fields=decode_motor_frame(frame,self.runtime.config.rov.motor_channels,self.runtime.config.rov.motor_signs)
            source='successful_motor_tx'
            if fields['armed']:
                self.motor_output_active=True
            elif all(v==1500 for v in fields['motor_pwm'].values()):
                self.motor_output_active=False
                self.pending_motor_neutral=False
        elif frame[23]==170:
            axes=[(int.from_bytes(frame[k:k+2],'big')-1500)/500 for k in (1,5,7,3)]
            fields=dict(protocol='channel',command=axes,armed=bool(frame[18]),gear=frame[12],
                        heading_hold=bool(frame[10]),depth_hold=bool(frame[11]))
            source='successful_channel_tx'
        else:return
        self.sequence+=1
        self.events.append(dict(sequence=self.sequence,stamp_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),**fields,
            frame_hex=frame.hex(),source=source,mode=self.mode))

    def state(self):
        return dict(mode=self.mode,fault=self.fault,eligible=self.eligible(),
            max_pose_age_s=self.max_pose_age_s,
            motor_output_active=self.motor_output_active,pending_motor_neutral=self.pending_motor_neutral,
            autonomy_protocol='motor',actuation_contract=CONTRACT_ID,
            motor_channels=list(self.runtime.config.rov.motor_channels),motor_signs=list(self.runtime.config.rov.motor_signs),
            unused_motor_pwm={'L4':1500,'R4':1500},
            generation=self.generation,pressure_stamp_ns=self.pressure_stamp_ns,
            pressure_depth_m=(self.runtime.main_telemetry or {}).get('depth_m'),
            proposal_age_s=None if self.proposal is None else self.clock()-self.received,
            last_tx_sequence=self.sequence,last_tx=self.events[-1] if self.events else None)


def create_app(config, *, start_rov=True, control_config=None):
    if config.rov.command_protocol!='channel':raise ValueError('This adapter requires the existing channel protocol')
    app=server.create_app(config,start_rov=start_rov);runtime=app[server.RUNTIME_KEY]
    mux=ControlMux(runtime,control_config=control_config);runtime.rov.command_provider=mux.output
    runtime.rov._pack_command=mux.pack
    original_frame=runtime.rov.on_frame
    def frame(parsed):
        before=runtime.main_telemetry
        original_frame(parsed)
        if runtime.main_telemetry is not before:mux.pressure_stamp_ns=time.time_ns()
    runtime.rov.on_frame=frame
    original_write=runtime.rov._write
    async def write(payload, *, heartbeat=False):
        await original_write(payload,heartbeat=heartbeat)
        if not heartbeat:mux.transmitted(payload)
    runtime.rov._write=write
    async def neutral():
        if runtime.rov.config.mode!='control' or runtime.rov._writer is None:return
        use_motor=mux.motor_output_active or mux.pending_motor_neutral or (mux.events and mux.events[-1]['protocol']=='motor')
        command=MotorControlCommand.neutral() if use_motor else ControlCommand.neutral()
        try:await runtime.rov._write(mux.pack(command))
        except (ConnectionError,OSError):pass
    runtime.rov.send_neutral_now=neutral
    async def status(request):return web.json_response(mux.state())
    async def tx(request):
        try:after=int(request.query.get('after','0'))
        except ValueError:return web.json_response({'error':'Invalid TX cursor'},status=400)
        return web.json_response(dict(events=[x for x in mux.events if x['sequence']>after],**mux.state()))
    async def proposal(request):
        try:mux.propose(await request.json());return web.json_response(mux.state())
        except (ValueError,TypeError):return web.json_response({'error':'Invalid controller proposal'},status=400)
    async def mode(request):
        try:
            mux.set_mode((await request.json())['mode'])
            if mux.mode!='auto' and (mux.motor_output_active or mux.pending_motor_neutral):
                await runtime.neutral_now()
            return web.json_response(mux.state())
        except (ValueError,KeyError,TypeError) as error:return web.json_response({'error':str(error)},status=409)
    async def neutral_probe(request):
        safety=runtime.safety.snapshot()
        if safety.get('armed') or safety.get('arm_pending') or safety.get('has_control') or mux.mode=='auto':
            return web.json_response({'error':'Neutral probe requires DISARM, no control owner, and manual/shadow mode'},status=409)
        if not runtime.rov.state.connected or runtime.rov._writer is None:
            return web.json_response({'error':'ROV disconnected'},status=503)
        try:await runtime.rov._write(mux.pack(MotorControlCommand.neutral()))
        except (OSError,ConnectionError):return web.json_response({'error':'Neutral write failed'},status=503)
        return web.json_response(mux.state())
    app.router.add_get('/api/autonomy/state',status)
    app.router.add_get('/api/autonomy/tx',tx)
    app.router.add_post('/api/autonomy/proposal',proposal)
    app.router.add_post('/api/autonomy/mode',mode)
    app.router.add_post('/api/autonomy/neutral_probe',neutral_probe)
    return app,mux


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path.home()/'.config/hmu/default.toml')
    p.add_argument('--control-config',type=Path,default=(Path(__file__).resolve().parent.parent / 'config/config.json'))
    p.add_argument('--port',type=int)
    p.add_argument('--no-rov',action='store_true')
    args=p.parse_args();config=load_config(args.config)
    app,_=create_app(config,start_rov=not args.no_rov,
                     control_config=json.loads(args.control_config.read_text()))
    web.run_app(app,host=config.server.host,port=args.port or config.server.port,access_log=None)


if __name__=='__main__':main()
