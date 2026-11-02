import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
import json
from pathlib import Path
import unittest
import asyncio
from dataclasses import replace
from hmu_bridge.config import AppConfig,RovConfig
from hmu_bridge.protocol import ControlCommand,pack_channel_command
from hmu_bridge.server import BridgeRuntime
from hmu_extension import ControlMux,create_app,MotorControlCommand
from motor_contract import CONTRACT_ID,MODEL_ORDER,decode_motor_frame
from aiohttp.test_utils import TestClient,TestServer
from unittest.mock import patch


class Tests(unittest.TestCase):
    def setUp(self):
        self.t=10.;self.runtime=BridgeRuntime(AppConfig(rov=RovConfig(command_protocol='channel',motor_channels=MODEL_ORDER)),start_rov=False)
        self.manual=ControlCommand();self.runtime.rov.command_provider=lambda:self.manual
        self.mux=ControlMux(self.runtime,clock=lambda:self.t);self.runtime.rov.state.connected=True
        self.mux.transmitted(self.mux.pack(MotorControlCommand.neutral()))
        self.mux.events.clear();self.mux.sequence=0

    def proposal(self):
        return dict(command=[.1,.02,-.01,.01],eligible=True,collision_radius_m=.35,
                    map_age_s=.2,source_age_s=.1,solve_age_s=.03,
                    actuation_contract=CONTRACT_ID,motor_frame_contract_verified=True,action_history_verified=True)

    def test_manual_is_unchanged(self):
        self.manual=ControlCommand(sway=.2,armed=True)
        self.assertEqual(self.mux.output(),self.manual)

    def test_pose_age_budget_accepts_measured_delay_and_expires_in_transit(self):
        self.assertEqual(self.mux.state()['max_pose_age_s'],1.0)
        p=self.proposal();p['source_age_s']=.8;self.mux.propose(p)
        self.assertTrue(self.mux.eligible())
        self.t+=.19
        self.assertTrue(self.mux.eligible())
        self.t+=.02
        self.assertFalse(self.mux.eligible())
        p['source_age_s']=1.;self.mux.propose(p)
        self.assertFalse(self.mux.eligible())

    def test_pose_age_budget_can_change_without_editing_gate(self):
        mux=ControlMux(self.runtime,clock=lambda:self.t,control_config={'max_pose_age_s':.7})
        p=self.proposal();p['source_age_s']=.69;mux.propose(p)
        self.assertTrue(mux.eligible())
        p['source_age_s']=.71;mux.propose(p)
        self.assertFalse(mux.eligible())
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                ControlMux(self.runtime,control_config={'max_pose_age_s':value})

    def test_auto_requires_fresh_verified_clearance(self):
        self.mux.propose(self.proposal());self.mux.set_mode('auto')
        self.manual=ControlCommand(armed=True)
        self.assertAlmostEqual(self.mux.output().surge,.1)
        self.t+=.31
        self.assertEqual(self.mux.output().surge,0)
        self.assertEqual(self.mux.mode,'shadow')

    def test_manual_overrides_auto(self):
        self.mux.propose(self.proposal());self.mux.set_mode('auto')
        self.manual=ControlCommand(sway=-.2,armed=True)
        self.assertEqual(self.mux.output(),self.manual)
        self.assertEqual(self.mux.mode,'manual')

    def test_small_preview_radius_rejected(self):
        p=self.proposal();p['collision_radius_m']=.1;self.mux.propose(p)
        with self.assertRaises(ValueError):self.mux.set_mode('auto')

    def test_tx_axes_are_wire_quantized_and_armed(self):
        c=ControlCommand(surge=.11,sway=-.12,heave=.13,yaw=-.02,armed=True)
        self.mux.transmitted(pack_channel_command(c));x=self.mux.events[-1]
        self.assertEqual(x['command'],[.11,-.12,.13,-.02]);self.assertTrue(x['armed'])

    def test_auto_motor_manual_channel_and_unused_slots(self):
        self.mux.propose(self.proposal());self.mux.set_mode('auto');self.manual=ControlCommand(armed=True)
        auto=self.mux.output();self.assertIsInstance(auto,MotorControlCommand)
        frame=self.mux.pack(auto);self.assertEqual(frame[23],0xBB);self.assertEqual(frame[21],1)
        self.assertEqual(frame[7:9],bytes.fromhex('05dc'));self.assertEqual(frame[15:17],bytes.fromhex('05dc'))
        self.mux.transmitted(frame);e=self.mux.events[-1]
        self.assertEqual(e['source'],'successful_motor_tx');self.assertTrue(e['armed'])
        self.manual=ControlCommand(sway=.2,armed=True)
        neutral=self.mux.pack(self.mux.output())
        self.assertEqual(neutral[23],0xBB);self.assertEqual(neutral[21],0)
        self.assertTrue(all(int.from_bytes(neutral[i:i+2],'big')==1500 for i in range(1,17,2)))
        self.mux.transmitted(neutral)
        self.assertEqual(self.mux.pack(self.mux.output())[23],0xAA)

    def test_model_positions_map_to_vendor_right_first(self):
        self.assertEqual(MODEL_ORDER,('R1','L1','R3','L3','R2','L2'))
        frame=self.mux.pack(MotorControlCommand(yaw=.1,armed=True))
        slots=decode_motor_frame(frame)['motor_pwm']
        self.assertEqual([slots[k] for k in ('R1','L1','R3','L3')],[1550,1450,1450,1550])

    def test_expired_auto_clears_motor_frame_before_manual(self):
        self.mux.propose(self.proposal());self.mux.set_mode('auto');self.manual=ControlCommand(armed=True)
        self.mux.transmitted(self.mux.pack(self.mux.output()))
        self.assertTrue(self.mux.motor_output_active)
        self.t+=.31
        neutral=self.mux.pack(self.mux.output())
        self.assertEqual(self.mux.mode,'shadow')
        self.assertEqual(neutral[23],0xBB);self.assertEqual(neutral[21],0)
        self.assertTrue(self.mux.motor_output_active)
        self.mux.transmitted(neutral)
        self.assertFalse(self.mux.motor_output_active)
        self.assertEqual(self.mux.pack(self.mux.output())[23],0xAA)

    def test_startup_and_explicit_shadow_require_motor_neutral(self):
        mux=ControlMux(self.runtime,clock=lambda:self.t)
        first=mux.pack(mux.output())
        self.assertEqual(first[23],0xBB);self.assertEqual(first[21],0)
        mux.transmitted(first)
        mux.propose(self.proposal());mux.set_mode('auto');self.manual=ControlCommand(armed=True)
        mux.transmitted(mux.pack(mux.output()));mux.set_mode('shadow')
        neutral=mux.pack(mux.output());self.assertEqual(neutral[23],0xBB);self.assertEqual(neutral[21],0)

    def test_vendor_motor_field_layout_matches_literal_values(self):
        frame=self.mux.pack(MotorControlCommand(surge=.1,sway=.2,heave=.05,yaw=.03,
                                             light=66,gimbal_1=17,gimbal_2=32,gripper=1,armed=True))
        expected=[1335,1475,1465,1500,1565,1475,1635,1500]
        self.assertEqual([int.from_bytes(frame[i:i+2],'big') for i in range(1,17,2)],expected)
        self.assertEqual(list(frame[17:]),[66,17,32,1,1,0,0xBB])
        broken=bytearray(frame);broken[7:9]=(1600).to_bytes(2,'big')
        with self.assertRaisesRegex(ValueError,'L4/R4'):decode_motor_frame(broken)

    def test_channel_history_cannot_enable_auto(self):
        p=self.proposal();p['action_history_verified']=False;self.mux.propose(p)
        with self.assertRaises(ValueError):self.mux.set_mode('auto')

    def test_measured_motor_polarities_pack_and_reconstruct_each_axis(self):
        config=json.loads((TaskPath(__file__).resolve().parents[2] / 'config/config.json').read_text())
        self.assertEqual(config['motor_channels'],['R1','L1','R3','L3','R2','L2'])
        self.assertEqual(config['motor_signs'],[1,1,-1,-1,-1,1])
        rov=replace(self.runtime.config.rov,motor_signs=tuple(config['motor_signs']))
        runtime=BridgeRuntime(AppConfig(rov=rov),start_rov=False)
        mux=ControlMux(runtime)
        # Literal manufacturer wire slots: L1,L2,L3,L4,R1,R2,R3,R4.
        cases=[
            (dict(surge=.1),[1450,1500,1450,1500,1450,1500,1450,1500],[.1,0,0,0]),
            (dict(sway=.1),[1450,1500,1550,1500,1550,1500,1450,1500],[0,.1,0,0]),
            (dict(heave=.1),[1500,1450,1500,1500,1500,1550,1500,1500],[0,0,.1,0]),
            (dict(yaw=.1),[1450,1500,1450,1500,1550,1500,1550,1500],[0,0,0,.1]),
        ]
        for axes,expected,command in cases:
            frame=mux.pack(MotorControlCommand(**axes,armed=True))
            self.assertEqual([int.from_bytes(frame[i:i+2],'big') for i in range(1,17,2)],expected)
            decoded=decode_motor_frame(frame,MODEL_ORDER,tuple(config['motor_signs']))
            for actual,target in zip(decoded['command'],command):
                self.assertAlmostEqual(actual,target,places=6)
            self.assertTrue(decoded['armed'])


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposal_api_and_successful_drain_only(self):
        config=AppConfig(rov=RovConfig(command_protocol='channel',mode='control',motor_channels=MODEL_ORDER))
        app,mux=create_app(config,start_rov=False)
        mux.transmitted(mux.pack(MotorControlCommand.neutral()))
        mux.events.clear();mux.sequence=0
        async with TestClient(TestServer(app)) as client:
            r=await client.get('/api/autonomy/state');self.assertEqual((await r.json())['mode'],'shadow')
            p=dict(command=[.1,0,0,0],eligible=True,collision_radius_m=.35,map_age_s=.1,
                   source_age_s=.1,solve_age_s=.03,actuation_contract=CONTRACT_ID,
                   motor_frame_contract_verified=True,action_history_verified=True)
            self.assertEqual((await client.post('/api/autonomy/proposal',json=p)).status,200)
            self.assertEqual((await client.post('/api/autonomy/mode',json={'mode':'auto'})).status,200)
            class Writer:
                fail=False
                def write(self,payload):self.payload=payload
                async def drain(self):
                    if self.fail:raise ConnectionError('write failed')
            writer=Writer();mux.runtime.rov._writer=writer
            mux.manual=lambda:ControlCommand(armed=True)
            mux.runtime.rov.state.connected=True
            frame=mux.runtime.rov._pack_command(mux.runtime.rov.command_provider())
            self.assertEqual(frame[23],0xBB);self.assertEqual(frame[21],1)
            await mux.runtime.rov._write(frame)
            self.assertEqual(mux.sequence,1)
            self.assertEqual(mux.events[-1]['protocol'],'motor')
            await mux.runtime.rov.send_neutral_now()
            self.assertEqual(writer.payload[23],0xBB);self.assertEqual(writer.payload[21],0)
            self.assertTrue(all(int.from_bytes(writer.payload[i:i+2],'big')==1500 for i in range(1,17,2)))
            self.assertEqual(mux.sequence,2)
            writer.fail=True
            with self.assertRaises(ConnectionError):await mux.runtime.rov._write(frame)
            self.assertEqual(mux.sequence,2)
            self.assertEqual((await client.post('/api/autonomy/neutral_probe',json={})).status,409)
            mux.set_mode('shadow');writer.fail=False;mux.manual=lambda:ControlCommand()
            with patch.object(mux.runtime.safety,'snapshot',return_value={'armed':True}):
                self.assertEqual((await client.post('/api/autonomy/neutral_probe',json={})).status,409)
            self.assertEqual((await client.post('/api/autonomy/neutral_probe',json={})).status,200)
            self.assertEqual(writer.payload[23],0xBB);self.assertEqual(writer.payload[21],0)
            self.assertEqual(mux.events[-1]['motor_pwm']['L4'],1500)
            self.assertEqual(mux.events[-1]['motor_pwm']['R4'],1500)
            mux.runtime.rov._writer=None


if __name__=='__main__':unittest.main()
