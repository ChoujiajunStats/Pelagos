"""Focused checks for the actual Beitong layout, input ABI and stale commands."""
from pathlib import Path
import os
import struct
import sys
import unittest

SIM=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(SIM/'adapters'),str(SIM/'tools')]
from joystick_core import command,packet_error,select_mapping
from joystick_host import Gamepad
from manual_envelope import manual_operating_envelope,state_diagnostic
import yaml


class JoystickChecks(unittest.TestCase):
    def test_beitong_events_and_vehicle_directions(self):
        # The detected device advertises ABS_X,Y,Z,RZ,GAS,BRAKE,HAT0X,HAT0Y.
        axes=[0,1,2,5,9,10,16,17]
        buttons=list(range(0x130,0x13f))
        pad=Gamepad.__new__(Gamepad)
        pad.fd,writer=os.pipe()
        os.set_blocking(pad.fd,False)
        pad.axes=[0.]*8;pad.buttons=[False]*15;pad.pressed_events=set()
        mapping=select_mapping(axes,buttons)
        events=[(0,-32767,0x82,1),(0,32767,0x82,0),
                (0,-32767,0x82,3),(0,32767,0x82,2),
                (0,1,0x81,buttons.index(0x136)),
                (0,1,0x01,buttons.index(0x131)),(0,0,0x01,buttons.index(0x131))]
        try:
            os.write(writer,b''.join(struct.pack('=IhBB',*event) for event in events))
            pad.read()
            result=command(pad.axes,pad.buttons,mapping)
            self.assertEqual(result['velocity'],[.75,-.75,.50,-1.25])
            self.assertTrue(result['deadman'])
            self.assertIn(buttons.index(0x131),pad.pressed_events)
            self.assertFalse(pad.buttons[buttons.index(0x131)])
            # Xbox right stick uses different ABS codes and array positions.
            xbox=select_mapping([0,1,2,3,4,5],buttons)
            self.assertEqual((xbox['yaw_axis'],xbox['up_axis']),(3,4))
        finally:pad.close();os.close(writer)

    def test_deadzone_and_original_input_deadline(self):
        mapping=select_mapping([0,1,2,5],list(range(0x130,0x13f)))
        output=command([.05,-.03,.10,-.08],[False]*15,mapping)
        self.assertEqual(output['velocity'],[0.]*4)
        packet=dict(output,connected=True,session='desktop-gamepad',sequence=3,issued_ns=1_000_000_000)
        self.assertIsNone(packet_error(packet,1_050_000_000,'desktop-gamepad',2))
        self.assertEqual(packet_error(packet,1_251_000_000,'desktop-gamepad',2),'input_expired')
        self.assertEqual(packet_error(packet,1_050_000_000,'desktop-gamepad',3),'replayed_input')
        packet['velocity'][0]=float('nan')
        self.assertEqual(packet_error(packet,1_050_000_000,'desktop-gamepad',2),'velocity_limit')

    def test_manual_control_has_no_position_box(self):
        original=dict(depth_enu_m=[-12.5,.5],horizontal_position_abs_m=8.,roll_pitch_peak_deg=25.,
                      linear_speed_peak_m_s=1.2,angular_speed_peak_rad_s=1.2)
        # Round-trip through the same YAML loader used by the retained Guard.
        envelope=yaml.safe_load(yaml.safe_dump(manual_operating_envelope(original)))
        # Reproduce the reported stop location, then travel beyond the old
        # box in every axis. No finite location should be a position fault.
        for position in ([1.46,9.91,-6.26],[80.,-120.,-30.],[-300.,400.,1.]):
            snapshot=state_diagnostic(position,[0.,0.,0.,1.],[0.]*6,envelope)
            self.assertEqual(snapshot['violations'],[])
            self.assertTrue(envelope['depth_enu_m'][0]<position[2]<envelope['depth_enu_m'][1])
            self.assertLess(max(abs(p) for p in position[:2]),envelope['horizontal_position_abs_m'])
        self.assertTrue(state_diagnostic([0.,0.,-4.],[.5,0.,0.,3**.5/2],[0.]*6,envelope)['violations'])


if __name__=='__main__':unittest.main()
