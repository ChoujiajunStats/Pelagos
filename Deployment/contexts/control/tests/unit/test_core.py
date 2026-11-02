import sys
from pathlib import Path as TaskPath
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskPath(__file__).resolve().parents[2] / _role))
import unittest
import numpy as np
from geometry import LocalFrame,timed_reference,minimum_centered_sphere_radius
from history import RealHistory,command_to_modal
from clock_sync import ClockAlignment
from collision import swept_sphere_status
from motor_contract import CONTRACT_ID


class Tests(unittest.TestCase):
    def test_pose_age_budget_replaces_half_second_gate(self):
        from runtime import Runtime
        runtime=Runtime.__new__(Runtime);runtime.max_pose_age_s=1.0
        for age in (.49,.51,.9,.999):
            self.assertTrue(runtime.pose_age_valid(age))
        for age in (1.,1.01,float('nan'),float('inf'),-.1):
            self.assertFalse(runtime.pose_age_valid(age))
        runtime.max_pose_age_s=.7
        self.assertTrue(runtime.pose_age_valid(.69))
        self.assertFalse(runtime.pose_age_valid(.71))

    def test_sphere_encloses_user_dimensions(self):
        r=minimum_centered_sphere_radius([.55,.55,.5])
        self.assertAlmostEqual(r,np.linalg.norm([.275,.275,.25]))
        self.assertGreater(r,.46);self.assertLess(r,.47)
        with self.assertRaises(ValueError):minimum_centered_sphere_radius([.55,-.55,.5])

    def test_clock_alignment_and_step_rejection(self):
        clock=ClockAlignment()
        packet=dict(robot_send_ns=9_950_000_000,echo_host_send_ns=10_000_000_000,
                    robot_receive_ns=9_002_000_000,body=dict(stamp_ns=9_900_000_000))
        out=clock.align(packet,10_952_000_000)
        self.assertEqual(clock.offset_ns,1_000_000_000)
        self.assertEqual(out['body']['stamp_ns'],10_900_000_000)
        self.assertEqual(packet['body']['stamp_ns'],9_900_000_000)
        packet.update(robot_send_ns=9_450_000_000,robot_receive_ns=8_502_000_000)
        with self.assertRaisesRegex(ValueError,'clock_step'):
            clock.align(packet,10_952_000_000)
        self.assertIsNone(ClockAlignment().align(packet,12_000_000_000))

    def test_flu_frd_ned_gravity(self):
        frame=LocalFrame([1,2,3],[0,0,0,1],.4,np.diag([1,-1,-1]))
        body=dict(position=[1,2,3],quaternion=[0,0,0,1],velocity=[.1,.2,.3])
        imu=dict(acceleration=[0,0,9.80665],gyro=[.01,.02,.03])
        obs,pos,r=frame.state(body,imu,.5)
        np.testing.assert_allclose(r,np.eye(3),atol=1e-9)
        np.testing.assert_allclose(obs[6:9],[.1,-.2,-.3])
        np.testing.assert_allclose(obs[12:15],[0,0,0],atol=1e-9)
        self.assertAlmostEqual(pos[2],.1)

    def test_reference_speed_and_braking(self):
        times=np.arange(31)*.1
        p,v=timed_reference([[0,0,0],[.08,0,0]],[0,0,0],[0,0,0],times)
        self.assertLessEqual(np.max(np.linalg.norm(v,axis=1)),.050001)
        self.assertLessEqual(np.max(p[:,0]),.080001)
        self.assertLess(np.max(np.diff(p[:,0])),.0051)

    def test_real_history_does_not_seed(self):
        h=RealHistory();obs=np.r_[1.,0,0,0,1,0,np.zeros(10)]
        for i in range(30):
            t=10+i*.1
            h.append_tx([dict(sequence=i+1,stamp_ns=int((t-.1)*1e9),command=[.1,0,0,0],armed=True,gear=1,heading_hold=False,depth_hold=False)])
            h.append(t,obs,[i*.01,0,0])
            if i<25:
                with self.assertRaises(ValueError):h.arrays()
        x,u,pos,t=h.arrays();self.assertEqual(x.shape,(26,16));self.assertEqual(u.shape,(26,5))
        np.testing.assert_allclose(u[:,0],.1,atol=1e-6)

    def test_mixer_modal_round_trip(self):
        np.testing.assert_allclose(command_to_modal([.1,-.1,.05,.02]),[.1,-.1,.05,.02,0],atol=1e-7)

    def test_history_rejects_unsupported_firmware_hold(self):
        h=RealHistory();obs=np.r_[1.,0,0,0,1,0,np.zeros(10)]
        for i in range(32):
            t=1792900000.+i*.1
            h.append_tx([dict(sequence=i+1,stamp_ns=round((t-.1)*1e9),command=[0,0,0,0],armed=True,gear=1,heading_hold=False,depth_hold=True)])
            h.append(t,obs,[0,0,0])
        with self.assertRaisesRegex(ValueError,'unmodelled_hold'):h.arrays()

    def test_motor_history_uses_actual_quantized_demands(self):
        h=RealHistory();obs=np.r_[1.,0,0,0,1,0,np.zeros(10)]
        for i in range(32):
            t=1792900000.+i*.1
            h.append_tx([dict(sequence=i+1,stamp_ns=round((t-.1)*1e9),command=[.9]*4,
                armed=True,gear=1,heading_hold=False,depth_hold=False,protocol='motor',
                actuation_contract=CONTRACT_ID,model_motor_normalized=[-.1,-.1,.1,.1,-.05,-.05])])
            h.append(t,obs,[0,0,0])
        x,u,p,t=h.arrays();self.assertTrue(h.action_contract_verified)
        np.testing.assert_allclose(u,np.tile([.1,0,.05,0,0],(26,1)),atol=1e-6)

    def test_armed_channel_is_only_shadow_history(self):
        h=RealHistory();obs=np.r_[1.,0,0,0,1,0,np.zeros(10)]
        for i in range(32):
            t=1792900000.+i*.1
            h.append_tx([dict(sequence=i+1,stamp_ns=round((t-.1)*1e9),command=[0]*4,
                armed=True,gear=1,heading_hold=False,depth_hold=False,protocol='channel')])
            h.append(t,obs,[0,0,0])
        h.arrays();self.assertFalse(h.action_contract_verified)
        for e in h.tx:e['armed']=False
        h.arrays();self.assertTrue(h.action_contract_verified)

    def test_swept_sphere_diagonal_unknown_and_crop(self):
        data=np.ones((15,15,15));m=dict(resolution=.1,origin=[0.,0.,0.])
        self.assertEqual(swept_sphere_status(data,m,[[.4,.4,.4],[1.,1.,1.]],.1),'free')
        data[7,7,7]=0
        self.assertEqual(swept_sphere_status(data,m,[[.4,.4,.4],[1.,1.,1.]],.1),'in_collision')
        data[7,7,7]=np.nan
        self.assertEqual(swept_sphere_status(data,m,[[.4,.4,.4],[1.,1.,1.]],.1),'in_unknown_space')
        data[:]=1
        self.assertEqual(swept_sphere_status(data,m,[[.05,.4,.4],[.1,.5,.5]],.1),'in_unknown_space')


if __name__=='__main__':unittest.main()
