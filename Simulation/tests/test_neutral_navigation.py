"""Exercise neutral adaptation with the actual source Tracker and guide."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from uw_interfaces.msg import ControlRequest, NavigationGoal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'adapters'))
from estimated_navigation import neutral_navigation_tracker

SOURCE_ROOT = next(path for path in [Path('/stonefish-workspace/navigation/uw_navigation'),
                                    Path('/home/hong/Desktop/MY_Stonefish_ws/stonefish_slam_ws/navigation/uw_navigation')]
                   if path.is_dir())
spec = importlib.util.spec_from_file_location('retained_navigation_core', SOURCE_ROOT/'core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
NOW = 280_000_000_000_000
tree = ast.parse((SOURCE_ROOT/'node.py').read_text())
tracker = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'Tracker')
namespace = dict(Node=object, time=SimpleNamespace(monotonic_ns=lambda: NOW),
                 Path=Path, json=json, math=math, String=String, ControlRequest=ControlRequest,
                 guide=core.guide, yaw=core.yaw, settling_reference=core.settling_reference)
exec(compile(ast.Module(body=[tracker], type_ignores=[]), str(SOURCE_ROOT/'node.py'), 'exec'), namespace)
SourceTracker = namespace['Tracker']
NeutralTracker = neutral_navigation_tracker(SourceTracker)


def fixture(cls=NeutralTracker, waypoint_index=0):
    node = cls.__new__(cls)
    node.out = Path('/runs/run')
    node.ns = 'rov01'
    node.cfg = dict(planner_profile='ethz_nbvplanner', controller_backend='jepa_mppi')
    node.p = dict(position_gain=.55, yaw_gain=.8, horizontal_speed_m_s=.48,
                  vertical_speed_m_s=.06, yaw_rate_rad_s=.12, position_tolerance_m=.06,
                  yaw_tolerance_rad=.1, settle_speed_m_s=.04, settle_yaw_rate_rad_s=.04,
                  dwell_sec=1.)
    state = Odometry()
    state.header.frame_id, state.child_frame_id = 'rov01/odom', 'rov01/base_link'
    state.header.stamp.sec = 12
    state.pose.pose.position.x, state.pose.pose.position.y, state.pose.pose.position.z = 1.2, -.7, .3
    state.pose.pose.orientation.z, state.pose.pose.orientation.w = math.sin(.2), math.cos(.2)
    state.twist.twist.linear.x = .02
    node.state, node.state_wall = state, NOW-20_000_000
    node.guard = dict(state='ARMED', source='navigation', generation=2, epoch=1)
    node.guard_wall = NOW-10_000_000
    node.sequence, node.goal_sequence, node.identity = 17, 7, ('mission', 2, 'token')
    node.dwell, node.braking = None, False
    goal = NavigationGoal(run_id='run', mission_id='mission', token='token', generation=2,
                          epoch=1, sequence=7, waypoint_index=waypoint_index,
                          issued_steady_ns=NOW-30_000_000, deadline_steady_ns=NOW+170_000_000)
    goal.target.header.frame_id = 'rov01/odom'
    goal.target.pose.orientation.z, goal.target.pose.orientation.w = math.sin(-.1), math.cos(-.1)
    node.goal = goal
    node.outputs, node.statuses = [], []
    node.pub = SimpleNamespace(publish=node.outputs.append)
    node.status = SimpleNamespace(publish=lambda message: node.statuses.append(json.loads(message.data)))
    return node


def command(message):
    t = message.command.twist
    return [t.linear.x, t.linear.y, t.linear.z, t.angular.z]


class NeutralNavigationChecks(unittest.TestCase):
    def setUp(self):
        namespace['time'].monotonic_ns=lambda:NOW

    def test_neutral_reached_requires_actual_speed_yaw_rate_and_source_dwell(self):
        node=fixture()
        original_clock=namespace['time'].monotonic_ns
        def step(offset,speed,yaw_rate):
            now=NOW+round(offset*1e9)
            namespace['time'].monotonic_ns=lambda:now
            node.state_wall=node.guard_wall=now
            node.goal.issued_steady_ns=now
            node.goal.deadline_steady_ns=now+200_000_000
            node.state.twist.twist.linear.x=speed
            node.state.twist.twist.angular.z=yaw_rate
            node.tick()
            self.assertEqual(command(node.outputs[-1]),[0.]*4)
            return node.statuses[-1]['reached']
        try:
            self.assertFalse(step(0.,.2,0.))
            self.assertFalse(step(2.,.2,0.))
            self.assertFalse(step(3.,.02,0.))
            self.assertFalse(step(3.9,.02,0.))
            self.assertTrue(step(4.01,.02,0.))
            self.assertFalse(step(4.1,.02,.05))
            self.assertFalse(step(5.5,.02,.05))
            self.assertFalse(step(5.6,.02,.01))
            self.assertTrue(step(6.61,.02,.01))
        finally:
            namespace['time'].monotonic_ns=original_clock

    def test_neutral_is_exact_zero_despite_old_held_pose_and_yaw(self):
        node = fixture()
        original_goal, original_pose = node.goal, node.goal.target.pose
        source = fixture(SourceTracker)
        source.tick()
        self.assertNotEqual(command(source.outputs[0]), [0.]*4)
        node.tick()
        self.assertEqual(command(node.outputs[0]), [0.]*4)
        self.assertIs(node.goal, original_goal)
        self.assertIs(node.goal.target.pose, original_pose)
        self.assertEqual(node.sequence, 18)
        self.assertEqual(len(node.outputs), 1)
        self.assertEqual(node.outputs[0].issued_steady_ns, original_goal.issued_steady_ns)
        self.assertEqual(node.outputs[0].valid_until_steady_ns, original_goal.deadline_steady_ns)
        self.assertEqual((node.outputs[0].generation, node.outputs[0].token), (2, 'token'))
        self.assertEqual(node.statuses[-1]['waypoint_index'], 0)

    def test_nonproject_waypoints_retain_source_command_and_status(self):
        for waypoint in [0, 1, 12, 35]:
            node, source = fixture(waypoint_index=waypoint), fixture(SourceTracker, waypoint)
            node.cfg['planner_profile']=source.cfg['planner_profile']='another_planner'
            original = node.goal.target.pose
            node.tick()
            source.tick()
            self.assertEqual(node.outputs[0], source.outputs[0])
            self.assertEqual(node.statuses, source.statuses)
            self.assertIs(node.goal.target.pose, original)

    def test_nonproject_waypoint_zero_retains_source_guidance(self):
        node, source = fixture(), fixture(SourceTracker)
        node.cfg['planner_profile'] = source.cfg['planner_profile'] = 'another_planner'
        node.tick()
        source.tick()
        self.assertEqual(node.outputs, source.outputs)
        self.assertNotEqual(command(node.outputs[0]), [0.]*4)

    def test_expired_goal_remains_unpublished_and_restored(self):
        for old_issue in [False, True]:
            node = fixture()
            if old_issue:
                node.goal.issued_steady_ns = NOW-250_000_000
            else:
                node.goal.deadline_steady_ns = NOW-1
            pose = node.goal.target.pose
            node.tick()
            self.assertFalse(node.outputs)
            self.assertEqual(node.sequence, 17)
            self.assertIs(node.goal.target.pose, pose)
            self.assertEqual(node.statuses[-1]['state'], 'IDLE')

    def test_source_guard_identity_checks_still_block_output(self):
        for key, value in [('state', 'DISARMED'), ('source', 'cli'), ('generation', 3)]:
            node = fixture()
            node.guard[key] = value
            node.tick()
            self.assertFalse(node.outputs)

    def test_source_goal_identity_and_epoch_checks_are_unchanged(self):
        for changed in ['run', 'frame', 'generation', 'epoch', 'expired']:
            node = fixture()
            previous = node.goal
            goal = copy.deepcopy(previous)
            goal.sequence += 1
            if changed == 'run': goal.run_id = 'wrong_run'
            elif changed == 'frame': goal.target.header.frame_id = 'wrong_frame'
            elif changed == 'generation': goal.generation += 1
            elif changed == 'epoch': goal.epoch += 1
            else: goal.deadline_steady_ns = NOW-1
            node.on_goal(goal)
            self.assertIs(node.goal, previous)

    def test_neutral_id_cannot_reuse_previous_edge_reached_status(self):
        node = fixture(waypoint_index=12)
        old_edge = node.goal
        node.dwell = NOW-2_000_000_000
        neutral = copy.deepcopy(old_edge)
        neutral.sequence += 1
        neutral.waypoint_index = 0
        node.on_goal(neutral)
        self.assertIsNone(node.dwell)
        node.tick()
        self.assertEqual(node.statuses[-1]['waypoint_index'], 0)
        self.assertNotEqual(node.statuses[-1]['waypoint_index'], old_edge.waypoint_index)
        self.assertFalse(node.statuses[-1]['reached'])
        self.assertEqual(old_edge.waypoint_index, 12)

    def test_original_goal_is_restored_if_source_clears_goal_and_raises(self):
        class ClearsGoal:
            def tick(self):
                self.goal = None
                raise RuntimeError('source failure')
        cls = neutral_navigation_tracker(ClearsGoal)
        node = fixture(cls)
        goal, pose = node.goal, node.goal.target.pose
        with self.assertRaisesRegex(RuntimeError, 'source failure'):
            node.tick()
        self.assertIsNone(node.goal)
        self.assertIs(goal.target.pose, pose)


class HeadingNavigationChecks(unittest.TestCase):
    def setUp(self):
        namespace['time'].monotonic_ns=lambda:NOW

    def node(self,waypoint=14):
        node=fixture(waypoint_index=waypoint)
        node.state.pose.pose.orientation.z=0.;node.state.pose.pose.orientation.w=1.
        node.state.twist.twist.linear.x=0.
        node.goal.target.pose.position=copy.deepcopy(node.state.pose.pose.position)
        node.goal.target.pose.position.y+=1.
        node.goal.target.pose.orientation.z=0.;node.goal.target.pose.orientation.w=1.
        return node

    def tick(self,node,offset=0.,yaw=None,speed=None,yaw_rate=None):
        now=NOW+round(offset*1e9)
        namespace['time'].monotonic_ns=lambda:now
        node.state_wall=node.guard_wall=now
        node.goal.issued_steady_ns=now
        node.goal.deadline_steady_ns=now+200_000_000
        if yaw is not None:
            node.state.pose.pose.orientation.z=math.sin(yaw/2.)
            node.state.pose.pose.orientation.w=math.cos(yaw/2.)
        if speed is not None:node.state.twist.twist.linear.x=speed
        if yaw_rate is not None:node.state.twist.twist.angular.z=yaw_rate
        node.tick()
        return node.statuses[-1]

    def aligned(self,node):
        self.tick(node,0.,yaw=math.pi/2.)
        self.tick(node,1.01,yaw=math.pi/2.)
        self.assertEqual(node._heading_phase,'translate')

    def test_turn_is_zero_translation_and_preserves_original_goal_and_lease(self):
        node=self.node();goal=node.goal;pose=goal.target.pose
        report=self.tick(node)
        actual=command(node.outputs[-1])
        self.assertEqual(actual[:3],[0.,0.,0.])
        self.assertGreater(actual[3],0.)
        self.assertEqual(report['execution_phase'],'align')
        self.assertEqual((report['waypoint_index'],report['source_waypoint_index']),(1,14))
        self.assertFalse(report['reached'])
        self.assertIs(node.goal,goal);self.assertIs(node.goal.target.pose,pose)
        self.assertEqual(node.goal.waypoint_index,14)
        request=node.outputs[-1]
        self.assertEqual((request.generation,request.token),(goal.generation,goal.token))
        self.assertEqual((request.issued_steady_ns,request.valid_until_steady_ns),
                         (goal.issued_steady_ns,goal.deadline_steady_ns))
        self.assertEqual(len(node.outputs),1)

    def test_alignment_requires_real_speed_yaw_rate_and_full_source_dwell(self):
        node=self.node()
        self.tick(node,0.,yaw=math.pi/2.,speed=.1)
        self.tick(node,2.,yaw=math.pi/2.,speed=.02,yaw_rate=.05)
        self.assertEqual(node._heading_phase,'align')
        self.tick(node,3.,speed=.02,yaw_rate=.01)
        self.tick(node,3.99)
        self.assertEqual(node._heading_phase,'align')
        report=self.tick(node,4.01)
        self.assertEqual(node._heading_phase,'translate')
        self.assertTrue(report['phase_reached']);self.assertFalse(report['reached'])
        self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])
        report=self.tick(node,4.02)
        self.assertEqual(report['waypoint_index'],2)
        self.assertGreater(command(node.outputs[-1])[0],.47)

    def test_translate_arrival_precedes_original_view_and_only_view_can_finish_edge(self):
        node=self.node();self.aligned(node)
        self.tick(node,1.02)
        self.assertEqual(node.statuses[-1]['execution_phase'],'translate')
        self.assertEqual(command(node.outputs[-1])[3],0.)
        node.state.pose.pose.position=copy.deepcopy(node.goal.target.pose.position)
        self.tick(node,2.,speed=.02,yaw_rate=.01)
        report=self.tick(node,3.01)
        self.assertEqual(node._heading_phase,'observe')
        self.assertTrue(report['phase_reached']);self.assertFalse(report['reached'])
        report=self.tick(node,3.02)
        self.assertEqual(report['execution_phase'],'observe')
        self.assertEqual(report['waypoint_index'],14)
        self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])
        self.assertLess(command(node.outputs[-1])[3],0.)
        self.tick(node,4.,yaw=0.,speed=.01,yaw_rate=.01)
        report=self.tick(node,5.01)
        self.assertTrue(report['reached'])
        self.assertEqual(report['source_waypoint_index'],14)

    def test_observation_keeps_actual_endpoint_instead_of_hiding_position_error(self):
        node=self.node();self.aligned(node)
        node._heading_phase='observe'
        node.dwell=NOW-2_000_000_000
        report=self.tick(node,2.,yaw=0.)
        self.assertGreater(report['distance_m'],.9)
        self.assertFalse(report['reached'])
        self.assertGreater(math.hypot(*command(node.outputs[-1])[:2]),.47)

    def test_loss_of_existing_heading_tolerance_brakes_and_realigns(self):
        node=self.node();self.aligned(node)
        report=self.tick(node,1.02,yaw=math.pi/2.-.2)
        self.assertEqual(node._heading_phase,'align')
        self.assertEqual(report['waypoint_index'],1)
        self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])
        self.assertFalse(report['reached'])

    def test_rolling_same_index_goal_renewal_does_not_restart_alignment(self):
        node=self.node(13);self.aligned(node)
        old_identity=node._heading_identity
        renewed=copy.deepcopy(node.goal);renewed.sequence+=1
        renewed.target.pose.position.y+=1.
        node.on_goal(renewed)
        report=self.tick(node,1.02)
        self.assertEqual(node._heading_identity,old_identity)
        self.assertEqual(node._heading_phase,'translate')
        self.assertEqual(report['waypoint_index'],2)
        self.assertGreater(command(node.outputs[-1])[0],.47)

    def test_neutral_interlude_preserves_edge_phase_but_cannot_lend_its_dwell(self):
        node=self.node();self.aligned(node)
        original=copy.deepcopy(node.goal)
        neutral=copy.deepcopy(node.goal);neutral.sequence+=1;neutral.waypoint_index=0
        node.on_goal(neutral)
        self.tick(node,2.,speed=0.,yaw_rate=0.)
        self.tick(node,3.01)
        self.assertTrue(node.statuses[-1]['reached'])
        self.assertEqual(node.statuses[-1]['waypoint_index'],0)
        self.assertEqual(command(node.outputs[-1]),[0.]*4)
        self.assertEqual(node._heading_phase,'translate')
        original.sequence=neutral.sequence+1
        original.issued_steady_ns=NOW+3_010_000_000
        original.deadline_steady_ns=NOW+3_210_000_000
        node.on_goal(original)
        self.assertIsNone(node.dwell)
        node.state.pose.pose.position=copy.deepcopy(original.target.pose.position)
        report=self.tick(node,3.02)
        self.assertFalse(report['phase_reached']);self.assertFalse(report['reached'])
        self.assertEqual(node._heading_phase,'translate')

    def test_new_real_waypoint_requires_new_alignment_and_direction(self):
        node=self.node();self.aligned(node)
        goal=copy.deepcopy(node.goal);goal.sequence+=1;goal.waypoint_index+=1
        goal.target.pose.position=copy.deepcopy(node.state.pose.pose.position)
        goal.target.pose.position.x+=1.
        node.on_goal(goal)
        report=self.tick(node,1.02)
        self.assertEqual(node._heading_phase,'align')
        self.assertEqual(node._heading_yaw,0.)
        self.assertFalse(report['phase_reached'])
        self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])

    def test_same_edge_new_generation_and_token_preserve_stage_and_direction(self):
        for phase in ('translate','observe'):
            with self.subTest(phase=phase):
                node=self.node();self.aligned(node)
                node._heading_phase=phase
                direction=node._heading_yaw;path_identity=node._heading_identity
                old_goal=copy.deepcopy(node.goal)
                goal=copy.deepcopy(node.goal);goal.sequence+=1
                goal.generation+=1;goal.token='renewed-token'
                node.guard['generation']=goal.generation
                node.on_goal(goal)
                self.assertIsNone(node.dwell)
                report=self.tick(node,1.02)
                self.assertEqual(node._heading_phase,phase)
                self.assertEqual(node._heading_identity,path_identity)
                self.assertEqual(node._heading_yaw,direction)
                self.assertEqual(report['generation'],3)
                self.assertEqual(node.outputs[-1].token,'renewed-token')
                old_goal.sequence=goal.sequence+1
                node.on_goal(old_goal)
                self.assertIs(node.goal,goal)

    def test_endpoint_overshoot_and_rearm_do_not_turn_toward_reverse_bearing(self):
        node=self.node();self.aligned(node)
        direction=node._heading_yaw
        node.state.pose.pose.position=copy.deepcopy(node.goal.target.pose.position)
        node.state.pose.pose.position.y+=.2
        goal=copy.deepcopy(node.goal);goal.sequence+=1
        goal.generation+=1;goal.token='renewed-token'
        node.guard['generation']=goal.generation
        node.on_goal(goal)
        report=self.tick(node,1.02)
        self.assertEqual(node._heading_phase,'translate')
        self.assertEqual(node._heading_yaw,direction)
        self.assertEqual(report['segment_yaw_rad'],math.pi/2.)
        self.assertLess(command(node.outputs[-1])[0],0.)
        self.assertAlmostEqual(command(node.outputs[-1])[3],0.)
        self.assertFalse(report['reached'])

    def test_realign_after_actual_yaw_deviation_keeps_original_path_direction(self):
        node=self.node();self.aligned(node)
        direction=node._heading_yaw
        node.state.pose.pose.position=copy.deepcopy(node.goal.target.pose.position)
        node.state.pose.pose.position.y+=.2
        report=self.tick(node,1.02,yaw=math.pi/2.+.2)
        self.assertEqual(node._heading_phase,'align')
        self.assertEqual(node._heading_yaw,direction)
        self.assertEqual(report['segment_yaw_rad'],direction)
        self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])
        self.assertLess(command(node.outputs[-1])[3],0.)

    def test_stale_or_revoked_state_never_publishes_or_advances_any_phase(self):
        for phase in ('align','translate','observe'):
            for failure in ('expired','old_issue','manual','disarmed','generation','stale_state'):
                with self.subTest(phase=phase,failure=failure):
                    node=self.node();self.tick(node)
                    node._heading_phase=phase
                    if phase=='translate':
                        node.state.pose.pose.orientation.z=math.sin(node._heading_yaw/2.)
                        node.state.pose.pose.orientation.w=math.cos(node._heading_yaw/2.)
                    node.outputs.clear();node.statuses.clear()
                    node.dwell=NOW-2_000_000_000
                    if failure=='expired':node.goal.deadline_steady_ns=NOW-1
                    elif failure=='old_issue':node.goal.issued_steady_ns=NOW-250_000_000
                    elif failure=='manual':node.guard['source']='cli'
                    elif failure=='disarmed':node.guard['state']='DISARMED'
                    elif failure=='generation':node.guard['generation']=3
                    else:node.state_wall=NOW-250_000_000
                    original=node.goal
                    node.tick()
                    self.assertFalse(node.outputs)
                    self.assertEqual(node.statuses[-1]['state'],'IDLE')
                    self.assertFalse(node.statuses[-1]['reached'])
                    self.assertEqual(node._heading_phase,phase)
                    self.assertIsNone(node.dwell)
                    self.assertIs(node.goal,original)

    def test_real_edge_goal_and_status_publisher_restore_without_reviving_cleared_goal(self):
        class ClearsGoal:
            def tick(self):
                self.goal=None
                raise RuntimeError('source failure')
        node=fixture(neutral_navigation_tracker(ClearsGoal),waypoint_index=14)
        goal,publisher=node.goal,node.status
        pose=copy.deepcopy(goal.target.pose)
        with self.assertRaisesRegex(RuntimeError,'source failure'):node.tick()
        self.assertIsNone(node.goal)
        self.assertIs(node.status,publisher)
        self.assertEqual(goal.target.pose,pose)
        self.assertEqual(goal.waypoint_index,14)

    def test_invalid_actual_quaternion_still_uses_source_invalid_state_rejection(self):
        node=self.node()
        node.state.pose.pose.orientation.w=0.
        node.tick()
        self.assertIsNone(node.goal)
        self.assertFalse(node.outputs)
        self.assertEqual(node.statuses[-1]['state'],'INVALID_STATE')


class TurnCruiseNavigationChecks(unittest.TestCase):
    def setUp(self):
        namespace['time'].monotonic_ns=lambda:NOW

    def tick(self,node,offset=0.):
        now=NOW+round(offset*1e9)
        namespace['time'].monotonic_ns=lambda:now
        node.state_wall=node.guard_wall=now
        node.goal.issued_steady_ns=now
        node.goal.deadline_steady_ns=now+200_000_000
        node.tick()
        return node.statuses[-1]

    def test_external_turn_tracks_latest_xyz_and_preserves_requested_yaw_and_lease(self):
        for mission in ('mission','nbv-reactive'):
            with self.subTest(mission=mission):
                node=fixture(waypoint_index=1);node.goal.mission_id=mission
                goal,pose=node.goal,node.goal.target.pose
                self.tick(node)
                self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])
                self.assertLess(command(node.outputs[-1])[3],0.)
                self.assertAlmostEqual(node.statuses[-1]['yaw_error_rad'],-.6)
                node.state.pose.pose.position.x+=3.
                node.state.pose.pose.position.y-=2.
                node.state.pose.pose.position.z+=.5
                report=self.tick(node,.05)
                self.assertEqual(command(node.outputs[-1])[:3],[0.,0.,0.])
                self.assertEqual(report['distance_m'],0.)
                self.assertEqual((report['mission_id'],report['waypoint_index']),(mission,1))
                self.assertIs(node.goal,goal);self.assertIs(goal.target.pose,pose)
                request=node.outputs[-1]
                self.assertEqual((request.token,request.generation),('token',2))
                self.assertEqual((request.issued_steady_ns,request.valid_until_steady_ns),
                                 (goal.issued_steady_ns,goal.deadline_steady_ns))

    def test_external_turn_reached_requires_actual_xyz_speed_yaw_rate_and_full_dwell(self):
        node=fixture(waypoint_index=1);node.goal.mission_id='nbv-reactive'
        node.goal.target.pose.orientation=copy.deepcopy(node.state.pose.pose.orientation)
        def step(offset,x,z,rate):
            node.state.twist.twist.linear.x=x;node.state.twist.twist.linear.z=z
            node.state.twist.twist.angular.z=rate
            report=self.tick(node,offset)
            self.assertEqual(command(node.outputs[-1]),[0.]*4)
            return report['reached']
        self.assertFalse(step(0.,.1,0.,0.))
        self.assertFalse(step(2.,.1,0.,0.))
        self.assertFalse(step(3.,.02,.05,0.))
        self.assertFalse(step(4.,.02,0.,.05))
        self.assertFalse(step(6.,.02,0.,.05))
        self.assertFalse(step(7.,.02,0.,.01))
        self.assertFalse(step(7.99,.02,0.,.01))
        self.assertTrue(step(8.01,.02,0.,.01))
        node.state.pose.pose.orientation.z=math.sin(.51/2.)
        node.state.pose.pose.orientation.w=math.cos(.51/2.)
        self.assertFalse(self.tick(node,8.02)['reached'])

    def test_reactive_rolling_goals_use_exact_source_guidance_without_heading_stages(self):
        node=fixture(waypoint_index=14);source=fixture(SourceTracker,14)
        for tracker in (node,source):
            tracker.goal.mission_id='nbv-reactive'
            tracker.goal.target.pose.position=copy.deepcopy(tracker.state.pose.pose.position)
            tracker.goal.target.pose.position.x+=math.cos(.4)
            tracker.goal.target.pose.position.y+=math.sin(.4)
            tracker.goal.target.pose.orientation=copy.deepcopy(tracker.state.pose.pose.orientation)
        for offset in (0.,.05,.1):
            if offset:
                for tracker in (node,source):
                    goal=copy.deepcopy(tracker.goal);goal.sequence+=1
                    goal.target.pose.position.x+=math.cos(.4)
                    goal.target.pose.position.y+=math.sin(.4)
                    tracker.on_goal(goal)
            self.tick(node,offset);self.tick(source,offset)
            self.assertEqual(node.outputs[-1],source.outputs[-1])
            self.assertEqual(node.statuses[-1],source.statuses[-1])
            self.assertGreater(command(node.outputs[-1])[0],.47)
            self.assertEqual(node.statuses[-1]['mission_id'],'nbv-reactive')
            self.assertEqual(node.statuses[-1]['waypoint_index'],14)
            self.assertNotIn('execution_phase',node.statuses[-1])
            self.assertFalse(hasattr(node,'_heading_phase'))

    def test_reactive_near_endpoint_retains_source_command_without_observe_or_stop_gate(self):
        node=fixture(waypoint_index=14);source=fixture(SourceTracker,14)
        for tracker in (node,source):
            tracker.goal.mission_id='nbv-reactive'
            tracker.goal.target.pose.position=copy.deepcopy(tracker.state.pose.pose.position)
            tracker.goal.target.pose.position.x+=.02*math.cos(.4)
            tracker.goal.target.pose.position.y+=.02*math.sin(.4)
            tracker.goal.target.pose.orientation=copy.deepcopy(tracker.state.pose.pose.orientation)
        self.tick(node);self.tick(source)
        self.assertEqual(node.outputs[-1],source.outputs[-1])
        self.assertGreater(command(node.outputs[-1])[0],0.)
        self.assertNotIn('execution_phase',node.statuses[-1])
        self.assertFalse(hasattr(node,'_heading_phase'))

    def test_reactive_neutral_keeps_full_zero_intent(self):
        node=fixture();node.goal.mission_id='nbv-reactive'
        report=self.tick(node)
        self.assertEqual(command(node.outputs[-1]),[0.]*4)
        self.assertEqual((report['mission_id'],report['waypoint_index']),('nbv-reactive',0))

    def test_source_authority_and_lease_rejection_remain_for_external_turn_and_cruise(self):
        for waypoint in (1,14):
            for failure in ('expired','old_issue','manual','disarmed','generation','stale_state','stale_guard'):
                with self.subTest(waypoint=waypoint,failure=failure):
                    node=fixture(waypoint_index=waypoint);node.goal.mission_id='nbv-reactive'
                    if failure=='expired':node.goal.deadline_steady_ns=NOW-1
                    elif failure=='old_issue':node.goal.issued_steady_ns=NOW-250_000_000
                    elif failure=='manual':node.guard['source']='cli'
                    elif failure=='disarmed':node.guard['state']='DISARMED'
                    elif failure=='generation':node.guard['generation']=3
                    elif failure=='stale_state':node.state_wall=NOW-250_000_000
                    else:node.guard_wall=NOW-250_000_000
                    original=node.goal;pose=node.goal.target.pose
                    node.tick()
                    self.assertFalse(node.outputs)
                    self.assertEqual(node.statuses[-1]['state'],'IDLE')
                    self.assertIs(node.goal,original);self.assertIs(node.goal.target.pose,pose)

    def test_source_goal_identity_epoch_and_sequence_checks_remain_for_turn_and_cruise(self):
        for waypoint in (1,14):
            for changed in ('run','frame','generation','epoch','expired','sequence'):
                with self.subTest(waypoint=waypoint,changed=changed):
                    node=fixture(waypoint_index=waypoint);node.goal.mission_id='nbv-reactive'
                    node.identity=('nbv-reactive',2,'token')
                    previous=node.goal;goal=copy.deepcopy(previous);goal.sequence+=1
                    if changed=='run':goal.run_id='wrong_run'
                    elif changed=='frame':goal.target.header.frame_id='wrong_frame'
                    elif changed=='generation':goal.generation+=1
                    elif changed=='epoch':goal.epoch+=1
                    elif changed=='expired':goal.deadline_steady_ns=NOW-1
                    else:goal.sequence=node.goal_sequence
                    node.on_goal(goal)
                    self.assertIs(node.goal,previous)

    def test_external_turn_does_not_restore_goal_cleared_by_source_failure(self):
        class ClearsGoal:
            def tick(self):
                self.goal=None
                raise RuntimeError('source failure')
        node=fixture(neutral_navigation_tracker(ClearsGoal),waypoint_index=1)
        goal,pose=node.goal,node.goal.target.pose
        with self.assertRaisesRegex(RuntimeError,'source failure'):node.tick()
        self.assertIsNone(node.goal);self.assertIs(goal.target.pose,pose)


if __name__ == '__main__':
    unittest.main(verbosity=2)
