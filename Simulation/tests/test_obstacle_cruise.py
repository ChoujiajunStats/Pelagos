"""Exercise the actual bridge state methods for continuous obstacle-triggered cruise.

ROS transport and publishers are replaced by sinks; the production method bodies
are parsed from nbv_bridge.py so decisions and task-revision handling are real.
"""
import ast
import json
import math
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOW = 100_000_000_000
BODY = [0., 0., 0., 0., 0., 0., 1.]


class ObstacleCruiseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT/'adapters/nbv_bridge.py'
        tree = ast.parse(source.read_text())
        bridge = next(item for item in tree.body if isinstance(item, ast.ClassDef)
                      and item.name == 'NbvBridge')
        methods = [item for item in bridge.body if isinstance(item, ast.FunctionDef)
                   and item.name in ('poll', 'tick', 'disable', 'mission_request', 'enable',
                                     'apply_cruise_result', 'map_source_fresh', 'on_map_source',
                                     'navigation_mission_id','begin_avoidance_turn','tick_avoidance_turn')]
        methods.extend(item for item in bridge.body if isinstance(item, ast.FunctionDef)
                       and item.name == 'publish_neutral_goal')
        cls.namespace = dict(
            time=NS(monotonic_ns=lambda: NOW), json=json, math=math,
            ns=lambda stamp: stamp.sec*10**9+stamp.nanosec,
            pose_values=lambda pose: pose[:],
            SourceInitialization=lambda: NS(index=0),
            String=lambda **kwargs: NS(**kwargs),
            SetBool=NS(Request=lambda **kwargs: NS(**kwargs),
                       Response=lambda: NS(success=False, message='')))
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                     str(source), 'exec'), cls.namespace)

    def node(self):
        self.namespace['time'].monotonic_ns = lambda: NOW
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        events, authorities, goals, responses, plans, submissions = [], [], [], [], [], []
        goal_calls, actions = [], []

        def publish_goal(target, now, waypoint_index=None):
            goals.append(target[:])
            goal_calls.append((target[:], waypoint_index))
            actions.append(('goal', waypoint_index))

        def request_authority(action, *args):
            authorities.append(action)
            actions.append(('authority', action))
            return True
        target = [10., 0., 0., 0., 0., 0., 1.]
        node = NS(
            out=Path(temporary.name), task_revision=7,
            enabled=True, token='lease', generation=1, initialized=False,
            last_goal_deadline_ns=NOW+200_000_000,
            neutral_renewal_margin_ns=100_000_000,
            planning_only=False, mode='explore', control_mode='autonomy',
            pending_control_mode=None, authority_future=None,
            state=NS(pose=NS(pose=BODY[:]),
                     header=NS(stamp=NS(sec=100, nanosec=0))),
            guard=dict(state='ARMED', generation=1),
            localization=dict(epoch=1, state='TRACKING'),
            controller=dict(ready_for_arm=True), controller_wall=NOW,
            healthy=lambda now: True, state_healthy=lambda now: True,
            map_revision=10, validated_revision=10,
            map_freshness_budget_ns=250_000_000+round(1.e9/5.),
            startup_profile='obstacle_triggered', startup_map_status={},
            cruise_active=True, obstacle_phase='cruise',
            obstacle_switch_distance_m=3., cruise_map_status={}, cruise_requests=0,
            turn_path=[],turn_yaw=None,turn_started_wall=0,turn_retry_source_revision=-1,
            map_source_fresh=lambda now: True,
            startup_limit_segment=False, forward_start_active=True,
            forward_start_required=False,
            path=[BODY[:], target[:]], target=target[:], path_stamp=0,
            hold_target=None, hold_index=11, waypoint_index=12,
            route=[], route_index=0, route_complete=False,
            initialization=NS(index=0), navigation={},
            params=dict(pcl_throttle=.36, pelagos_obstacle_plan_distance_m=3.,
                        pelagos_start_unknown_policy='obstacle_triggered'), backend_state=None,
            preview_requested=None, preview_result=None, reset_execution=False,
            future=None, job=None, pending=None, last_plan_wall=0,
            plans_requested=0, paths_generated=0, last_plan_seconds=None,
            event=lambda name, **kwargs: events.append((name, kwargs)),
            request_authority=request_authority,
            publish_goal=publish_goal,
            publish_plan=lambda *args: plans.append(args),
            submit=lambda operation, **kwargs: submissions.append((operation, kwargs)),
            publish_status=lambda *args: None,
            response_pub=NS(publish=lambda message: responses.append(json.loads(message.data))),
            debug_pub=NS(publish=lambda message: None))
        for name in ('poll', 'tick', 'disable', 'mission_request', 'enable', 'apply_cruise_result',
                     'publish_neutral_goal','navigation_mission_id','begin_avoidance_turn','tick_avoidance_turn'):
            if name in self.namespace:
                setattr(node, name, MethodType(self.namespace[name], node))
        node.record = NS(events=events, authorities=authorities, goals=goals,
                         responses=responses, plans=plans, submissions=submissions,
                         goal_calls=goal_calls, actions=actions)
        return node

    def request(self, node, action):
        node.mission_request(NS(data=json.dumps(dict(run_id=node.out.name, epoch=1,
            request_id='request', issued_steady_ns=NOW, action=action))))
        self.assertTrue(node.record.responses[-1]['accepted'], node.record.responses)

    def response(self, node, operation, result, task_revision=None):
        node.future = NS(done=lambda: True, result=lambda: result)
        node.job = dict(operation=operation,
                        task_revision=node.task_revision if task_revision is None else task_revision,
                        body=BODY[:], stamp_ns=NOW, submitted_wall_ns=NOW-1)
        node.poll(NOW)

    def cruise_response(self, node, *, distance=None, revision=10, valid=True,
                        body_known=False, exits=0, progress=0.):
        diagnostics = dict(forward_obstacle_found=distance is not None,
                           obstacle_observation='known_occupied' if distance is not None else 'none_in_current_map',
                           startup_body_known=body_known, observed_forward_exit_directions=exits,
                           startup_progress_m=progress, startup_max_distance_m=5.,
                           startup_limit_segment=progress >= 5.)
        if distance is not None:
            diagnostics['nearest_obstacle_front_clearance_m'] = distance
        body = node.state.pose.pose[:]
        target = [body[0]+1., *body[1:]]
        self.response(node, 'plan_forward_cruise', dict(
            valid=valid, map_revision=revision, source_revision=NOW,
            path=[body, target] if valid else [],
            reason='cruise_corridor_valid' if valid else 'cruise_corridor_occupied',
            diagnostics=diagnostics))

    def test_pause_revokes_motion_and_late_obstacle_response_cannot_resume(self):
        node = self.node()
        old_revision = node.task_revision
        self.request(node, 'pause')
        self.response(node, 'plan_forward_cruise',
                      dict(valid=True, map_revision=10, diagnostics=dict(
                          forward_obstacle_found=True, nearest_obstacle_front_clearance_m=.1)),
                      task_revision=old_revision)
        self.assertFalse(node.enabled)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'inactive')
        self.assertFalse(node.token)
        self.assertIsNone(node.target)
        self.assertIsNone(node.hold_target)
        self.assertEqual(node.path, [])
        self.assertEqual(node.record.authorities, ['DISARM'])
        self.assertEqual(node.record.goals, [])

    def test_manual_takeover_revokes_and_late_plan_cannot_restore_target(self):
        node = self.node()
        old_revision = node.task_revision
        self.request(node, 'select_manual')
        self.response(node, 'plan', dict(valid=True, map_revision=10,
                      path=[BODY[:], [1., 0., 0., 0., 0., 0., 1.]], reason='planned'),
                      task_revision=old_revision)
        self.assertEqual(node.pending_control_mode, 'manual')
        self.assertFalse(node.enabled)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'inactive')
        self.assertFalse(node.token)
        self.assertIsNone(node.target)
        self.assertEqual(node.path, [])
        self.assertEqual(node.record.authorities, ['DISARM'])
        self.assertEqual(node.record.goals, [])

    def test_selecting_autonomy_after_manual_does_not_restart_cruise(self):
        node = self.node()
        node.enabled = False
        node.token = ''
        node.target = None
        node.path = []
        node.control_mode = 'manual'
        node.guard['state'] = 'DISARMED'
        self.request(node, 'select_autonomy')
        self.assertEqual(node.pending_control_mode, 'autonomy')
        self.assertFalse(node.enabled)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'inactive')
        self.assertFalse(node.token)
        self.assertIsNone(node.target)
        self.assertEqual(node.record.authorities, ['DISARM'])
        self.assertEqual(node.record.goals, [])

    def test_positive_gain_plan_waits_for_current_collision_validation(self):
        node = self.node()
        node.initialized = True
        node.forward_start_active = False
        node.cruise_active = False
        node.obstacle_phase = 'tracking'
        node.startup_profile = 'source_scan'
        (node.out/'nbv.sock').touch()
        self.response(node, 'plan', dict(valid=True, map_revision=10, reason='planned',
            path=[BODY[:], [1., 0., 0., 0., 0., 0., 1.]],
            diagnostics=dict(positive_gain_found=True, path_collision_valid=True)))
        self.assertIsNotNone(node.target)
        self.assertEqual(node.validated_revision, -1)
        node.tick()
        self.assertEqual(node.record.submissions[0][0], 'validate')
        self.assertEqual(node.record.submissions[0][1]['path'], [node.target])
        self.assertEqual(node.record.goals, [])

    def test_occupied_local_path_response_revokes_motion(self):
        node = self.node()
        node.initialized = True
        node.forward_start_active = False
        node.cruise_active = False
        node.obstacle_phase = 'tracking'
        self.response(node, 'validate', dict(valid=False, map_revision=10,
                                            reason='known_occupied_on_path'))
        self.assertFalse(node.enabled)
        self.assertFalse(node.token)
        self.assertIsNone(node.target)
        self.assertEqual(node.path, [])
        self.assertEqual(node.record.authorities, ['DISARM'])
        self.assertEqual(node.record.goals, [])

    def test_known_surroundings_and_free_exit_do_not_trigger_local_planning(self):
        node = self.node()
        self.cruise_response(node, body_known=True, exits=1)
        self.assertTrue(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'cruise')
        self.assertFalse(node.initialized)
        self.assertIsNotNone(node.target)
        self.assertEqual(node.validated_revision, 10)
        self.assertEqual(node.record.authorities, [])
        self.assertNotIn('obstacle_triggered_planning', [name for name, _ in node.record.events])

    def test_cruise_continues_past_five_meters_without_distance_cap(self):
        node = self.node()
        node.state.pose.pose = [12., 0., 0., 0., 0., 0., 1.]
        self.cruise_response(node, progress=12.)
        self.assertTrue(node.enabled)
        self.assertTrue(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'cruise')
        self.assertEqual(node.target, [13., 0., 0., 0., 0., 0., 1.])
        self.assertEqual(node.record.authorities, [])

    def test_obstacle_at_three_meters_replaces_cruise_with_active_braking(self):
        node = self.node()
        node.hold_target = [-1., 0., 0., 0., 0., 0., 1.]
        self.cruise_response(node, distance=3.)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'braking')
        self.assertFalse(node.initialized)
        self.assertEqual(node.token, 'lease')
        self.assertIsNone(node.target)
        self.assertIsNone(node.hold_target)
        self.assertEqual(node.path, [])
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])
        self.assertIn('obstacle_triggered_planning', [name for name, _ in node.record.events])

    def test_obstacle_beyond_three_meters_keeps_cruising(self):
        node = self.node()
        self.cruise_response(node, distance=3.01)
        self.assertTrue(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'cruise')
        self.assertIsNotNone(node.target)
        self.assertEqual(node.record.authorities, [])

    def test_stale_map_query_cannot_trigger_obstacle_or_occupied_decision(self):
        for valid in (True, False):
            with self.subTest(valid=valid):
                node = self.node()
                target = node.target[:]
                self.cruise_response(node, distance=0., revision=9, valid=valid)
                self.assertTrue(node.cruise_active)
                self.assertEqual(node.obstacle_phase, 'cruise')
                self.assertEqual(node.target, target)
                self.assertEqual(node.token, 'lease')
                self.assertEqual(node.record.authorities, [])
                self.assertNotIn('obstacle_triggered_planning',
                                 [name for name, _ in node.record.events])

    def test_stalled_map_source_cannot_trigger_obstacle_decision(self):
        node = self.node()
        node.map_source_fresh = lambda now: False
        target = node.target[:]
        self.cruise_response(node, distance=0.)
        self.assertTrue(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'cruise')
        self.assertEqual(node.target, target)
        self.assertEqual(node.token, 'lease')
        self.assertEqual(node.record.authorities, [])

    def test_new_map_heartbeat_does_not_refresh_stalled_source_progress(self):
        node = self.node()
        source = NOW-610_000_000
        node.map_source_wall = NOW-450_000_000
        node.map_source_revision_wall = NOW-450_000_000
        node.map_source_status = dict(ready=True, source_revision=source)
        node.map_source_acquisition_revision = source
        self.namespace['on_map_source'](node, NS(data=json.dumps(dict(
            run=node.out.name, ready=True, map_revision=10, source_revision=source))))
        self.assertEqual(node.map_source_wall, NOW)
        self.assertEqual(node.map_source_revision_wall, NOW-450_000_000)
        self.assertFalse(self.namespace['map_source_fresh'](node, NOW))

    def test_advancing_source_progress_is_fresh_despite_pipeline_acquisition_lag(self):
        node = self.node()
        source = NOW-610_000_000
        node.map_source_wall = NOW-300_000_000
        node.map_source_revision_wall = NOW-300_000_000
        node.map_source_status = dict(ready=True, source_revision=source-10_000_000)
        node.map_source_acquisition_revision = source-10_000_000
        self.namespace['on_map_source'](node, NS(data=json.dumps(dict(
            run=node.out.name, ready=True, map_revision=11, source_revision=source))))
        self.assertEqual(node.map_source_wall, NOW)
        self.assertEqual(node.map_source_revision_wall, NOW)
        self.assertEqual(node.map_source_status['source_revision'], source)
        self.assertTrue(self.namespace['map_source_fresh'](node, NOW))
        self.assertEqual(node.map_freshness_budget_ns, 450_000_000)
        node.map_source_wall = NOW-449_999_999
        node.map_source_revision_wall = NOW-449_999_999
        self.assertTrue(self.namespace['map_source_fresh'](node, NOW))
        node.map_source_wall = NOW-450_000_000
        self.assertFalse(self.namespace['map_source_fresh'](node, NOW))
        node.map_source_wall = NOW
        node.map_source_revision_wall = NOW-450_000_000
        self.assertFalse(self.namespace['map_source_fresh'](node, NOW))

    def test_occupied_cruise_path_brakes_even_without_distance_detection(self):
        node = self.node()
        self.cruise_response(node, valid=False)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.token, 'lease')
        self.assertIsNone(node.target)
        self.assertEqual(node.path, [])
        self.assertEqual(node.obstacle_phase, 'braking')
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])

    def test_all_exploration_start_actions_select_obstacle_cruise(self):
        for action in ('start_exploration', 'start_forward_exploration'):
            with self.subTest(action=action):
                node = self.node()
                node.enabled = False
                node.token = ''
                node.guard['state'] = 'DISARMED'
                node.cruise_active = False
                node.obstacle_phase = 'inactive'
                self.request(node, action)
                self.assertTrue(node.enabled)
                self.assertTrue(node.cruise_active)
                self.assertEqual(node.startup_profile, 'obstacle_triggered')
                self.assertEqual(node.obstacle_phase, 'cruise')

    def test_enable_service_used_by_cli_selects_obstacle_cruise(self):
        node = self.node()
        node.enabled = False
        node.token = ''
        node.guard['state'] = 'DISARMED'
        node.cruise_active = False
        node.obstacle_phase = 'inactive'
        result = node.enable(NS(data=True), NS(success=False, message=''))
        self.assertTrue(result.success)
        self.assertTrue(node.cruise_active)
        self.assertEqual(node.startup_profile, 'obstacle_triggered')
        self.assertEqual(node.obstacle_phase, 'cruise')

    def test_cruise_refreshes_segments_without_waiting_for_arrival(self):
        for reached in (False, True):
            with self.subTest(reached=reached):
                node = self.node()
                node.forward_start_active = False
                node.navigation = dict(reached=reached, generation=1, waypoint_index=12)
                (node.out/'nbv.sock').touch()
                node.tick()
                self.assertEqual(node.record.submissions[0][0], 'plan_forward_cruise')
                self.assertTrue(node.cruise_active)
                self.assertEqual(node.obstacle_phase, 'cruise')
                self.assertEqual(node.record.goals, [node.target])
                self.assertEqual(node.record.authorities, [])

    def test_obstacle_switch_waits_for_actual_stop_without_scheduling_motion(self):
        node = self.node()
        node.forward_start_active = False
        (node.out/'nbv.sock').touch()
        self.cruise_response(node, distance=3.)
        node.tick()
        self.assertEqual(node.record.submissions, [])
        self.assertEqual(node.record.authorities, [])
        self.assertTrue(all(index==0 for _,index in node.record.goal_calls))
        self.assertEqual(node.token, 'lease')
        self.assertIsNone(node.hold_target)

    def settled_navigation(self,node,**changes):
        report=dict(run=node.out.name,mission_id='nbv-reactive',state='TRACKING',
                    reached=True,generation=node.generation,waypoint_index=0,wall_ns=NOW)
        report.update(changes)
        return report

    def test_fresh_current_neutral_settle_allows_planning_from_latest_body(self):
        node=self.node()
        node.forward_start_active=False
        (node.out/'nbv.sock').touch()
        self.cruise_response(node,distance=3.)
        node.state.pose.pose=[.4,0.,0.,0.,0.,0.,1.]
        node.navigation=self.settled_navigation(node)
        node.tick()
        self.assertEqual(node.obstacle_phase,'planning')
        self.assertTrue(node.initialized)
        self.assertEqual(node.hold_target,node.state.pose.pose)
        self.assertEqual(node.record.submissions,[])
        self.assertIn('obstacle_braking_complete',[name for name,_ in node.record.events])
        node.tick()
        self.assertEqual(node.record.submissions,[('plan_turn',{})])
        self.assertTrue(all(index==0 for _,index in node.record.goal_calls))

    def test_old_foreign_or_unsettled_status_cannot_end_braking(self):
        for changes in [dict(wall_ns=NOW-1),dict(wall_ns=NOW+1),
                        dict(wall_ns=NOW-250_000_000),dict(reached=False),
                        dict(run='other'),dict(mission_id='other'),dict(generation=2),
                        dict(waypoint_index=12),dict(state='IDLE')]:
            with self.subTest(changes=changes):
                node=self.node()
                self.cruise_response(node,distance=3.)
                node.navigation=self.settled_navigation(node,**changes)
                node.tick()
                self.assertEqual(node.obstacle_phase,'braking')
                self.assertFalse(node.initialized)
                self.assertFalse(node.record.submissions)
                self.assertTrue(all(index==0 for _,index in node.record.goal_calls))

    def test_map_dropout_keeps_neutral_braking_without_authorizing_translation(self):
        node=self.node()
        self.cruise_response(node,distance=3.)
        node.healthy=lambda now:False
        node.map_source_fresh=lambda now:False
        node.tick()
        self.assertEqual(node.token,'lease')
        self.assertEqual(node.record.authorities,[])
        self.assertEqual(node.obstacle_phase,'braking')
        self.assertFalse(node.record.submissions)
        self.assertTrue(all(index==0 for _,index in node.record.goal_calls))

    def test_braking_without_token_can_request_neutral_authority(self):
        node=self.node()
        node.token='';node.guard['state']='DISARMED'
        self.cruise_response(node,distance=3.)
        node.tick()
        self.assertEqual(node.record.authorities,['ARM'])
        self.assertEqual(node.record.goal_calls,[])
        self.assertIsNone(node.target)

    def test_pause_during_braking_revokes_and_late_arm_cannot_resume(self):
        node=self.node()
        self.cruise_response(node,distance=3.)
        self.request(node,'pause')
        accepted=NS(accepted=True,token='late',generation=2)
        node.authority_action='ARM'
        node.authority_future=NS(done=lambda:True,result=lambda:accepted)
        before=len(node.record.goal_calls)
        node.tick()
        self.assertFalse(node.enabled)
        self.assertFalse(node.token)
        self.assertEqual(node.obstacle_phase,'inactive')
        self.assertEqual(len(node.record.goal_calls),before)
        self.assertEqual(node.record.authorities,['DISARM','DISARM'])

    def test_fault_during_braking_stops_even_before_authority_arrives(self):
        for token in ['lease','']:
            node=self.node()
            self.cruise_response(node,distance=3.)
            node.token=token
            node.guard.update(state='FAULT',reason='estimator_LOST')
            before=len(node.record.goal_calls)
            node.tick()
            self.assertFalse(node.enabled)
            self.assertFalse(node.token)
            self.assertEqual(node.obstacle_phase,'inactive')
            self.assertEqual(len(node.record.goal_calls),before)
            self.assertEqual(node.record.authorities,['DISARM'])

    def test_stale_map_progress_renews_only_neutral_with_current_authority(self):
        node = self.node()
        node.forward_start_active = False
        node.healthy = lambda now: False
        node.tick()
        self.assertEqual(node.token,'lease')
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])
        self.assertEqual(node.record.actions, [('goal', 0)])
        self.assertTrue(node.enabled)
        self.assertTrue(node.cruise_active)
        self.assertTrue(node.map_waiting)
        node.tick()
        self.assertEqual(node.record.goal_calls,[(BODY,0),(BODY,0)])
        self.assertEqual([kind for kind,_ in node.record.events],['map_progress_unavailable'])

    def test_map_progress_recovery_waits_for_current_path_validation(self):
        node=self.node()
        node.initialized=True
        node.forward_start_active=node.cruise_active=False
        node.obstacle_phase='tracking'
        node.healthy=lambda now:False
        node.tick()
        self.assertEqual(node.record.goal_calls,[(BODY,0)])
        node.healthy=lambda now:True
        node.map_revision=11
        node.last_goal_deadline_ns=NOW+90_000_000
        (node.out/'nbv.sock').touch()
        node.tick()
        self.assertFalse(node.map_waiting)
        self.assertEqual(node.record.submissions,[('validate',dict(path=[node.target]))])
        self.assertTrue(all(index==0 for _,index in node.record.goal_calls))
        self.response(node,'validate',dict(valid=True,map_revision=11))
        node.tick()
        self.assertEqual(node.record.goal_calls[-1],(node.target,None))
        self.assertEqual(node.token,'lease')
        self.assertEqual(node.record.authorities,[])
        self.assertIn('map_progress_restored',[kind for kind,_ in node.record.events])

    def test_revision_gap_keeps_authority_with_neutral_before_old_goal_deadline(self):
        node = self.node()
        node.forward_start_active = False
        node.map_revision = 11
        node.validated_revision = 10
        node.last_goal_deadline_ns = NOW+1
        node.tick()
        self.assertEqual(node.token, 'lease')
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])
        self.assertTrue(node.enabled)

    def test_pending_local_plan_renews_neutral_instead_of_old_hold_target(self):
        node = self.node()
        node.initialized = True
        node.forward_start_active = node.cruise_active = False
        node.target = None
        node.path = []
        node.hold_target = [99., 1., 0., 0., 0., 0., 1.]
        node.tick()
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])
        self.assertEqual(node.token, 'lease')
        self.assertEqual(node.record.authorities, [])

    def test_current_map_validation_restores_only_the_checked_motion_goal(self):
        node = self.node()
        node.forward_start_active = False
        node.map_revision = 11
        node.last_goal_deadline_ns = NOW+40_000_000
        node.tick()
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])
        self.response(node, 'validate', dict(valid=True, map_revision=11))
        node.tick()
        self.assertEqual(node.record.goal_calls, [(BODY, 0), (node.target, None)])
        self.assertEqual(node.token, 'lease')
        self.assertEqual(node.record.authorities, [])

    def test_short_revision_gap_keeps_existing_lease_without_toggling_target(self):
        node = self.node()
        node.forward_start_active = False
        node.map_revision = 11
        node.last_goal_deadline_ns = NOW+180_000_000
        node.tick()
        self.assertEqual(node.record.goal_calls, [])
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.token, 'lease')
        node.last_goal_deadline_ns = NOW+40_000_000
        node.tick()
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])

    def test_unhealthy_body_state_disarms_without_publishing_neutral(self):
        node = self.node()
        node.forward_start_active = False
        node.healthy = node.state_healthy = lambda now: False
        node.tick()
        self.assertEqual(node.record.goal_calls, [])
        self.assertEqual(node.record.authorities, ['DISARM'])
        self.assertFalse(node.token)
        self.assertFalse(node.enabled)

    def test_reserved_stage_reached_cannot_advance_real_path(self):
        for index,phase in [(0,'neutral'),(1,'align'),(2,'translate')]:
            with self.subTest(index=index):
                node = self.node()
                node.initialized = True
                node.forward_start_active = node.cruise_active = False
                node.navigation = dict(reached=True, generation=1, waypoint_index=index,
                                       source_waypoint_index=12,execution_phase=phase)
                target = node.target[:]
                node.tick()
                self.assertEqual(node.waypoint_index, 12)
                self.assertEqual(node.target, target)
                self.assertEqual(node.record.goal_calls, [(target, None)])
                self.assertEqual(node.record.events, [])

    def test_legacy_profile_map_revision_gap_does_not_renew_any_goal(self):
        node = self.node()
        node.startup_profile = 'source_scan'
        node.initialized = True
        node.forward_start_active = node.cruise_active = False
        node.map_revision = 11
        node.tick()
        self.assertEqual(node.record.goal_calls, [])
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.token, 'lease')

    def test_late_arm_ack_after_obstacle_switch_can_only_start_neutral_braking(self):
        node = self.node()
        node.forward_start_active = False
        node.token = ''
        accepted = NS(accepted=True, token='late-arm-token', generation=2)
        node.authority_action = 'ARM'
        node.authority_future = NS(done=lambda: False, result=lambda: accepted)

        def authority(action, *args):
            if node.authority_future is not None:
                return False
            node.record.authorities.append(action)
            return True

        node.request_authority = authority
        self.cruise_response(node, distance=3.)
        self.assertFalse(node.cruise_active)
        self.assertIsNone(node.target)
        self.assertEqual(node.record.authorities, [])
        node.authority_future = NS(done=lambda: True, result=lambda: accepted)
        node.tick()
        self.assertEqual(node.token, 'late-arm-token')
        self.assertEqual(node.generation, 2)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase, 'braking')
        self.assertIsNone(node.target)
        self.assertEqual(node.record.authorities, [])
        self.assertEqual(node.record.goal_calls, [(BODY, 0)])

    def turn_plan(self,node,*,valid=True,reason='planned',source_revision=10):
        self.response(node,'plan_turn',dict(valid=valid,map_revision=10,
            source_revision=source_revision,reason=reason,diagnostics={},
            path=[BODY[:],[1.,1.,0.,0.,0.,0.,1.]] if valid else []))

    def test_reactive_plan_selects_heading_without_tracking_source_endpoint(self):
        node=self.node()
        self.turn_plan(node)
        self.assertEqual(node.obstacle_phase,'turning')
        self.assertAlmostEqual(node.turn_yaw,math.pi/4.)
        self.assertIsNone(node.target)
        self.assertFalse(node.initialized)
        self.assertEqual(len(node.turn_path),2)
        self.assertEqual(node.path,node.turn_path)
        self.assertEqual(node.record.goals,[])
        self.assertTrue(next(data for kind,data in node.record.events if kind=='nbv_plan')['valid'])

    def test_turn_waits_for_current_path_validation_then_requests_only_rotation(self):
        node=self.node()
        self.turn_plan(node)
        node.tick()
        self.assertEqual(node.record.submissions,[('validate',dict(path=[node.turn_path[-1]]))])
        self.assertEqual(node.record.goal_calls,[(BODY,0)])
        self.response(node,'validate',dict(valid=True,map_revision=10))
        node.state.pose.pose=[.1,.2,.3,0.,0.,0.,1.]
        node.tick()
        target,index=node.record.goal_calls[-1]
        self.assertEqual(index,1)
        self.assertEqual(target[:3],[.1,.2,.3])
        self.assertAlmostEqual(target[5],math.sin(math.pi/8.))
        self.assertAlmostEqual(target[6],math.cos(math.pi/8.))
        self.assertEqual(node.turn_started_wall,NOW)

    def test_turn_reached_hands_off_direction_without_endpoint_arrival(self):
        node=self.node()
        self.turn_plan(node)
        node.validated_revision=10
        node.tick()
        node.navigation=self.settled_navigation(node,waypoint_index=1)
        # Position is still at the plan root, 1.4 m from the old endpoint.
        node.tick()
        self.assertEqual(node.obstacle_phase,'setting_cruise_direction')
        self.assertEqual(node.record.submissions[-1],('set_cruise_direction',dict(path=node.turn_path)))
        self.assertTrue(all(index in (0,1) for _,index in node.record.goal_calls))
        self.response(node,'set_cruise_direction',dict(valid=True,map_revision=11,
            source_revision=11,reason='cruise_direction_set_from_source_edge',diagnostics={}))
        self.assertTrue(node.cruise_active)
        self.assertEqual(node.obstacle_phase,'cruise')
        self.assertFalse(node.initialized)
        self.assertIsNone(node.target)
        self.assertEqual(node.turn_path,[])
        self.assertEqual(node.token,'lease')
        self.assertEqual(node.record.authorities,[])
        self.assertIn('avoidance_turn_complete',[kind for kind,_ in node.record.events])
        node.future=NS(done=lambda:False)
        node.tick()
        self.assertEqual(node.record.goal_calls[-1],(BODY,0))

    def test_old_or_foreign_turn_reached_cannot_start_cruise(self):
        for changes in [dict(waypoint_index=0),dict(waypoint_index=12),
                        dict(generation=2),dict(mission_id='nbv-explore'),
                        dict(wall_ns=NOW-1),dict(wall_ns=NOW+1),dict(reached=False)]:
            with self.subTest(changes=changes):
                node=self.node();self.turn_plan(node);node.validated_revision=10
                node.tick()
                node.navigation=self.settled_navigation(node,waypoint_index=1)
                node.navigation.update(changes)
                node.tick()
                self.assertEqual(node.obstacle_phase,'turning')
                self.assertFalse(node.cruise_active)
                self.assertEqual(node.record.submissions,[])

    def test_no_cruisable_turn_waits_for_real_new_map_before_retry(self):
        node=self.node()
        node.cruise_active=False;node.initialized=True;node.obstacle_phase='planning'
        node.map_source_status=dict(source_revision=10)
        self.turn_plan(node,valid=False,reason='no_cruisable_turn_direction')
        (node.out/'nbv.sock').touch()
        self.namespace['time'].monotonic_ns=lambda:NOW+2_000_000_000
        node.tick()
        self.assertEqual(node.record.submissions,[])
        self.assertTrue(all(index==0 for _,index in node.record.goal_calls))
        node.map_source_status['source_revision']=11
        node.tick()
        self.assertEqual(node.record.submissions,[('plan_turn',{})])

    def test_direction_rejected_does_not_resume_old_cruise(self):
        node=self.node();self.turn_plan(node)
        node.obstacle_phase='setting_cruise_direction'
        self.response(node,'set_cruise_direction',dict(valid=False,map_revision=11,
            source_revision=11,reason='new_cruise_direction_too_close_to_obstacle',diagnostics={}))
        node.tick()
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase,'planning')
        self.assertIsNone(node.target)
        self.assertTrue(all(index==0 for _,index in node.record.goal_calls))

    def test_pause_rejects_late_direction_handoff(self):
        node=self.node();self.turn_plan(node)
        old_revision=node.task_revision
        self.request(node,'pause')
        self.response(node,'set_cruise_direction',dict(valid=True,map_revision=11,
            diagnostics={}),task_revision=old_revision)
        self.assertFalse(node.enabled)
        self.assertFalse(node.cruise_active)
        self.assertEqual(node.obstacle_phase,'inactive')
        self.assertEqual(node.turn_path,[])
        self.assertEqual(node.record.goal_calls,[])

    def test_started_stationary_turn_is_not_interrupted_by_map_progress_gap(self):
        node=self.node();self.turn_plan(node);node.validated_revision=10
        node.tick()
        before=len(node.record.goal_calls)
        node.healthy=lambda now:False
        node.map_source_fresh=lambda now:False
        node.tick()
        self.assertEqual(node.obstacle_phase,'turning')
        self.assertEqual(node.token,'lease')
        self.assertEqual(node.record.authorities,[])
        self.assertTrue(all(index==1 for _,index in node.record.goal_calls[before:]))
        self.assertEqual(node.record.submissions,[])

    def test_turn_does_not_start_on_stale_map(self):
        node=self.node();self.turn_plan(node)
        node.healthy=lambda now:False
        node.tick()
        self.assertEqual(node.turn_started_wall,0)
        self.assertEqual(node.record.goal_calls,[(BODY,0)])
        self.assertEqual(node.record.submissions,[])

    def test_aligned_turn_holds_neutral_until_fresh_map_can_commit(self):
        node=self.node();self.turn_plan(node);node.validated_revision=10
        node.tick()
        node.navigation=self.settled_navigation(node,waypoint_index=1)
        node.healthy=lambda now:False
        node.tick()
        self.assertEqual(node.obstacle_phase,'setting_cruise_direction')
        self.assertEqual(node.record.goal_calls[-1],(BODY,0))
        self.assertEqual(node.record.submissions,[])
        node.tick()
        self.assertEqual(node.record.submissions,[])
        node.healthy=lambda now:True
        node.map_revision=11
        node.tick()
        self.assertEqual(node.record.submissions,[('set_cruise_direction',dict(path=node.turn_path))])
        self.assertFalse(node.cruise_active)

    def test_turn_map_exemption_does_not_bypass_stale_body_revocation(self):
        node=self.node();self.turn_plan(node);node.validated_revision=10
        node.tick()
        before=len(node.record.goal_calls)
        node.state_healthy=lambda now:False
        node.healthy=lambda now:False
        node.tick()
        self.assertFalse(node.enabled)
        self.assertFalse(node.token)
        self.assertEqual(node.record.authorities,['DISARM'])
        self.assertEqual(len(node.record.goal_calls),before)


if __name__ == '__main__':
    unittest.main()
