#!/usr/bin/env python3
"""ROS2 acquisition, NBV service transport, and leased navigation goals."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import math
from pathlib import Path
import time
import threading
import socket
import struct

import numpy as np
import yaml
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock as ClockMsg
from std_msgs.msg import String
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray
from uw_interfaces.msg import NavigationGoal
from uw_interfaces.srv import Control
from geometry import compose, optical_transform
from nbv_protocol import exchange, interpolate_pose, project_depth
from nbv_initialization import SourceInitialization
from mission_geometry import route_poses


def ns(stamp): return stamp.sec*10**9+stamp.nanosec


def pose_values(message):
    p,q=message.position,message.orientation
    return [p.x,p.y,p.z,q.x,q.y,q.z,q.w]


def fill(pose,values):
    pose.position.x,pose.position.y,pose.position.z=map(float,values[:3])
    pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,values[3:])


class NbvBridge(Node):
    def __init__(self):
        super().__init__('pelagos_nbv_bridge')
        self.out=Path(self.declare_parameter('run_dir','').value)
        profile=yaml.safe_load((self.out/'robot_profile.yaml').read_text())
        geometry=json.loads((self.out/'stereo-geometry.json').read_text())
        self.body_camera=optical_transform(profile['cameras']['left'])
        self.k=np.asarray(geometry['K_left'])
        self.shape=(geometry['height'],geometry['width'])
        self.params=yaml.safe_load((self.out/'nbv.yaml').read_text())
        self.map_source=self.params.get('pelagos_map_source','wavemap')
        self.map_source_status={};self.map_source_wall=0
        self.map_source_revision_wall=0;self.map_source_acquisition_revision=0
        self.map_waiting=False
        mapping=yaml.safe_load((self.out/'wavemap.yaml').read_text())['/**']['ros__parameters']
        map_publish_hz=float(mapping.get('publish_rate_hz',0.))
        # The map/relay is a slower stream than body/control feedback. Its
        # existing receive budget must include one configured publish period.
        self.map_freshness_budget_ns=250_000_000+(round(1.e9/map_publish_hz) if map_publish_hz>0 else 0)
        navigation=yaml.safe_load((self.out/'navigation-parameters.yaml').read_text())
        self.neutral_renewal_margin_ns=round(2.e9/float(navigation['rate_hz']))
        self.planning_only=yaml.safe_load((self.out/'resolved_config.yaml').read_text()).get('planning_only',False)
        self.preview_requested=None;self.preview_result=None
        self.preview_plan_operation='plan_turn'
        self.startup_profile='source_scan'
        self.forward_start_required=False;self.forward_start_active=False
        self.startup_limit_segment=False
        self.startup_map_status={}
        self.cruise_active=False;self.obstacle_phase='inactive'
        self.obstacle_switch_distance_m=float(self.params.get('pelagos_obstacle_plan_distance_m',3.0))
        self.cruise_map_status={};self.cruise_requests=0
        self.turn_path=[];self.turn_yaw=None;self.turn_started_wall=0
        self.turn_retry_source_revision=-1
        self.bridge=CvBridge()
        self.clock_socket=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
        self.clock_socket.setblocking(False)
        self.create_subscription(ClockMsg,'/clock',self.forward_clock,1)
        self.history=deque(maxlen=600)
        self.state=None;self.state_wall=0
        self.localization={};self.localization_wall=0
        self.guard={};self.guard_wall=0
        self.controller={};self.controller_wall=0
        self.control_mode='autonomy';self.pending_control_mode=None
        self.joystick={};self.joystick_wall=0
        self.navigation={}
        self.pending=None;self.depth_wall=0;self.depth_stamp=0;self.processed_depth_stamp=0
        self.last_observe_stamp=0;self.last_plan_wall=0
        self.sequence=0;self.goal_sequence=0;self.map_revision=0
        self.last_goal_deadline_ns=0
        self.future=None;self.job=None;self.pool=ThreadPoolExecutor(max_workers=1)
        self.stop_event=threading.Event()
        self.path=[];self.path_stamp=0;self.target=None;self.validated_revision=-1
        self.hold_target=None;self.hold_index=12;self.waypoint_index=12
        self.enabled=False;self.reason='mapping_preview'
        self.backend_state=None
        self.mode='explore';self.route=[];self.route_index=0;self.route_complete=False
        self.task_revision=0;self.reset_execution=False
        self.plans_requested=0;self.paths_generated=0;self.last_plan_seconds=None
        self.initialized=False;self.initialization=SourceInitialization()
        self.authority=self.create_client(Control,'/rov01/control/authority')
        self.authority_future=None;self.authority_action=None
        self.token='';self.generation=0;self.reached_wall=0
        self.status_pub=self.create_publisher(String,'/pelagos/planner/status',1)
        self.path_pub=self.create_publisher(RosPath,'/pelagos/planner/path',1)
        self.map_pub=self.create_publisher(MarkerArray,'/pelagos/planner/map',1)
        self.goal_pub=self.create_publisher(NavigationGoal,'/rov01/navigation/goal',1)
        self.goal_marker=self.create_publisher(Marker,'/pelagos/planner/goal',1)
        self.text_pub=self.create_publisher(Marker,'/pelagos/planner/text',1)
        durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.debug_pub=self.create_publisher(String,'/pelagos/planner/debug',durable)
        self.route_pub=self.create_publisher(RosPath,'/pelagos/mission/route',durable)
        self.response_pub=self.create_publisher(String,'/pelagos/mission/response',10)
        self.create_subscription(String,'/pelagos/mission/request',self.mission_request,10)
        self.create_subscription(Odometry,'/rov01/state/odometry',self.on_state,1)
        if self.map_source=='native_depth':
            self.create_subscription(Image,'/stereo/depth',self.on_depth,qos_profile_sensor_data)
        else:self.create_subscription(String,'/pelagos/nbv/map_status',self.on_map_source,1)
        self.create_subscription(String,'/rov01/localization/status',self.on_localization,1)
        self.create_subscription(String,'/rov01/control/status',self.on_guard,1)
        self.create_subscription(String,'/rov01/control/controller_status',self.on_controller,1)
        self.create_subscription(String,'/pelagos/joystick/status',self.on_joystick,1)
        self.create_subscription(String,'/rov01/navigation/status',
                                 lambda m:setattr(self,'navigation',json.loads(m.data)),1)
        self.create_service(SetBool,'/pelagos/exploration/enable',self.enable)
        self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_localization(self,m):
        self.localization=json.loads(m.data);self.localization_wall=time.monotonic_ns()

    def on_map_source(self,message):
        value=json.loads(message.data)
        if value.get('run')==self.out.name:
            self.map_source_status=value
            self.map_source_wall=time.monotonic_ns()
            revision=int(value.get('source_revision',0))
            if revision>self.map_source_acquisition_revision:
                self.map_source_acquisition_revision=revision
                self.map_source_revision_wall=self.map_source_wall
            if value.get('ready'):self.map_revision=max(self.map_revision,int(value.get('map_revision',0)))

    def forward_clock(self,message):
        try:self.clock_socket.sendto(struct.pack('=q',ns(message.clock)),str(self.out/'nbv-clock.sock'))
        except OSError:pass  # Backend creates the socket while ROS1 starts.

    def on_guard(self,m):
        self.guard=json.loads(m.data);self.guard_wall=time.monotonic_ns()

    def on_controller(self,m):
        self.controller=json.loads(m.data);self.controller_wall=time.monotonic_ns()

    def on_joystick(self,m):
        value=json.loads(m.data)
        if value.get('run')==self.out.name:self.joystick=value;self.joystick_wall=time.monotonic_ns()

    def on_state(self,m):
        if m.header.frame_id!='rov01/odom' or m.child_frame_id!='rov01/base_link':return
        stamp=ns(m.header.stamp)
        if self.history and stamp<=self.history[-1][0]:
            if stamp<self.history[-1][0]:self.disable('state_time_rewind')
            return
        values=pose_values(m.pose.pose)
        if not all(math.isfinite(x) for x in values):return
        self.history.append((stamp,values))
        self.state=m;self.state_wall=time.monotonic_ns()

    def on_depth(self,m):
        stamp=ns(m.header.stamp)
        if (m.encoding!='32FC1' or (m.height,m.width)!=self.shape or
                m.header.frame_id!='cam0_rect_optical' or stamp<=self.depth_stamp):return
        self.depth_stamp=stamp
        # Keep only the newest frame. Inference/transport cannot grow a queue.
        self.pending=m

    def enable(self,request,response):
        if request.data:
            if self.planning_only:
                response.success=False;response.message='当前为纯规划实验，不启动控制；请使用生成规划预览';return response
            if self.control_mode!='autonomy' or self.pending_control_mode:
                response.success=False;response.message='请先选择自主任务控制模式';return response
            if self.enabled:
                response.success=True;response.message='Exploration is already enabled';return response
            if self.guard.get('state')=='FAULT':
                response.success=False;response.message='Guard is faulted: '+str(self.guard.get('reason'));return response
            if self.localization.get('epoch',1)!=1:
                response.success=False;response.message='Estimator reset: restart simulation and NBV map';return response
            backend=self.out/'nbv-backend-status.json'
            if backend.exists():
                response.success=False;response.message='NBV backend exited; restart this simulation run';return response
            self.enabled=True
            self.startup_profile='obstacle_triggered';self.forward_start_required=False;self.forward_start_active=False
            self.cruise_active=True;self.obstacle_phase='cruise';self.cruise_map_status={}
            self.initialized=False;self.initialization=SourceInitialization()
            self.task_revision+=1;self.mode='explore';self.reset_execution=True
            self.reason='waiting_for_cruise_state_map_and_jepa_history'
            response.message='Straight cruise requested; local planning begins at about 3 m obstacle clearance'
        else:
            self.disable('explicit_stop');response.message='Exploration stopped; DISARM requested'
        response.success=True;return response

    def event(self,kind,**data):
        with (self.out/'mission-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(wall_ns=time.monotonic_ns(),kind=kind,
                mode=self.mode,task_revision=self.task_revision,**data),allow_nan=False)+'\n')

    def mission_request(self,message):
        request={}
        try:
            request=json.loads(message.data)
            if not isinstance(request,dict):request={};raise ValueError('任务消息必须是 JSON 对象')
            if request.get('run_id')!=self.out.name or request.get('epoch')!=1:
                raise ValueError('任务的运行编号或估计会话不匹配')
            action=request['action']
            if self.planning_only and action not in ('plan_preview','forward_start_preview','pause'):
                raise ValueError('当前为纯规划实验，只允许生成路径预览或暂停')
            if action=='pause':self.disable('operator_pause')
            elif action in ('plan_preview','forward_start_preview'):
                if self.enabled or self.token:raise ValueError('请先暂停运动任务，再生成纯规划路径')
                if self.preview_requested or (self.job and self.job.get('preview')):
                    raise ValueError('纯规划正在计算，请等待本次结果')
                if not self.healthy(time.monotonic_ns()):raise ValueError('等待实际估计位姿和规划地图就绪')
                if not self.planning_only and self.guard.get('state')!='DISARMED':
                    raise ValueError('纯规划预览要求 Guard 处于 DISARMED')
                self.disable('planning_only_preview');self.preview_requested='reset_execution';self.preview_result=None
                self.preview_plan_operation='plan_forward_start' if action=='forward_start_preview' else 'plan_turn'
            elif action=='reset_guard':
                if self.enabled or self.token:raise ValueError('请先暂停任务')
                if self.authority_future is not None:raise ValueError('控制服务正在应答，请稍后重试')
                if not self.request_authority('CLEAR_FAULT'):raise ValueError('控制授权服务尚未就绪')
            elif action in ('select_manual','select_autonomy'):
                if not 0<=time.monotonic_ns()-int(request.get('issued_steady_ns',0))<250_000_000:
                    raise ValueError('控制模式指令已过期，请重新选择')
                if self.pending_control_mode:raise ValueError('控制模式正在切换，请等待完成')
                desired='manual' if action=='select_manual' else 'autonomy'
                if desired!=self.control_mode:
                    self.pending_control_mode=desired
                    self.disable('manual_takeover' if desired=='manual' else 'autonomy_selected')
            elif action in ('start_route','resume_route','start_exploration','start_forward_exploration'):
                # One-shot operator intent uses the existing control protocol's
                # 250 ms freshness budget; autonomous task execution has no new timeout.
                if not 0<=time.monotonic_ns()-int(request.get('issued_steady_ns',0))<250_000_000:
                    raise ValueError('任务指令已过期，请重新点击开始')
                if self.enabled:raise ValueError('请先暂停当前任务再切换任务')
                if self.guard.get('state')=='FAULT':raise ValueError('Guard 故障尚未清除，请查看故障详情')
                if not self.healthy(time.monotonic_ns()):raise ValueError('等待估计状态及规划地图就绪')
                route=None
                if action=='start_route':
                    if request.get('frame_id')!='rov01/odom':raise ValueError('路径点必须在本次估计坐标系中')
                    route=route_poses(request['poses'])
                if action=='resume_route' and (not self.route or self.route_complete):raise ValueError('没有可继续的路线')
                response=self.enable(SetBool.Request(data=True),SetBool.Response())
                if not response.success:raise ValueError(response.message)
                if action not in ('start_exploration','start_forward_exploration'):
                    self.cruise_active=False;self.obstacle_phase='inactive'
                    self.startup_profile='waypoints';self.initialized=True
                    self.mode='waypoints';self.reset_execution=False
                    if route is not None:
                        self.route=route;self.route_index=0;self.route_complete=False
                        data=dict(run_id=self.out.name,frame_id='rov01/odom',epoch=1,poses=route)
                        (self.out/'mission-route.json').write_text(json.dumps(data,indent=2))
                        self.publish_route()
            else:raise ValueError('不支持的任务操作')
            self.event('operator_command',action=action,request_id=request.get('request_id'))
            result=dict(accepted=True,reason='已接收任务操作：'+action)
        except (ValueError,KeyError,TypeError) as error:
            result=dict(accepted=False,reason=str(error))
        self.response_pub.publish(String(data=json.dumps(dict(
            request_id=request.get('request_id'),run_id=self.out.name,**result),ensure_ascii=False)))

    def publish_route(self):
        path=RosPath();path.header.frame_id='rov01/odom'
        path.header.stamp=self.get_clock().now().to_msg()
        for values in self.route:
            pose=PoseStamped();pose.header=path.header;fill(pose.pose,values);path.poses.append(pose)
        self.route_pub.publish(path)

    def request_authority(self,action,source=None):
        if self.authority_future is not None or not self.authority.service_is_ready():return False
        req=Control.Request()
        req.run_id=self.out.name;req.action=action
        req.source=source or ('cli' if self.control_mode=='manual' else 'navigation')
        self.authority_future=self.authority.call_async(req);self.authority_action=action
        return True

    def disable(self,reason):
        self.task_revision+=1
        self.enabled=False;self.token='';self.reason=reason
        self.hold_target=None;self.target=None;self.path=[];self.validated_revision=-1
        self.forward_start_required=False;self.forward_start_active=False
        self.cruise_active=False;self.obstacle_phase='inactive'
        self.turn_path=[];self.turn_yaw=None;self.turn_started_wall=0
        self.turn_retry_source_revision=-1
        self.map_waiting=False
        self.preview_requested=None
        self.request_authority('DISARM')
        self.event('stopped',reason=reason)
        if not self.initialized:
            self.initialization=SourceInitialization()

    def healthy(self,now):
        return (self.state_healthy(now) and self.map_revision>0 and
            (self.map_source=='native_depth' or
             self.map_source_fresh(now)))

    def map_source_fresh(self,now):
        # A ready heartbeat cannot renew a stalled source. Check actual frame
        # progress in wall time; the mapping pipeline has its own capture-time
        # interpolation, projection and publishing latency, unlike body state.
        source=int(self.map_source_status.get('source_revision',0))
        return (self.state is not None and source>0 and
            now-self.map_source_wall<self.map_freshness_budget_ns and self.map_source_status.get('ready',False) and
            now-self.map_source_revision_wall<self.map_freshness_budget_ns)

    def state_healthy(self,now):
        return (self.state is not None and now-self.state_wall<250_000_000 and
                now-self.localization_wall<250_000_000 and (self.planning_only or now-self.guard_wall<250_000_000) and
                self.localization.get('state')=='TRACKING' and self.localization.get('epoch')==1)

    def submit(self,operation,stamp_ns=None,body=None,**kwargs):
        self.sequence+=1
        selected_body=pose_values(self.state.pose.pose) if body is None else body
        if 'camera' not in kwargs:
            camera=compose((selected_body[:3],selected_body[3:]),self.body_camera)
            kwargs['camera']=[*camera[0],*camera[1]]
        request=dict(run=self.out.name,epoch=1,sequence=self.sequence,operation=operation,
                     task_revision=self.task_revision,
                     stamp_ns=ns(self.state.header.stamp) if stamp_ns is None else stamp_ns,
                     body=selected_body,**kwargs)
        self.job=request
        if operation in ('plan','plan_turn','plan_forward_start'):
            self.plans_requested+=1;request['submitted_wall_ns']=time.monotonic_ns()
        if operation=='plan_forward_cruise':self.cruise_requests+=1
        self.future=self.pool.submit(exchange,self.out/'nbv.sock',request,self.stop_event)

    def poll(self,now):
        if self.future is None or not self.future.done():return
        future,job=self.future,self.job
        self.future=None;self.job=None
        try: result=future.result()
        except Exception as error:
            if job['operation']!='observe' and job['task_revision']!=self.task_revision:return
            self.reason='nbv_transport: '+str(error)
            self.path=[];self.target=None;self.validated_revision=-1
            backend=self.out/'nbv-backend-status.json'
            if backend.exists():
                try:
                    state=json.loads(backend.read_text()).get('state')
                    if state=='finished':self.reason='upstream_planner_finished'
                except (ValueError,OSError):pass
            if self.enabled:self.disable(self.reason)
            return
        # Imports now arrive independently of planning/validation responses.
        # A completed query must not roll back a newer installed-map report.
        self.map_revision=max(self.map_revision,result.get('map_revision',self.map_revision))
        operation=job['operation']
        if operation in ('plan_forward_start','validate_forward_start') and result.get('diagnostics'):
            self.startup_map_status=dict(map_revision=result['map_revision'],**{
                k:result['diagnostics'][k] for k in ('startup_body_known','startup_can_plan_forward','observed_forward_exit_directions','root_unknown_samples','root_occupied_samples',
                    'startup_progress_m','startup_max_distance_m','startup_limit_segment') if k in result['diagnostics']})
        if operation!='observe' and job['task_revision']!=self.task_revision:return
        if operation=='plan_forward_cruise':
            self.apply_cruise_result(result,job,now);return
        if operation in ('plan','plan_turn','plan_forward_start','inspect') and result.get('diagnostics'):
            self.debug_pub.publish(String(data=json.dumps(dict(
                run_id=self.out.name,requested_body=job['body'],execution_authorized=bool(self.enabled),
                origin='actual_estimated_body',**result))))
        if job.get('preview'):
            if operation=='reset_execution':
                self.preview_requested='inspect';return
            self.preview_result=result
            (self.out/'planner-diagnostics.json').write_text(json.dumps(result,indent=2))
            if operation=='inspect':self.preview_requested=self.preview_plan_operation;self.reason='planning_preview_computing'
            else:
                self.path=result.get('path',[]) if result.get('valid') else []
                self.path_stamp=job['stamp_ns'];self.target=None
                self.reason='planning_preview_ready' if self.path else 'planning_preview_no_valid_path'
                self.last_plan_seconds=(now-job['submitted_wall_ns'])*1e-9
                if self.path:self.paths_generated+=1
                self.publish_plan(result.get('occupied',[]))
                evidence=dict(run_id=self.out.name,frame_id='rov01/odom',execution_authorized=False,
                    requested_body=job['body'],**result)
                (self.out/'planner-preview.json').write_text(json.dumps(evidence,indent=2))
                self.event('planning_preview',valid=bool(self.path),points=len(self.path),diagnostics=result.get('diagnostics'))
            return
        if operation=='observe' and result['valid']:
            self.depth_wall=now;self.processed_depth_stamp=job['stamp_ns']
        elif operation in ('plan','plan_turn','plan_forward_start'):
            # Keep the actual source tree/termination diagnostics for every
            # executed planning request, including empty-path responses.
            with (self.out/'planner-execution-results.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(run_id=self.out.name,wall_ns=now,
                    requested_body=job['body'],requested_stamp_ns=job['stamp_ns'],
                    operation=operation,**result),allow_nan=False)+'\n')
            self.path=result['path'] if result['valid'] else []
            self.target=self.path[-1] if self.path else None
            self.path_stamp=job['stamp_ns']
            # Check from the latest estimated body pose after computation;
            # the request-time body pose may have moved in the meantime.
            self.validated_revision=-1
            if self.target:self.waypoint_index+=1
            self.reason=result['reason']
            self.last_plan_wall=now
            self.last_plan_seconds=(now-job['submitted_wall_ns'])*1e-9
            if operation=='plan' and not self.target and result['reason']=='no_forward_known_free_edge':
                self.disable(result['reason'])
            if self.target:self.paths_generated+=1
            if operation=='plan_forward_start':
                self.forward_start_required=False;self.forward_start_active=bool(self.target)
                self.startup_limit_segment=str(result.get('diagnostics',{}).get('startup_limit_segment','false')).lower()=='true'
                if not self.target:self.disable(result['reason'])
            self.event('nbv_plan',valid=bool(self.target),points=len(self.path),
                       computation_seconds=self.last_plan_seconds,map_revision=self.map_revision,
                       path_kind='startup_forward' if operation=='plan_forward_start' else 'nbv_rrt')
            self.publish_plan(result.get('occupied') or None)
            if operation=='plan_turn':
                self.turn_retry_source_revision=result.get('source_revision',self.map_revision)
                if self.target:self.begin_avoidance_turn(now)
                else:
                    self.obstacle_phase='planning';self.initialized=True
                    self.hold_target=pose_values(self.state.pose.pose)
                    self.reason=result['reason']
        elif operation=='set_cruise_direction':
            if not self.enabled or self.obstacle_phase!='setting_cruise_direction':return
            self.turn_path=[];self.turn_yaw=None;self.turn_started_wall=0
            self.target=None;self.path=[];self.validated_revision=-1
            if result['valid']:
                self.cruise_active=True;self.obstacle_phase='cruise';self.initialized=False
                # Renew neutral while the first rolling goal is computed. The
                # old turn's original 200 ms intent must not silently expire.
                self.hold_target=pose_values(self.state.pose.pose)
                self.reason='turn_complete_continuous_cruise'
                self.event('avoidance_turn_complete',body=pose_values(self.state.pose.pose),
                    diagnostics=result.get('diagnostics',{}),map_revision=result['map_revision'])
            else:
                self.cruise_active=False;self.obstacle_phase='planning';self.initialized=True
                self.hold_target=pose_values(self.state.pose.pose)
                self.turn_retry_source_revision=result.get('source_revision',self.map_revision)
                self.reason=result['reason'];self.last_plan_wall=now
                self.event('avoidance_turn_direction_rejected',reason=result['reason'],
                    diagnostics=result.get('diagnostics',{}))
            self.publish_plan()
        elif operation=='reset_execution':
            if not result['valid']:self.disable('planner_execution_reset_failed')
            else:self.reset_execution=False
        elif operation in ('validate','validate_forward_start'):
            if result['valid']:
                self.validated_revision=result['map_revision']
                if (operation=='validate_forward_start' and result['map_revision']==self.map_revision and
                    self.startup_profile=='forward_until_observed' and
                    any(str(result.get('diagnostics',{}).get(k,'false')).lower()=='true' for k in ('startup_body_known','startup_can_plan_forward'))):
                    self.forward_start_active=False;self.forward_start_required=False;self.initialized=True
                    self.hold_target=pose_values(self.state.pose.pose);self.hold_index=self.waypoint_index
                    self.target=None;self.path=[];self.validated_revision=-1
                    self.reason='startup_observed_corridor_replanning'
                    self.event('startup_observed_corridor_switch_to_nbv',body=self.hold_target,map_revision=self.map_revision,
                        startup_map_status=self.startup_map_status)
            else:
                self.target=None;self.path=[];self.validated_revision=-1
                if self.token:self.disable('path_blocked_or_unknown')
                else:self.reason='path_blocked_or_unknown'

    def apply_cruise_result(self,result,job,now):
        if not self.enabled or not self.cruise_active:return
        self.cruise_map_status=dict(result.get('diagnostics',{}))
        self.cruise_map_status.update(map_revision=result['map_revision'],
            source_revision=result.get('source_revision',0))
        if result['map_revision']!=self.map_revision or not self.map_source_fresh(now):
            self.reason='waiting_for_current_cruise_map';return
        found=str(self.cruise_map_status.get('forward_obstacle_found','false')).lower()=='true'
        value=self.cruise_map_status.get('nearest_obstacle_front_clearance_m')
        distance=float(value) if found and value is not None else None
        with (self.out/'cruise-evidence.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(wall_ns=now,requested_body=job['body'],
                map_revision=result['map_revision'],source_revision=result.get('source_revision',0),
                path_valid=bool(result['valid']),obstacle_clearance_m=distance,
                diagnostics=result.get('diagnostics',{})),allow_nan=False)+'\n')
        if (distance is not None and distance<=self.obstacle_switch_distance_m) or not result['valid']:
            # A DISARM/ARM round trip is not a stopped vehicle. Keep the
            # controller braking with explicit zero velocity until the retained
            # Tracker reports its existing speed/yaw-rate/dwell settle criteria.
            self.cruise_active=False;self.obstacle_phase='braking';self.initialized=False
            self.obstacle_braking_started_wall=now
            self.target=None;self.hold_target=None;self.path=[];self.validated_revision=-1
            if self.token:self.publish_neutral_goal(now)
            self.reason='obstacle_3m_braking_before_planning';self.last_plan_wall=0
            self.event('obstacle_triggered_planning',clearance_m=distance,
                trigger_distance_m=self.obstacle_switch_distance_m,map_revision=result['map_revision'],
                cruise_path_valid=bool(result['valid']),body=pose_values(self.state.pose.pose))
            self.publish_plan();return
        path=result.get('path',[])
        if not path:self.disable('cruise_backend_returned_empty_segment');return
        if self.target is None:self.waypoint_index+=1
        self.path=path;self.target=path[-1];self.path_stamp=job['stamp_ns']
        self.hold_target=None
        self.validated_revision=result['map_revision']
        self.reason='cruising_until_obstacle_3m';self.publish_plan()

    def navigation_mission_id(self):
        return 'nbv-reactive' if self.startup_profile=='obstacle_triggered' and self.mode=='explore' else 'nbv-explore'

    def begin_avoidance_turn(self,now):
        # Execute only the geometry-derived heading. The sampled NBV endpoint
        # and camera view remain planning evidence, not tracker destinations.
        self.turn_path=[pose[:] for pose in self.path]
        dx=self.turn_path[-1][0]-self.turn_path[0][0]
        dy=self.turn_path[-1][1]-self.turn_path[0][1]
        if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx,dy)==0.:
            self.disable('nbv_turn_has_no_horizontal_direction');return
        self.turn_yaw=math.atan2(dy,dx);self.turn_started_wall=0
        self.cruise_active=False;self.obstacle_phase='turning';self.initialized=False
        self.forward_start_active=False;self.forward_start_required=False
        self.target=None;self.validated_revision=-1
        self.hold_target=pose_values(self.state.pose.pose)
        self.event('nbv_turn_selected',yaw_rad=self.turn_yaw,source_path=self.turn_path,
            source_view_yaw_executed=False,body=self.hold_target)
        self.reason='turning_to_nbv_direction'

    def tick_avoidance_turn(self,now,healthy):
        if not self.state_healthy(now):return
        if self.obstacle_phase=='setting_cruise_direction':
            if self.token:self.publish_neutral_goal(now)
            self.reason='checking_new_cruise_direction' if healthy else 'waiting_for_fresh_map_to_resume'
            if healthy and self.future is None:self.submit('set_cruise_direction',path=self.turn_path)
            return
        if not healthy and not self.turn_started_wall:return
        if self.obstacle_phase=='turning' and not self.turn_started_wall:
            # Check the source edge once from the current body before starting
            # this stationary rotation. A later map update does not imply a
            # new moving goal; the final direction handoff rechecks the map.
            if self.validated_revision!=self.map_revision:
                if self.future is None:self.submit('validate',path=[self.turn_path[-1]])
                if self.token:self.publish_neutral_goal(now)
                self.reason='checking_nbv_turn_direction';return
        if not self.token:
            if (self.guard.get('state')=='DISARMED' and self.controller.get('ready_for_arm') and
                    now-self.controller_wall<250_000_000):self.request_authority('ARM')
            return
        if not self.turn_started_wall:self.turn_started_wall=now
        target=pose_values(self.state.pose.pose)
        target[3:]=[0.,0.,math.sin(self.turn_yaw/2.),math.cos(self.turn_yaw/2.)]
        self.publish_goal(target,now,1)
        self.reason='turning_to_nbv_direction'
        navigation=self.navigation
        aligned=(navigation.get('run')==self.out.name and
            navigation.get('mission_id')==self.navigation_mission_id() and
            navigation.get('state')=='TRACKING' and navigation.get('reached') and
            navigation.get('generation')==self.generation and navigation.get('waypoint_index')==1 and
            navigation.get('wall_ns',0)>=self.turn_started_wall and
            0<=now-navigation.get('wall_ns',0)<250_000_000)
        if aligned and self.obstacle_phase=='turning':
            self.obstacle_phase='setting_cruise_direction'
            self.event('nbv_turn_aligned',yaw_rad=self.turn_yaw,
                body=pose_values(self.state.pose.pose),navigation=navigation)
            self.publish_neutral_goal(now)
            if healthy and self.future is None:
                self.submit('set_cruise_direction',path=self.turn_path)
                self.reason='checking_new_cruise_direction'
            else:self.reason='waiting_for_fresh_map_to_resume'

    def publish_plan(self,occupied=None):
        header=self.state.header
        path=RosPath();path.header=header
        for values in self.path:
            p=PoseStamped();p.header=header;fill(p.pose,values);path.poses.append(p)
        self.path_pub.publish(path)
        if occupied is None:return
        array=MarkerArray()
        clear=Marker();clear.header=header;clear.action=Marker.DELETEALL;array.markers.append(clear)
        groups={}
        for x,y,z,size in occupied:groups.setdefault(round(size,6),[]).append(Point(x=x,y=y,z=z))
        for index,(size,points) in enumerate(groups.items()):
            marker=Marker();marker.header=header;marker.ns='LAS2_OCTOMAP';marker.id=index
            marker.type=Marker.CUBE_LIST;marker.action=Marker.ADD;marker.pose.orientation.w=1.
            marker.scale.x=marker.scale.y=marker.scale.z=size
            marker.color.r,marker.color.g,marker.color.b,marker.color.a=.2,.65,1.,.55
            marker.points=points;array.markers.append(marker)
        self.map_pub.publish(array)

    def initialization_target(self):
        reached=(self.token and self.navigation.get('generation')==self.generation and
                 self.navigation.get('waypoint_index')==self.initialization.index and
                 self.navigation.get('reached'))
        target=self.initialization.update(self.get_clock().now().nanoseconds*1e-9,
                                         pose_values(self.state.pose.pose),bool(reached))
        self.reason=('source_initialization_wait' if target is None else
                     'source_initialization_translation' if self.initialization.index==11 else
                     'source_initialization_scan')
        if self.initialization.done:
            self.initialized=True;self.target=None;self.validated_revision=-1
            self.hold_target=pose_values(self.state.pose.pose);self.hold_index=12
            self.reason='source_initialization_complete'
        return target

    def publish_goal(self,target,now,waypoint_index=None):
        self.goal_sequence+=1
        goal=NavigationGoal(run_id=self.out.name,mission_id=self.navigation_mission_id(),token=self.token,
            generation=self.generation,epoch=1,sequence=self.goal_sequence,
            issued_steady_ns=now,deadline_steady_ns=now+200_000_000,
            waypoint_index=self.waypoint_index if waypoint_index is None else waypoint_index)
        goal.target.header=self.state.header;fill(goal.target.pose,target)
        self.goal_pub.publish(goal)
        self.last_goal_deadline_ns=goal.deadline_steady_ns

    def publish_neutral_goal(self,now):
        # Reserved project index0 means zero velocity in estimated_navigation.
        # It cannot satisfy any real path's arrival index (real indices>=12).
        self.publish_goal(pose_values(self.state.pose.pose),now,0)

    def tick(self):
        now=time.monotonic_ns()
        self.poll(now)
        backend=self.out/'nbv-backend-status.json'
        if self.backend_state is None and backend.exists():
            try:self.backend_state=json.loads(backend.read_text())
            except (ValueError,OSError):pass
        if self.backend_state and self.enabled:
            self.disable('upstream_planner_finished' if self.backend_state.get('state')=='finished'
                         else 'nbv_backend_fault: '+self.backend_state.get('reason','see logs/nbv_native.log'))
        if self.authority_future is not None and self.authority_future.done():
            action=self.authority_action
            try:response=self.authority_future.result()
            except Exception as error:response=None;self.reason=str(error)
            self.authority_future=None
            arm_target_current=(self.startup_profile!='obstacle_triggered' or
                self.obstacle_phase=='braking' or
                (self.obstacle_phase in ('turning','setting_cruise_direction') and self.turn_path and
                 (self.turn_started_wall or self.validated_revision==self.map_revision)) or
                (self.target is not None and self.validated_revision==self.map_revision))
            if action=='ARM' and response and response.accepted and self.enabled and arm_target_current:
                self.token,self.generation=response.token,response.generation
            elif action=='ARM' and response and response.accepted:
                self.request_authority('DISARM')
            elif action=='ARM':self.reason='arm_wait: '+(response.reason if response else 'service_error')
            elif action=='CLEAR_FAULT':
                self.reason='guard_reset: '+(response.reason if response else 'service_error')
                self.event('guard_reset',accepted=bool(response and response.accepted),reason=self.reason)
            elif action=='SELECT_SOURCE':
                accepted=bool(response and response.accepted)
                if accepted:
                    self.control_mode=self.pending_control_mode
                    self.reason='manual_control_selected' if self.control_mode=='manual' else 'autonomy_control_selected'
                else:self.reason='control_mode_switch_failed: '+(response.reason if response else 'service_error')
                self.pending_control_mode=None
                self.event('control_mode_changed',accepted=accepted,control_mode=self.control_mode,reason=self.reason)
        if self.pending_control_mode and self.authority_future is None:
            if self.guard.get('state')=='ARMED':self.request_authority('DISARM')
            elif self.guard.get('state') in ('DISARMED','FAULT'):
                self.request_authority('SELECT_SOURCE','cli' if self.pending_control_mode=='manual' else 'navigation')
        healthy=self.healthy(now)
        authority_lost=(self.guard.get('generation')==self.generation and
                        self.guard.get('state')!='ARMED')
        if ((self.enabled and self.guard.get('state')=='FAULT') or
                (self.token and (not self.enabled or not self.state_healthy(now) or authority_lost))):
            self.disable('guard: '+str(self.guard.get('reason')) if self.guard.get('state')=='FAULT'
                         else 'motion_stopped_state_or_guard_unavailable')
        if (self.token and self.enabled and self.startup_profile=='obstacle_triggered' and
                self.obstacle_phase!='braking' and
                not (self.obstacle_phase=='turning' and self.turn_started_wall) and not healthy):
            # Keep an explicit zero-velocity intent while map progress is
            # unavailable. Never renew the old spatial goal; all map freshness,
            # state/Guard and original intent/output expiry checks still apply.
            if not getattr(self,'map_waiting',False):
                self.event('map_progress_unavailable',map_revision=self.map_revision,
                    validated_map_revision=self.validated_revision,obstacle_phase=self.obstacle_phase,
                    map_status=getattr(self,'map_source_status',{}),
                    receive_age_seconds=(now-self.map_source_wall)*1.e-9 if hasattr(self,'map_source_wall') else None,
                    progress_age_seconds=(now-self.map_source_revision_wall)*1.e-9
                        if hasattr(self,'map_source_revision_wall') else None)
            self.map_waiting=True
            if self.state_healthy(now):self.publish_neutral_goal(now)
            self.reason='waiting_for_fresh_validated_map'
        if self.enabled and healthy and getattr(self,'map_waiting',False):
            self.map_waiting=False
            self.event('map_progress_restored',map_revision=self.map_revision,
                validated_map_revision=self.validated_revision)
        if self.enabled and self.obstacle_phase=='braking':
            # Zero intent needs fresh state/authority, not free map coverage.
            # Map freshness is required again before any RRT/translation goal.
            if self.state_healthy(now):
                self.reason='obstacle_3m_braking_before_planning'
                if self.token:
                    self.publish_neutral_goal(now)
                    navigation=self.navigation
                    if (navigation.get('run')==self.out.name and navigation.get('mission_id')==self.navigation_mission_id() and
                            navigation.get('state')=='TRACKING' and navigation.get('reached') and
                            navigation.get('generation')==self.generation and
                            navigation.get('waypoint_index')==0 and
                            navigation.get('wall_ns',0)>=self.obstacle_braking_started_wall and
                            0<=now-navigation.get('wall_ns',0)<250_000_000):
                        self.obstacle_phase='planning';self.initialized=True
                        self.hold_target=pose_values(self.state.pose.pose)
                        self.reason='obstacle_stopped_local_planning';self.last_plan_wall=0
                        self.event('obstacle_braking_complete',body=self.hold_target,
                            navigation=navigation,map_revision=self.map_revision)
                elif (self.guard.get('state')=='DISARMED' and self.controller.get('ready_for_arm') and
                        now-self.controller_wall<250_000_000):
                    self.request_authority('ARM')
            self.publish_status(now,healthy);return
        if self.enabled and self.obstacle_phase in ('turning','setting_cruise_direction'):
            self.tick_avoidance_turn(now,healthy)
            self.publish_status(now,healthy);return
        init_target=None
        if (self.enabled and not self.initialized and self.startup_profile=='source_scan' and healthy and
            (self.token or self.controller.get('ready_for_arm'))):
            init_target=self.initialization_target()
            if init_target is not None and not self.token and self.guard.get('state')=='DISARMED':
                self.request_authority('ARM')
            if init_target is not None and self.token:self.publish_goal(init_target,now,self.initialization.index)
        # Advance only after the tracker has settled at the endpoint of the
        # original planner's best edge. Replan with newly observed free space.
        if ((self.initialized or self.forward_start_active) and self.token and self.target and self.navigation.get('reached') and
                self.validated_revision==self.map_revision and
                self.navigation.get('generation')==self.generation and
                self.navigation.get('waypoint_index')==self.waypoint_index):
            if self.mode=='explore' and self.initialized and not self.forward_start_active:
                self.event('nbv_edge_reached',waypoint_index=self.waypoint_index,
                    distance_m=self.navigation.get('distance_m'),yaw_error_rad=self.navigation.get('yaw_error_rad'),
                    body=pose_values(self.state.pose.pose),target=self.target)
            self.hold_target=self.target[:];self.hold_index=self.waypoint_index
            self.target=None;self.path=[];self.validated_revision=-1
            self.reason='edge_complete_replanning'
            if self.forward_start_active:
                self.forward_start_active=False
                if self.startup_profile=='forward_until_observed':
                    if self.startup_limit_segment:self.disable('startup_5m_limit_reached_surroundings_unknown')
                    else:
                        self.forward_start_required=True;self.reason='startup_continue_forward_unknown'
                        self.event('startup_forward_segment_reached',body=pose_values(self.state.pose.pose))
                else:
                    self.initialized=True;self.reason='forward_start_complete_replanning'
                    self.event('forward_start_reached',initialization_method=self.startup_profile)
            if self.mode=='waypoints':
                self.event('waypoint_reached',index=self.route_index)
                self.route_index+=1
                self.route_complete=self.route_index>=len(self.route)
        if (self.enabled and self.initialized and self.mode=='waypoints' and
                not self.route_complete and self.target is None):
            self.target=self.route[self.route_index][:];self.waypoint_index+=1
            self.path=[pose_values(self.state.pose.pose),self.target]
            self.validated_revision=-1;self.publish_plan()
        if (self.future is None and self.state is not None and self.localization.get('epoch')==1 and
                (self.out/'nbv.sock').exists()):
            if self.preview_requested:
                operation=self.preview_requested;self.preview_requested=None
                self.submit(operation,preview=True)
            elif self.enabled and self.reset_execution:
                self.submit('reset_execution')
            elif self.enabled and self.cruise_active and healthy:
                self.submit('plan_forward_cruise')
            elif self.enabled and self.forward_start_required and healthy:
                self.submit('plan_forward_start');self.last_plan_wall=now
            elif self.target is not None and healthy and self.validated_revision!=self.map_revision:
                self.submit('validate_forward_start' if self.forward_start_active else 'validate',path=[self.target])
            elif (self.pending is not None and
                  (ns(self.pending.header.stamp)-self.last_observe_stamp)*1e-9>=self.params['pcl_throttle']):
                message=self.pending
                body=interpolate_pose(self.history,ns(message.header.stamp))
                if body is not None:
                    self.pending=None;self.last_observe_stamp=ns(message.header.stamp)
                    depth=self.bridge.imgmsg_to_cv2(message,desired_encoding='passthrough')
                    points=project_depth(depth,self.k)
                    if len(points)>0:
                        camera=compose((body[:3],body[3:]),self.body_camera)
                        self.submit('observe',stamp_ns=ns(message.header.stamp),body=body,
                                    camera=[*camera[0],*camera[1]],
                                    points_f32_le=base64.b64encode(points.tobytes()).decode('ascii'))
                elif self.history and ns(message.header.stamp)<self.history[0][0]:self.pending=None
            elif (self.enabled and self.mode=='explore' and self.initialized and healthy and self.target is None and
                  now-self.last_plan_wall>1_000_000_000 and
                  (not self.token or self.hold_target is not None)):
                reactive=self.startup_profile=='obstacle_triggered'
                source=getattr(self,'map_source_status',{}).get('source_revision',self.map_revision)
                if not reactive or source>getattr(self,'turn_retry_source_revision',-1):
                    self.submit('plan_turn' if reactive else 'plan');self.last_plan_wall=now
        if (self.enabled and (self.initialized or self.forward_start_active or self.cruise_active) and healthy and self.target and self.validated_revision==self.map_revision and
                not self.token and self.guard.get('state')=='DISARMED' and
                self.controller.get('ready_for_arm') and now-self.controller_wall<250_000_000):
            self.request_authority('ARM')
        startup_holding=(self.startup_profile=='forward_until_observed' and self.hold_target is not None)
        if self.token and self.enabled and (self.initialized or self.forward_start_active or self.cruise_active or startup_holding) and healthy:
            if self.target is not None:
                if self.validated_revision==self.map_revision:
                    self.publish_goal(self.target,now)
                    self.reason=('cruising_until_obstacle_3m' if self.cruise_active else
                        'startup_forward_until_observed' if self.forward_start_active and
                        self.startup_profile=='forward_until_observed' else 'tracking_nbv_edge')
                    if (self.navigation.get('generation')==self.generation and
                            self.navigation.get('source_waypoint_index')==self.waypoint_index):
                        phase=self.navigation.get('execution_phase')
                        if phase=='align':self.reason='aligning_path_heading'
                        elif phase=='observe':self.reason='observing_nbv_viewpoint'
                    if self.startup_profile=='obstacle_triggered' and not self.cruise_active:self.obstacle_phase='tracking'
                else:
                    if (self.startup_profile=='obstacle_triggered' and
                            self.last_goal_deadline_ns-now<=self.neutral_renewal_margin_ns):
                        self.publish_neutral_goal(now)
                    self.reason='waiting_for_current_map_validation'
            elif self.hold_target is not None:
                if self.startup_profile=='obstacle_triggered':self.publish_neutral_goal(now)
                else:self.publish_goal(self.hold_target,now,self.hold_index)
                self.reason='route_complete_holding' if self.route_complete and self.mode=='waypoints' else 'planning_holding_pose'
        self.publish_status(now,healthy)

    def publish_status(self,now,healthy):
        report=dict(run=self.out.name,backend='ethz-asl/nbvplanner',source='OKVIS2 + LAS2-S',
                    map_source=self.map_source,native_depth_fusion=self.map_source=='native_depth',
                    map_source_status=self.map_source_status,
                    map_waiting=self.map_waiting,
                    map_acquisition_age_seconds=(ns(self.state.header.stamp)-int(self.map_source_status.get('source_revision',0)))*1e-9
                        if self.state is not None and self.map_source_status.get('source_revision') else None,
                    map_freshness_budget_seconds=self.map_freshness_budget_ns*1.e-9,
                    enabled=self.enabled,healthy=bool(healthy),reason=self.reason,map_revision=self.map_revision,
                    path_points=len(self.path),guard=self.guard.get('state','WAITING'),
                    jepa_ready=bool(self.token or self.controller.get('ready_for_arm',False)),
                    initialized=self.initialized,validated=self.validated_revision==self.map_revision,
                    validated_map_revision=self.validated_revision,wall_ns=now)
        report['initialization_waypoint']=self.initialization.index if not self.initialized else None
        report['backend_status']=self.backend_state
        report.update(mode=self.mode,route_index=self.route_index,route_total=len(self.route),
            startup_profile=self.startup_profile,forward_start_active=self.forward_start_active,
            startup_map_status=self.startup_map_status,
            cruise_active=self.cruise_active,obstacle_phase=self.obstacle_phase,
            obstacle_switch_distance_m=self.obstacle_switch_distance_m,
            cruise_requests=self.cruise_requests,cruise_map_status=self.cruise_map_status,
            turn_yaw_rad=self.turn_yaw,
            forward_start_required=self.forward_start_required,
            planning_only=self.planning_only,
            preview_busy=bool(self.preview_requested or (self.job and self.job.get('preview'))),
            planning_preview={key:self.preview_result[key] for key in ('valid','reason','map_revision','diagnostics')
                              if key in self.preview_result} if self.preview_result else None,
            control_mode=self.control_mode,control_mode_pending=self.pending_control_mode,
            joystick_connected=bool(now-self.joystick_wall<250_000_000 and self.joystick.get('connected')),
            joystick_reason=self.joystick.get('reason','not_connected'),
            mission_protocol=1,
            route_complete=self.route_complete,plans_requested=self.plans_requested,
            paths_generated=self.paths_generated,last_plan_seconds=self.last_plan_seconds,
            target=self.target,body_pose=pose_values(self.state.pose.pose) if self.state else None,
            guard_reason=self.guard.get('reason'),guard_fault=self.guard.get('fault_context'))
        report['guard_health']=self.guard.get('health')
        report['jepa']={key:self.controller[key] for key in (
            'backend','model','checkpoint_sha256','device','sample_count','history_samples',
            'model_rate_hz','reason','timing_seconds','solver_reason','motor_polarity',
            'roll_source','gpu_warmed','warmup_error') if key in self.controller}
        report['navigation']={key:self.navigation[key] for key in (
            'state','generation','waypoint_index','source_waypoint_index','execution_phase',
            'phase_reached','segment_yaw_rad','distance_m','yaw_error_rad','target_velocity','reached') if key in self.navigation}
        self.status_pub.publish(String(data=json.dumps(report)))
        if self.state is not None:
            marker=Marker();marker.header=self.state.header;marker.ns='nbv_status'
            marker.type=Marker.TEXT_VIEW_FACING;marker.action=Marker.ADD
            fill(marker.pose,pose_values(self.state.pose.pose));marker.pose.position.z+=.8
            marker.scale.z=.09;marker.color.r,marker.color.g,marker.color.b,marker.color.a=.09,.2,.36,1.
            marker.text=f'NBV map {self.map_revision} | {self.reason}'
            self.text_pub.publish(marker)
            goal=Marker();goal.header=self.state.header;goal.ns='nbv_goal'
            visible_target=self.target
            if self.obstacle_phase in ('turning','setting_cruise_direction') and self.turn_yaw is not None:
                visible_target=pose_values(self.state.pose.pose)
                visible_target[3:]=[0.,0.,math.sin(self.turn_yaw/2.),math.cos(self.turn_yaw/2.)]
            goal.type=Marker.ARROW;goal.action=Marker.ADD if visible_target else Marker.DELETE
            if visible_target:fill(goal.pose,visible_target)
            goal.scale.x,goal.scale.y,goal.scale.z=.35,.06,.06
            goal.color.r,goal.color.g,goal.color.b,goal.color.a=1.,.8,0.,1.
            self.goal_marker.publish(goal)
        temporary=self.out/'planner-status.json.tmp';temporary.write_text(json.dumps(report))
        temporary.replace(self.out/'planner-status.json')

    def close(self):
        self.disable('shutdown')
        self.stop_event.set()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.clock_socket.close()


def main():
    rclpy.init();node=NbvBridge()
    try:rclpy.spin(node)
    finally:node.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
