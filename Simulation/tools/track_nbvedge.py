#!/usr/bin/env python3
"""Finite test of a saved native NBV edge through the existing navigation/JEPA chain.

Uses Guard leases and NavigationGoal; never publishes velocity or actuator output.
Source contracts: Stonefish navigation/uw_navigation/node.py, interfaces/msg/
NavigationGoal.msg, guard/uw_guard/node.py; hashes are recorded with the result.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from nbv_protocol import exchange
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from uw_interfaces.msg import NavigationGoal
from uw_interfaces.srv import Control


def values(pose):
    p,q=pose.position,pose.orientation
    return [p.x,p.y,p.z,q.x,q.y,q.z,q.w]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--seconds',type=float,default=20.)
    args=parser.parse_args();out=args.run_dir;plan=json.loads(args.plan.read_text())
    if not 0<args.seconds<=30:raise ValueError('Finite test must be within 30 seconds')
    if plan.get('run_id')!=out.name or plan.get('epoch')!=1 or plan.get('frame_id')!='rov01/odom' or not plan.get('valid'):
        raise ValueError('Plan is not a valid native result from this estimator session')
    path=plan.get('path',[])
    if not path or plan.get('diagnostics',{}).get('path_kind')=='forward_start_not_rrt':
        raise ValueError('This test requires a native NBV edge, not a bootstrap segment')
    length=sum(math.dist(a[:3],b[:3]) for a,b in zip(path,path[1:]))
    if abs(length-math.dist(path[0][:3],path[-1][:3]))>1.e-5:
        raise ValueError('Endpoint tracker requires the source straight best edge')
    rclpy.init();node=Node('pelagos_finite_nbvedge_test')
    latest={};received={};operator_stop=[False]
    def string(name,message):
        data=json.loads(message.data)
        if data.get('run')==out.name:latest[name]=data;received[name]=time.monotonic_ns()
    for name,topic in (('planner','/pelagos/planner/status'),('guard','/rov01/control/status'),
        ('controller','/rov01/control/controller_status'),('localization','/rov01/localization/status'),
        ('navigation','/rov01/navigation/status'),('terminal','/rov01/sim/terminal/status')):
        node.create_subscription(String,topic,lambda m,k=name:string(k,m),1)
    def state(message):
        if message.header.frame_id=='rov01/odom' and message.child_frame_id=='rov01/base_link':
            latest['state']=message;received['state']=time.monotonic_ns()
    node.create_subscription(Odometry,'/rov01/state/odometry',state,1)
    def operator(message):
        data=json.loads(message.data)
        if data.get('run_id')==out.name:operator_stop[0]=True
    node.create_subscription(String,'/pelagos/mission/request',operator,10)
    publisher=node.create_publisher(NavigationGoal,'/rov01/navigation/goal',1)
    client=node.create_client(Control,'/rov01/control/authority')
    pool=ThreadPoolExecutor(max_workers=1);stop=threading.Event()
    record=dict(run_id=out.name,plan_file=str(args.plan),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        controller='jepa_mppi',target=path[-1],source='actual OKVIS2 body + native measured OctoMap',
        execute_requested=args.execute,events=[],samples=[])
    contract_root=Path('/stonefish-workspace')
    record['source_contract_sha256']={name:hashlib.sha256((contract_root/name).read_bytes()).hexdigest()
        for name in ('navigation/uw_navigation/node.py','guard/uw_guard/node.py',
                     'interfaces/msg/NavigationGoal.msg','controller/uw_controller/node.py')}
    result_path=out/'nbv-edge-tracking-test.json'
    token='';generation=0;future=None;reason='not_started';started=None;mission='nbv-edge-test-'+uuid.uuid4().hex
    def event(kind,**data):record['events'].append(dict(wall_ns=time.monotonic_ns(),kind=kind,**data))
    def fresh(name,now):return name in received and now-received[name]<250_000_000
    def environment(now):
        return (all(fresh(k,now) for k in ('state','planner','guard','localization','controller')) and
            latest['planner'].get('control_mode')=='autonomy' and not latest['planner'].get('enabled') and
            not latest['planner'].get('control_mode_pending') and
            latest['localization'].get('state')=='TRACKING' and latest['localization'].get('epoch')==1)
    def validate():
        s=latest['state'];request=dict(run=out.name,epoch=1,sequence=time.monotonic_ns(),stamp_ns=0,
            operation='validate',body=values(s.pose.pose),path=[path[-1]])
        return exchange(out/'nbv.sock',request,stop)
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02);now=time.monotonic_ns()
            if operator_stop[0]:reason='operator_changed_task';return
            if (environment(now) and latest['guard'].get('state')=='DISARMED' and
                    latest['guard'].get('ready') and latest['controller'].get('ready_for_arm') and
                    latest['controller'].get('backend')=='jepa_mppi' and publisher.get_subscription_count() and
                    client.service_is_ready()):break
        else:
            reason='readiness_not_met';record['readiness']={k:v for k,v in latest.items() if k!='state'};return
        record['initial_body']=values(latest['state'].pose.pose)
        future=pool.submit(validate)
        while not future.done():rclpy.spin_once(node,timeout_sec=.02)
        validation=future.result();future=None;record['initial_validation']=validation
        if not validation.get('valid'):reason='path_blocked_or_unknown';return
        if not args.execute:reason='checked_without_arming';return
        if not environment(time.monotonic_ns()):reason='state_changed_before_arm';return
        arm=client.call_async(Control.Request(run_id=out.name,action='ARM',source='navigation'))
        rclpy.spin_until_future_complete(node,arm,timeout_sec=2.)
        if not arm.done() or not arm.result().accepted:
            reason='arm_rejected';event(reason,detail=arm.result().reason if arm.done() else 'service_timeout');return
        token=arm.result().token;generation=arm.result().generation;event('armed',generation=generation)
        started=time.monotonic();record['started_wall_ns']=time.monotonic_ns()
        checked_revision=validation['map_revision'];sequence=0;last_sent=0.;last_sample=0.
        while time.monotonic()-started<args.seconds:
            rclpy.spin_once(node,timeout_sec=.01);now=time.monotonic_ns();wall=time.monotonic()
            g=latest.get('guard',{});p=latest.get('planner',{})
            if operator_stop[0]:reason='operator_changed_task';break
            if not environment(now):reason='state_or_mode_unavailable';break
            if g.get('state')=='FAULT':reason='guard_fault:'+str(g.get('reason'));break
            # The ARM answer can precede the next Guard status heartbeat.
            if g.get('generation',0)>=generation and (g.get('state')!='ARMED' or g.get('source')!='navigation'):
                reason='authority_changed';break
            if future is not None and future.done():
                answer=future.result();future=None
                if not answer.get('valid'):reason='path_blocked_or_unknown';break
                checked_revision=answer['map_revision']
            revision=p.get('map_revision',0)
            if future is None and revision>checked_revision:future=pool.submit(validate)
            if future is None and wall-last_sent>=.05:
                sequence+=1;m=NavigationGoal(run_id=out.name,mission_id=mission,token=token,
                    generation=generation,epoch=1,sequence=sequence,issued_steady_ns=now,
                    deadline_steady_ns=now+200_000_000,waypoint_index=1)
                m.target.header=latest['state'].header;t=path[-1]
                m.target.pose.position.x,m.target.pose.position.y,m.target.pose.position.z=t[:3]
                (m.target.pose.orientation.x,m.target.pose.orientation.y,
                 m.target.pose.orientation.z,m.target.pose.orientation.w)=t[3:]
                publisher.publish(m);last_sent=wall
            if wall-last_sample>=.05:
                pose=values(latest['state'].pose.pose)
                record['samples'].append(dict(wall_ns=now,body=pose,distance_m=math.dist(pose[:3],path[-1][:3]),
                    navigation=latest.get('navigation',{}),controller=latest.get('controller',{}),
                    terminal={k:latest.get('terminal',{}).get(k) for k in ('state','reason','applied_setpoint','request_seq')},
                    checked_map_revision=checked_revision))
                last_sample=wall
            nav=latest.get('navigation',{})
            if nav.get('mission_id')==mission and nav.get('generation')==generation and nav.get('reached'):
                reason='reached';break
        else:reason='finite_test_timeout'
    finally:
        # Never disarm a new manual owner after an operator takeover.
        g=latest.get('guard',{})
        if token and g.get('source')=='navigation' and g.get('generation')==generation:
            disarm=client.call_async(Control.Request(run_id=out.name,action='DISARM',source='navigation'))
            rclpy.spin_until_future_complete(node,disarm,timeout_sec=2.)
            event('disarm_requested',accepted=bool(disarm.done() and disarm.result().accepted))
            deadline=time.monotonic()+1.
            while time.monotonic()<deadline:
                rclpy.spin_once(node,timeout_sec=.02)
                if (latest.get('guard',{}).get('state')=='DISARMED' and
                        latest.get('terminal',{}).get('state')=='DISARMED'):break
        stop.set();pool.shutdown(wait=True,cancel_futures=True)
        record['reason']=reason;record['finished_wall_ns']=time.monotonic_ns()
        if 'state' in latest:record['final_body']=values(latest['state'].pose.pose)
        record['final_guard']={k:latest.get('guard',{}).get(k) for k in ('state','source','reason','generation')}
        result_path.write_text(json.dumps(record,indent=2));print(json.dumps(dict(reason=reason,
            samples=len(record['samples']),evidence=str(result_path)),ensure_ascii=False),flush=True)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
