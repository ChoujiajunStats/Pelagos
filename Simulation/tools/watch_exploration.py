#!/usr/bin/env python3
"""Observe a finite live exploration trial through existing status files."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=90.)
    parser.add_argument('--leave-running',action='store_true',help='End observation without pausing the operator-authorized task')
    args=parser.parse_args();out=args.run_dir
    initial=time.monotonic();last_status=initial;active=None;record=dict(run=out.name,samples=[],reason='waiting_for_start')
    target=out/'startup-exploration-trial.json';last_print=0.;g={}
    def read(name):return json.loads((out/name).read_text())
    while True:
        now=time.monotonic()
        try:p=read('planner-status.json');g=read('control-status.json');m=read('mapping-status.json')
        except (OSError,ValueError):
            if now-last_status>5:record['reason']='status_unavailable';break
            time.sleep(.1);continue
        last_status=now
        if p['enabled'] and active is None:active=now;record['started_wall_ns']=time.monotonic_ns()
        if active is None:
            if now-initial>30:record['reason']='task_not_started';break
            time.sleep(.1);continue
        record['samples'].append(dict(wall_ns=time.monotonic_ns(),planner=p,guard=g,
            mapping={k:m.get(k) for k in ['integration_hz','integrated_frames','octomap_projection_ms','octomap_snapshot_ms']}))
        if now-last_print>=10:
            print(json.dumps(dict(seconds=now-active,reason=p['reason'],guard=g['state'],
                points=p['path_points'],body=p.get('body_pose'),
                obstacle_phase=p.get('obstacle_phase'),
                front_clearance_m=p.get('cruise_map_status',{}).get('nearest_obstacle_front_clearance_m'),
                cruise_progress_m=p.get('cruise_map_status',{}).get('cruise_progress_m'),
                navigation=p.get('navigation'),jepa=p.get('jepa'))),flush=True)
            last_print=now;target.write_text(json.dumps(record,indent=2))
        if g['state']=='FAULT':record['reason']='guard_fault:'+g['reason'];break
        if p['control_mode']!='autonomy':record['reason']='operator_takeover';break
        if not p['enabled']:record['reason']='task_stopped:'+p['reason'];break
        if now-active>=args.seconds:record['reason']='finite_trial_complete';break
        time.sleep(.2)
    # Preserve an operator's new manual ownership. Otherwise end this finite
    # trial using the same task pause/Guard DISARM operation as the UI.
    try:
        p=read('planner-status.json')
        if args.leave_running and p['enabled']:
            record['task_left_running']=True
        elif p['enabled'] and p['control_mode']=='autonomy':
            result=subprocess.run([str(Path(__file__).resolve().parents[1]/'sim'),'pause'],
                capture_output=True,text=True,timeout=8)
            record['pause']=dict(code=result.returncode,output=result.stdout)
        # Task disable and Guard DISARM are asynchronous. Observe the actual
        # acknowledgement before labelling an early status as the final state.
        deadline=time.monotonic()+8.
        record['final_guard']=g
        while True:
            try:
                record['final_guard']=read('control-status.json')
                p=read('planner-status.json')
            except (OSError,ValueError):
                if time.monotonic()>=deadline:
                    record['stop_confirmation_timeout']=True;break
                time.sleep(.05);continue
            if record.get('task_left_running') or p['control_mode']!='autonomy' or record['final_guard']['state'] in ('DISARMED','FAULT'):break
            if time.monotonic()>=deadline:
                record['stop_confirmation_timeout']=True;break
            time.sleep(.05)
    finally:
        record['finished_wall_ns']=time.monotonic_ns();target.write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(reason=record['reason'],samples=len(record['samples']),evidence=str(target))),flush=True)


if __name__=='__main__':main()
