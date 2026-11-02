#!/usr/bin/env python3
"""Short, operator-authorized HMU motion acceptance; never a scan controller.

Default is preflight only. Execution needs --execute, explicit clearance and
the exact dedicated Cockpit PID. Uses existing channel bridge/SafetyGate.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--cockpit-pid',type=int)
    parser.add_argument('--confirmed-clearance-m',type=float,default=0.)
    parser.add_argument('--max-displacement-m',type=float,default=.10)
    parser.add_argument('--amplitude',type=float,default=.15);parser.add_argument('--pulse-seconds',type=float,default=2.)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not .02<=args.max_displacement_m<=.12 or not 0<args.amplitude<=.2 or not 0<args.pulse_seconds<=2.:
        parser.error('acceptance displacement/amplitude/time bounds exceeded')
    if args.execute and (args.confirmed_clearance_m<.3 or not args.cockpit_pid):parser.error('execution requires confirmed .3 m clearance and dedicated Cockpit PID')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result={'started_at':datetime.datetime.now().astimezone().isoformat(),'execution_requested':args.execute,
        'controller':'bounded manual-channel acceptance; not JEPA autonomous control','samples':[],'legs':[]}
    def api(payload=None):
        request=urllib.request.Request('http://127.0.0.1:8765/'+('api/control' if payload is not None else 'health'),
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(request,timeout=.7))
    state=api();result['initial_control']=state['control'];result['initial_telemetry']=state['main']
    if state['rov']['command_protocol']!='channel' or not state['rov']['connected']:raise RuntimeError('manufacturer channel bridge not connected')
    if state['control']['armed']:raise RuntimeError('preflight requires existing operator DISARM')
    stream=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','nvidia@192.168.1.202',
        'source /opt/ros/humble/setup.bash && python3 /home/nvidia/Dapeng_Deployment/Deployment/tests/acceptance/probes/control_pose_monitor.py'],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    latest={};lock=threading.Lock()
    def reader():
        for line in stream.stdout:
            try:
                value=json.loads(line)
                if value:
                    with lock:latest.update(value,received=time.monotonic())
            except (ValueError,TypeError):continue
    threading.Thread(target=reader,daemon=True).start()
    def pose():
        with lock:p=dict(latest)
        if not p or time.monotonic()-p['received']>.3 or not p.get('finite') or not -.1<=p['pose_age_s']<.25 or p['visual_age_s']>1.5:
            raise RuntimeError('fresh finite propagated and visual-optimized VIO required')
        return p
    token=None;seq=0;cockpit=None;closed=False;record_start=time.monotonic();anchor=None
    def command(sway=0.):
        nonlocal seq
        payload={'type':'control','token':token,'sequence':seq,'axes':{'surge':0.,'sway':sway,'heave':0.,'yaw':0.},
                 'gear':1,'heading_hold':False,'depth_hold':False}
        seq+=1;return api(payload)
    def checked_sample():
        p=pose();h=api();m=h['main']
        if not h['rov']['connected'] or not m or m.get('leak_detected'):raise RuntimeError('vehicle connection/leak guard')
        if (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(h['last_frame_at'])).total_seconds()>1.:
            raise RuntimeError('vehicle telemetry stale')
        if anchor is not None:
            if math.dist(p['position'],anchor['position'])>args.max_displacement_m:raise RuntimeError('displacement bound reached')
            if abs(m['depth_m']-result['depth_anchor_m'])>.04:raise RuntimeError('pressure depth deviation')
            heading=abs((m['heading_deg']-result['heading_anchor_deg']+180)%360-180)
            if heading>5:raise RuntimeError('heading deviation')
            if math.hypot(*p['velocity'])>.2:raise RuntimeError('speed bound reached')
        sample={'elapsed_s':time.monotonic()-record_start,'pose':{k:v for k,v in p.items() if k!='received'},
                'depth_m':m['depth_m'],'heading_deg':m['heading_deg'],'armed':h['control']['armed']}
        result['samples'].append(sample);return p
    def neutral_for(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:command();checked_sample();time.sleep(.05)
    try:
        deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            try:pose();break
            except RuntimeError:time.sleep(.1)
        anchor=pose();result['pose_anchor']=anchor;result['depth_anchor_m']=state['main']['depth_m'];result['heading_anchor_deg']=state['main']['heading_deg']
        checked_sample()
        if not args.execute:
            result['passed']=True;result['status']='preflight_passed_no_actuation';return
        p=Path('/proc')/str(args.cockpit_pid)
        argv=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        if not argv or Path(argv[0]).name!='hmu-cockpit':raise RuntimeError('PID is not the dedicated Cockpit application')
        env=dict(x.split('=',1) for x in p.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
        cockpit={'argv':argv,'environment':{k:env[k] for k in ['DISPLAY','XAUTHORITY','WAYLAND_DISPLAY','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'] if k in env}}
        os.kill(args.cockpit_pid,signal.SIGTERM);closed=True
        deadline=time.monotonic()+5
        while api()['control']['has_control'] and time.monotonic()<deadline:time.sleep(.1)
        if api()['control']['has_control']:raise RuntimeError('Cockpit did not release ownership; no takeover attempted')
        token=api({'type':'takeover'})['token'];command();api({'type':'arm_begin','token':token})
        neutral_for(1.7);api({'type':'arm_commit','token':token});result['arm_verified']=api()['control']['armed']
        if not result['arm_verified']:raise RuntimeError('ARM not acknowledged')
        neutral_for(.5)
        for sign in [1.,-1.]:
            start=pose();began=time.monotonic();sent=0
            while time.monotonic()-began<args.pulse_seconds:
                p=checked_sample()
                if math.dist(start['position'],p['position'])>=.04:break
                command(sign*args.amplitude);sent+=1;time.sleep(.05)
            command();neutral_for(1.5);end=pose()
            result['legs'].append({'sway':sign*args.amplitude,'duration_s':time.monotonic()-began,'sent_commands':sent,
                'estimated_displacement_m':math.dist(start['position'],end['position']),'start':start['position'],'end':end['position']})
        result['passed']=True;result['status']='bounded_channel_motion_check_completed'
    except Exception as e:
        result['passed']=False;result['status']='aborted';result['error']=str(e)
    finally:
        if token:
            for action in ['neutral','disarm','release']:
                try:
                    if action=='neutral':command()
                    else:api({'type':action,'token':token})
                except Exception as e:result.setdefault('cleanup_errors',[]).append(str(e))
        if closed and cockpit:
            restore=['systemd-run','--user','--collect','--unit=hmu-cockpit-after-visualization-check']
            restore.extend('--setenv='+k+'='+v for k,v in cockpit['environment'].items());restore.extend(cockpit['argv'])
            restored=subprocess.run(restore,capture_output=True,text=True);result['cockpit_restore_returncode']=restored.returncode
            result['cockpit_restore_message']=restored.stdout+restored.stderr
        stream.terminate()
        try:stream.wait(timeout=3)
        except subprocess.TimeoutExpired:stream.kill()
        try:result['final_control']=api()['control']
        except Exception as e:result['final_control_error']=str(e)
        result['ended_at']=datetime.datetime.now().astimezone().isoformat()
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:result.get(k) for k in ['passed','status','error','arm_verified','legs','final_control','cockpit_restore_returncode']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
