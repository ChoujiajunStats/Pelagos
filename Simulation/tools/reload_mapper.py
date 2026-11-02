#!/usr/bin/env python3
"""Reload only mapping processes, preserving the live simulator/VIO/Guard.

The original supervisor predates hot reload. Keep its two replaced children
stopped, so it does not tear down the live estimator on their exit. This scoped
supervisor owns the replacements until the original container is stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir
    planner=json.loads((out/'planner-status.json').read_text());guard=json.loads((out/'control-status.json').read_text())
    if planner['enabled'] or guard['state']!='DISARMED' or time.monotonic_ns()-guard['wall_ns']>250_000_000:
        raise RuntimeError('Mapper reload requires a paused task and fresh DISARMED Guard')
    manifest_path=out/'mapper-reload.json'
    if manifest_path.exists():raise RuntimeError('Mapper reload already exists; do not create duplicate fusion processes')
    originals={}
    targets={'wavemap':'/opt/pelagos/mapping/underwater_wavemap_ros/lib/underwater_wavemap_ros/wavemap_node',
        'relay':'/pelagos/Simulation/adapters/wavemap_nbv_relay.py'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:argv=(entry/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        for name,executable in targets.items():
            if executable.encode() in argv:
                if name in originals:raise RuntimeError('Ambiguous original '+name)
                originals[name]=int(entry.name)
    if originals.keys()!=targets.keys():raise RuntimeError('Both original managed processes must be present')
    snapshot=out/'same-session-reload.wvmp'
    shutil.copy2(out/'map.wvmp',snapshot)
    mapping=yaml.safe_load((out/'wavemap.yaml').read_text())
    params=mapping['/**']['ros__parameters']
    params.update(octomap_resolution_m=.1,octomap_topic='/mapping/octomap_native',same_session_map_input=str(snapshot))
    for name in ('wavemap.yaml','nbv.yaml'):
        shutil.copy2(out/name,out/(name+'.before-native-resolution'))
    nbv=yaml.safe_load((out/'nbv.yaml').read_text());nbv['resolution']=.1
    (out/'nbv.yaml').write_text(yaml.safe_dump(nbv,sort_keys=False))
    (out/'wavemap.yaml').write_text(yaml.safe_dump(mapping,sort_keys=False))
    record=dict(originals=originals,owner_pid=os.getpid(),run=out.name,epoch=1,
        resolution=.1,snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        reason='Preserve current vehicle, native VIO frame and manually collected map; replace only mapping workers',
        state='starting',replacements={})
    manifest_path.write_text(json.dumps(record,indent=2))
    for pid in originals.values():os.kill(pid,signal.SIGSTOP)
    ros=['--ros-args','-p','use_sim_time:=true'];python='/opt/research/bin/python3'
    commands=[('wavemap',[str(args.binary),*ros,'-r','__ns:=/pelagos/wavemap','-r','__node:=underwater_wavemap_native',
        '--params-file',str(out/'wavemap.yaml')]),
        ('relay',[python,targets['relay'],*ros,'-r','__node:=pelagos_wavemap_nbv_relay_native',
        '-p','run_dir:='+str(out),'-p','octomap_topic:=/mapping/octomap_native'])]
    children=[];stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        for name,command in commands:
            stream=(out/'logs'/(name+'-native.log')).open('w',buffering=1)
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((child,stream));record['replacements'][name]=child.pid
        record['state']='running';manifest_path.write_text(json.dumps(record,indent=2))
        while not stopping:
            for child,_ in children:
                if child.poll() is not None:raise RuntimeError('Replacement exited: '+str(child.returncode))
            time.sleep(.2)
    finally:
        # No automatic task/authorization resume on any process failure.
        subprocess.run([python,'/pelagos/Simulation/adapters/autonomy_cli.py','--run-dir',str(out),'pause'],timeout=8,check=False)
        for child,_ in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        for child,stream in children:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            stream.close()
        record['state']='stopped';manifest_path.write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
