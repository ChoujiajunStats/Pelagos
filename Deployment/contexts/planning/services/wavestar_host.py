#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Native Wavemap/Wavestar path preview worker; no ROS 1 or vehicle control."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import zlib
import numpy as np
import yaml
import sys
from pathlib import Path as DeploymentPath
sys.path.insert(0, str(DeploymentPath(__file__).resolve().parents[3] / 'contracts'))
from wire import receive_packet, send_packet
from planning_grid import sphere_path_status


class Preview:
    def __init__(self, args):
        self.args=args
        self.connection=None
        self.send_lock=threading.Lock()
        self.snapshot=None;self.odom=None;self.goal=None
        self.map_arrival=0.;self.odom_arrival=0.
        self.previous_key=None;self.previous_time=0.
        self.previous_start=None
        self.size=yaml.safe_load(Path(args.robot_config).read_text())['RobotParams']['size']
        self.configuration=yaml.safe_load(Path(args.navigation_config).read_text())
        self.configuration['/**']['ros__parameters']['planning_map']['robot_radius_m']=max(self.size)/2
        self.configuration['/**']['ros__parameters']['planner']['algorithm']=args.algorithm
        self.temp=tempfile.TemporaryDirectory(prefix='dapeng-wavestar-preview-')
        self.directory=Path(self.temp.name)
        self.config_path=self.directory/'navigation.yaml'
        self.config_path.write_text(yaml.safe_dump(self.configuration))
        self.cache_path=Path(os.environ.get('XDG_RUNTIME_DIR','/tmp'))/'dapeng-wavestar'/ 'planning_roi.npz'
        self.cache_path.parent.mkdir(parents=True,exist_ok=True)
        threading.Thread(target=self.network,daemon=True).start()

    def transmit(self, metadata):
        with self.send_lock:
            if self.connection:
                try:send_packet(self.connection,metadata)
                except (OSError,ValueError):pass

    def network(self):
        while True:
            try:
                with socket.create_connection(('127.0.0.1',self.args.port),timeout=3) as c:
                    c.settimeout(1)
                    with self.send_lock:self.connection=c
                    while True:
                        m,payload=receive_packet(c)
                        if m['type']=='map':
                            shape=m.get('shape',[])
                            if len(shape)!=3 or any(not isinstance(x,int) or not 0<x<=1000 for x in shape):
                                raise ValueError('invalid ROI shape')
                            size=math.prod(shape)*4
                            if not 0<size<=4000000 or size+m.get('native_map_bytes',0)!=len(payload):
                                raise ValueError('invalid Wavemap + ROI payload')
                            if m.get('native_map_codec')!='zlib' or not 0<m.get('native_map_raw_bytes',0)<=32*1024*1024:
                                raise ValueError('native Wavemap missing or oversized')
                            self.snapshot=(m,payload[:size],payload[size:])
                            self.map_arrival=time.monotonic()
                            temporary=self.cache_path.with_suffix('.tmp.npz')
                            np.savez(temporary,data=np.frombuffer(payload[:size],dtype='<f4').reshape(shape),
                                metadata=json.dumps(m),received_ns=time.time_ns())
                            os.replace(temporary,self.cache_path)
                        elif m['type']=='odometry':self.odom=m;self.odom_arrival=time.monotonic()
                        elif m['type']=='goal':self.goal=m
                        elif m['type']=='clear':self.goal=None
            except (OSError,EOFError,ValueError) as error:
                print('Wavestar connection:',error,flush=True)
            finally:
                with self.send_lock:self.connection=None
                self.snapshot=None;self.odom=None;self.goal=None;self.previous_key=None
            time.sleep(1)

    def plan_once(self):
        snapshot,odom,goal=self.snapshot,self.odom,self.goal
        if not snapshot or not odom or not goal:return
        now=time.monotonic()
        if now-self.map_arrival>4 or now-self.odom_arrival>1.5:return
        metadata,roi,native=snapshot
        key=(metadata['sequence'],goal['sequence'])
        moved=self.previous_start is None or math.dist(odom['pose'][:3],self.previous_start)>.15
        if now-self.previous_time<.5 or (key==self.previous_key and not moved):return
        self.previous_key=key;self.previous_time=now;self.previous_start=odom['pose'][:3]
        response=dict(goal_sequence=goal['sequence'],map_sequence=metadata['sequence'],map_stamp=metadata['stamp'])
        try:
            started=time.monotonic()
            decoder=zlib.decompressobj()
            raw=decoder.decompress(native,metadata['native_map_raw_bytes']+1)
            if not decoder.eof or decoder.unused_data or len(raw)!=metadata['native_map_raw_bytes']:
                raise ValueError('native Wavemap decompression size mismatch')
            map_path=self.directory/'input.wvmp';map_path.write_bytes(raw)
            output=self.directory/'result'
            command=[self.args.binary,str(map_path),str(self.config_path),str(output),
                     *map(str,odom['pose'][:3]),*map(str,goal['pose'][:3])]
            result=subprocess.run(command,capture_output=True,text=True,timeout=3)
            if result.returncode:raise ValueError(result.stderr[-500:])
            report=yaml.safe_load((output/'report.yaml').read_text())
            if not self.goal or self.goal['sequence']!=goal['sequence']:return
            if not self.snapshot or self.snapshot[0]['sequence']!=metadata['sequence']:return
            if report['planning_status']!='success':
                self.transmit(dict(response,type='status',state=report['planning_status'],
                    detail=dict(planner='wavestar',radius_m=max(self.size)/2,
                        start_native_log_odds=report.get('start_native_log_odds'),
                        goal_native_log_odds=report.get('goal_native_log_odds'),
                        start_collision_free=report.get('start_collision_free'))))
                return
            with (output/'path.csv').open() as f:
                poses=[[float(p['x']),float(p['y']),float(p['z']),0.,0.,0.,1.] for p in csv.DictReader(f)]
            grid=np.frombuffer(roi,dtype='<f4').reshape(metadata['shape'])
            checked=sphere_path_status(grid,metadata,poses,max(self.size)/2)
            if checked!='free':
                self.transmit(dict(response,type='status',state='path_validation_failed',detail=checked));return
            self.transmit(dict(response,type='path',state='path_ready',poses=poses,
                planning_ms=report['planning_ms'],remaining_distance_m=math.dist(poses[-1][:3],goal['pose'][:3])))
            print(json.dumps(dict(event='path_ready',map_sequence=metadata['sequence'],points=len(poses),
                search_ms=report['planning_ms'],total_ms=(time.monotonic()-started)*1000,
                length_m=sum(math.dist(a[:3],b[:3]) for a,b in zip(poses,poses[1:])))),flush=True)
        except (ValueError,OSError,subprocess.TimeoutExpired,yaml.YAMLError,KeyError,zlib.error) as error:
            self.transmit(dict(response,type='status',state='wavestar_failed',detail=str(error)))

    def run(self):
        try:
            while True:self.plan_once();time.sleep(.1)
        finally:self.temp.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=18771)
    p.add_argument('--binary',required=True)
    p.add_argument('--robot-config',required=True)
    p.add_argument('--navigation-config',required=True)
    p.add_argument('--algorithm',choices=('multi_res_theta_star','theta_star','a_star'),default='multi_res_theta_star')
    Preview(p.parse_args()).run()


if __name__=='__main__':main()
