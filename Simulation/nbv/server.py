#!/usr/bin/env python3
"""ROS1 side of the local Unix socket bridge; original NBV runs in C++."""
import argparse
import array
import base64
import json
from pathlib import Path
import signal
import socketserver
import socket
import struct
import threading
import subprocess
import time
import sys
import zlib

import rospy
import rosgraph
import yaml
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock
from pelagos_nbv.srv import Step, StepRequest


def pose(values):
    message = Pose()
    message.position.x, message.position.y, message.position.z = values[:3]
    (message.orientation.x, message.orientation.y,
     message.orientation.z, message.orientation.w) = values[3:]
    return message


def values(message):
    p, q = message.position, message.orientation
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        data={}
        self.request.settimeout(5.)
        try:
            data = json.loads(self.rfile.readline(36 * 1024 * 1024))
            if data['run'] != self.server.run or data['epoch'] != 1:
                raise ValueError('run_or_estimator_epoch_mismatch')
            req = StepRequest()
            req.header = Header(stamp=rospy.Time(data['stamp_ns']//10**9, data['stamp_ns']%10**9),
                                frame_id='rov01/odom')
            req.operation = data['operation']
            req.include_map=bool(data.get('preview',False))
            req.body = pose(data['body'])
            req.camera = pose(data.get('camera', data['body']))
            if req.operation=='replace_map':
                raw=base64.b64decode(data['map_data_b64'],validate=True)
                if zlib.crc32(raw)!=data['map_crc32']:raise ValueError('map_payload_checksum_mismatch')
                req.map.header=Header(stamp=req.header.stamp,frame_id=data['map_frame'])
                req.map.binary=bool(data['map_binary']);req.map.id=data['map_id']
                req.map.resolution=float(data['map_resolution']);req.map.data=array.array('b',raw)
                req.source_revision=int(data['source_revision'])
            if req.operation == 'observe':
                raw=base64.b64decode(data['points_f32_le'],validate=True)
                if len(raw)%12:raise ValueError('invalid XYZ point buffer')
                req.cloud=PointCloud2(header=Header(stamp=req.header.stamp,frame_id='cam0_rect_optical'),
                    height=1,width=len(raw)//12,is_dense=True,is_bigendian=False,point_step=12,
                    row_step=len(raw),data=raw,fields=[PointField(name=name,offset=i*4,
                    datatype=PointField.FLOAT32,count=1) for i,name in enumerate(('x','y','z'))])
            req.path = [pose(p) for p in data.get('path', [])]
            result = self.server.step(req)
            response = dict(run=data['run'], epoch=data['epoch'], sequence=data['sequence'],
                            valid=result.valid, reason=result.reason, map_revision=result.map_revision,
                            map_source=result.map_source,source_revision=result.source_revision,
                            depth_fusion_enabled=result.depth_fusion_enabled,
                            path=[values(p) for p in result.path],
                            diagnostics=json.loads(result.diagnostics_json or '{}'),
                            occupied=[[p.x,p.y,p.z,s] for p,s in zip(result.occupied,result.sizes)])
        except Exception as error:
            response = dict(valid=False, reason=str(error),path=[],occupied=[],
                            **{key:data.get(key) for key in ('run','epoch','sequence')})
        self.wfile.write(json.dumps(response, allow_nan=False).encode()+b'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--master-port',type=int,default=11311)
    args=parser.parse_args();out=args.run_dir
    if not 1024<=args.master_port<=65535:raise ValueError('Invalid local ROS master port')
    import os
    os.environ['ROS_MASTER_URI']='http://127.0.0.1:'+str(args.master_port)
    while not (out/'nbv.yaml').exists(): time.sleep(.1)
    logs = out/'logs'; logs.mkdir(exist_ok=True)
    children = []
    stop_clock=threading.Event();clock_socket=None;clock_thread=None
    try:
        with (logs/'nbv_roscore.log').open('w') as stream:
            children.append(subprocess.Popen(['roscore','-p',str(args.master_port)], stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic()+30
        while not rosgraph.is_master_online():
            if time.monotonic()>deadline: raise RuntimeError('ROS1 master startup timeout')
            time.sleep(.1)
        rospy.init_node('pelagos_nbv_transport', disable_signals=True)
        rospy.set_param('/use_sim_time', True)
        rospy.set_param('/pelagos_nbv', yaml.safe_load((out/'nbv.yaml').read_text()))
        clock_path=out/'nbv-clock.sock';clock_path.unlink(missing_ok=True)
        clock_socket=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
        clock_socket.bind(str(clock_path));clock_socket.settimeout(.25)
        clock_pub=rospy.Publisher('/clock',Clock,queue_size=1)
        def receive_clock():
            while not stop_clock.is_set():
                try:data=clock_socket.recv(8)
                except socket.timeout:continue
                except OSError:break
                if len(data)!=8:continue
                stamp=struct.unpack('=q',data)[0]
                if stamp>=0:clock_pub.publish(Clock(clock=rospy.Time(stamp//10**9,stamp%10**9)))
        clock_thread=threading.Thread(target=receive_clock,daemon=True);clock_thread.start()
        with (logs/'nbv_native.log').open('w') as stream:
            children.append(subprocess.Popen(['rosrun','pelagos_nbv','pelagos_nbv_node'],
                                             stdout=stream, stderr=subprocess.STDOUT))
        rospy.wait_for_service('/pelagos_step', timeout=30)
        address = out/'nbv.sock'
        address.unlink(missing_ok=True)
        class Server(socketserver.UnixStreamServer):
            timeout = .25
        with Server(str(address), Handler) as server:
            server.run = out.name
            server.step = rospy.ServiceProxy('/pelagos_step', Step)
            stopping = [False]
            for sig in (signal.SIGINT,signal.SIGTERM):
                signal.signal(sig, lambda *_: stopping.__setitem__(0,True))
            while not stopping[0] and not rospy.is_shutdown():
                native_code=children[-1].poll()
                if native_code is not None:
                    (out/'nbv-backend-status.json').write_text(json.dumps(dict(
                        state='finished' if native_code==0 else 'fault',native_returncode=native_code)))
                    break
                if children[0].poll() is not None:raise RuntimeError('ROS1 master exited')
                server.handle_request()
        address.unlink(missing_ok=True)
    finally:
        stop_clock.set()
        if clock_socket is not None:clock_socket.close()
        if clock_thread is not None:clock_thread.join(timeout=1.)
        (out/'nbv-clock.sock').unlink(missing_ok=True)
        for child in reversed(children):
            if child.poll() is None: child.send_signal(signal.SIGINT)
        for child in reversed(children):
            try: child.wait(timeout=4)
            except subprocess.TimeoutExpired: child.kill(); child.wait()


if __name__ == '__main__':
    try:main()
    except Exception as error:
        (Path(sys.argv[1])/'nbv-backend-status.json').write_text(json.dumps(
            dict(state='fault',reason=str(error))))
        raise
