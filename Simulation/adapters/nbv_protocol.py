"""ROS-independent geometry and bounded, session-checked NBV transport."""
import json
import socket

from feedback_sampling import interpolate_feedback


def interpolate_pose(history, stamp, maximum_gap_ns=100_000_000):
    """Interpolate at depth acquisition time; never pair with latest pose."""
    for (a,pa),(b,pb) in zip(history,list(history)[1:]):
        if a<=stamp<=b and 0<b-a<=maximum_gap_ns:
            p,q,_=interpolate_feedback((a,pa[:3],pa[3:],[]),(b,pb[:3],pb[3:],[]),stamp)
            return [*p,*q]
    for value,pose in history:
        if value==stamp: return pose[:]
    return None


def project_depth(depth,k,stride=1):
    import numpy as np
    v,u=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[::stride,::stride]
    good=np.isfinite(z)&(z>0)
    return np.column_stack(((u[good]-k[0,2])*z[good]/k[0,0],
                            (v[good]-k[1,2])*z[good]/k[1,1],z[good])).astype('<f4')


def exchange(address,request,stop_event=None):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
        stream.settimeout(5.)
        stream.connect(str(address))
        stream.sendall(json.dumps(request,allow_nan=False).encode()+b'\n')
        # Planner computation has its original iteration budget, without a new
        # wall-time cutoff. Poll only to allow clean process cancellation.
        stream.settimeout(.25)
        chunks=[];size=0
        while True:
            if stop_event is not None and stop_event.is_set():raise OSError('NBV transport stopped')
            try:data=stream.recv(65536)
            except socket.timeout:continue
            if not data:raise OSError('NBV backend closed the transport')
            chunks.append(data);size+=len(data)
            if size>36*1024*1024:raise ValueError('NBV response exceeds transport capacity')
            if data.endswith(b'\n'):break
        response=json.loads(b''.join(chunks))
        if any(response.get(key)!=request[key] for key in ('run','epoch','sequence')):
            raise ValueError('NBV response session or request sequence mismatch')
        return response
