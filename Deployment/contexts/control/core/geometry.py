"""Explicit OKVIS world/body to local NED/FRD transforms and slow path references."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def minimum_centered_sphere_radius(dimensions):
    dimensions=np.asarray(dimensions,dtype=float)
    if dimensions.shape!=(3,) or not np.isfinite(dimensions).all() or np.any(dimensions<=0):
        raise ValueError('Expected positive length, width and height in meters')
    return float(np.linalg.norm(dimensions)*.5)


class LocalFrame:
    def __init__(self, position, quaternion, pressure_depth, body_to_frd):
        self.body_to_frd=np.asarray(body_to_frd,dtype=float)
        if not np.allclose(self.body_to_frd.T@self.body_to_frd,np.eye(3)) or np.linalg.det(self.body_to_frd)<.999:
            raise ValueError('body_to_frd must be a proper rigid rotation')
        forward=Rotation.from_quat(quaternion).as_matrix()@self.body_to_frd.T[:,0]
        forward[2]=0.;forward/=np.linalg.norm(forward)
        down=np.array([0.,0.,-1.]);right=np.cross(down,forward)
        self.world_to_ned=np.stack((forward,right,down))
        self.origin=np.asarray(position,dtype=float);self.depth_origin=float(pressure_depth)

    def state(self, body, imu, pressure_depth):
        r=self.world_to_ned@Rotation.from_quat(body['quaternion']).as_matrix()@self.body_to_frd.T
        position=self.world_to_ned@(np.asarray(body['position'])-self.origin)
        position[2]=pressure_depth-self.depth_origin
        velocity=self.body_to_frd@np.asarray(body['velocity'])
        gyro=self.body_to_frd@np.asarray(imu['gyro'])
        force=self.body_to_frd@np.asarray(imu['acceleration'])
        acceleration=force+r.T@np.array([0.,0.,9.80665])
        observation=np.r_[r[:,0],r[:,1],velocity,gyro,acceleration,position[2]]
        if not np.isfinite(observation).all():raise ValueError('nonfinite sensor state')
        return observation,position,r

    def path_to_ned(self, path, body_world, current_ned):
        points=(np.asarray(path)-self.origin)@self.world_to_ned.T
        body_ned=self.world_to_ned@(np.asarray(body_world)-self.origin)
        points[:,2]+=current_ned[2]-body_ned[2]
        return points

    def predicted_to_world(self, delta, body_world):
        return np.asarray(delta)@self.world_to_ned+np.asarray(body_world)


def timed_reference(points, current, current_velocity_ned, times, speed=.05, acceleration=.05):
    points=np.asarray(points,dtype=float);current=np.asarray(current)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2:raise ValueError('No valid planner path')
    segments=np.diff(points,axis=0);lengths=np.linalg.norm(segments,axis=1)
    keep=np.r_[True,lengths>1e-6];points=points[keep]
    if len(points)<2:raise ValueError('Degenerate planner path')
    segments=np.diff(points,axis=0);lengths=np.linalg.norm(segments,axis=1);arc=np.r_[0.,np.cumsum(lengths)]
    best=None
    for i,(a,d,l) in enumerate(zip(points[:-1],segments,lengths)):
        u=float(np.clip(np.dot(current-a,d)/(l*l),0,1));distance=np.linalg.norm(current-(a+u*d))
        if best is None or distance<best[0]:best=(distance,arc[i]+u*l,i)
    s=best[1];v=min(speed,max(0.,np.dot(current_velocity_ned,segments[best[2]]/lengths[best[2]])))
    position=[];velocity=[];last=0.
    for t in times:
        dt=float(t-last);last=float(t);remaining=arc[-1]-s
        target=min(speed,math.sqrt(max(0.,2*acceleration*remaining)))
        new_v=v+float(np.clip(target-v,-acceleration*dt,acceleration*dt))
        s=min(arc[-1],s+(v+new_v)*dt*.5);v=new_v if s<arc[-1]-1e-6 else 0.
        i=min(len(lengths)-1,int(np.searchsorted(arc,s,side='right')-1));u=(s-arc[i])/lengths[i]
        position.append(points[i]+u*segments[i]);velocity.append(v*segments[i]/lengths[i])
    return np.asarray(position,dtype=np.float32),np.asarray(velocity,dtype=np.float32)
