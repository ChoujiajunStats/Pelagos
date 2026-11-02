"""Face the validated path before translating; preserve every spatial vertex."""
import math
from .omniplanner_protocol import normalize_pose,yaw_only_pose,distance


def heading(p):
    q=yaw_only_pose(p)[3:]
    return math.atan2(2*q[3]*q[2],1-2*q[2]*q[2])


def wrap(angle):return math.atan2(math.sin(angle),math.cos(angle))


def forward_path(start,path,*,final_heading=None,maximum_turn=.55):
    if not path or not 0<maximum_turn<=math.pi/2:raise ValueError('Invalid forward path')
    previous=normalize_pose(start);angle=heading(previous);result=[]
    def rotate(target_angle):
        nonlocal angle
        change=wrap(target_angle-angle)
        if abs(change)<.02:return
        steps=max(1,math.ceil(abs(change)/maximum_turn));initial=angle
        for i in range(1,steps+1):
            a=initial+change*i/steps
            result.append(previous[:3]+[0.,0.,math.sin(a/2),math.cos(a/2)])
        angle=wrap(target_angle)
    for item in path:
        target=normalize_pose(item);dx,dy=target[0]-previous[0],target[1]-previous[1]
        if math.hypot(dx,dy)>.03:rotate(math.atan2(dy,dx))
        target=target[:3]+[0.,0.,math.sin(angle/2),math.cos(angle/2)]
        result.append(target);previous=target
    if final_heading is not None:
        if not math.isfinite(final_heading):raise ValueError('Invalid final heading')
        rotate(final_heading)
    return result
