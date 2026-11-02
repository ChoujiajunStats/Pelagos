"""Waypoint input geometry; no exploration domain or motion limits."""
import math
from geometry import compose,rotate,q_normalize


def waypoint(x,y,z,yaw_degrees):
    values=[float(v) for v in (x,y,z,yaw_degrees)]
    if not all(math.isfinite(v) for v in values):raise ValueError('路径点必须是有限数值')
    angle=math.radians(values[3])/2
    return [*values[:3],0.,0.,math.sin(angle),math.cos(angle)]


def route_poses(values):
    if not isinstance(values,list) or not values:raise ValueError('请先添加路径点')
    result=[]
    for pose in values:
        if len(pose)!=7:raise ValueError('路径点需要 XYZ 和 xyzw 四元数')
        pose=[float(v) for v in pose]
        if not all(math.isfinite(v) for v in pose):raise ValueError('路径点含非有限数值')
        if abs(sum(v*v for v in pose[3:])-1)>1e-4:raise ValueError('路径点旋转无效')
        if abs(pose[3])+abs(pose[4])>1e-6:raise ValueError('路径点只指定航向，横滚和俯仰由车辆自身稳定')
        result.append([*pose[:3],*q_normalize(pose[3:])])
    return result


def clicked_waypoint(pose,transform,z):
    """Project the RViz XY selection onto the operator's estimated-world Z.

    The display alignment may include roll/pitch. Solve the plane intersection
    before transforming, rather than treating the display's z=0 as body depth.
    """
    p,q=pose;translation,rotation=transform
    ex,ey,ez=[rotate(rotation,axis) for axis in ((1,0,0),(0,1,0),(0,0,1))]
    if abs(ez[2])<1e-8:raise ValueError('此视图平面无法选择水平路径点')
    plane_z=(float(z)-translation[2]-ex[2]*p[0]-ey[2]*p[1])/ez[2]
    position,orientation=compose(transform,((p[0],p[1],plane_z),q))
    x,y,a,w=orientation
    yaw=math.atan2(2*(w*a+x*y),1-2*(y*y+a*a))
    return [position[0],position[1],float(z),0.,0.,math.sin(yaw/2),math.cos(yaw/2)]
