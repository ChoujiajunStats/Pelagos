"""Distance sampling and camera-frustum geometry without ROS dependencies."""
import math
from geometry import compose,q_normalize


class DistanceSampler:
    def __init__(self,spacing=.75):
        if not math.isfinite(spacing) or spacing<=0:raise ValueError('invalid pose spacing')
        self.spacing=spacing;self.previous=None;self.distance=0.;self.next_distance=0.

    def accept(self,position):
        position=tuple(position)
        if self.previous is not None:self.distance+=math.dist(position,self.previous)
        self.previous=position
        if self.distance+1e-9<self.next_distance:return False
        self.next_distance=self.distance+self.spacing
        return True


def frustum_edges(horizontal_fov,aspect,depth):
    width=depth*math.tan(horizontal_fov/2)
    height=width*aspect
    corners=[(-width,-height,depth),(width,-height,depth),
             (width,height,depth),(-width,height,depth)]
    return [((0.,0.,0.),c) for c in corners]+[(corners[i],corners[(i+1)%4]) for i in range(4)]


def transformed_frustum(transform,horizontal_fov,aspect,depth=.25):
    return [compose(transform,(point,(0.,0.,0.,1.)))[0]
            for edge in frustum_edges(horizontal_fov,aspect,depth) for point in edge]


def pose_error(reference,estimated):
    a,b=q_normalize(reference[1]),q_normalize(estimated[1])
    dot=min(1.,abs(sum(x*y for x,y in zip(a,b))))
    return dict(position_error_m=math.dist(reference[0],estimated[0]),
                rotation_error_deg=math.degrees(2*math.acos(dot)))
