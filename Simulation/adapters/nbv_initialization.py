"""Original RotorS scan waypoints, executed with actual ROV arrival feedback."""
import math


class SourceInitialization:
    def __init__(self):
        self.started=None
        self.origin=None
        self.index=0
        self.issued=None
        self.done=False

    def update(self,seconds,pose,reached=False):
        if self.done:return None
        if self.started is None:
            self.started=seconds;self.origin=list(pose)
        if seconds-self.started<5.:return None  # Source GUI wait.
        if self.issued is None:self.issued=seconds
        # Keep the source's minimum one-second hold. A water vehicle must also
        # actually arrive before advancing to the next scan waypoint.
        if reached and seconds-self.issued>=1.:
            self.index+=1;self.issued=seconds
            if self.index==12:self.done=True;return None
        x,y,z,w=self.origin[3:]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        angle=yaw-math.pi*.1*min(self.index,10)
        target=[*self.origin[:3],0.,0.,math.sin(angle/2),math.cos(angle/2)]
        if self.index==11:target[0]-=.5;target[1]-=.5
        return target
