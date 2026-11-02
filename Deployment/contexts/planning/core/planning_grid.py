"""Independent conservative validation of planner outputs on immutable snapshots."""
import math
import numpy as np


def body_summary(data, metadata, center, radius=.35):
    """Native cells intersecting the collision sphere, including UNKNOWN."""
    resolution = metadata['resolution']
    origin = np.asarray(metadata['origin'])
    low = np.floor((np.asarray(center)-radius-origin)/resolution).astype(int)
    high = np.floor((np.asarray(center)+radius-origin)/resolution).astype(int)
    counts = dict(free=0,occupied=0,unknown=0)
    for index in np.ndindex(tuple(high-low+1)):
        key=low+index
        cell_min=origin+key*resolution
        closest=np.clip(center,cell_min,cell_min+resolution)
        if np.linalg.norm(np.asarray(center)-closest)>radius:
            continue
        if np.any(key<0) or np.any(key>=data.shape):
            counts['unknown']+=1;continue
        value=data[tuple(key)]
        name='unknown' if not np.isfinite(value) else 'occupied' if value<=resolution*.5 else 'free'
        counts[name]+=1
    return counts


def box_status(data, metadata, center, size):
    resolution = metadata['resolution']
    low = np.floor((np.asarray(center)-np.asarray(size)/2-metadata['origin'])/resolution).astype(int)
    high = np.floor((np.asarray(center)+np.asarray(size)/2-metadata['origin'])/resolution).astype(int)
    if np.any(low < 0) or np.any(high >= data.shape):
        return 'outside_map'
    region = data[low[0]:high[0]+1,low[1]:high[1]+1,low[2]:high[2]+1]
    if np.any(region[np.isfinite(region)] <= resolution*.5):
        return 'in_collision'
    if not np.all(np.isfinite(region)):
        return 'in_unknown_space'
    return 'free'


def path_status(data, metadata, poses, size):
    if (len(poses) < 2 or len(poses) > 2000
            or any(len(p) != 7 or not all(math.isfinite(v) for v in p) for p in poses)):
        return 'invalid_path'
    # Padding by half a sampling step bounds every swept point between samples.
    step = metadata['resolution']*.5
    padded_size = np.asarray(size)+step
    for a, b in zip(poses[:-1], poses[1:]):
        a, b = np.asarray(a[:3]), np.asarray(b[:3])
        distance = np.linalg.norm(b-a)
        if distance > 20:
            return 'invalid_segment'
        count = max(1, int(math.ceil(distance/step)))
        for t in np.linspace(0, 1, count+1):
            status = box_status(data, metadata, a+(b-a)*t, padded_size)
            if status != 'free':
                return status
    return 'free'


def _segment_box_distance_squared(a, b, minimum, maximum):
    direction=b-a
    cuts=[0.,1.]
    for axis in range(3):
        if abs(direction[axis])>1e-12:
            for boundary in (minimum[axis],maximum[axis]):
                t=(boundary-a[axis])/direction[axis]
                if 0<t<1:cuts.append(t)
    cuts.sort()
    def distance(t):
        p=a+t*direction
        return float(np.sum((p-np.clip(p,minimum,maximum))**2))
    best=float('inf')
    for left,right in zip(cuts,cuts[1:]):
        middle=a+(left+right)*.5*direction
        aa=bb=0.
        for axis in range(3):
            if middle[axis]<minimum[axis] or middle[axis]>maximum[axis]:
                boundary=minimum[axis] if middle[axis]<minimum[axis] else maximum[axis]
                aa+=direction[axis]**2;bb+=direction[axis]*(a[axis]-boundary)
        t=float(np.clip(-bb/aa,left,right)) if aa>0 else left
        best=min(best,distance(left),distance(right),distance(t))
    return best


def sphere_path_status(data, metadata, poses, radius):
    """Exact swept-sphere vs native-cell AABBs; UNKNOWN is blocked."""
    if (not 0<radius<=2 or not 2<=len(poses)<=2000
            or any(len(p)!=7 or not all(math.isfinite(v) for v in p) for p in poses)):
        return 'invalid_path'
    resolution=metadata['resolution'];origin=np.asarray(metadata['origin'])
    checked=0
    for aa,bb in zip(poses,poses[1:]):
        a=np.asarray(aa[:3]);b=np.asarray(bb[:3])
        low=np.floor((np.minimum(a,b)-radius-origin)/resolution).astype(int)
        high=np.floor((np.maximum(a,b)+radius-origin)/resolution).astype(int)
        count=int(np.prod(high-low+1));checked+=count
        if count<=0 or checked>2000000:return 'collision_budget_exceeded'
        for local in np.ndindex(tuple(high-low+1)):
            key=low+local
            inside=not(np.any(key<0) or np.any(key>=data.shape))
            value=data[tuple(key)] if inside else float('nan')
            if np.isfinite(value) and value>resolution*.5:continue
            minimum=origin+key*resolution
            if _segment_box_distance_squared(a,b,minimum,minimum+resolution)<=radius**2+1e-10:
                return 'in_unknown_space' if not np.isfinite(value) else 'in_collision'
    return 'free'
