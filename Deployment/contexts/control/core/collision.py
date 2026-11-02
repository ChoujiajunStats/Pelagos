"""Vectorized exact swept-sphere / voxel AABB checks on a bounded immutable ROI."""
import math
import numpy as np


def swept_sphere_status(data, metadata, points, radius):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not 2<=len(points)<=2000 or not np.isfinite(points).all():return 'invalid_path'
    if not 0<radius<=2:return 'invalid_path'
    resolution=float(metadata['resolution']);origin=np.asarray(metadata['origin']);shape=np.asarray(data.shape)
    checked=0
    for a,b in zip(points[:-1],points[1:]):
        low=np.floor((np.minimum(a,b)-radius-origin)/resolution).astype(int)
        high=np.floor((np.maximum(a,b)+radius-origin)/resolution).astype(int)
        count=int(np.prod(high-low+1));checked+=count
        if count<=0 or checked>2000000:return 'collision_budget_exceeded'
        keys=np.indices(tuple(high-low+1)).reshape(3,-1).T+low
        inside=np.all((keys>=0)&(keys<shape),axis=1)
        values=np.full(len(keys),np.nan)
        values[inside]=data[tuple(keys[inside].T)]
        blocked=~np.isfinite(values)|(values<=resolution*.5)
        minimum=origin+keys[blocked]*resolution;values=values[blocked]
        if not len(minimum):continue
        maximum=minimum+resolution;direction=b-a;n=len(minimum)
        cuts=np.zeros((n,8));cuts[:,1]=1.
        for axis in range(3):
            if abs(direction[axis])>1e-12:
                cuts[:,2+axis*2]=np.clip((minimum[:,axis]-a[axis])/direction[axis],0.,1.)
                cuts[:,3+axis*2]=np.clip((maximum[:,axis]-a[axis])/direction[axis],0.,1.)
        cuts.sort(axis=1);distance=np.full(n,np.inf)
        for i in range(7):
            left,right=cuts[:,i],cuts[:,i+1]
            midpoint=a+((left+right)*.5)[:,None]*direction
            outside=(midpoint<minimum)|(midpoint>maximum)
            boundary=np.where(midpoint<minimum,minimum,maximum)
            aa=np.sum(outside*direction**2,axis=1)
            bb=np.sum(outside*direction*(a-boundary),axis=1)
            optimum=np.divide(-bb,aa,out=left.copy(),where=aa>0)
            t=np.clip(optimum,left,right);p=a+t[:,None]*direction
            distance=np.minimum(distance,np.sum((p-np.clip(p,minimum,maximum))**2,axis=1))
        intersect=distance<=radius**2+1e-10
        if np.any(intersect):return 'in_unknown_space' if not np.isfinite(values[intersect]).all() else 'in_collision'
    return 'free'
