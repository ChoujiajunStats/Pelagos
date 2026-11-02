#!/usr/bin/env python3
"""Offline robot visual LOD; quadric decimation with source-surface UV reprojection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import fast_simplification


def load(p):
    v=[];uv=[];f=[];ft=[]
    for line in p.open():
        x=line.split()
        if not x:continue
        if x[0]=='v':v.append([float(a) for a in x[1:4]])
        elif x[0]=='vt':uv.append([float(a) for a in x[1:3]])
        elif x[0]=='f':
            face=[a.split('/') for a in x[1:]]
            for i in range(1,len(face)-1):
                tri=[face[0],face[i],face[i+1]];f.append([int(a[0])-1 for a in tri]);ft.append([int(a[1])-1 if len(a)>1 and a[1] else -1 for a in tri])
    return np.asarray(v),np.asarray(f,dtype=np.int32),np.asarray(uv),np.asarray(ft)


def project(points,triangles):
    # points: N,3; candidates: N,K,3,3. Exact nearest point for each triangle.
    p=points[:,None,:];a,b,c=triangles[:,:,0],triangles[:,:,1],triangles[:,:,2]
    ab=b-a;ac=c-a;ap=p-a
    d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1)
    d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1);den=d00*d11-d01*d01
    safe=np.where(abs(den)>1e-24,den,1.)
    u=(d11*d20-d01*d21)/safe;w=(d00*d21-d01*d20)/safe
    inside=(u>=0)&(w>=0)&(u+w<=1)&(abs(den)>1e-24)
    bary=np.stack([1-u-w,u,w],axis=-1);q=a+u[:,:,None]*ab+w[:,:,None]*ac
    distance=((p-q)**2).sum(-1);distance[~inside]=np.inf
    for i,j in [(0,1),(1,2),(2,0)]:
        start=triangles[:,:,i];edge=triangles[:,:,j]-start
        t=np.clip(((p-start)*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-24),0,1)
        d=((p-(start+t[:,:,None]*edge))**2).sum(-1);use=d<distance
        b0=np.zeros_like(bary);b0[:,:,i]=1-t;b0[:,:,j]=t
        bary[use]=b0[use];distance[use]=d[use]
    best=np.argmin(distance,axis=1);rows=np.arange(len(points))
    return best,bary[rows,best],np.sqrt(distance[rows,best])


def simplify(path,target,material,mtl):
    verts,faces,uv,ft=load(path);unique,inverse=np.unique(np.round(verts,8),axis=0,return_inverse=True)
    geometry=inverse[faces].astype(np.int32)
    p,f=fast_simplification.simplify(unique,geometry,target_count=target,agg=4.)
    original=verts[faces];centroids=original.mean(axis=1);tree=cKDTree(centroids)
    all_uv=[];errors=[]
    points=p[f].reshape(-1,3)
    for begin in range(0,len(points),3000):
        pts=points[begin:begin+3000];_,neighbors=tree.query(pts,k=64)
        best,bary,dist=project(pts,original[neighbors]);source_faces=neighbors[np.arange(len(pts)),best]
        # A centroid shortlist can miss a large source triangle whose edge is
        # close to a small detail. Resolve those conservative outliers exactly.
        for j in np.flatnonzero(dist>.002):
            exact,weight,distance=project(pts[j:j+1],original[None,:,:,:])
            source_faces[j]=exact[0];bary[j]=weight[0];dist[j]=distance[0]
        if len(uv) and (ft>=0).all():tex=(uv[ft[source_faces]]*bary[:,:,None]).sum(axis=1)
        else:tex=np.zeros((len(pts),2))
        all_uv.append(tex);errors.extend(dist.tolist())
    tex=np.concatenate(all_uv);normals=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
    destination=path.with_name(path.stem+'_lod.obj')
    with destination.open('w') as out:
        out.write('# Display-only quadric LOD of the supplied BlueROV2 asset.\nmtllib '+mtl+'\nusemtl '+material+'\n')
        for v in p:out.write('v '+' '.join(f'{x:.8f}' for x in v)+'\n')
        for t in tex:out.write('vt '+' '.join(f'{x:.8f}' for x in t)+'\n')
        for n in normals:out.write('vn '+' '.join(f'{x:.8f}' for x in n)+'\n')
        for i,tri in enumerate(f):out.write('f '+' '.join(f'{v+1}/{3*i+j+1}/{i+1}' for j,v in enumerate(tri))+'\n')
    bbox=float(max(np.abs(p.min(0)-verts.min(0)).max(),np.abs(p.max(0)-verts.max(0)).max()))
    print(path.name,'faces',len(f),'bbox_m',bbox,'max_forward_m',max(errors),flush=True)
    if not np.isfinite(p).all() or bbox>.006 or max(errors)>.006:raise ValueError('robot visual LOD exceeded 6 mm forward/source or bounding-box error')
    return dict(source=path.name,lod=destination.name,source_triangles=len(faces),lod_triangles=len(f),
                bbox_max_difference_m=bbox,lod_to_source_max_distance_m=max(errors),lod_to_source_p95_distance_m=float(np.percentile(errors,95)),
                note='Forward surface and bounding-box checks only; not a symmetric Hausdorff bound. UVs reprojected; visual only.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args();root=args.directory
    results=[simplify(root/'bluerov2.obj',18000,'hull','hull.mtl'),simplify(root/'cw.obj',800,'ID69','propeller.mtl'),simplify(root/'ccw.obj',800,'ID955','propeller.mtl')]
    p=root/'assembly.json';a=json.loads(p.read_text());a['hull']='bluerov2_lod.obj'
    for t in a['thrusters']:t['mesh']=Path(t['mesh']).stem.replace('_lod','')+'_lod.obj'
    a['visual_lod']=results;p.write_text(json.dumps(a,indent=2)+'\n')
    report=dict(converter='fast-simplification '+fast_simplification.__version__,results=results,
                total_display_triangles=results[0]['lod_triangles']+3*results[1]['lod_triangles']+3*results[2]['lod_triangles'])
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
