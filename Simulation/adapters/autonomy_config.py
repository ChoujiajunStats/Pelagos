"""Source parameters and the user's approved geometric adaptations."""
import hashlib
import itertools
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


def source_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested(flat):
    result={}
    for path,value in flat.items():
        node=result;parts=path.split('/')
        for part in parts[:-1]:node=node.setdefault(part,{})
        node[parts[-1]]=value
    return result


def scene_bounds(asset,box):
    c,s=math.cos(asset['cave_yaw_ned']),math.sin(asset['cave_yaw_ned'])
    points=[]
    for corner in itertools.product(*zip(*asset['original_bounds_ned'])):
        x,y,z=[value*asset['scale'] for value in corner]
        n=c*x-s*y+asset['cave_offset_ned'][0]
        e=s*x+c*y+asset['cave_offset_ned'][1]
        d=z+asset['cave_offset_ned'][2]
        points.append((e-asset['spawn_enu'][0],n-asset['spawn_enu'][1],-d-asset['spawn_enu'][2]))
    r=math.ceil(max(math.hypot(p[0],p[1]) for p in points)+math.hypot(box['x'],box['y'])/2+box['overshoot'])
    zpad=box['z']/2+box['overshoot']
    return dict(minX=-float(r),maxX=float(r),minY=-float(r),maxY=float(r),
                minZ=float(math.floor(min(p[2] for p in points)-zpad)),
                maxZ=float(math.ceil(max(p[2] for p in points)+zpad)),softBounds=False)


def nbv_config(root,settings,asset,camera):
    root=Path(root);upstream=root/'TP/Planner/nbvplanner'
    source=upstream/settings['upstream_profile']
    params=nested(yaml.safe_load(source.read_text()))
    launch=upstream/'interface_nbvp_rotors/launch/flat_exploration.launch'
    include=ET.parse(launch).getroot().findall('include')[-1]
    args={arg.get('name'):arg.get('value') for arg in include.findall('arg')}
    for key in ('resolution','sensor_max_range'):
        params[key]=float(args[key])
    if 'map_resolution_m' in settings:
        params['resolution']=float(settings['map_resolution_m'])
    inspector=upstream/'interface_nbvp_rotors/launch/mav_inspector.launch'
    node=next(n for n in ET.parse(inspector).getroot().iter('node') if n.get('name')=='nbvPlanner')
    params['map_publish_frequency']=float(next(p for p in node.findall('param') if p.get('name')=='map_publish_frequency').get('value'))
    params['tf_frame']='rov01/odom';params['robot_frame']='rov01/base_link'
    if settings['planning_bounds']!='porth_full_scene_conservative':raise ValueError('No approved NBV bounds policy')
    params['bbx']=scene_bounds(asset,params['system']['bbx'])
    if settings['camera_parameters']!='simulation_calibration':raise ValueError('No approved camera adaptation')
    hfov=float(camera['horizontal_fov_deg'])
    vfov=math.degrees(2*math.atan(math.tan(math.radians(hfov)/2)*camera['height']/camera['width']))
    params['system']['camera']=dict(pitch=[0.],horizontal=[hfov],vertical=[vfov])
    provenance=dict(upstream_commit='5c07657af18f967d7049d187db0607024746a42c',
        sources={str(p.relative_to(root)):source_hash(p) for p in (source,launch,inspector)},
        approved_adaptations=['full Porth scale-derived sampling bounds','validated simulation camera calibration'],
        planning_bounds=params['bbx'],bounds_are_motion_geofence=False,
        transport='Unix socket; existing ROS domain unchanged',algorithm_overrides={})
    if 'map_resolution_m' in settings:
        provenance['approved_adaptations'].append('2026-10-31 user approved NBV query resolution 0.4 -> 0.1 m')
    return params,provenance
