#!/usr/bin/env python3
"""Isolated-domain acceptance of native incremental TRIANGLE_LIST surfaces."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import yaml
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from visualization_msgs.msg import Marker, MarkerArray
from smoke_incremental_visualization import Probe


class SurfaceProbe(Probe):
    def __init__(self):
        super().__init__()
        self.surfaces = {};self.surface_packets = [];self.surface_colors = []
        self.create_subscription(MarkerArray, '/visualization/surface_mesh', self.surface,
                                 QoSProfile(depth=20, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def surface(self, msg):
        self.surface_packets.append(msg)
        for m in msg.markers:
            key=(m.ns,m.id)
            if m.action==Marker.DELETEALL:self.surfaces.clear()
            elif m.action==Marker.DELETE:self.surfaces.pop(key,None)
            else:
                self.surfaces[key]=m
                self.surface_colors.append([m.color.r,m.color.g,m.color.b])

    def parameter(self, name, value):
        c=self.create_client(SetParameters,'/wavemap_navigation/set_parameters');assert c.wait_for_service(timeout_sec=5)
        p=ParameterValue()
        if isinstance(value,str):p.type=ParameterType.PARAMETER_STRING;p.string_value=value
        elif isinstance(value,bool):p.type=ParameterType.PARAMETER_BOOL;p.bool_value=value
        elif isinstance(value,float):p.type=ParameterType.PARAMETER_DOUBLE;p.double_value=value
        else:p.type=ParameterType.PARAMETER_INTEGER;p.integer_value=value
        f=c.call_async(SetParameters.Request(parameters=[Parameter(name=name,value=p)]))
        self.wait('set '+name,f.done);result=f.result().results[0];self.destroy_client(c);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if int(os.environ.get('ROS_DOMAIN_ID','0'))==0:ap.error('requires isolated nonzero ROS_DOMAIN_ID')
    args.output.mkdir(parents=True,exist_ok=True);os.environ['ROS_LOG_DIR']=str(args.output/'ros_logs')
    cfg=yaml.safe_load(args.config.read_text());p=cfg['/**']['ros__parameters']
    p['mapping'].update(width=32,height=24,fx=24.,fy=24.,cx=15.5,cy=11.5,min_cell_width_m=.1,
        tree_height=4,max_range_m=6.,map_input_path='',map_output_path=str(args.output/'online.wvmp'),calibration_path='')
    p['inputs'].update(body_odometry='/test/body',camera_odometry='/test/camera',depth='/test/depth')
    p['management'].update(crop_radius_m=4.,crop_period_s=.3,threshold_period_s=2.,prune_period_s=5.)
    p['visualization'].update(enabled=True,mode='surface',require_empty_start=True,update_hz=3.,mesh_resolution_m=.1,
        mesh_read_budget_ms=20.,max_mesh_blocks_per_update=4,update_radius_m=6.,pose_max_age_s=.6,keyframe_seconds=2.)
    config=args.output/'test.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    log=(args.output/'node.log').open('w');child=subprocess.Popen(['ros2','launch','underwater_navigation_ros',
        'incremental_mapping.launch.py','rviz:=false','config:='+str(config),'map_output:='+str(args.output/'online.wvmp')],stdout=log,stderr=subprocess.STDOUT)
    probe=None;error=None;checks=[];report={};started=time.monotonic()
    try:
        rclpy.init();probe=SurfaceProbe()
        probe.wait('empty online map has no surfaces',lambda:probe.metrics and probe.metrics[-1].message=='native_incremental_surface' and not probe.surfaces)
        checks.append('empty start, no environment preload')
        probe.depth_enabled=True
        probe.wait('real depth integrator produces triangle surface',lambda:any(m.points for m in probe.surfaces.values()))
        for m in probe.surfaces.values():
            assert m.type==Marker.TRIANGLE_LIST and m.header.frame_id=='odom'
            assert len(m.points)%3==0 and not m.colors and m.pose.orientation.w==1.
            assert all(math.isfinite(v) for q in m.points for v in (q.x,q.y,q.z))
            assert all(1.5<=q.x<=2.5 for q in m.points), 'surface outside actually observed synthetic wall range'
        assert not probe.cubes()
        checks.append('native-only wall geometry, TRIANGLE_LIST, normals-compatible winding/material and retired cubes')
        probe.depth_enabled=False;probe.spin(1.)
        saved=probe.save();shutil.copy2(saved,args.output/'observed-before-crop.wvmp')
        report['peak_triangles']=sum(len(m.points)//3 for m in probe.surfaces.values())
        probe.wait('new cyan geometry fades to stable blue-gray',lambda:probe.surfaces and all(abs(m.color.r-.32)<.001 for m in probe.surfaces.values()),seconds=8)
        checks.append('time-based cyan highlight fades without occupancy or geometry change')
        assert not probe.parameter('visualization.mesh_resolution_m',.15).successful
        assert not probe.parameter('visualization.surface_alpha',1.1).successful
        assert probe.parameter('visualization.update_radius_m',.2).successful
        probe.wait('display radius removes out-of-window geometry',lambda:not probe.surfaces)
        assert probe.parameter('visualization.update_radius_m',6.).successful
        probe.wait('re-entering display window rebuilds actual geometry',lambda:any(m.points for m in probe.surfaces.values()))
        checks.append('live parameter validation, radius deletion and rebuild')
        assert probe.parameter('visualization.mode','points').successful
        probe.wait('low cost mode has only native occupied points',lambda:not probe.surfaces and probe.clouds and probe.clouds[-1].width>0)
        assert not probe.cubes()
        assert probe.parameter('visualization.mode','surface').successful
        probe.wait('surface mode restored',lambda:any(m.points for m in probe.surfaces.values()))
        checks.append('low cost point mode without cubes or second map')
        probe.y=.4;probe.spin(.7);probe.y=.7
        probe.wait('real odometry trajectory extends',lambda:probe.paths and len(probe.paths[-1].poses)>=3 and abs(probe.paths[-1].poses[-1].pose.position.y-.7)<1e-6)
        expected=set(probe.surfaces);late=[]
        sub=probe.create_subscription(MarkerArray,'/visualization/surface_mesh',late.append,QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        probe.wait('late RViz receives complete periodic keyframe',lambda:any(any(m.action==Marker.DELETEALL for m in a.markers) and expected.issubset({(m.ns,m.id) for m in a.markers if m.action==Marker.ADD}) for a in late),seconds=6)
        probe.destroy_subscription(sub);checks.append('odom trajectory and complete late-join keyframes')
        probe.y=20.
        probe.wait('rolling crop deletes all old geometry',lambda:not probe.surfaces and probe.metrics[-1].message=='native_incremental_surface')
        probe.spin(1.);assert not probe.surfaces
        checks.append('cropped native blocks and dependent seam geometry removed; no historical occupancy retained')
        probe.body_enabled=False
        probe.wait('stale pose clears surfaces',lambda:probe.metrics[-1].message=='waiting_for_fresh_robot_pose')
        checks.append('stale-pose handling')
        report['metrics']=[{'message':m.message,'values':{v.key:v.value for v in m.values}} for m in probe.metrics]
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    finally:
        child.send_signal(signal.SIGINT)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()
        if probe:probe.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    if error is None:
        guard=subprocess.run([str(args.binary),'--ros-args','--params-file',str(config),'-p',
            'mapping.map_input_path:='+str(args.output/'observed-before-crop.wvmp')],capture_output=True,text=True,timeout=10)
        (args.output/'preload-rejection.log').write_text(guard.stdout+guard.stderr)
        if guard.returncode==0 or 'empty map_input' not in guard.stderr+guard.stdout:error='preloaded map was not rejected'
        else:checks.append('online rendering rejects preloaded environment maps')
    report.update(passed=error is None and child.returncode==0,error=error,node_returncode=child.returncode,checks=checks,
        duration_s=time.monotonic()-started,domain=os.environ.get('ROS_DOMAIN_ID'),
        scope='isolated synthetic sensor fixture; zero physical commands; not field VIO or mapping accuracy validation')
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','error','checks']},ensure_ascii=False,indent=2),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
