#!/usr/bin/env python3
"""Read-only live acceptance check for the single Wavemap fusion pipeline."""
import argparse
import json
from pathlib import Path
import sys
import time
import yaml

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from nbv_protocol import exchange


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir
    parameters=yaml.safe_load((out/'wavemap.yaml').read_text())['/**']['ros__parameters']
    map_topic=parameters.get('octomap_topic','/mapping/octomap')
    rclpy.init();node=Node('pelagos_map_acceptance_audit');states=[]
    node.create_subscription(Odometry,'/rov01/state/odometry',states.append,1)
    try:
        def sample():
            return {name:json.loads((out/(name+'.json')).read_text()) for name in
                ('mapping-status','map-source-status','planner-status','control-status')}
        first=sample();deadline=time.monotonic()+3.
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        second=sample()
        if not states:raise RuntimeError('No actual estimated body received')
        state=states[-1];p,q=state.pose.pose.position,state.pose.pose.orientation
        request=dict(run=out.name,epoch=1,sequence=time.monotonic_ns(),operation='inspect',
            stamp_ns=state.header.stamp.sec*10**9+state.header.stamp.nanosec,
            body=[p.x,p.y,p.z,q.x,q.y,q.z,q.w])
        native=exchange(out/'nbv.sock',request)
        endpoint=lambda e:dict(node=e.node_name,namespace=e.node_namespace,topic_type=e.topic_type)
        publishers=[endpoint(e) for e in node.get_publishers_info_by_topic(map_topic)]
        depth_topic=parameters.get('depth_topic','/stereo/depth')
        depth_subscribers=[endpoint(e) for e in node.get_subscriptions_info_by_topic(depth_topic)]
        display_depth_subscribers=[endpoint(e) for e in node.get_subscriptions_info_by_topic('/stereo/depth')]
        depth_publishers=[endpoint(e) for e in node.get_publishers_info_by_topic(depth_topic)]
        display_depth_publishers=[endpoint(e) for e in node.get_publishers_info_by_topic('/stereo/depth')]
        checks=dict(one_projection_publisher=len(publishers)==1 and publishers[0]['node'].startswith('underwater_wavemap'),
            mapper_uses_configured_depth_topic=any(e['node'].startswith('underwater_wavemap') for e in depth_subscribers),
            one_shared_depth_source=(len(depth_publishers)==len(display_depth_publishers)==1 and
                depth_publishers[0]==display_depth_publishers[0]),
            nbv_bridge_has_no_depth_subscription=all(e['node']!='pelagos_nbv_bridge'
                for e in depth_subscribers+display_depth_subscribers),
            real_fusion_advances=second['mapping-status']['integrated_frames']>first['mapping-status']['integrated_frames'],
            imported_revision_advances=second['map-source-status']['map_revision']>first['map-source-status']['map_revision'],
            external_map_ready=second['map-source-status']['ready'],
            native_depth_disabled=not native['depth_fusion_enabled'] and int(native['diagnostics']['native_depth_insertions'])==0,
            nonempty_query_map=int(native['diagnostics']['free_leaves'])>0 and int(native['diagnostics']['occupied_leaves'])>0,
            task_remains_paused=not first['planner-status']['enabled'] and not second['planner-status']['enabled'],
            guard_remains_disarmed=first['control-status']['state']==second['control-status']['state']=='DISARMED',
            resolution_matches_config=abs(float(native['diagnostics']['resolution_m'])-parameters['octomap_resolution_m'])<1.e-6)
        evidence=dict(checks=checks,projection_publishers=publishers,depth_topic=depth_topic,
            depth_publishers=depth_publishers,display_depth_publishers=display_depth_publishers,
            depth_subscribers=depth_subscribers,display_depth_subscribers=display_depth_subscribers,
            samples=[first,second],native_inspection=native,execution_authorized=False)
        (out/'wavemap-nbv-live-audit.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(checks))
        if not all(checks.values()):raise RuntimeError('Live acceptance checks failed')
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
