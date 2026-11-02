#!/usr/bin/env python3
"""Approved planning-only comparison on the same measured OctoMap and root."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from nbv_protocol import exchange


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir.resolve()
    import yaml
    if not yaml.safe_load((out/'resolved_config.yaml').read_text()).get('planning_only'):
        raise RuntimeError('This experiment requires a planning-only run without control processes')
    component=json.loads((out/'planner-component-test.json').read_text())
    evidence=dict(run_id=out.name,execution_authorized=False,
        approval='2026-10-30 user approved exact measured-map sampling-window contrast',
        snapshot_sha256=hashlib.sha256((out/'nbv-frozen.bt').read_bytes()).hexdigest(),
        actual_body=component['actual_body'],test_root=component['test_root'],
        note='Virtual component-test root is not the actual vehicle pose. No motion is sent.',results={})
    def request(operation,body):
        value=dict(run=out.name,epoch=1,sequence=time.monotonic_ns(),operation=operation,
            task_revision=0,stamp_ns=0,body=body)
        started=time.monotonic();result=exchange(out/'nbv.sock',value)
        result['wall_seconds']=time.monotonic()-started
        return result
    def save():
        (out/'planner-map-ab.json').write_text(json.dumps(evidence,indent=2))
    root=component['test_root']
    loaded=request('load_snapshot',root);evidence['snapshot_load']=loaded;save()
    if not loaded['valid']:raise RuntimeError(loaded['reason'])
    inspected=request('inspect',root);evidence['frozen_inspect']=inspected;save()
    d=inspected['diagnostics']
    actual_bounds=list(map(float,d['map_min']))+list(map(float,d['map_max']))
    expected=[0.,-3.2,-2.8,5.2,3.2,2.8]
    evidence['frozen_map_bounds']=actual_bounds
    evidence['approved_sampling_window']=expected
    save()
    for name,operation,body in (
        ('global_virtual_root','plan',root),
        ('map_window_virtual_root','plan_map_window',root),
        ('map_window_actual_root','plan_map_window',component['actual_body'])):
        reset=request('reset_execution',body)
        if not reset['valid']:raise RuntimeError(reset['reason'])
        result=request(operation,body);result['requested_body']=body
        evidence['results'][name]=result;save()
        diagnostic=result.get('diagnostics',{})
        print(json.dumps(dict(case=name,valid=result['valid'],reason=result['reason'],
            seconds=result['wall_seconds'],nodes=diagnostic.get('accepted_tree_nodes'),
            path_points=len(result.get('path',[])),positive_gain=diagnostic.get('positive_gain_found'),
            root_unknown_samples=diagnostic.get('root_unknown_samples')),ensure_ascii=False),flush=True)
    # Keep the snapshot frozen for review; live fusion requires explicit unfreeze.
    print(f'Evidence: {out}/planner-map-ab.json',flush=True)


if __name__=='__main__':main()
