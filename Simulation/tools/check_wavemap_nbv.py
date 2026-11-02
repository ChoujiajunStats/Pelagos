#!/usr/bin/env python3
"""Offline full-map import/replacement checks; never connect to a live vehicle."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from nbv_protocol import exchange


def payload(path):
    header,raw=path.read_bytes().split(b'data\n',1)
    if b'id OcTree\n' not in header or b'res 0.4\n' not in header:
        raise ValueError('Expected full 0.4 m OcTree file')
    return raw


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir.resolve();source=args.source_run.resolve()
    if not (out/'offline-source.json').is_file() or (out/'resolved_config.yaml').exists():
        raise RuntimeError('Requires a dedicated offline backend, without a vehicle/pipeline')
    body=json.loads((source/'planner-preview-after-user-map.json').read_text())['requested_body']
    raw=payload(out/'wavemap-projection.ot')
    digest=lambda b:hashlib.sha256(b).hexdigest()
    evidence=dict(execution_authorized=False,source_run=source.name,historical_body=body,
        source_wavemap_sha256=digest((source/'map.wvmp').read_bytes()),
        octomap_payload_sha256=digest(raw),octomap_bytes=len(raw),checks={})
    def save():
        (out/'import-checks.json').write_text(json.dumps(evidence,indent=2))
    def call(operation,**kw):
        request=dict(run=out.name,epoch=1,sequence=time.monotonic_ns(),stamp_ns=1_000_000_000,
            operation=operation,body=body)
        request.update(kw)
        return exchange(out/'nbv.sock',request)
    def map_args(data,revision):
        return dict(map_frame='rov01/odom',map_binary=False,map_id='OcTree',map_resolution=.4,
            source_revision=revision,map_data_b64=base64.b64encode(data).decode(),map_crc32=zlib.crc32(data))
    def check(name,condition,result):
        evidence['checks'][name]=dict(passed=bool(condition),response=result);save()
        if not condition:raise AssertionError(name)
    imported=call('replace_map',**map_args(raw,1))
    check('full_import',imported['valid'] and not imported['depth_fusion_enabled'],imported)
    first=call('inspect');d=first['diagnostics']
    check('nonempty_free_and_occupied',int(d['free_leaves'])>0 and int(d['occupied_leaves'])>0,first)
    check('signed_bytes_and_probabilities_roundtrip',payload(out/'nbv-map.ot')==raw,{})
    second=call('inspect')
    check('inspect_does_not_mutate_live_map',payload(out/'nbv-map.ot')==raw,second)
    ignored=call('observe',points_f32_le='')
    check('depth_fusion_disabled',ignored['valid'] and ignored['map_revision']==imported['map_revision'] and
        not ignored['depth_fusion_enabled'],ignored)
    for name,changes in (
        ('reject_wrong_frame',dict(map_frame='world')),
        ('reject_wrong_resolution',dict(map_resolution=.1)),
        ('reject_binary',dict(map_binary=True)),
        ('reject_wrong_tree_type',dict(map_id='ColorOcTree')),
        ('reject_checksum_mismatch',dict(map_crc32=zlib.crc32(raw)^1))):
        kwargs=map_args(raw,2);kwargs.update(changes)
        answer=call('replace_map',**kwargs)
        check(name,not answer['valid'],answer)
    stale=call('replace_map',**map_args(b'',1))
    check('old_revision_does_not_clear_map',stale['valid'] and stale['map_revision']==imported['map_revision'],stale)
    cleared=call('replace_map',**map_args(b'',2));empty=call('inspect')
    check('replacement_removes_old_free_and_occupied',cleared['valid'] and
        int(empty['diagnostics']['free_leaves'])==0 and int(empty['diagnostics']['occupied_leaves'])==0 and
        empty['diagnostics']['root_point']=='unknown',empty)
    restored=call('replace_map',**map_args(raw,3));again=call('inspect')
    check('restored_snapshot_matches_source',restored['valid'] and payload(out/'nbv-map.ot')==raw and
        int(again['diagnostics']['native_depth_insertions'])==0,again)
    call('reset_execution')
    started=time.monotonic();plan=call('plan');plan['wall_seconds']=time.monotonic()-started
    evidence['historical_root_plan']=plan;save()
    print(json.dumps(dict(passed=len(evidence['checks']),map_bytes=len(raw),
        native_depth_insertions=again['diagnostics']['native_depth_insertions'],
        plan_valid=plan['valid'],plan_reason=plan['reason'],path_points=len(plan['path'])),ensure_ascii=False))


if __name__=='__main__':main()
