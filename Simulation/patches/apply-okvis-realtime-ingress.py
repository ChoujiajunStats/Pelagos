#!/usr/bin/env python3
"""Apply the reviewed ROS scheduling patch only to an exact build-copy snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def apply(source):
    package=Path(__file__).resolve().parent
    manifest=json.loads((package/'okvis-realtime-ingress-sha256.json').read_text())
    patch=package/'okvis-realtime-ingress.patch'
    changed=subprocess.check_output(['git','apply','--numstat',str(patch)],text=True)
    paths={line.split('\t',2)[2] for line in changed.splitlines()}
    if paths!=set(manifest['files']):raise RuntimeError('Patch paths differ from reviewed manifest')
    def verify(stage):
        for name,expected in manifest['files'].items():
            actual=hashlib.sha256((source/name).read_bytes()).hexdigest()
            if actual!=expected[stage+'_sha256']:
                raise RuntimeError('OKVIS '+stage+' SHA256 mismatch: '+name)
    verify('before')
    subprocess.run(['git','apply','--check',str(patch)],cwd=source,check=True)
    subprocess.run(['git','apply',str(patch)],cwd=source,check=True)
    verify('after')
    print('Applied exact OKVIS ROS ingress patch; algorithms, parameters and acquisition stamps retained')


if __name__=='__main__':apply(Path(sys.argv[1]).resolve())
