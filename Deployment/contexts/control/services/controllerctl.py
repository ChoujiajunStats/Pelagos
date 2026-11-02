#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Inspect the HMU manual/shadow/auto mux; this tool never arms the vehicle."""
import argparse
import json
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('status','manual','shadow','auto','neutral-probe'),default='status',nargs='?')
    args=p.parse_args();payload=None if args.action=='status' else json.dumps(dict(mode=args.action)).encode()
    route='state' if payload is None else 'mode'
    if args.action=='neutral-probe':route='neutral_probe';payload=b'{}'
    request=urllib.request.Request('http://127.0.0.1:8765/api/autonomy/'+route,data=payload,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=2) as response:print(json.dumps(json.load(response),indent=2,ensure_ascii=False))
    except urllib.error.HTTPError as e:raise SystemExit(e.read().decode())


if __name__=='__main__':main()
