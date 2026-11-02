#!/usr/bin/env python3
"""Copy the requested BlueROV hull/propeller assets and assemble a six-thruster visual."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args();src=args.source/'source';dest=args.destination;dest.mkdir(parents=True,exist_ok=True)
    scene=(src/'bluerov2.scn').read_text();root=ET.fromstring(scene)
    horizontal=['FrontRight','FrontLeft','BackRight','BackLeft'];thrusters=[]
    for name in horizontal:
        a=root.find(f'.//actuator[@name="{name}"]');o=a.find('origin');mesh=a.find('propeller/mesh')
        thrusters.append(dict(name=name,mesh=Path(mesh.attrib['filename']).name,
            xyz=[float(x) for x in o.attrib['xyz'].split()],rpy=[float(x) for x in o.attrib['rpy'].split()],kind='horizontal'))
    note=scene.split('Bluerov 6 thrusters Z',1)[1].split('-->',2)[1]
    origins=re.findall(r'<origin xyz="([^"]+)" rpy="([^"]+)"',note)
    if len(origins)!=2:raise ValueError('expected exactly two source six-thruster vertical origins')
    for name,(xyz,rpy),mesh in zip(['VerticalLeft','VerticalRight'],origins,['ccw.obj','cw.obj']):
        thrusters.append(dict(name=name,mesh=mesh,xyz=[float(x) for x in xyz.split()],rpy=[float(x) for x in rpy.split()],kind='vertical'))
    assert len(thrusters)==6
    hashes={}
    for name in ['bluerov2.obj','ccw.obj','cw.obj','br2.png']:
        raw=(src/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        if name=='bluerov2.obj':
            # The supplied OBJ has UVs but no material declaration. Bind its supplied texture.
            (dest/name).write_bytes(b'mtllib hull.mtl\nusemtl hull\n'+raw)
        else:shutil.copy2(src/name,dest/name)
    (dest/'hull.mtl').write_text('newmtl hull\nKa 0.5 0.5 0.5\nKd 1 1 1\nKs 0.1 0.1 0.1\nNs 24\nd 1\nmap_Kd br2.png\n')
    for mesh,material in [('cw.obj','ID69'),('ccw.obj','ID955')]:
        p=dest/mesh;text=p.read_text();text=re.sub(r'^mtllib .*\n','',text,flags=re.M)
        p.write_text('mtllib propeller.mtl\n'+text)
    (dest/'propeller.mtl').write_text(''.join(f'newmtl {name}\nKa 0.05 0.21 0.34\nKd 0 0.42 0.68\nKs 0.15 0.15 0.15\nNs 24\nd 1\n\n' for name in ['ID69','ID955']))
    assembly=dict(hull='bluerov2.obj',thrusters=thrusters,scale_m=1.,source=str(args.source),source_sha256=hashes,
        note='Source USD is eight-thruster Heavy. Omit HeavyFrame and its four vertical propellers; use two vertical positions from the source six-thruster note. Horizontal positions follow the supplied hull mesh/active SCN coordinate convention. Visual only; no physics or control changes.')
    (dest/'assembly.json').write_text(json.dumps(assembly,indent=2)+'\n')
    shutil.copy2(args.source/'LICENSE',dest/'LICENSE')
    shutil.copy2(args.source/'THIRD_PARTY.md',dest/'UPSTREAM_PROVENANCE.md')
    print(json.dumps(dict(thrusters=len(thrusters),horizontal=4,vertical=2,destination=str(dest)),indent=2))


if __name__=='__main__':main()
