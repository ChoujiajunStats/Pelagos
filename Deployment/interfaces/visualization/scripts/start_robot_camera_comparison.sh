#!/usr/bin/env bash
# Start a display-only image comparison; the running VIO and map are retained.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
robot=${DAPENG_JETSON:-nvidia@192.168.1.202}
remote_root=${DAPENG_REMOTE_ROOT:-/home/nvidia/Dapeng_Deployment}
printf -v command 'bash -s -- %q' "$remote_root"
ssh "$robot" "$command" <<'REMOTE'
set -eo pipefail
root=$1
python3 - "$root" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'Deployment/application'))
from mapper_backend import inspect_backend
info=inspect_backend(sys.argv[1])
if info['mapper']!='wavemap' or not info['navigation']:
    raise SystemExit('An active integrated Wavemap navigation pipeline is required.')
PY
REMOTE
rsync -a --exclude=__pycache__ "$root/Deployment/interfaces/visualization/" "$robot:$remote_root/Deployment/interfaces/visualization/"
ssh "$robot" "$command" <<'REMOTE'
set -eo pipefail
root=$1
calibration=$(python3 - "$root" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
for cmd in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=cmd.read_bytes().decode().strip('\0').split('\0')
    except (OSError,UnicodeError): continue
    if any(a.endswith('/wavemap_navigation_node') for a in args):
        for a in args:
            if a.startswith('mapping.calibration_path:='):
                p=Path(a.split(':=',1)[1]).resolve()
                if p.is_relative_to(root/'runs') and p.is_file():
                    print(p);raise SystemExit(0)
raise SystemExit('Frozen calibration not found for the active run.')
PY
)
source /opt/ros/humble/setup.bash
"$root/venv-las2/bin/python" -c 'import cv2,numpy,rclpy'
if systemctl --user is-active --quiet dapeng-camera-comparison.service; then
  systemctl --user stop dapeng-camera-comparison.service
fi
systemd-run --user --collect --unit=dapeng-camera-comparison \
  --property=KillMode=mixed --property=TimeoutStopSec=10 \
  /usr/bin/bash -c 'source /opt/ros/humble/setup.bash; exec "$1/venv-las2/bin/python" -u "$1/Deployment/interfaces/visualization/adapters/camera_map_overlay.py" --calibration "$2"' bash "$root" "$calibration"
# Restart only the independent viewer to load the new Image display.
systemctl --user stop dapeng-rviz-remote.service
systemd-run --user --collect --unit=dapeng-rviz-remote \
  --property=KillMode=mixed --property=TimeoutStopSec=15 \
  --setenv=DISPLAY=:1 --setenv=XAUTHORITY=/home/nvidia/.Xauthority \
  --setenv=LIBGL_ALWAYS_SOFTWARE=1 --setenv=QT_X11_NO_MITSHM=1 \
  --setenv=DAPENG_RVIZ_CAMERA_COMPARISON=1 \
  /usr/bin/bash "$root/Deployment/interfaces/visualization/scripts/run_robot_viewer.sh" wavemap_navigation
echo 'Camera comparison enabled.'
REMOTE
