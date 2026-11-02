#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
robot=${DAPENG_ROBOT_SSH:-nvidia@192.168.1.202}
model_python=${DAPENG_JEPA_PYTHON:-$root/venv-jepa/bin/python}
python3 - <<'PY'
import urllib.request,json
s=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/autonomy/state',timeout=2))
if s['mode']=='auto':raise SystemExit('Select manual/shadow before restarting the controller')
PY
rsync -a --exclude=__pycache__ "$root/Deployment/contexts/control/" \
  "$robot:/home/nvidia/Dapeng_Deployment/Deployment/contexts/control/"
ssh -o BatchMode=yes "$robot" 'bash -s' <<'SH'
set -e
systemctl --user stop dapeng-control-gateway.service 2>/dev/null || true
systemd-run --user --collect --unit=dapeng-control-gateway \
  --property=MemoryMax=160M --property=CPUQuota=35% \
  /usr/bin/bash /home/nvidia/Dapeng_Deployment/Deployment/contexts/control/scripts/run_robot_gateway.sh
SH
systemctl --user stop dapeng-jepa-controller.service dapeng-control-tunnel.service 2>/dev/null || true
systemd-run --user --collect --unit=dapeng-control-tunnel \
  /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlPath=none -o ControlMaster=no \
  -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18781:127.0.0.1:18781 "$robot"
systemd-run --user --collect --unit=dapeng-jepa-controller --property=MemoryMax=3G \
  --property=CPUQuota=150% --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 --setenv=PYTHONNOUSERSITE=1 \
  "$model_python" -u "$root/Deployment/contexts/control/services/runtime.py"
