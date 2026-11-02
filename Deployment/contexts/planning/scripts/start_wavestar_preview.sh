#!/usr/bin/env bash
# Planner-only Wavestar preview: keep the current camera, VIO, LAS2 and Wavemap.
set -euo pipefail
planning_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root_dir=$(cd "$planning_dir/../../.." && pwd)
robot=${DAPENG_ROBOT_SSH:-nvidia@192.168.1.202}
info=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$robot" \
  'python3 /home/nvidia/Dapeng_Deployment/Deployment/application/services/mapper_backend.py')
python3 - "$info" <<'PY'
import json,sys
i=json.loads(sys.argv[1])
if i['mapper']!='wavemap' or i['state']!='active' or i.get('navigation'):
    raise SystemExit('An active standalone Wavemap pipeline is required.')
PY
build_dir=${DAPENG_NAVIGATION_BUILD:-$root_dir/runs/20261022-wavestar-integration/build-native}
if [[ ! -f $build_dir/CMakeCache.txt ]]; then
  cmake -S "$root_dir/Deployment/contexts/navigation/core" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$build_dir" --target offline_check -j2
python3 -c 'import numpy,yaml'
rsync -a --exclude=bin --exclude=__pycache__ "$planning_dir/" \
  "$robot:/home/nvidia/Dapeng_Deployment/Deployment/contexts/planning/"
ssh -o BatchMode=yes "$robot" \
  'bash /home/nvidia/Dapeng_Deployment/Deployment/contexts/planning/scripts/build_wavemap_export.sh'
bash "$planning_dir/scripts/stop_preview.sh"
ssh -o BatchMode=yes "$robot" 'bash -s' <<'SH'
set -e
systemd-run --user --collect --unit=dapeng-wavestar-preview-bridge \
  --property=MemoryMax=256M --property=MemorySwapMax=64M --property=CPUQuota=50% \
  /usr/bin/bash /home/nvidia/Dapeng_Deployment/Deployment/contexts/planning/scripts/run_robot_bridge.sh \
  --backend wavemap --planner wavestar
SH
systemd-run --user --collect --unit=dapeng-wavestar-preview-tunnel \
  /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlPath=none -o ControlMaster=no \
  -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18771:127.0.0.1:18771 "$robot"
systemd-run --user --collect --unit=dapeng-wavestar-preview-host \
  --property=MemoryMax=512M --property=MemorySwapMax=128M --property=CPUQuota=150% \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  /usr/bin/python3 -u "$planning_dir/services/wavestar_host.py" \
  --binary "$build_dir/offline_check" --robot-config "$planning_dir/config/planner.yaml" \
  --navigation-config "$root_dir/Deployment/contexts/navigation/config/navigation.yaml"
