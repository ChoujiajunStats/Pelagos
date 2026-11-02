#!/usr/bin/env bash
set -euo pipefail
planning_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
robot_target=${DAPENG_ROBOT_SSH:-nvidia@192.168.1.202}
ssh_socket=${DAPENG_SSH_SOCKET:-/tmp/dapeng-orin-deploy.sock}
ssh_options=(-o BatchMode=yes -o ConnectTimeout=5 -o "ControlPath=$ssh_socket")
backend_info=$(ssh "${ssh_options[@]}" "$robot_target" \
  'python3 /home/nvidia/Dapeng_Deployment/Deployment/application/services/mapper_backend.py --format shell')
read -r active_unit active_mapper active_state <<< "$backend_info"
if [[ $active_state != active || ! $active_mapper =~ ^(nvblox|wavemap)$ ]]; then
  echo 'An active Wavemap or nvblox pipeline is required.' >&2
  exit 1
fi
# An integrated Wavestar node already owns /planning/*; avoid two planners.
navigation_active=$(ssh "${ssh_options[@]}" "$robot_target" \
  'python3 /home/nvidia/Dapeng_Deployment/Deployment/application/services/mapper_backend.py --format json')
if [[ $navigation_active == *'"navigation": true'* ]]; then
  echo 'Integrated Wavestar is active. Use navigation/set_goal.sh.' >&2
  exit 1
fi
docker image inspect dapeng/omniplanner:20261021 >/dev/null
bash "$planning_dir/scripts/stop_preview.sh"
rsync -a --exclude=bin --exclude=__pycache__ -e "ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlPath=$ssh_socket" \
  "$planning_dir/" "$robot_target:/home/nvidia/Dapeng_Deployment/Deployment/contexts/planning/"
if [[ $active_mapper == wavemap ]]; then
  ssh "${ssh_options[@]}" "$robot_target" \
    'set -e; export PYTHONNOUSERSITE=1; python3 -c "from scipy.ndimage import distance_transform_edt"; bash /home/nvidia/Dapeng_Deployment/Deployment/contexts/planning/scripts/build_wavemap_export.sh'
fi
ssh "${ssh_options[@]}" "$robot_target" bash -s -- "$active_mapper" <<'SH'
set -e
systemctl --user stop dapeng-omniplanner-bridge.service 2>/dev/null || true
systemd-run --user --collect --unit=dapeng-omniplanner-bridge \
  --property=MemoryMax=256M --property=MemorySwapMax=64M --property=CPUQuota=50% \
  /usr/bin/bash /home/nvidia/Dapeng_Deployment/Deployment/contexts/planning/scripts/run_robot_bridge.sh --backend "$1"
SH
systemctl --user stop dapeng-omniplanner-host.service dapeng-omniplanner-tunnel.service 2>/dev/null || true
ssh "${ssh_options[@]}" -O cancel -L 127.0.0.1:18771:127.0.0.1:18771 "$robot_target" 2>/dev/null || true
systemd-run --user --collect --unit=dapeng-omniplanner-tunnel \
  /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlPath=none -o ControlMaster=no \
  -N -T -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -L 127.0.0.1:18771:127.0.0.1:18771 "$robot_target"
systemd-run --user --collect --unit=dapeng-omniplanner-host --property=TimeoutStopSec=20 \
  /usr/bin/docker run --rm --name dapeng-omniplanner --network=host --memory=2g \
  --cpus=2 --pids-limit=512 --log-opt=max-size=20m --log-opt=max-file=2 \
  -v "$planning_dir/../..:/integration:ro" -v "$planning_dir/config/planner.yaml:/config/planner.yaml:ro" \
  dapeng/omniplanner:20261021 /bin/bash /integration/contexts/planning/scripts/run_host_container.sh
