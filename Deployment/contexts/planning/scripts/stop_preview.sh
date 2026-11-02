#!/usr/bin/env bash
set -euo pipefail
robot_target=${DAPENG_ROBOT_SSH:-nvidia@192.168.1.202}
ssh_socket=${DAPENG_SSH_SOCKET:-/tmp/dapeng-orin-deploy.sock}
ssh -T -o BatchMode=yes -o ConnectTimeout=5 -o "ControlPath=$ssh_socket" "$robot_target" 'bash -s' <<'REMOTE'
set -e
if systemctl --user is-active --quiet dapeng-omniplanner-bridge.service || systemctl --user is-active --quiet dapeng-wavestar-preview-bridge.service; then
  source /opt/ros/humble/setup.bash
  timeout 4 ros2 service call /omniplanner/clear std_srvs/srv/Trigger '{}' >/dev/null 2>&1 || true
fi
for unit in dapeng-omniplanner-bridge.service dapeng-wavestar-preview-bridge.service; do
  if [[ $(systemctl --user show "$unit" -p LoadState --value) == loaded ]]; then
    systemctl --user stop "$unit"
  fi
done
REMOTE
systemctl --user stop dapeng-omniplanner-host.service dapeng-omniplanner-tunnel.service \
  dapeng-wavestar-preview-host.service dapeng-wavestar-preview-tunnel.service 2>/dev/null || true
