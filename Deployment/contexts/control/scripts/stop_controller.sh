#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
python3 "$root/Deployment/contexts/control/services/controllerctl.py" shadow || true
systemctl --user stop dapeng-jepa-controller.service dapeng-control-tunnel.service
ssh -o BatchMode=yes -o ConnectTimeout=5 "${DAPENG_ROBOT_SSH:-nvidia@192.168.1.202}" \
  systemctl --user stop dapeng-control-gateway.service
