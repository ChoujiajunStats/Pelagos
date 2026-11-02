#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
python3 - <<'PY'
import json,urllib.request
s=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=2))
if s['control']['armed'] or s['control']['arm_pending']:
    raise SystemExit('HMU is ARM: operator DISARM is required before loading the bridge extension.')
if s['rov']['command_protocol']!='channel':raise SystemExit('Expected the restored manufacturer channel protocol')
PY
mkdir -p "$root/runs/20261025-jepa-controller/backup" "$HOME/.config/systemd/user/hmu-bridge.service.d"
if [[ ! -e $root/runs/20261025-jepa-controller/backup/hmu-bridge-unit.txt ]]; then
  systemctl --user cat hmu-bridge.service > "$root/runs/20261025-jepa-controller/backup/hmu-bridge-unit.txt"
fi
cat > "$HOME/.config/systemd/user/hmu-bridge.service.d/80-dapeng-controller.conf" <<EOF
[Service]
ExecStart=
ExecStart=$root/venv-hmu/bin/python $root/Deployment/contexts/control/adapters/hmu_extension.py --config $root/Deployment/config/hmu/default.toml
EOF
systemctl --user daemon-reload
systemctl --user restart hmu-bridge.service
python3 - <<'PY'
import json,time,urllib.request,urllib.error
deadline=time.monotonic()+8
while time.monotonic()<deadline:
    try:
        state=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/autonomy/state',timeout=.5))
        if state.get('autonomy_protocol')=='motor' and state.get('mode')=='shadow':break
    except (OSError,ValueError):pass
    time.sleep(.1)
else:raise SystemExit('HMU extension did not become ready; inspect hmu-bridge.service')
PY
