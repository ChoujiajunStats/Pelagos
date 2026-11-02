#!/usr/bin/env bash
set -euo pipefail
robot=${DAPENG_JETSON:-nvidia@192.168.1.202}
remote_root=${DAPENG_REMOTE_ROOT:-/home/nvidia/Dapeng_Deployment}
viewer=${DAPENG_VNC_VIEWER:-$(command -v xtigervncviewer || command -v vncviewer || true)}
viewer_libs=$HOME/.local/share/dapeng-vnc-client/usr/lib/x86_64-linux-gnu
local_port=${DAPENG_VNC_PORT:-15901}
socket=${XDG_RUNTIME_DIR:-/tmp}/dapeng-rviz-ssh.sock
cache_dir=${XDG_CACHE_HOME:-$HOME/.cache}/dapeng-rviz
quality=${DAPENG_VNC_QUALITY:-9}
compression=${DAPENG_VNC_COMPRESSION:-1}
mapper=${DAPENG_MAPPER:-auto}
while (($#)); do
  case "$1" in
    --mapper) (($# >= 2)) || { echo '--mapper requires a value.' >&2; exit 2; }; mapper=$2; shift ;;
    --mapper=*) mapper=${1#*=} ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ $mapper == auto || $mapper == wavemap || $mapper == nvblox ]] || { echo 'Expected mapper auto|wavemap|nvblox.' >&2; exit 2; }

if [[ ! $quality =~ ^[0-9]$ || ! $compression =~ ^[0-9]$ ]]; then
  echo 'VNC quality and compression must be integers in [0, 9].' >&2
  exit 2
fi

ensure_remote_rviz() {
  local remote_command
  printf -v remote_command 'bash -s -- %q %q' "$remote_root" "$mapper"
  ssh -S "$socket" "$robot" "$remote_command" <<'REMOTE'
set -eo pipefail
root=$1
mapper=$2
info=$(python3 "$root/Deployment/application/services/mapper_backend.py" --root "$root")
read -r active_mapper active_navigation <<< "$(python3 -c 'import json,sys; i=json.loads(sys.argv[1]); print(i["mapper"] or "wavemap",int(i.get("navigation",False)))' "$info")"
[[ $mapper != auto ]] || mapper=$active_mapper
if [[ $mapper == wavemap && $active_navigation == 1 ]]; then mapper=wavemap_navigation; fi
running_mapper=
if systemctl --user is-active --quiet dapeng-rviz-remote.service; then
  pid=$(systemctl --user show dapeng-rviz-remote.service -p MainPID --value)
  running_mapper=$(python3 - "$pid" <<'PY'
from pathlib import Path
import sys
try:
    arguments=Path(f'/proc/{int(sys.argv[1])}/cmdline').read_bytes().decode().strip('\0').split('\0')
    index=next(i for i,value in enumerate(arguments) if value.endswith('/run_robot_viewer.sh'))
    print(arguments[index+1] if index+1<len(arguments) else 'nvblox')
except (OSError,StopIteration,ValueError):
    print('unknown')
PY
)
fi
if [[ $running_mapper != "$mapper" ]]; then
  if [[ $(systemctl --user show dapeng-rviz-remote.service -p LoadState --value) == loaded ]]; then
    systemctl --user stop dapeng-rviz-remote.service
  fi
  systemd-run --user --collect --unit=dapeng-rviz-remote \
    --property=KillMode=mixed --property=TimeoutStopSec=15 \
    --setenv=DISPLAY=:1 --setenv=XAUTHORITY=/home/nvidia/.Xauthority \
    --setenv=LIBGL_ALWAYS_SOFTWARE=1 --setenv=QT_X11_NO_MITSHM=1 \
    /usr/bin/bash "$root/Deployment/interfaces/visualization/scripts/run_robot_viewer.sh" "$mapper"
fi
echo "RViz backend: $mapper"
REMOTE
}

if ssh -S "$socket" -O check "$robot" >/dev/null 2>&1; then
  ensure_remote_rviz
  echo 'RViz ready in the existing VNC session.'
  exit 0
fi
if [[ ! -x $viewer ]]; then
  viewer=$(command -v xtigervncviewer || command -v vncviewer || true)
fi
if [[ -z $viewer || ! -x $viewer ]]; then
  echo 'Install tigervnc-viewer or set DAPENG_VNC_VIEWER.' >&2
  exit 1
fi
mkdir -p "$cache_dir"
chmod 700 "$cache_dir"
ssh -M -S "$socket" -fNT -o ControlPersist=no -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:$local_port:127.0.0.1:5901" "$robot"
cleanup() {
  ssh -S "$socket" -O exit "$robot" >/dev/null 2>&1 || true
}
trap cleanup EXIT
ensure_remote_rviz
scp -q -o "ControlPath=$socket" "$robot:.vnc/passwd" "$cache_dir/vnc.passwd"
chmod 600 "$cache_dir/vnc.passwd"
echo 'Remote RViz ready.'
echo "VNC quality=$quality compression=$compression"
LD_LIBRARY_PATH="$viewer_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$viewer" -Shared -RemoteResize=1 -DesktopSize=1440x900 \
  -AutoSelect=0 -PreferredEncoding=Tight -FullColor=1 \
  -QualityLevel="$quality" -CompressLevel="$compression" \
  -SecurityTypes=VncAuth -passwd "$cache_dir/vnc.passwd" \
  -geometry 1440x900 "127.0.0.1::$local_port"
