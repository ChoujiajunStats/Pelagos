#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: bash Deployment/contexts/navigation/scripts/set_goal.sh x y z (odom frame, meters)' >&2; exit 2; }
robot=${DAPENG_JETSON:-nvidia@192.168.1.202}
remote_root=${DAPENG_REMOTE_ROOT:-/home/nvidia/Dapeng_Deployment}
printf -v remote_command 'bash -s -- %q %q %q %q' "$remote_root" "$1" "$2" "$3"
ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$robot" "$remote_command" <<'REMOTE'
set -eo pipefail
root=$1
shift
source /opt/ros/humble/setup.bash
exec python3 "$root/Deployment/contexts/navigation/adapters/publish_goal.py" "$@"
REMOTE
