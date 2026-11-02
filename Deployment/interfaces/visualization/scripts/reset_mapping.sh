#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
if [[ -v DAPENG_JETSON ]]; then robot=$DAPENG_JETSON; else robot=nvidia@192.168.1.202; fi
if [[ -v DAPENG_REMOTE_ROOT ]]; then remote_root=$DAPENG_REMOTE_ROOT; else remote_root=/home/nvidia/Dapeng_Deployment; fi

health=$(curl -fsS --max-time 3 http://127.0.0.1:8765/health)
armed=$(printf '%s' "$health" | jq -r '.control.armed // empty')
if [[ $armed != false ]]; then
  echo 'DISARM required before resetting mapping.' >&2
  exit 1
fi

if ! docker image inspect osrf/ros:humble-desktop >/dev/null 2>&1; then
  echo 'Missing image: osrf/ros:humble-desktop' >&2
  exit 1
fi

echo 'Saving map and restarting VIO/nvblox.'
ssh -T -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes "$robot" \
  bash -s -- "$remote_root" <<'REMOTE'
set -euo pipefail
root=$1
source /opt/ros/humble/setup.bash

if systemctl --user is-active --quiet hmu-stereo.service; then
  echo 'Camera is owned by hmu-stereo.service.' >&2
  exit 1
fi
if ! systemctl --user is-active --quiet dapeng-nvblox.service; then
  echo 'dapeng-nvblox.service is not running.' >&2
  exit 1
fi

current_run=
for manifest in "$root"/runs/*/manifest.json; do
  [[ -f $manifest ]] || continue
  if [[ $(jq -r '.mapper // empty' "$manifest") == nvblox ]] &&
     [[ $(jq -r '.status // empty' "$manifest") == running ]]; then
    current_run=${manifest%/manifest.json}
  fi
done
if [[ -z $current_run ]]; then
  echo 'Active nvblox run not found.' >&2
  exit 1
fi

map_path="$current_run/map-before-reset-$(date -u +%Y%m%dT%H%M%SZ).nvblx"
save_output=$(mktemp)
if timeout 50 ros2 service call /nvblox_node/save_map nvblox_msgs/srv/FilePath \
    "{file_path: $map_path}" | tee "$save_output" &&
   grep -Eq 'success(:|=)[[:space:]]*(true|True)' "$save_output" &&
   [[ -s $map_path ]]; then
  echo "Map saved: $map_path"
else
  echo 'WARNING: Map save failed; reset will continue.'
  cat "$save_output"
fi
rm -f "$save_output"

echo 'Restarting VIO, LAS2, pose adapter and nvblox.'
systemctl --user restart dapeng-nvblox.service

new_run=
for _ in $(seq 1 90); do
  for manifest in "$root"/runs/*/manifest.json; do
    [[ -f $manifest ]] || continue
    if [[ $(jq -r '.mapper // empty' "$manifest") == nvblox ]] &&
       [[ $(jq -r '.status // empty' "$manifest") == running ]]; then
      candidate=${manifest%/manifest.json}
      [[ $candidate != "$current_run" ]] && new_run=$candidate
    fi
  done
  [[ -n $new_run ]] && break
  sleep 1
done
if [[ -z $new_run ]]; then
  echo 'nvblox startup timed out after 90s.' >&2
  exit 1
fi
echo "Run directory: $new_run"
REMOTE

if docker container inspect dapeng-rviz-live >/dev/null 2>&1; then
  docker rm -f dapeng-rviz-live >/dev/null
fi
RVIZ_CONFIG=nvblox.rviz bash "$root_dir/Deployment/interfaces/visualization/scripts/start_rviz.sh"
echo 'Mapping reset complete.'
