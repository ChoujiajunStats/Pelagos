#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash

python3 /configuration/adapters/rviz_pose.py &
pose_pid=$!
rviz2 -d "/configuration/rviz/${RVIZ_CONFIG:-nvblox.rviz}" &
rviz_pid=$!

cleanup() {
  trap - INT TERM EXIT
  kill -INT "$pose_pid" "$rviz_pid" 2>/dev/null || true
  wait "$pose_pid" "$rviz_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait -n "$pose_pid" "$rviz_pid"
