#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /ws/install/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11331
export ROS_IP=127.0.0.1
roscore -p 11331 >/tmp/omniplanner-roscore.log 2>&1 &
master_pid=$!
planner_pid=""
bridge_pid=""
cleanup() {
  trap - INT TERM EXIT
  for pid in "$bridge_pid" "$planner_pid" "$master_pid"; do
    if [[ -n "$pid" ]]; then kill -INT "$pid" 2>/dev/null || true; fi
  done
  sleep 1
  for pid in "$bridge_pid" "$planner_pid" "$master_pid"; do
    if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  wait || true
}
trap cleanup INT TERM EXIT
for attempt in {1..40}; do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep .25
done
rosparam load /config/planner.yaml /dapeng_omniplanner
rosrun gbplanner dapeng_omniplanner_node odometry:=/omniplanner/odometry \
  local_navigation_goal:=/omniplanner/local_goal nvblox_esdf:=/omniplanner/esdf \
  >/tmp/omniplanner-core.log 2>&1 &
planner_pid=$!
python3 -u /integration/contexts/planning/adapters/host_bridge.py &
bridge_pid=$!
wait -n "$planner_pid" "$bridge_pid" "$master_pid"
