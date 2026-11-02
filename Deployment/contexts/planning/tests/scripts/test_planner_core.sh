#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /ws/install/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11332 ROS_IP=127.0.0.1
roscore -p 11332 >/tmp/omniplanner-test-master.log 2>&1 &
master_pid=$!
planner_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$planner_pid" ]]; then kill -INT "$planner_pid" 2>/dev/null || true; fi
  kill -INT "$master_pid" 2>/dev/null || true
  sleep 1
  if [[ -n "$planner_pid" ]]; then kill -TERM "$planner_pid" 2>/dev/null || true; fi
  wait || true
}
trap cleanup EXIT INT TERM
for attempt in {1..40}; do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep .25
done
rosparam load /config/planner.yaml /dapeng_omniplanner
rosrun gbplanner dapeng_omniplanner_node odometry:=/omniplanner/odometry \
  local_navigation_goal:=/omniplanner/local_goal nvblox_esdf:=/omniplanner/esdf \
  >/tmp/omniplanner-test-core.log 2>&1 &
planner_pid=$!
python3 /integration/contexts/planning/tests/unit/test_planner_core.py
