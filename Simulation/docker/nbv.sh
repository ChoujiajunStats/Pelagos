#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /opt/pelagos_nbv_ws/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1
mkdir -p "$PELAGOS_RUN/logs"
exec python3 /pelagos/Simulation/nbv/server.py "$PELAGOS_RUN" >"$PELAGOS_RUN/logs/nbv_transport.log" 2>&1
