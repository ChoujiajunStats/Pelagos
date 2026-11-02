#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec ros2 run nvblox_ros nvblox_node --ros-args \
  --params-file "${1:-$root_dir/Deployment/config/profiles/orin-nano/nvblox.yaml}" \
  -p use_sim_time:="${DAPENG_USE_SIM_TIME:-false}" \
  -r camera_0/depth/image:=/stereo/depth \
  -r camera_0/depth/camera_info:=/stereo/camera_info \
  -r /tf:=/nvblox/tf -r /tf_static:=/nvblox/tf_static
