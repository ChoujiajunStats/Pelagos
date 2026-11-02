#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source "${DAPENG_ZED_WS:-$HOME/zed_ws}/install/setup.bash"
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zedx serial_number:=42312885 \
  ros_params_override_path:="$root_dir/Deployment/config/profiles/orin-nano/zed-x.yaml" \
  node_log_type:=screen
