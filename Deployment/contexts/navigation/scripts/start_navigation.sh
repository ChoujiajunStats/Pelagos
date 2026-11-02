#!/usr/bin/env bash
set -eo pipefail
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
source /opt/ros/humble/setup.bash
source "$root_dir/Deployment/infrastructure/ros2_ws/install-navigation/setup.bash"
exec "$root_dir/Deployment/infrastructure/ros2_ws/install-navigation/underwater_navigation_ros/lib/underwater_navigation_ros/wavemap_navigation_node" \
  --ros-args --params-file "${2:?Frozen navigation.yaml required}" \
  -p "mapping.map_output_path:=${1:?Map output path required}" \
  -p "mapping.calibration_path:=${3:?Frozen camera geometry required}"
