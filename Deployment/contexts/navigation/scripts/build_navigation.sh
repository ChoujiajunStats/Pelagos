#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$root_dir/Deployment/infrastructure/ros2_ws"
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
colcon --log-base log-navigation build --packages-select underwater_navigation_ros \
  --build-base build-navigation --install-base install-navigation --symlink-install \
  --parallel-workers 1 --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG'
