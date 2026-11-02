#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$root_dir/Deployment/infrastructure/ros2_ws"
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
colcon build --symlink-install --packages-select underwater_wavemap_ros \
  --parallel-workers 1 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DWAVEMAP_SOURCE_DIR="$root_dir/TP/Mapping/wavemap"
