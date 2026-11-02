#!/usr/bin/env bash
set -euo pipefail
planning_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root_dir=$(cd "$planning_dir/../../.." && pwd)
wavemap_build=${DAPENG_WAVEMAP_BUILD:-$root_dir/Deployment/infrastructure/ros2_ws/build/underwater_wavemap_ros/wavemap}
mkdir -p "$root_dir/build/wavemap_export/bin"
g++ -std=c++17 -O2 -DNDEBUG -I"$root_dir/TP/Mapping/wavemap/library/cpp/include" \
  -I/usr/include/eigen3 "$planning_dir/adapters/map_export/wavemap_export.cpp" \
  "$wavemap_build/src/io/libwavemap_io.a" "$wavemap_build/src/core/libwavemap_core.a" \
  -lglog -pthread -o "$root_dir/build/wavemap_export/bin/wavemap_export"
