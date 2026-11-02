#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
source "$root_dir/Deployment/infrastructure/ros2_ws/install/setup.bash"
map_output_path=${1:?Absolute map output path required}
params_path=${2:-$root_dir/Deployment/config/profiles/orin-nano/wavemap.yaml}
if [[ "$map_output_path" != /* ]]; then
  printf '%s\n' 'Map output path must be absolute.' >&2
  exit 2
fi
exec ros2 run underwater_wavemap_ros wavemap_node --ros-args \
  -r __node:=wavelet_mapper --params-file "$params_path" \
  -p use_sim_time:="${DAPENG_USE_SIM_TIME:-false}" \
  -p map_output_path:="$map_output_path"
