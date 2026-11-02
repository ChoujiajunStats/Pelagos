#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
if [[ -f "$root_dir/install/okvis2/local_setup.bash" ]]; then
  source "$root_dir/install/okvis2/local_setup.bash"
fi
exec "$root_dir/install/okvis2/lib/okvis/okvis_node_subscriber" --ros-args \
  -r __ns:=/okvis \
  -p config_filename:="$root_dir/Deployment/config/profiles/orin-nano/okvis2.yaml" \
  -p imu_propagated_state_publishing_rate:=30.0 \
  -r /okvis/cam0/image_raw:=/zed/zed_node/left/gray/raw/image \
  -r /okvis/cam1/image_raw:=/zed/zed_node/right/gray/raw/image \
  -r /okvis/imu0:=/zed/zed_node/imu/data_raw "$@"
