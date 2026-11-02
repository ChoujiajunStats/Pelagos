#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec python3 "$root_dir/Deployment/contexts/mapping/adapters/nvblox_pose_tf.py" --ros-args \
  -p use_sim_time:="${DAPENG_USE_SIM_TIME:-false}" \
  -r /tf:=/nvblox/tf -r /tf_static:=/nvblox/tf_static
