#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec /usr/bin/python3 "$root_dir/Deployment/contexts/localization/adapters/okvis_pose_adapter.py" "$@"
