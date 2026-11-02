#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
export PYTHONPATH="$root_dir/TP/Stereo/LiteAnyStereoV2${PYTHONPATH:+:$PYTHONPATH}"
exec "$root_dir/venv-las2/bin/python" -u "$root_dir/Deployment/contexts/depth/adapters/las2_node.py" \
  --checkpoint "$root_dir/Deployment/artifacts/weights/LAS2_S.pth" \
  --calibration "$root_dir/Deployment/config/profiles/orin-nano/stereo-geometry.json" \
  --variant s "$@"
