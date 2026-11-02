#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1
exec /usr/bin/python3 -u "$(dirname -- "${BASH_SOURCE[0]}")/../adapters/ros_gateway.py" "$@"
