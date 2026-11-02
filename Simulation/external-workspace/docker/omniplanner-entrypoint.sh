#!/bin/bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /opt/omniplanner_ws/devel/setup.bash
exec "$@"
