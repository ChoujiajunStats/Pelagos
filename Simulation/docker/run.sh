#!/usr/bin/env bash
set -eo pipefail
# ROS/ament setup scripts read optional environment variables without defaults.
source /opt/ros/jazzy/setup.bash
source /opt/uw_underlay/setup.bash
source /opt/pelagos/interfaces/setup.bash
if [[ -f /opt/pelagos/mapping/setup.bash ]]; then
  source /opt/pelagos/mapping/setup.bash
fi
if [[ -f /opt/pelagos/visualization/share/pelagos_rviz/local_setup.bash ]]; then
  source /opt/pelagos/visualization/share/pelagos_rviz/local_setup.bash
fi
set -u
export PYTHONPATH="/pelagos/Simulation/adapters:/pelagos/TP/Stereo/LiteAnyStereoV2:/pelagos/TP/Controller:/stonefish-workspace/simulations:/stonefish-workspace/perception:/stonefish-workspace/robot:/stonefish-workspace/guard:/stonefish-workspace/controller:/stonefish-workspace/navigation:/stonefish-workspace/runtime${PYTHONPATH:+:$PYTHONPATH}"
export ROS_LOG_DIR="$PELAGOS_RUN/ros-log"
mkdir -p "$ROS_LOG_DIR"
exec /opt/research/bin/python3 /pelagos/Simulation/tools/run_pipeline.py \
  --run-dir "$PELAGOS_RUN" --scene "$PELAGOS_SCENE" --ros-domain "$ROS_DOMAIN_ID"
