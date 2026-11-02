#!/usr/bin/env bash
set -eo pipefail
# Keep nounset disabled while sourcing the ROS/ament environment.
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
export PYTHONPATH="/pelagos/Simulation/adapters:/stonefish-workspace/robot:/stonefish-workspace/guard:/stonefish-workspace/runtime${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == rviz || "${1:-}" == ui ]]; then
  if [[ ! -f /opt/pelagos/visualization/share/pelagos_rviz/plugins.xml ]]; then
    echo '当前镜像缺少 ROS2 RViz 任务插件；请执行 ./Simulation/sim build。' >&2
    exit 1
  fi
  export LANG=C.UTF-8 LC_ALL=C.UTF-8 LANGUAGE=en
  exec rviz2 -d /pelagos/Simulation/config/pelagos.rviz --ros-args -p use_sim_time:=true
fi
if [[ "${1:-}" == joystick ]]; then
  exec /opt/research/bin/python3 /pelagos/Simulation/adapters/joystick_receiver.py --run-dir "$PELAGOS_RUN"
fi
if [[ "${1:-}" == planner || "${1:-}" == explore || "${1:-}" == pause || "${1:-}" == navigate || "${1:-}" == manual || "${1:-}" == autonomous || "${1:-}" == plan-preview || "${1:-}" == start-preview || "${1:-}" == explore-forward ]]; then
  exec /opt/research/bin/python3 /pelagos/Simulation/adapters/autonomy_cli.py --run-dir "$PELAGOS_RUN" "$1"
fi
if [[ "${1:-}" == metrics ]]; then
  exec ros2 topic echo --once /pelagos/depth/metrics std_msgs/msg/String
fi
if [[ "${1:-}" == mapping ]]; then
  exec ros2 topic echo --once /mapping/status std_msgs/msg/String
fi
if [[ "${1:-}" == map-save ]]; then
  exec ros2 service call /pelagos/wavemap/save_map std_srvs/srv/Trigger '{}'
fi
exec /opt/research/bin/python3 -c 'from uw_guard.cli import main; main()' --run-dir "$PELAGOS_RUN" "$@"
