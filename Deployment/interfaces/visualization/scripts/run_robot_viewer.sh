#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
viewer_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$viewer_dir/../../infrastructure/ros2_ws/install-navigation/local_setup.bash" ]]; then
  source "$viewer_dir/../../infrastructure/ros2_ws/install-navigation/local_setup.bash"
fi
case "${1:-nvblox}" in
  nvblox) configuration="$viewer_dir/rviz/robot_vio_mapping.rviz" ;;
  wavemap) configuration="$viewer_dir/rviz/robot_vio_wavemap.rviz" ;;
  wavemap_navigation) configuration="$viewer_dir/rviz/robot_vio_navigation.rviz" ;;
  *) echo 'Expected mapper wavemap|nvblox.' >&2; exit 2 ;;
esac
if [[ ${DAPENG_RVIZ_CAMERA_COMPARISON:-0} == 1 && ${1:-nvblox} == wavemap_navigation ]]; then
  configuration="$viewer_dir/rviz/robot_camera_comparison.rviz"
fi
if [[ ${DAPENG_RVIZ_LOW_LOAD:-0} == 1 && ${1:-nvblox} == wavemap_navigation ]]; then
  configuration="$viewer_dir/rviz/robot_vio_navigation_light.rviz"
fi

python3 "$viewer_dir/adapters/rviz_pose.py" &
pose_pid=$!
model_pid=
if [[ ${1:-nvblox} == wavemap_navigation ]]; then
  python3 "$viewer_dir/adapters/robot_model.py" &
  model_pid=$!
fi
rviz2 -d "$configuration" --ros-args -r __node:=dapeng_robot_rviz &
rviz_pid=$!

cleanup() {
  trap - INT TERM EXIT
  kill -INT "$pose_pid" "$rviz_pid" ${model_pid:+"$model_pid"} 2>/dev/null || true
  wait "$pose_pid" "$rviz_pid" ${model_pid:+"$model_pid"} 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait -n "$pose_pid" "$rviz_pid" ${model_pid:+"$model_pid"}
