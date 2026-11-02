#!/usr/bin/env bash
# Host entry: supervised real pipeline, opt-in mesh, robot-side RViz over VNC.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
show_rviz=1
check_only=0
for arg in "$@"; do
  case "$arg" in
    --no-rviz) show_rviz=0 ;;
    --check-only) check_only=1 ;;
    --help|-h) echo 'Usage: bash Deployment/interfaces/visualization/scripts/start_robot_surface_mesh.sh [--check-only] [--no-rviz]'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if ((check_only)); then
  bash "$root/Deployment/contexts/navigation/scripts/start_robot_navigation.sh" --check-only --no-rviz
  exit 0
fi
bash "$root/Deployment/contexts/navigation/scripts/start_robot_navigation.sh" --no-rviz
bash "$root/Deployment/interfaces/visualization/scripts/set_robot_visualization.sh" surface "${DAPENG_MESH_HZ:-2.5}"
if ((show_rviz)); then
  bash "$root/Deployment/interfaces/visualization/scripts/open_robot_rviz.sh" --mapper wavemap
fi
