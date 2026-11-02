#!/usr/bin/env bash
set -eo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
container_name=dapeng-rviz-live
rviz_image=${RVIZ_IMAGE:-osrf/ros:humble-desktop}
xauthority_file=${XAUTHORITY:-$HOME/.Xauthority}

if ! command -v docker >/dev/null; then
  echo 'Docker and a Humble RViz image are required.' >&2
  exit 1
fi
if [[ ! -f "$xauthority_file" ]]; then
  echo "Xauthority not found: $xauthority_file" >&2
  exit 1
fi
if [[ ! -f "$root_dir/Deployment/interfaces/visualization/rviz/mapping.rviz" ]]; then
  echo 'mapping.rviz not found.' >&2
  exit 1
fi
if docker container inspect "$container_name" >/dev/null 2>&1; then
  if [[ $(docker inspect -f '{{.State.Running}}' "$container_name") == true ]]; then
    echo 'RViz already active.'
  else
    docker start "$container_name" >/dev/null
    echo 'RViz resumed.'
  fi
  exit 0
fi

if ! docker image inspect "$rviz_image" >/dev/null 2>&1; then
  echo "Image not found. Run: docker pull $rviz_image" >&2
  exit 1
fi

docker run -d --name "$container_name" --init \
  --network host \
  --env "DISPLAY=${DISPLAY:-:1}" \
  --env XAUTHORITY=/tmp/rviz.xauthority \
  --env QT_X11_NO_MITSHM=1 \
  --env LIBGL_ALWAYS_SOFTWARE=1 \
  --env LC_ALL=C.UTF-8 \
  --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
  --env ROS_LOCALHOST_ONLY=0 \
  --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  --env FASTRTPS_DEFAULT_PROFILES_FILE=/configuration/config/fastdds-orin.xml \
  --volume "$xauthority_file:/tmp/rviz.xauthority:ro" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --volume /usr/share/fonts:/usr/share/fonts:ro \
  --volume "$root_dir/Deployment/interfaces/visualization:/configuration:ro" \
  --entrypoint /bin/bash "$rviz_image" -lc \
  'exec bash /configuration/scripts/run_viewer.sh' >/dev/null

echo 'RViz started.'
echo 'Logs: docker logs --tail 30 dapeng-rviz-live'
echo 'Stop viewer: docker stop dapeng-rviz-live'
