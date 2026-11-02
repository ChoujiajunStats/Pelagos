#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash Deployment/application/scripts/start_robot_mapping.sh [--mapper wavemap|nvblox] [--check-only] [--no-rviz]
  --mapper     Mapping backend (default: wavemap).
  --check-only  Validate prerequisites without starting services.
  --no-rviz     Skip RViz and VNC.
  --help        Show help.

Environment: DAPENG_MAPPER, DAPENG_JETSON, DAPENG_REMOTE_ROOT, DAPENG_VIO_WAIT_SECONDS.
HELP
}

check_only=0
open_rviz=1
mapper=${DAPENG_MAPPER:-wavemap}
while (($#)); do
  case "$1" in
    --mapper) (($# >= 2)) || { usage >&2; exit 2; }; mapper=$2; shift ;;
    --mapper=*) mapper=${1#*=} ;;
    --check-only) check_only=1 ;;
    --no-rviz) open_rviz=0 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ $mapper == wavemap || $mapper == nvblox ]] || { echo 'Expected mapper wavemap|nvblox.' >&2; exit 2; }

workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
robot=${DAPENG_JETSON:-nvidia@192.168.1.202}
remote_root=${DAPENG_REMOTE_ROOT:-/home/nvidia/Dapeng_Deployment}
wait_seconds=${DAPENG_VIO_WAIT_SECONDS:-75}
if [[ ! $wait_seconds =~ ^[0-9]+$ ]] || ((wait_seconds < 5 || wait_seconds > 300)); then
  echo 'DAPENG_VIO_WAIT_SECONDS must be an integer in [5, 300].' >&2
  exit 2
fi

echo "Checking SSH: $robot"
if ! ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$robot" true; then
  echo 'SSH unavailable. Check the robot address and USB network.' >&2
  exit 1
fi
run_remote() {
  local mode=$1 target_epoch remote_command
  target_epoch=$(date -u +%s)
  printf -v remote_command 'bash -s -- %q %q %q %q %q' \
    "$remote_root" "$target_epoch" "$mode" "$wait_seconds" "$mapper"

ssh -T -o BatchMode=yes -o ConnectTimeout=8 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$robot" "$remote_command" <<'REMOTE'
set -eo pipefail
root=$1
requested_epoch=$2
check_only=$3
wait_seconds=$4
mapper=$5
begin_seconds=$SECONDS
camera_unit=zed-x-ingress.service
pipeline_unit=dapeng-$mapper.service

die() { echo "ERROR: $*" >&2; exit 1; }
unit_state() { systemctl --user show "$1" -p ActiveState --value; }
busy_state() {
  case "$1" in active|activating|reloading|deactivating) return 0 ;; *) return 1 ;; esac
}

[[ -d $root ]] || die "Deployment directory not found: $root"
for command in systemctl systemd-run python3 flock; do
  command -v "$command" >/dev/null || die "Command not found: $command"
done
[[ -r /opt/ros/humble/setup.bash ]] || die 'ROS 2 Humble not found.'
for relative in \
  Deployment/application/services/run_pipeline.py \
  Deployment/application/services/mapper_backend.py \
  Deployment/contexts/sensing/scripts/start_zed_x.sh \
  Deployment/contexts/localization/scripts/start_okvis2.sh \
  Deployment/config/profiles/orin-nano/okvis2.yaml \
  Deployment/config/profiles/orin-nano/stereo-geometry.json \
  "Deployment/config/profiles/orin-nano/$mapper.yaml" \
  Deployment/config/profiles/orin-nano/zed-x.yaml \
  Deployment/artifacts/weights/LAS2_S.pth; do
  [[ -s $root/$relative ]] || die "Missing deployment file: $root/$relative"
done
[[ -x $root/install/okvis2/lib/okvis/okvis_node_subscriber ]] || die 'OKVIS2 binary not found. Run build okvis.'
[[ -r /home/nvidia/zed_ws/install/setup.bash ]] || die 'ZED ROS workspace not found.'
if [[ $mapper == wavemap ]]; then
  [[ -r $root/Deployment/infrastructure/ros2_ws/install/setup.bash &&
     -x $root/Deployment/infrastructure/ros2_ws/install/underwater_wavemap_ros/lib/underwater_wavemap_ros/wavemap_node ]] || die 'Wavemap ROS binary not found. Run build wavemap.'
else
  [[ -x /opt/ros/humble/lib/nvblox_ros/nvblox_node ]] || die 'nvblox node not found.'
fi
[[ $(systemctl --user show "$camera_unit" -p LoadState --value) == loaded ]] || die 'Camera service not installed.'

if [[ $check_only == 0 ]]; then
  mkdir -p "$root/runs"
  exec 9>"$root/runs/.start-robot-vio.lock"
  flock -n 9 || die 'Another startup operation is in progress.'
fi

camera_state=$(unit_state "$camera_unit")
[[ $camera_state != deactivating ]] || die 'Camera service is stopping.'
backend_info=$(python3 "$root/Deployment/application/services/mapper_backend.py" --root "$root" --format shell)
read -r active_unit active_mapper pipeline_state <<< "$backend_info"
if [[ $active_unit != - ]]; then
  [[ $active_mapper == "$mapper" ]] || die "Backend conflict: $active_unit runs $active_mapper. Save and stop it before selecting $mapper."
  pipeline_unit=$active_unit
fi

if ! busy_state "$pipeline_state"; then
  python3 - "$root/runs/.algorithm-pipeline.lock" <<'PY'
import fcntl, pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.exists():
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('Pipeline lock is held by another process.')
PY
fi

if ! busy_state "$camera_state" && \
   busy_state "$(unit_state hmu-stereo.service)"; then
  die 'Camera is owned by hmu-stereo.service.'
fi

target_epoch=$((requested_epoch + SECONDS - begin_seconds))
offset=$((target_epoch - $(date -u +%s)))
absolute_offset=${offset#-}
echo "Backend=$mapper camera=$camera_state pipeline=$pipeline_state clock_offset=${offset}s"
if ((absolute_offset > 2)); then
  if busy_state "$camera_state" || busy_state "$pipeline_state"; then
    die 'Clock offset exceeds 2s. Stop camera and pipeline before correcting time.'
  fi
  sudo -n true || die 'Clock correction requires passwordless sudo or manual synchronization.'
  if [[ $check_only == 0 ]]; then
    target_epoch=$((requested_epoch + SECONDS - begin_seconds))
    sudo -n date -u -s "@$target_epoch" >/dev/null
    echo 'Robot clock synchronized.'
  else
    echo 'Clock correction required before startup.'
  fi
fi

if [[ $check_only == 1 ]]; then
  echo 'Preflight passed.'
  exit 0
fi

if ! busy_state "$camera_state"; then
  echo 'Starting camera and IMU.'
  systemctl --user start "$camera_unit"
else
  echo 'Camera already active.'
fi
if ! busy_state "$pipeline_state"; then
  echo "Starting VIO, LAS2 and $mapper."
  if [[ $(systemctl --user show "$pipeline_unit" -p LoadState --value) == loaded ]]; then
    existing_command=$(systemctl --user show "$pipeline_unit" -p ExecStart --value)
    [[ $existing_command == *run_pipeline.py* && $existing_command == *"--mapper $mapper"* &&
       $existing_command == *"$root/"* ]] || die "Unexpected ExecStart in $pipeline_unit."
    systemctl --user start "$pipeline_unit"
  else
    systemd-run --user --collect --unit="${pipeline_unit%.service}" \
      --property=KillMode=mixed --property=TimeoutStopSec=45 \
      --property='SuccessExitStatus=130 143' \
      --property=Requires=zed-x-ingress.service --property=After=zed-x-ingress.service \
      --property="WorkingDirectory=$root" --setenv=PYTHONUNBUFFERED=1 \
      /usr/bin/python3 "$root/Deployment/application/services/run_pipeline.py" --mapper "$mapper"
  fi
else
  echo "Pipeline already active: $pipeline_unit"
fi

echo "Waiting for pipeline readiness (timeout: ${wait_seconds}s)."
source /opt/ros/humble/setup.bash
if ! python3 - "$wait_seconds" "$pipeline_unit" "$camera_unit" "$mapper" <<'PY'
import math, subprocess, sys, time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt64

timeout = int(sys.argv[1])
rclpy.init()
node = Node('dapeng_vio_start_check')
received = []
map_frames = [0]
if sys.argv[4] == 'wavemap':
    node.create_subscription(UInt64, '/mapping/integrated_frames',
        lambda message: map_frames.__setitem__(0, int(message.data)),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

def observe(message):
    p, q = message.pose.pose.position, message.pose.pose.orientation
    values = [p.x,p.y,p.z,q.x,q.y,q.z,q.w]
    stamp = message.header.stamp.sec*1000000000 + message.header.stamp.nanosec
    age = (node.get_clock().now().nanoseconds-stamp)*1e-9
    if (message.header.frame_id.lstrip('/') == 'world'
            and message.child_frame_id.lstrip('/') == 'body'
            and all(math.isfinite(v) for v in values)
            and max(abs(v) for v in values[:3]) < 30
            and .9 < sum(v*v for v in values[3:]) < 1.1 and -1 < age < 3):
        if not received or received[-1][0] != stamp:
            received.append((stamp,values[:3]))

node.create_subscription(Odometry, '/okvis/okvis_odometry', observe,
                         QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
start = time.monotonic()
next_check = start
try:
    while time.monotonic()-start < timeout:
        rclpy.spin_once(node, timeout_sec=.2)
        if time.monotonic() >= next_check:
            for unit in sys.argv[2:4]:
                state = subprocess.check_output(
                    ['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()
                if state not in ('active','activating','reloading'):
                    sys.exit(f'{unit} entered state {state}.')
            next_check = time.monotonic()+2
        if len(received) >= 3 and (sys.argv[4] != 'wavemap' or map_frames[0] > 0):
            p = received[-1][1]
            print(f'VIO ready: position=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})m')
            if sys.argv[4] == 'wavemap':
                print(f'Wavemap ready: integrated_frames={map_frames[0]}')
            print('Pipeline readiness check passed.')
            break
    else:
        sys.exit('Pipeline readiness timed out. Check camera and pipeline logs.')
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
then
  systemctl --user --no-pager --full status "$camera_unit" "$pipeline_unit" || true
  echo "Logs: journalctl --user -u $pipeline_unit -n 50 --no-pager" >&2
  exit 1
fi
echo 'Pipeline running.'
REMOTE
}

if [[ $mapper == wavemap && $check_only == 0 ]]; then
  run_remote 1
  echo 'Stopping previous planner preview.'
  DAPENG_ROBOT_SSH="$robot" bash "$workspace/Deployment/contexts/planning/scripts/stop_preview.sh"
fi
run_remote "$check_only"

if [[ $check_only == 0 && $open_rviz == 1 ]]; then
  echo 'Opening robot RViz.'
  bash "$workspace/Deployment/interfaces/visualization/scripts/open_robot_rviz.sh" --mapper "$mapper"
fi
