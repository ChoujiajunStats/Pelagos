#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
robot=${DAPENG_JETSON:-nvidia@192.168.1.202}
remote_root=${DAPENG_REMOTE_ROOT:-/home/nvidia/Dapeng_Deployment}
mode=run
open_rviz=1
for argument in "$@"; do
  case "$argument" in
    --check-only) mode=check ;;
    --no-rviz) open_rviz=0 ;;
    --help|-h) echo 'Usage: bash Deployment/contexts/navigation/scripts/start_robot_navigation.sh [--check-only] [--no-rviz]'; exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done
bash "$root_dir/Deployment/application/scripts/start_robot_mapping.sh" --mapper wavemap --check-only --no-rviz
remote_start() {
local requested_mode=$1 remote_command
printf -v remote_command 'bash -s -- %q %q %q' "$remote_root" "$requested_mode" "$(date -u +%s)"
ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$robot" "$remote_command" <<'REMOTE'
set -euo pipefail
root=$1
mode=$2
target_epoch=$3
binary="$root/Deployment/infrastructure/ros2_ws/install-navigation/underwater_navigation_ros/lib/underwater_navigation_ros/wavemap_navigation_node"
[[ -x $binary ]] || { echo 'Navigation binary not found. Run build navigation.' >&2; exit 1; }
for relative in Deployment/contexts/navigation/scripts/start_navigation.sh Deployment/config/profiles/orin-nano/navigation.yaml Deployment/contexts/navigation/provenance/source-lock.json; do
  [[ -s $root/$relative ]] || { echo "Missing file: $relative" >&2; exit 1; }
done
info=$(python3 "$root/Deployment/application/services/mapper_backend.py" --root "$root")
python3 -c 'import json,sys; i=json.loads(sys.argv[1]);
if i["unit"] and (i["mapper"]!="wavemap" or not i.get("navigation")):sys.exit("Another pipeline is active. Save and stop it before starting integrated navigation.")' "$info"
if [[ $mode == check ]]; then
  echo 'Navigation preflight passed.'
  exit 0
fi
exec 9>"$root/runs/.start-robot-vio.lock"
flock -n 9 || { echo 'Another startup operation is in progress.' >&2; exit 1; }
info=$(python3 "$root/Deployment/application/services/mapper_backend.py" --root "$root")
python3 -c 'import json,sys; i=json.loads(sys.argv[1]);
if i["unit"] and (i["mapper"]!="wavemap" or not i.get("navigation")):sys.exit("Another pipeline is active. Save and stop it first.")' "$info"
active=$(systemctl --user show dapeng-wavemap.service -p ActiveState --value)
if [[ $active != active ]]; then
  # Correct time only before either camera or algorithms start.
  offset=$((target_epoch-$(date -u +%s)))
  if ((offset > 2 || offset < -2)); then
    [[ $(systemctl --user show zed-x-ingress.service -p ActiveState --value) != active ]] || {
      echo 'Clock correction blocked while the camera is active.' >&2; exit 1; }
    sudo -n date -u -s "@$target_epoch" >/dev/null
  fi
  systemctl --user start zed-x-ingress.service
  if [[ $(systemctl --user show dapeng-wavemap.service -p LoadState --value) == loaded ]]; then
    command=$(systemctl --user show dapeng-wavemap.service -p ExecStart --value)
    [[ $command == *run_pipeline.py* && $command == *--navigation* ]] || {
      echo 'Existing unit has an unexpected ExecStart.' >&2; exit 1; }
    systemctl --user start dapeng-wavemap.service
  else
    systemd-run --user --collect --unit=dapeng-wavemap \
      --property=KillMode=mixed --property=TimeoutStopSec=45 --property='SuccessExitStatus=130 143' \
      --property=Requires=zed-x-ingress.service --property=After=zed-x-ingress.service \
      --property="WorkingDirectory=$root" --setenv=PYTHONUNBUFFERED=1 \
      /usr/bin/python3 "$root/Deployment/application/services/run_pipeline.py" --mapper wavemap --navigation
  fi
else
  echo 'Navigation already active.'
fi
set +u
source /opt/ros/humble/setup.bash
python3 - <<'PY'
import subprocess,time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import UInt64
rclpy.init();node=rclpy.create_node('navigation_start_check')
state={'fusion':0,'planning':0,'odom':0}
durable=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
node.create_subscription(UInt64,'/mapping/integrated_frames',lambda m:state.__setitem__('fusion',m.data),durable)
node.create_subscription(UInt64,'/mapping/planning_map_version',lambda m:state.__setitem__('planning',m.data),durable)
def odom(m):
 age=(node.get_clock().now().nanoseconds-(m.header.stamp.sec*10**9+m.header.stamp.nanosec))*1e-9
 if m.header.frame_id=='world' and m.child_frame_id=='body' and -.5<age<3:state['odom']+=1
node.create_subscription(Odometry,'/okvis/okvis_odometry',odom,QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT))
begin=time.monotonic()
while time.monotonic()-begin<90:
 rclpy.spin_once(node,timeout_sec=.2)
 if all(state.values()):break
 if int(time.monotonic()-begin)%3==0:
  unit=subprocess.check_output(['systemctl','--user','show','dapeng-wavemap.service','-p','ActiveState','--value'],text=True).strip()
  if unit not in ('active','activating'):raise SystemExit('Navigation supervisor exited. Check pipeline logs.')
else:raise SystemExit('Navigation readiness timed out. Check pipeline logs.')
print('Navigation ready:',state)
node.destroy_node();rclpy.shutdown()
PY
REMOTE
}
remote_start check
if [[ $mode == run ]]; then
  echo 'Stopping previous planner preview.'
  DAPENG_ROBOT_SSH="$robot" bash "$root_dir/Deployment/contexts/planning/scripts/stop_preview.sh"
  remote_start run
fi
if [[ $mode == run && $open_rviz == 1 ]]; then
  bash "$root_dir/Deployment/interfaces/visualization/scripts/open_robot_rviz.sh" --mapper wavemap
fi
