#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then echo "Usage: bash $0 x y z (world frame, meters)" >&2; exit 2; fi
# Parse numeric arguments before placing them in the remote ROS command.
goal_yaml=$(python3 - "$@" <<'PY'
import json,math,sys
v=[float(x) for x in sys.argv[1:]]
if not all(math.isfinite(x) and abs(x)<1000 for x in v):raise ValueError('Invalid goal')
print(json.dumps(dict(header=dict(frame_id='world'),pose=dict(
    position=dict(zip(('x','y','z'),v)),orientation=dict(w=1.)))))
PY
)
ssh -T -o BatchMode=yes -o ConnectTimeout=5 "${DAPENG_ROBOT_SSH:-nvidia@192.168.1.202}" \
  "source /opt/ros/humble/setup.bash; ros2 topic pub --once /planning/goal geometry_msgs/msg/PoseStamped '$goal_yaml'"
