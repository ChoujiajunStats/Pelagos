#!/usr/bin/env bash
# Change only renderer parameters; no camera, VIO, depth or map restart.
set -euo pipefail
mode=${1:-surface}
hz=${2:-2.0}
case "$mode" in surface|points|off) ;; *) echo 'Usage: set_robot_visualization.sh [surface|points|off] [Hz]' >&2; exit 2;; esac
printf -v command 'bash -s -- %q %q' "$mode" "$hz"
ssh -T -o BatchMode=yes -o ConnectTimeout=8 "${DAPENG_JETSON:-nvidia@192.168.1.202}" "$command" <<'REMOTE'
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
python3 - "$1" "$2" <<'PY'
import sys
import rclpy
from rcl_interfaces.msg import Parameter,ParameterValue,ParameterType
from rcl_interfaces.srv import SetParametersAtomically
rclpy.init();node=rclpy.create_node('change_native_renderer')
client=node.create_client(SetParametersAtomically,'/wavemap_navigation/set_parameters_atomically')
try:
    if not client.wait_for_service(timeout_sec=10):raise RuntimeError('Navigation parameter service unavailable.')
    mode,hz=sys.argv[1],float(sys.argv[2]);values=[Parameter(name='visualization.enabled',value=ParameterValue(type=ParameterType.PARAMETER_BOOL,bool_value=mode!='off')),
        Parameter(name='visualization.update_hz',value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,double_value=hz))]
    if mode!='off':values.append(Parameter(name='visualization.mode',value=ParameterValue(type=ParameterType.PARAMETER_STRING,string_value=mode)))
    future=client.call_async(SetParametersAtomically.Request(parameters=values));rclpy.spin_until_future_complete(node,future,timeout_sec=15)
    if not future.done():raise RuntimeError('Parameter update timed out.')
    result=future.result().result
    if not result.successful:raise RuntimeError(result.reason)
    print('Renderer:',mode,'Hz:',hz,'')
finally:
    node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
PY
REMOTE
