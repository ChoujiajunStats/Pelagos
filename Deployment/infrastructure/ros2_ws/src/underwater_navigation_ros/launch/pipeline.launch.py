"""Robot-side unified launch; uses the existing frozen-input supervisor."""
import json
import os
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def start(context):
    root = LaunchConfiguration('deployment_root').perform(context)
    info = json.loads(subprocess.check_output([
        'python3', os.path.join(root, 'Deployment/application/services/mapper_backend.py'), '--root', root], text=True))
    if info['unit']:
        raise RuntimeError('An existing algorithm supervisor is running; reuse it or save and stop it first.')
    bridge = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'dapeng-omniplanner-bridge.service'])
    if bridge.returncode == 0:
        raise RuntimeError('Stop OmniPlanner before launching the Wavemap navigation pipeline.')
    subprocess.run(['systemctl', '--user', 'start', 'zed-x-ingress.service'], check=True)
    pipeline = ExecuteProcess(cmd=['python3', os.path.join(root, 'Deployment/application/services/run_pipeline.py'),
                                  '--mapper', 'wavemap', '--navigation'], output='screen',
                              sigterm_timeout='40', sigkill_timeout='5')
    return [pipeline, RegisterEventHandler(OnProcessExit(target_action=pipeline,
            on_exit=[Shutdown(reason='The supervised navigation pipeline exited.')]))]


def generate_launch_description():
    share = get_package_share_directory('underwater_navigation_ros')
    return LaunchDescription([
        DeclareLaunchArgument('deployment_root', default_value=os.environ.get('DAPENG_ROOT', '/home/nvidia/Dapeng_Deployment')),
        DeclareLaunchArgument('rviz', default_value='false'),
        OpaqueFunction(function=start),
        Node(package='underwater_navigation_ros', executable='robot_model.py', name='bluerov2_six_visual',
             condition=IfCondition(LaunchConfiguration('rviz'))),
        Node(package='rviz2', executable='rviz2', name='dapeng_incremental_rviz',
             arguments=['-d', os.path.join(share, 'rviz', 'robot_incremental_mapping.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')),
             additional_env={'DISPLAY': os.environ.get('DISPLAY', ':1'),
                             'XAUTHORITY': os.path.expanduser('~/.Xauthority'),
                             'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_X11_NO_MITSHM': '1'}),
    ])
