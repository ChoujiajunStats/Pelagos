"""Robot-side native incremental renderer; source odometry/depth are external."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('underwater_navigation_ros')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(share, 'config', 'navigation.yaml')),
        DeclareLaunchArgument('map_output', default_value=''),
        DeclareLaunchArgument('rviz', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'navigation.launch.py')),
            launch_arguments={'config': LaunchConfiguration('config'), 'map_input': '',
                              'map_output': LaunchConfiguration('map_output')}.items()),
        Node(package='underwater_navigation_ros', executable='robot_model.py', name='bluerov2_six_visual',
             condition=IfCondition(LaunchConfiguration('rviz'))),
        Node(package='rviz2', executable='rviz2', name='dapeng_incremental_rviz',
             arguments=['-d', os.path.join(share, 'rviz', 'robot_incremental_mapping.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')),
             additional_env={'DISPLAY': os.environ.get('DISPLAY', ':1'),
                             'XAUTHORITY': os.path.expanduser('~/.Xauthority'),
                             'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_X11_NO_MITSHM': '1'}),
    ])
