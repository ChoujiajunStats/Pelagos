from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default = os.path.join(get_package_share_directory('underwater_navigation_ros'), 'config', 'navigation.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default),
        DeclareLaunchArgument('map_input', default_value=''),
        DeclareLaunchArgument('map_output', default_value=''),
        Node(package='underwater_navigation_ros', executable='wavemap_navigation_node',
             name='wavemap_navigation', output='screen', parameters=[LaunchConfiguration('config'), {
                 'mapping.map_input_path': ParameterValue(LaunchConfiguration('map_input'), value_type=str),
                 'mapping.map_output_path': ParameterValue(LaunchConfiguration('map_output'), value_type=str),
             }]),
    ])
