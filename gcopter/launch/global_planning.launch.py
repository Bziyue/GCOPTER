from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    gcopter_share = get_package_share_directory('gcopter')
    params_file = os.path.join(gcopter_share, 'config', 'global_planning_ros2.yaml')

    return LaunchDescription([
        Node(
            package='mockamap',
            executable='mockamap_node',
            name='mockamap_node',
            output='screen',
            parameters=[{
                'seed': 1024,
                'update_freq': 1.0,
                'resolution': 0.25,
                'x_length': 50,
                'y_length': 50,
                'z_length': 5,
                'type': 1,
                'complexity': 0.025,
                'fill': 0.3,
                'fractal': 1,
                'attenuation': 0.1,
            }],
            remappings=[('mock_map', 'voxel_map')],
        ),
        Node(
            package='gcopter',
            executable='global_planning',
            name='global_planning_node',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(gcopter_share, 'config', 'global_planning.rviz')],
        ),
    ])
