from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    gcopter_share = get_package_share_directory('gcopter')
    params_file = os.path.join(gcopter_share, 'config', 'global_planning_ros2.yaml')
    drone_count = LaunchConfiguration('drone_count')
    use_random_map = LaunchConfiguration('use_random_map')
    random_map_seed = LaunchConfiguration('random_map_seed')
    random_cylinder_count = LaunchConfiguration('random_cylinder_count')
    random_box_count = LaunchConfiguration('random_box_count')

    return LaunchDescription([
        DeclareLaunchArgument(
            'drone_count',
            default_value='80',
            description='Number of drones to plan sequentially in chain-planning mode.',
        ),
        DeclareLaunchArgument(
            'use_random_map',
            default_value='false',
            description='Enable internal random column map generation instead of mockamap.',
        ),
        DeclareLaunchArgument(
            'random_map_seed',
            default_value='1024',
            description='Seed used by the internal random map generator.',
        ),
        DeclareLaunchArgument(
            'random_cylinder_count',
            default_value='24',
            description='Number of random cylindrical column obstacles.',
        ),
        DeclareLaunchArgument(
            'random_box_count',
            default_value='24',
            description='Number of random rectangular column obstacles.',
        ),
        Node(
            package='mockamap',
            executable='mockamap_node',
            name='mockamap_node',
            output='screen',
            condition=UnlessCondition(use_random_map),
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
            parameters=[params_file, {
                'DroneCount': drone_count,
                'UseRandomMap': use_random_map,
                'RandomMapSeed': random_map_seed,
                'RandomCylinderCount': random_cylinder_count,
                'RandomBoxCount': random_box_count,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(gcopter_share, 'config', 'global_planning.rviz')],
        ),
    ])
