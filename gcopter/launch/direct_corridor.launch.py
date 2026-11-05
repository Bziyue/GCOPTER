"""Original GCOPTER mockamap + Informed RRT* + C3/H3 corridor."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def setup(context):
    scene = LaunchConfiguration('scene').perform(context)
    if scene not in ('perlin', 'maze'):
        raise ValueError('scene must be perlin or maze')
    corridor = LaunchConfiguration('corridor').perform(context)
    if corridor not in ('c3', 'h3'):
        raise ValueError('corridor must be c3 or h3')
    share = get_package_share_directory('gcopter')
    size = 50 if scene == 'perlin' else 30
    generator = dict(seed=int(LaunchConfiguration('map_seed').perform(context)),
                     update_freq=1.0, resolution=0.25,
                     x_length=size, y_length=size, z_length=5)
    if scene == 'perlin':
        generator.update(type=1, complexity=0.025, fill=0.3, fractal=1, attenuation=0.1)
    else:
        generator.update(type=3, road_width=3.0, maze_type=1, add_wall_x=1, add_wall_y=1)
    planner = dict(frame='odom', resolution=0.25, shape_mode=corridor,
                   bounds=[-size/2, -size/2, 0., size/2, size/2, 5.],
                   height=float(LaunchConfiguration('height').perform(context)),
                   height_mode=LaunchConfiguration('height_mode').perform(context),
                   search_timeout=float(LaunchConfiguration('search_timeout').perform(context)),
                   search_range=2.0, search_seed=7, robot_radius=0.2, safety_margin=0.05,
                   reference_clearance=0.2, control_spacing=0.4, min_radius=0.03,
                   max_radius=1.5, corridor_budget_ms=1000.,
                   local_range=float(LaunchConfiguration('local_range').perform(context)),
                   local_progress=7.0,
                   output_dir=LaunchConfiguration('output_dir').perform(context))
    return [
        Node(package='mockamap', executable='mockamap_node', name='mockamap_node',
             parameters=[generator], remappings=[('mock_map', '/voxel_map')], output='screen'),
        Node(package='gcopter', executable='direct_corridor_demo', name='direct_demo',
             parameters=[planner], output='screen'),
        Node(package='rviz2', executable='rviz2', name='direct_corridor_rviz', output='screen',
             arguments=['-d', os.path.join(share, 'config', 'direct_corridor.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='perlin'),
        DeclareLaunchArgument('corridor', default_value='c3',
                              description='c3 (C2) or h3 (C1, generation only)'),
        DeclareLaunchArgument('map_seed', default_value='1024'),
        DeclareLaunchArgument('height', default_value='2.5'),
        DeclareLaunchArgument('height_mode', default_value='fixed'),
        DeclareLaunchArgument('search_timeout', default_value='0.5'),
        DeclareLaunchArgument('local_range', default_value='3.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('output_dir', default_value='/tmp/gcopter_direct_demo'),
        OpaqueFunction(function=setup),
    ])
