from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot_explorer',
            executable='reactive_controller',
            name='reactive_controller',
            output='screen'
        ),
        Node(
            package='turtlebot_explorer',
            executable='frontier_detector',
            name='frontier_detector',
            output='screen'
        ),
        Node(
            package='turtlebot_explorer',
            executable='path_planner',
            name='path_planner',
            output='screen'
        ),
        Node(
            package='turtlebot_explorer',
            executable='explorer_controller',
            name='explorer_controller',
            output='screen'
        ),
    ])