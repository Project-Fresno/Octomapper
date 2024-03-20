from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="octomap_mapper",
            executable="pcl_oct",
            name="octomapper",
            output="screen",
            parameters=[
                {"depth_topic": "/depth_camera/points",
                 "ground_cutoff_height": 0.8}
            ]
        )
    ])