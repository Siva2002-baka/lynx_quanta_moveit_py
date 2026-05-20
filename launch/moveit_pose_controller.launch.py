from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    moveit_pose_controller = Node(
        package="lynx_quanta_moveit_py",
        executable="moveit_pose_controller",
        name="moveit_piper_pose_controller",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "planning_group": "piper_arm",
                "ik_link_name": "arm_link6",
                "gripper_duration_s": 4.0,
                "position_tolerance_m": 0.015,
                "orientation_tolerance_rad": 3.14,
                "planning_time_s": 8.0,
                "planning_attempts": 10,
                "velocity_scale": 0.10,
                "acceleration_scale": 0.10,
            }
        ],
    )

    return LaunchDescription([
        moveit_pose_controller,
    ])