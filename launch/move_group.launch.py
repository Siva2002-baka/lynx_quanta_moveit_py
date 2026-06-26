import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_file(package_name, relative_path):
    package_path = get_package_share_directory(package_name)
    file_path = os.path.join(package_path, relative_path)

    with open(file_path, "r") as f:
        data = f.read()

    if not data.strip():
        raise RuntimeError(f"File is empty: {file_path}")

    return data


def load_yaml(package_name, relative_path):
    package_path = get_package_share_directory(package_name)
    file_path = os.path.join(package_path, relative_path)

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise RuntimeError(f"YAML file is empty or invalid: {file_path}")

    return data


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    pkg_lynx = get_package_share_directory("lynx_quanta")
    urdf_path = os.path.join(
        pkg_lynx,
        "urdf",
        "m20_with_arm",
        "m20_with_piper_v3.urdf",
    )

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", urdf_path]),
            value_type=str,
        )
    }

    robot_description_semantic = {
        "robot_description_semantic": load_file(
            "lynx_quanta_moveit_py",
            "config/piper.srdf",
        )
    }

    robot_description_kinematics = {
        "robot_description_kinematics": load_yaml(
            "lynx_quanta_moveit_py",
            "config/kinematics.yaml",
        )
    }

    joint_limits_yaml = load_yaml(
        "lynx_quanta_moveit_py",
        "config/joint_limits.yaml",
    )

    moveit_controllers = load_yaml(
        "lynx_quanta_moveit_py",
        "config/moveit_controllers.yaml",
    )

    ompl_planning = load_yaml(
        "lynx_quanta_moveit_py",
        "config/ompl_planning.yaml",
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            joint_limits_yaml,
            moveit_controllers,
            ompl_planning,
            {
                "use_sim_time": use_sim_time,
                "publish_robot_description": True,
                "publish_robot_description_semantic": True,
                "allow_trajectory_execution": True,
                "monitor_dynamics": False,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
        ),
        move_group,
    ])