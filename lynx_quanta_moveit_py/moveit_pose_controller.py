#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float32, String
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration as RosDuration

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    RobotState,
    PlanningOptions,
    WorkspaceParameters,
)
from geometry_msgs.msg import Vector3
from shape_msgs.msg import SolidPrimitive


ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

GRIPPER_JOINT_NAMES = [
    "arm_joint7",
    "arm_joint8",
]

NAMED_POSES = {
    "home": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "ready": [
        0.0,
        math.pi / 4.0,
        -math.pi / 4.0,
        0.0,
        math.pi / 4.0,
        0.0,
    ],
    "stow": [
        0.0,
        math.pi / 2.0,
        -math.pi * 2.0 / 3.0,
        0.0,
        math.pi / 6.0,
        0.0,
    ],
}

# Your URDF has arm_joint7 and arm_joint8 both:
# lower=0.0, upper=0.035, positive axis.
GRIPPER_OPEN = [0.035, 0.035]
GRIPPER_CLOSE = [0.0, 0.0]


class MoveItPoseController(Node):

    def __init__(self):
        super().__init__("moveit_piper_pose_controller")

        self.declare_parameter("planning_group", "piper_arm")
        self.declare_parameter("ik_link_name", "arm_link6")
        self.declare_parameter("gripper_duration_s", 4.0)
        self.declare_parameter("position_tolerance_m", 0.015)
        self.declare_parameter("orientation_tolerance_rad", 3.14)
        self.declare_parameter("planning_time_s", 8.0)
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("velocity_scale", 0.10)
        self.declare_parameter("acceleration_scale", 0.10)

        self.planning_group = str(self.get_parameter("planning_group").value)
        self.ik_link_name = str(self.get_parameter("ik_link_name").value)
        self.gripper_duration_s = float(self.get_parameter("gripper_duration_s").value)
        self.position_tolerance_m = float(self.get_parameter("position_tolerance_m").value)
        self.orientation_tolerance_rad = float(
            self.get_parameter("orientation_tolerance_rad").value
        )
        self.planning_time_s = float(self.get_parameter("planning_time_s").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.velocity_scale = float(self.get_parameter("velocity_scale").value)
        self.acceleration_scale = float(self.get_parameter("acceleration_scale").value)

        self._latest_joint_state: JointState | None = None

        self._arm_busy = False
        self._gripper_busy = False

        self._last_gripper = GRIPPER_CLOSE.copy()
        self._last_gripper_cmd_value: float | None = None
        self._last_gripper_command_time = 0.0

        self._plan_candidates: list[PoseStamped] = []
        self._plan_candidate_index = 0

        self._move_group_ac = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        self._grip_ac = ActionClient(
            self,
            FollowJointTrajectory,
            "gripper_controller/follow_joint_trajectory",
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            20,
        )

        self.create_subscription(
            PoseStamped,
            "/arm/target_pose",
            self._target_pose_cb,
            10,
        )

        self.create_subscription(
            String,
            "/arm/named_pose",
            self._named_pose_cb,
            10,
        )

        self.create_subscription(
            Float32,
            "/arm/gripper_cmd",
            self._gripper_cb,
            10,
        )

        self.get_logger().info(
            "MoveItPoseController ready.\n"
            "  /arm/target_pose -> MoveGroup full plan+execute\n"
            "  /arm/named_pose  -> MoveGroup joint goal\n"
            "  /arm/gripper_cmd -> gripper_controller\n"
            f"  planning_group={self.planning_group}\n"
            f"  ik_link_name={self.ik_link_name}\n"
            f"  position_tolerance_m={self.position_tolerance_m:.3f}\n"
            f"  orientation_tolerance_rad={self.orientation_tolerance_rad:.3f}\n"
            f"  planning_time_s={self.planning_time_s:.2f}\n"
            f"  velocity_scale={self.velocity_scale:.2f}\n"
            f"  acceleration_scale={self.acceleration_scale:.2f}"
        )

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        self._latest_joint_state = msg

    def _ros_dur(self, seconds: float) -> RosDuration:
        seconds = max(0.0, float(seconds))
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return RosDuration(sec=sec, nanosec=nanosec)

    def _rpy_to_quat(self, roll: float, pitch: float, yaw: float):
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw

    def _make_pose(self, x, y, z, roll_deg, pitch_deg, yaw_deg, frame_id):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        qx, qy, qz, qw = self._rpy_to_quat(
            math.radians(float(roll_deg)),
            math.radians(float(pitch_deg)),
            math.radians(float(yaw_deg)),
        )

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def _make_start_state(self):
        state = RobotState()
        if self._latest_joint_state is not None:
            state.joint_state = self._latest_joint_state
        return state

    # ─────────────────────────────────────────────
    # Manual pose target using full MoveIt planning
    # ─────────────────────────────────────────────

    def _target_pose_cb(self, msg: PoseStamped):
        if self._arm_busy:
            self.get_logger().warn("Arm is busy. Ignoring new target pose.")
            return

        if self._latest_joint_state is None:
            self.get_logger().warn("No /joint_states received yet. Cannot plan.")
            return

        if not self._move_group_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "/move_action not available. Is move_group running?"
            )
            return

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        frame_id = msg.header.frame_id if msg.header.frame_id else "arm_base_link"

        self.get_logger().info(
            f"Target pose received: frame={frame_id}, "
            f"xyz=({x:.3f}, {y:.3f}, {z:.3f})"
        )

        # Broad typo guard.
        if not (-0.63 <= x <= 0.63 and -0.63 <= y <= 0.63 and 0.00 <= z <= 0.70):
            self.get_logger().warn(
                f"Target rejected: outside numeric bounds. "
                f"xyz=({x:.3f}, {y:.3f}, {z:.3f})"
            )
            return

        # 3-D reach guard — J1 axis is at z=0.123 m inside arm_base_link.
        # Piper X official reach 626 mm measured from J1.
        J1_Z = 0.123
        REACH = 0.626
        reach = math.sqrt(x ** 2 + y ** 2 + (z - J1_Z) ** 2)
        if reach > REACH:
            max_x = math.sqrt(max(0.0, REACH ** 2 - y ** 2 - (z - J1_Z) ** 2))
            max_y = math.sqrt(max(0.0, REACH ** 2 - x ** 2 - (z - J1_Z) ** 2))
            max_z = J1_Z + math.sqrt(max(0.0, REACH ** 2 - x ** 2 - y ** 2))
            self.get_logger().warn(
                f"Target rejected: 3-D reach = {reach * 1000:.0f} mm > 626 mm. "
                f"xyz=({x:.3f}, {y:.3f}, {z:.3f})\n"
                f"  With your y={y:.3f} and z={z:.3f} → max |x| = {max_x:.3f} m\n"
                f"  With your x={x:.3f} and z={z:.3f} → max |y| = {max_y:.3f} m\n"
                f"  With your x={x:.3f} and y={y:.3f} → max z  = {max_z:.3f} m"
            )
            return

        candidates: list[PoseStamped] = []

        exact = PoseStamped()
        exact.header.frame_id = frame_id
        exact.header.stamp = self.get_clock().now().to_msg()
        exact.pose.position.x = x
        exact.pose.position.y = y
        exact.pose.position.z = z
        exact.pose.orientation = msg.pose.orientation
        candidates.append(exact)

        # These are fallback tool orientations for the same xyz.
        # Since orientation tolerance is broad, this mainly helps seed feasible poses.
        auto_rpy_deg = [
            (0, 0, 0),
            (0, 0, 90),
            (0, 0, -90),
            (0, 0, 180),
            (0, 30, 0),
            (0, -30, 0),
            (0, 45, 0),
            (0, -45, 0),
            (0, 60, 0),
            (0, -60, 0),
            (0, 30, 90),
            (0, 30, -90),
            (0, 30, 180),
            (0, -30, 90),
            (0, -30, -90),
            (0, -30, 180),
            (90, 0, 0),
            (-90, 0, 0),
            (90, 0, 90),
            (-90, 0, -90),
            (0, 90, 0),
            (0, -90, 0),
            (0, 90, 90),
            (0, -90, -90),
        ]

        for roll_deg, pitch_deg, yaw_deg in auto_rpy_deg:
            candidates.append(
                self._make_pose(
                    x,
                    y,
                    z,
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                    frame_id,
                )
            )

        self._plan_candidates = candidates
        self._plan_candidate_index = 0
        self._try_next_pose_plan_candidate()

    def _make_pose_constraints(self, pose_msg: PoseStamped) -> Constraints:
        constraints = Constraints()
        constraints.name = "pose_goal"

        # Position constraint as a small box around target.
        pos_constraint = PositionConstraint()
        pos_constraint.header = pose_msg.header
        pos_constraint.link_name = self.ik_link_name
        pos_constraint.weight = 1.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        side = max(0.002, self.position_tolerance_m * 2.0)
        box.dimensions = [side, side, side]

        box_pose = Pose()
        box_pose.position = pose_msg.pose.position
        box_pose.orientation.w = 1.0

        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(box_pose)

        constraints.position_constraints.append(pos_constraint)

        # Broad orientation constraint.
        # This avoids exact-orientation IK failures but still gives MoveIt a tool pose.
        ori_constraint = OrientationConstraint()
        ori_constraint.header = pose_msg.header
        ori_constraint.link_name = self.ik_link_name
        ori_constraint.orientation = pose_msg.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = self.orientation_tolerance_rad
        ori_constraint.absolute_y_axis_tolerance = self.orientation_tolerance_rad
        ori_constraint.absolute_z_axis_tolerance = self.orientation_tolerance_rad
        ori_constraint.weight = 0.1

        constraints.orientation_constraints.append(ori_constraint)

        return constraints

    def _try_next_pose_plan_candidate(self):
        if self._plan_candidate_index >= len(self._plan_candidates):
            self.get_logger().warn(
                "MoveGroup planning failed for all pose candidates."
            )
            return

        pose = self._plan_candidates[self._plan_candidate_index]
        idx = self._plan_candidate_index
        self._plan_candidate_index += 1

        request = MotionPlanRequest()
        request.group_name = self.planning_group
        request.start_state = self._make_start_state()
        request.num_planning_attempts = self.planning_attempts
        request.allowed_planning_time = self.planning_time_s
        request.max_velocity_scaling_factor = self.velocity_scale
        request.max_acceleration_scaling_factor = self.acceleration_scale
        request.goal_constraints.append(self._make_pose_constraints(pose))

        ws = WorkspaceParameters()
        ws.header.frame_id = "arm_base_link"
        ws.min_corner = Vector3(x=-0.65, y=-0.65, z=-0.35)
        ws.max_corner = Vector3(x=0.65, y=0.65, z=0.75)
        request.workspace_parameters = ws

        goal = MoveGroup.Goal()
        goal.request = request

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay = 0.2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self._arm_busy = True

        self.get_logger().info(
            f"Trying MoveGroup pose plan {idx + 1}/{len(self._plan_candidates)} "
            f"for link '{self.ik_link_name}'"
        )

        future = self._move_group_ac.send_goal_async(goal)
        future.add_done_callback(self._pose_plan_goal_response_cb)

    def _pose_plan_goal_response_cb(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self._arm_busy = False
            self.get_logger().error(f"MoveGroup pose goal failed before acceptance: {e}")
            return

        if not handle.accepted:
            self._arm_busy = False
            self.get_logger().warn("MoveGroup pose goal rejected. Trying next candidate.")
            self._try_next_pose_plan_candidate()
            return

        self.get_logger().info("MoveGroup pose goal accepted, planning/executing...")
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._pose_plan_result_cb)

    def _pose_plan_result_cb(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self._arm_busy = False
            self.get_logger().error(f"MoveGroup pose result failed: {e}")
            return

        if result.error_code.val == 1:
            self._arm_busy = False
            self.get_logger().info("MoveGroup pose plan+execute complete")
            return

        self.get_logger().warn(
            f"MoveGroup pose failed. error_code={result.error_code.val}. "
            "Trying next candidate."
        )

        self._arm_busy = False
        self._try_next_pose_plan_candidate()

    # ─────────────────────────────────────────────
    # Named joint targets using MoveGroup planning
    # ─────────────────────────────────────────────

    def _named_pose_cb(self, msg: String):
        name = msg.data.strip().lower()

        if name not in NAMED_POSES:
            self.get_logger().error(
                f"Unknown named pose '{name}'. Valid: {list(NAMED_POSES.keys())}"
            )
            return

        if self._arm_busy:
            self.get_logger().warn(f"Arm is busy. Ignoring named pose '{name}'.")
            return

        if self._latest_joint_state is None:
            self.get_logger().warn("No /joint_states received yet. Cannot plan.")
            return

        if not self._move_group_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "/move_action not available. Is move_group running?"
            )
            return

        self.get_logger().info(f"Named pose via MoveGroup: {name}")
        self._send_joint_goal_via_move_group(NAMED_POSES[name])

    def _send_joint_goal_via_move_group(self, q):
        request = MotionPlanRequest()
        request.group_name = self.planning_group
        request.start_state = self._make_start_state()
        request.num_planning_attempts = self.planning_attempts
        request.allowed_planning_time = self.planning_time_s
        request.max_velocity_scaling_factor = self.velocity_scale
        request.max_acceleration_scaling_factor = self.acceleration_scale

        constraints = Constraints()
        constraints.name = "joint_goal"

        for joint_name, joint_value in zip(ARM_JOINT_NAMES, q):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joint_value)
            jc.tolerance_above = 0.02
            jc.tolerance_below = 0.02
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request.goal_constraints.append(constraints)

        goal = MoveGroup.Goal()
        goal.request = request

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay = 0.2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self._arm_busy = True

        future = self._move_group_ac.send_goal_async(goal)
        future.add_done_callback(self._joint_goal_response_cb)

    def _joint_goal_response_cb(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self._arm_busy = False
            self.get_logger().error(f"MoveGroup joint goal failed before acceptance: {e}")
            return

        if not handle.accepted:
            self._arm_busy = False
            self.get_logger().error("MoveGroup joint goal rejected")
            return

        self.get_logger().info("MoveGroup joint goal accepted, planning/executing...")
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._joint_goal_result_cb)

    def _joint_goal_result_cb(self, future):
        self._arm_busy = False

        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f"MoveGroup joint result failed: {e}")
            return

        if result.error_code.val == 1:
            self.get_logger().info("MoveGroup joint plan+execute complete")
        else:
            self.get_logger().warn(
                f"MoveGroup joint goal failed. error_code={result.error_code.val}"
            )

    # ─────────────────────────────────────────────
    # Gripper direct controller action
    # ─────────────────────────────────────────────

    def _gripper_cb(self, msg: Float32):
        t = max(0.0, min(1.0, float(msg.data)))

        now = time.monotonic()

        if self._last_gripper_cmd_value is not None:
            same_command = abs(t - self._last_gripper_cmd_value) < 1e-6
            too_soon = (now - self._last_gripper_command_time) < 0.75

            if same_command and too_soon:
                self.get_logger().info(
                    f"Gripper command {t:.0%} ignored: repeated too fast."
                )
                return

        if self._gripper_busy:
            self.get_logger().warn("Gripper is busy. Ignoring new gripper command.")
            return

        self._last_gripper_cmd_value = t
        self._last_gripper_command_time = now

        self._last_gripper = [
            GRIPPER_OPEN[0] * t,
            GRIPPER_OPEN[1] * t,
        ]

        self.get_logger().info(
            f"Gripper command: {t:.0%}, joints={self._last_gripper}"
        )

        self._send_gripper(self._last_gripper)

    def _send_gripper(self, q):
        if not self._grip_ac.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("gripper_controller action server not available")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.velocities = [0.0, 0.0]
        point.time_from_start = self._ros_dur(self.gripper_duration_s)

        goal.trajectory.points = [point]

        self._gripper_busy = True

        future = self._grip_ac.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response_cb)

    def _gripper_goal_response_cb(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self._gripper_busy = False
            self.get_logger().error(f"Gripper goal failed before acceptance: {e}")
            return

        if not handle.accepted:
            self._gripper_busy = False
            self.get_logger().error("Gripper goal rejected")
            return

        self.get_logger().info("Gripper goal accepted, executing...")
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._gripper_result_cb)

    def _gripper_result_cb(self, future):
        self._gripper_busy = False

        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f"Gripper result failed: {e}")
            return

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Gripper motion complete")
        else:
            self.get_logger().warn(
                f"Gripper error code={result.error_code}: {result.error_string}"
            )


def main():
    rclpy.init()
    node = MoveItPoseController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()