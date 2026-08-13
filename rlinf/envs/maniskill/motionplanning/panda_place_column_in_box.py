# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MPlib expert for ``PandaPlaceColumnInBox-v1``.

The expert uses ManiSkill's single-environment Panda planner. It approaches the
column horizontally and places the TCP at the upper boundary of the middle red
segment before closing the gripper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.structs.pose import to_sapien_pose

from rlinf.envs.maniskill.tasks.panda_place_column_in_box import (
    PandaPlaceColumnInBoxEnv,
)


@dataclass(frozen=True)
class LateralGraspCandidate:
    """A horizontal approach candidate for the column grasp."""

    approaching: np.ndarray
    closing: np.ndarray
    pregrasp_pose: sapien.Pose
    grasp_pose: sapien.Pose


class MotionPlanningFailure(RuntimeError):
    """Report the stage at which an expert trajectory failed."""

    def __init__(self, stage: str, detail: str | None = None):
        message = f"Motion planning failed during {stage}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.stage = stage
        self.detail = detail


def single_sapien_pose(pose: Any) -> sapien.Pose:
    """Convert a single-environment ManiSkill pose to a SAPIEN pose."""
    if isinstance(pose, sapien.Pose):
        return pose
    raw_pose = pose.raw_pose
    if raw_pose.ndim == 1:
        value = raw_pose.detach().cpu().numpy()
    elif raw_pose.shape[0] == 1:
        value = raw_pose[0].detach().cpu().numpy()
    else:
        raise ValueError("The motion-planning expert requires num_envs=1")
    return sapien.Pose(p=value[:3], q=value[3:])


def red_upper_boundary_position(
    column_pose: sapien.Pose, column_half_length: float
) -> np.ndarray:
    """Return the upper boundary of the column's middle red segment."""
    boundary_pose = column_pose * sapien.Pose(
        p=[column_half_length / 3.0, 0.0, 0.0]
    )
    return np.asarray(boundary_pose.p)


def lateral_pregrasp_pose(
    grasp_pose: sapien.Pose, pregrasp_distance: float
) -> sapien.Pose:
    """Shift a grasp pose backward along its horizontal approach axis."""
    if pregrasp_distance <= 0:
        raise ValueError("pregrasp_distance must be positive")
    return grasp_pose * sapien.Pose(p=[0.0, 0.0, -pregrasp_distance])


def linear_pose_waypoints(
    start_pose: sapien.Pose,
    target_pose: sapien.Pose,
    max_translation: float,
) -> list[sapien.Pose]:
    """Split a fixed-orientation Cartesian move into short translations."""
    if max_translation <= 0:
        raise ValueError("max_translation must be positive")
    start = np.asarray(start_pose.p)
    target = np.asarray(target_pose.p)
    segment_count = max(
        1, int(np.ceil(np.linalg.norm(target - start) / max_translation))
    )
    return [
        sapien.Pose(
            p=start + (target - start) * step / segment_count,
            q=target_pose.q,
        )
        for step in range(1, segment_count + 1)
    ]


def tcp_pose_for_attached_column(
    column_position: np.ndarray,
    tcp_orientation: np.ndarray,
    tcp_to_column: sapien.Pose,
) -> sapien.Pose:
    """Place an attached column while enforcing the requested TCP rotation."""
    tcp_rotation = sapien.Pose(q=tcp_orientation)
    column_offset = np.asarray(
        (tcp_rotation * sapien.Pose(p=tcp_to_column.p)).p
    )
    return sapien.Pose(
        p=np.asarray(column_position) - column_offset,
        q=tcp_orientation,
    )


def retreat_along_tcp_negative_z(
    tcp_pose: sapien.Pose, retreat_distance: float
) -> sapien.Pose:
    """Translate backward along the TCP local Z axis."""
    if retreat_distance <= 0:
        raise ValueError("retreat_distance must be positive")
    return tcp_pose * sapien.Pose(p=[0.0, 0.0, -retreat_distance])


def qpos_target_to_normalized_delta(
    target_qpos: np.ndarray,
    current_qpos: np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Convert an absolute qpos target to a normalized delta action."""
    target_qpos = np.asarray(target_qpos)
    current_qpos = np.asarray(current_qpos)
    lower = np.broadcast_to(np.asarray(lower), target_qpos.shape)
    upper = np.broadcast_to(np.asarray(upper), target_qpos.shape)
    if target_qpos.shape != current_qpos.shape:
        raise ValueError(
            "target_qpos and current_qpos must have the same shape, got "
            f"{target_qpos.shape} and {current_qpos.shape}"
        )
    if np.any(upper <= lower):
        raise ValueError("delta action upper bounds must exceed lower bounds")

    delta_qpos = target_qpos - current_qpos
    normalized = 2.0 * (delta_qpos - lower) / (upper - lower) - 1.0
    clipped = bool(np.any((normalized < -1.0) | (normalized > 1.0)))
    return np.clip(normalized, -1.0, 1.0), clipped


def build_lateral_grasp_candidates(
    env: PandaPlaceColumnInBoxEnv,
    column_pose: sapien.Pose,
    pregrasp_distance: float,
) -> list[LateralGraspCandidate]:
    """Build the rolled world-Y grasp at the red segment's upper boundary."""
    center = red_upper_boundary_position(column_pose, env.column_half_length)
    wrist_roll = sapien.Pose(q=[0.0, 0.0, 0.0, 1.0])
    approach_directions = (
        np.array([0.0, 1.0, 0.0]),
        np.array([0.707, 0.707, 0.0]),
        np.array([-0.707, 0.707, 0.0]),
    )
    candidates = []
    for approaching in approach_directions:
        closing = np.array([-approaching[1], approaching[0], 0.0])
        grasp_pose = env.agent.build_grasp_pose(
            approaching=approaching,
            closing=closing,
            center=center,
        )
        # Roll the TCP 180 degrees around its local approach axis. This is the
        # wrist orientation reached by rotating Franka joint 7 by pi from the
        # unrolled lateral grasp and matches the preferred initial wrist branch.
        grasp_pose = grasp_pose * wrist_roll
        candidates.append(
            LateralGraspCandidate(
                approaching=approaching,
                closing=-closing,
                pregrasp_pose=lateral_pregrasp_pose(
                    grasp_pose, pregrasp_distance
                ),
                grasp_pose=grasp_pose,
            )
        )
    return candidates


class AttachedObjectPandaPlanner(PandaArmMotionPlanningSolver):
    """Panda planner that enables MPlib's attached-object collision flag."""

    max_screw_planning_attempts = 5
    max_delta_refinement_steps = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_attached_collision = False
        self.joint7_reference: float | None = None
        self.max_joint7_drift = np.pi / 2.0
        self.num_clipped_delta_actions = 0

    def _current_qpos(self) -> np.ndarray:
        return self.robot.get_qpos().cpu().numpy()[0]

    def _action_for_qpos(
        self, qpos: np.ndarray, qvel: np.ndarray | None = None
    ) -> tuple[np.ndarray, bool]:
        """Build a controller-native action for an absolute arm target."""
        if self.control_mode == "pd_joint_pos":
            return np.hstack([qpos, self.gripper_state]), False
        if self.control_mode == "pd_joint_pos_vel":
            if qvel is None:
                qvel = np.zeros_like(qpos)
            return np.hstack([qpos, qvel, self.gripper_state]), False
        if self.control_mode != "pd_joint_delta_pos":
            raise ValueError(f"Unsupported control mode: {self.control_mode}")

        arm_controller = self.base_env.agent.controller.controllers["arm"]
        if not (
            arm_controller.config.use_delta
            and arm_controller.config.normalize_action
        ):
            raise ValueError(
                "pd_joint_delta_pos requires a normalized delta arm controller"
            )
        current_qpos = arm_controller.qpos.cpu().numpy()[0]
        arm_action, clipped = qpos_target_to_normalized_delta(
            qpos,
            current_qpos,
            arm_controller.config.lower,
            arm_controller.config.upper,
        )
        return np.hstack([arm_action, self.gripper_state]), clipped

    def _step_action(self, action: np.ndarray):
        result = self.env.step(action)
        self.elapsed_steps += 1
        if self.print_env_info:
            _, reward, _, _, info = result
            print(
                f"[{self.elapsed_steps:3}] Env Output: reward={reward} "
                f"info={info}"
            )
        if self.vis:
            self.base_env.render_human()
        return result

    def _follow_qpos_target(
        self, qpos: np.ndarray, qvel: np.ndarray | None = None
    ):
        refinements = (
            self.max_delta_refinement_steps
            if self.control_mode == "pd_joint_delta_pos"
            else 0
        )
        result = None
        for _ in range(refinements + 1):
            action, clipped = self._action_for_qpos(qpos, qvel)
            if clipped:
                self.num_clipped_delta_actions += 1
            result = self._step_action(action)
            if not clipped:
                break
        return result

    def follow_path(self, result, refine_steps: int = 0):
        """Execute an MPlib path with the configured joint controller."""
        num_steps = result["position"].shape[0]
        step_result = None
        for index in range(num_steps + refine_steps):
            path_index = min(index, num_steps - 1)
            qpos = result["position"][path_index]
            qvel = (
                result["velocity"][path_index]
                if self.control_mode == "pd_joint_pos_vel"
                else None
            )
            step_result = self._follow_qpos_target(qpos, qvel)
        return step_result

    def _move_gripper(self, gripper_state: float, steps: int):
        self.gripper_state = gripper_state
        step_result = None
        arm_dof = len(self.planner.joint_vel_limits)
        for _ in range(steps):
            qpos = self.robot.get_qpos()[0, :arm_dof].cpu().numpy()
            step_result = self._follow_qpos_target(qpos)
        return step_result

    def open_gripper(self, t=6, gripper_state=None):
        """Open the gripper while holding the arm in the active mode."""
        if gripper_state is None:
            gripper_state = self.OPEN
        return self._move_gripper(gripper_state, t)

    def close_gripper(self, t=6, gripper_state=None):
        """Close the gripper while holding the arm in the active mode."""
        if gripper_state is None:
            gripper_state = self.CLOSED
        return self._move_gripper(gripper_state, t)

    def _prepare_target(self, pose: sapien.Pose) -> sapien.Pose:
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        return self._transform_pose_for_planning(pose)

    def _plan_screw_from_qpos(
        self, pose: sapien.Pose, qpos: np.ndarray
    ) -> dict[str, Any]:
        target = self._prepare_target(pose)
        return self.planner.plan_screw(
            np.concatenate([target.p, target.q]),
            qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            use_attach=self.use_attached_collision,
            wrt_world=True,
        )

    def _plan_screw_with_retries(
        self, pose: sapien.Pose, qpos: np.ndarray
    ) -> dict[str, Any]:
        """Retry screw planning when MPlib does not return success."""
        result = None
        for _ in range(self.max_screw_planning_attempts):
            result = self._plan_screw_from_qpos(pose, qpos)
            if result["status"] == "Success":
                return result
        assert result is not None
        return result

    def _joint7_is_continuous(self, result: dict[str, Any]) -> bool:
        """Reject a plan that changes wrist branch after grasping."""
        if self.joint7_reference is None or result["status"] != "Success":
            return True
        positions = np.asarray(result["position"])
        return bool(
            positions.ndim == 2
            and positions.shape[0] > 0
            and positions.shape[1] > 6
            and np.max(np.abs(positions[:, 6] - self.joint7_reference))
            <= self.max_joint7_drift
        )

    def lock_joint7_branch(self) -> None:
        """Use the current wrist angle as the post-grasp branch reference."""
        self.joint7_reference = float(self._current_qpos()[6])

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Plan and optionally execute a Cartesian screw motion."""
        result = self._plan_screw_with_retries(pose, self._current_qpos())
        if result["status"] != "Success" or not self._joint7_is_continuous(
            result
        ):
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_rrt_connect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Plan and optionally execute an RRT-Connect motion."""
        target = self._prepare_target(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([target.p, target.q]),
            self._current_qpos(),
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            use_attach=self.use_attached_collision,
            wrt_world=True,
        )
        if result["status"] != "Success" or not self._joint7_is_continuous(
            result
        ):
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def can_follow_with_screw(
        self, pose: sapien.Pose, preceding_plan: dict[str, Any]
    ) -> bool:
        """Check a screw motion from the final state of a dry-run plan."""
        qpos = preceding_plan["position"][-1]
        result = self._plan_screw_with_retries(pose, qpos)
        return result["status"] == "Success"

    def attach_column_proxy(
        self,
        tcp_pose: sapien.Pose,
        column_pose: sapien.Pose,
        env: PandaPlaceColumnInBoxEnv,
    ) -> None:
        """Attach a conservative box proxy for the grasped column."""
        tcp_to_column = tcp_pose.inv() * column_pose
        half_axis = env.column_half_length + 2.0 * env.column_tip_half_length
        size = np.array(
            [2.0 * half_axis, 2.0 * env.column_radius, 2.0 * env.column_radius]
        )
        self.planner.update_attached_box(
            size,
            np.concatenate([tcp_to_column.p, tcp_to_column.q]),
        )
        self.use_attached_collision = True


class PandaPlaceColumnMotionPlanningExpert:
    """Execute a staged motion-planning solution for the column task."""

    pregrasp_distance = 0.1
    rim_clearance = 0.03
    insertion_clearance = 0.002
    retreat_distance = 0.1
    transfer_waypoint_spacing = 0.16

    def __init__(
        self,
        env: BaseEnv,
        *,
        debug: bool = False,
        vis: bool = False,
    ):
        self.env = env
        self.base_env: PandaPlaceColumnInBoxEnv = env.unwrapped
        if self.base_env.num_envs != 1:
            raise ValueError("The motion-planning expert requires num_envs=1")
        if self.base_env.control_mode not in {
            "pd_joint_pos",
            "pd_joint_delta_pos",
            "pd_joint_pos_vel",
        }:
            raise ValueError(
                "The motion-planning expert requires pd_joint_pos, "
                "pd_joint_delta_pos, or pd_joint_pos_vel control"
            )
        self.planner = AttachedObjectPandaPlanner(
            env,
            debug=debug,
            vis=vis,
            base_pose=self.base_env.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
        )

    @staticmethod
    def _require_motion(stage: str, result):
        if isinstance(result, int) and result == -1:
            raise MotionPlanningFailure(stage)
        return result

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if hasattr(value, "item"):
            return bool(value.item())
        return bool(value)

    def _add_static_collisions(self) -> None:
        env = self.base_env
        self.planner.add_box_collision(
            extents=np.array([1.2, 1.2, 0.02]),
            pose=sapien.Pose(p=[0.0, 0.0, -0.01]),
        )

        box_pose = single_sapien_pose(env.box.pose)
        t = env.box_wall_thickness
        inner_x = env.box_inner_half_width
        inner_y = env.box_inner_half_length
        outer_x = inner_x + t
        outer_y = inner_y + t
        h = env.box_wall_half_height
        wall_z = t + h
        box_parts = (
            ([0.0, 0.0, t * 0.5], [2 * outer_x, 2 * outer_y, t]),
            ([outer_x - t * 0.5, 0.0, wall_z], [t, 2 * outer_y, 2 * h]),
            ([-outer_x + t * 0.5, 0.0, wall_z], [t, 2 * outer_y, 2 * h]),
            ([0.0, outer_y - t * 0.5, wall_z], [2 * inner_x, t, 2 * h]),
            ([0.0, -outer_y + t * 0.5, wall_z], [2 * inner_x, t, 2 * h]),
        )
        for position, extents in box_parts:
            self.planner.add_box_collision(
                extents=np.asarray(extents),
                pose=box_pose * sapien.Pose(p=position),
            )

    def _select_grasp(
        self,
    ) -> tuple[LateralGraspCandidate, np.ndarray]:
        env = self.base_env
        column_pose = single_sapien_pose(env.column.pose)
        candidates = build_lateral_grasp_candidates(
            env, column_pose, self.pregrasp_distance
        )
        # The first candidate is the canonical world +Y approach. Use its
        # rotation for placement even if a diagonal fallback grasp is selected.
        placement_tcp_orientation = np.asarray(candidates[0].grasp_pose.q)
        for candidate in candidates:
            pregrasp_plan = self.planner.move_to_pose_with_rrt_connect(
                candidate.pregrasp_pose, dry_run=True
            )
            if isinstance(pregrasp_plan, int) and pregrasp_plan == -1:
                continue
            if self.planner.can_follow_with_screw(
                candidate.grasp_pose, pregrasp_plan
            ):
                return candidate, placement_tcp_orientation
        raise MotionPlanningFailure("pregrasp_ik", "no lateral grasp is reachable")

    def execute(self):
        """Run one expert trajectory and return the final environment step."""
        env = self.base_env
        self._add_static_collisions()
        self.planner.open_gripper(t=4)

        candidate, placement_tcp_orientation = self._select_grasp()
        self._require_motion(
            "pregrasp",
            self.planner.move_to_pose_with_rrt_connect(candidate.pregrasp_pose),
        )
        self._require_motion(
            "horizontal_grasp",
            self.planner.move_to_pose_with_screw(candidate.grasp_pose),
        )
        self.planner.close_gripper(t=8)
        if not self._as_bool(env.agent.is_grasping(env.column)):
            raise MotionPlanningFailure("grasp", "the column was not secured")
        self.planner.lock_joint7_branch()

        tcp_pose = single_sapien_pose(env.agent.tcp_pose)
        column_pose = single_sapien_pose(env.column.pose)
        tcp_to_column = tcp_pose.inv() * column_pose
        self.planner.attach_column_proxy(tcp_pose, column_pose, env)

        box_pose = single_sapien_pose(env.box.pose)
        wall_top = (
            box_pose.p[2]
            + env.box_wall_thickness
            + 2.0 * env.box_wall_half_height
        )
        transfer_column_z = (
            wall_top + env.column_half_length + self.rim_clearance
        )
        lift_distance = max(transfer_column_z - column_pose.p[2], 0.0)
        lift_pose = sapien.Pose(p=[0.0, 0.0, lift_distance]) * tcp_pose
        self._require_motion(
            "lift", self.planner.move_to_pose_with_screw(lift_pose)
        )
        transfer_start_pose = single_sapien_pose(env.agent.tcp_pose)

        transfer_column_position = np.array(
            [box_pose.p[0], box_pose.p[1], transfer_column_z]
        )
        transfer_tcp_pose = tcp_pose_for_attached_column(
            transfer_column_position,
            placement_tcp_orientation,
            tcp_to_column,
        )
        for waypoint_index, waypoint in enumerate(
            linear_pose_waypoints(
                transfer_start_pose,
                transfer_tcp_pose,
                self.transfer_waypoint_spacing,
            ),
            start=1,
        ):
            self._require_motion(
                f"transfer_waypoint_{waypoint_index}",
                self.planner.move_to_pose_with_screw(waypoint),
            )

        insertion_column_position = np.array(
            [
                box_pose.p[0],
                box_pose.p[1],
                box_pose.p[2]
                + env._column_rest_z()
                + self.insertion_clearance,
            ]
        )
        insertion_tcp_pose = tcp_pose_for_attached_column(
            insertion_column_position,
            placement_tcp_orientation,
            tcp_to_column,
        )
        self.planner.use_point_cloud = False
        self._require_motion(
            "insertion",
            self.planner.move_to_pose_with_screw(
                insertion_tcp_pose, refine_steps=2
            ),
        )

        release_result = self.planner.open_gripper(t=10)
        self.planner.use_attached_collision = False

        current_tcp_pose = single_sapien_pose(env.agent.tcp_pose)
        retreat_pose = retreat_along_tcp_negative_z(
            current_tcp_pose, self.retreat_distance
        )
        retreat_result = self.planner.move_to_pose_with_screw(retreat_pose)
        if not (isinstance(retreat_result, int) and retreat_result == -1):
            release_result = retreat_result
        final_result = self.planner.open_gripper(t=10)
        if final_result is not None:
            release_result = final_result
        evaluation = env.evaluate()
        if not self._as_bool(evaluation["success"]):
            diagnostics = self._success_diagnostics(evaluation)
            raise MotionPlanningFailure(
                "success_check",
                f"the released column did not satisfy success ({diagnostics})",
            )
        return release_result

    def _success_diagnostics(self, evaluation: dict[str, Any]) -> str:
        fields = (
            "is_xy_inside",
            "is_height_valid",
            "is_upright",
            "is_static",
            "is_grasped",
            "x_offset",
            "y_offset",
            "height_error",
            "axis_up",
            "linear_speed",
            "angular_speed",
        )
        values = []
        for field in fields:
            value = evaluation[field]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float):
                values.append(f"{field}={value:.5f}")
            else:
                values.append(f"{field}={value}")
        return ", ".join(values)

    def close(self) -> None:
        """Release planner resources."""
        self.planner.close()


def solve(
    env: BaseEnv,
    seed: int | None = None,
    debug: bool = False,
    vis: bool = False,
    column_xy_index: int | None = None,
):
    """Reset and solve one ``PandaPlaceColumnInBox-v1`` episode."""
    reset_options = (
        {"column_xy_index": column_xy_index}
        if column_xy_index is not None
        else None
    )
    env.reset(seed=seed, options=reset_options)
    expert = PandaPlaceColumnMotionPlanningExpert(
        env,
        debug=debug,
        vis=vis,
    )
    try:
        return expert.execute()
    finally:
        expert.close()
