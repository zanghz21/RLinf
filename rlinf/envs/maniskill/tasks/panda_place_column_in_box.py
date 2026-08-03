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

"""Pick-and-place: place a column into an open-top box with a sideways-mounted Panda.

The Franka Panda root frame normally has +Z upward. In this task the robot is
mounted so that root +Z is horizontal (along world -Y), as if the arm were
wall-mounted facing the table workspace.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.quaternions import mat2quat


@register_agent()
class PandaPlaceColumnWristCam(PandaWristCam):
    """Panda wrist-cam agent with task-tunable ``hand_camera`` parameters."""

    uid = "panda_place_column_wristcam"

    # Tune these to configure the wrist / hand camera.
    # ``hand_camera_pose`` is local on ``camera_link``:
    #   world_pose = camera_link.pose * hand_camera_pose
    hand_camera_width = 224
    hand_camera_height = 224
    hand_camera_fov = np.pi / 2
    hand_camera_near = 0.01
    hand_camera_far = 100
    hand_camera_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=self.hand_camera_pose,
                width=self.hand_camera_width,
                height=self.hand_camera_height,
                fov=self.hand_camera_fov,
                near=self.hand_camera_near,
                far=self.hand_camera_far,
                mount=self.robot.links_map["camera_link"],
            )
        ]


@register_env("PandaPlaceColumnInBox-v1", max_episode_steps=100)
class PandaPlaceColumnInBoxEnv(BaseEnv):
    """Place an upright cylindrical column into an open-top box with a horizontally mounted Panda.

    **Task Description:**
    Grasp a cylindrical column on the table and place it inside an open box
    (bottom + four walls, no lid). The Panda base is mounted with root +Z
    horizontal along world -Y. Uses ``PandaPlaceColumnWristCam`` so a wrist/hand
    camera is available in addition to the scene ``base_camera``.

    **Randomizations:**
    - Column xy is randomized in a left region of the table.
    - Box xy is randomized in a right region of the table.
    - Column yaw about world z is randomized.

    **Success Conditions:**
    - Column xy is inside the box inner opening.
    - Column rests near the box floor (not hovering high above the rim).
    - Column cylinder axis (local +X) points upward (world +Z).
    - Column is static and not grasped.
    """

    SUPPORTED_ROBOTS = ["panda_place_column_wristcam"]
    agent: Union[PandaPlaceColumnWristCam]

    # Cylindrical column: diameter 6cm, height 20cm.
    # Sapien cylinders are aligned with local +X by default (lying along X).
    # Rotate so the cylinder axis stands along world +Z.
    column_radius = 0.03
    column_half_length = 0.1
    # Small red tip sitting on top of the 3-color column.
    column_tip_radius = 0.015
    column_tip_half_length = 0.005
    # Ry=-90deg: local +X -> world +Z (upright).
    column_upright_rot = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    column_upright_quat = mat2quat(column_upright_rot)  # wxyz
    # Candidate XY reset anchors; a uniform perturbation in
    # [-column_xy_noise, +column_xy_noise] is added after sampling.
    column_xy_list = [
        [-0.4, -0.1],
        [-0.1, -0.1],
        [0.2, -0.1],
        [-0.1, -0.5],
        [0.2, -0.5],
    ]
    column_xy_noise = 0.02

    # Open box (no top). Inner half-extents are independently configurable:
    #   width  -> local X, length -> local Y.
    box_inner_half_width = 0.1   # half-size along X (full inner width = 10cm)
    box_inner_half_length = 0.05  # half-size along Y (full inner length = 8cm)
    box_wall_thickness = 0.005
    box_wall_half_height = 0.04
    # Success tolerances account for small contact-solver offsets after release.
    success_xy_tolerance = 0.005
    success_z_lower_tolerance = 0.01
    success_z_upper_tolerance = 0.02
    # Uniform yaw perturbation about world Z, in radians.
    box_yaw_noise = 0.15

    # Wall-mounted Panda base pose. Columns of ``robot_base_rot`` are the root
    # frame axes expressed in world coordinates:
    #   root +X -> world +X
    #   root +Y -> world +Z
    #   root +Z -> world -Y  (horizontal)
    robot_base_pos = [-0.55, -0.2, 0.5]
    robot_base_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    robot_base_quat = mat2quat(robot_base_rot)  # wxyz
    # Ready configuration expressed in the root frame.
    robot_init_qpos = [
        0.0,
        -0.8,
        0.0,
        -2.2,
        0.0,
        1.2,
        - np.pi / 4,
        # gripper
        0.04,
        0.04,
    ]

    def __init__(
        self,
        *args,
        robot_uids="panda_place_column_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Scene camera. Wrist/hand camera is configured on
        # ``PandaPlaceColumnWristCam`` (``hand_camera``).
        pose = sapien_utils.look_at(eye=[-0.35, 0.0, 0.7], target=[0.0, 0.0, 0.08])
        return [
            CameraConfig("base_camera", pose, 224, 224, np.pi * 3 / 4, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Wide side view that shows the wall-mounted base, column, and open box.
        pose = sapien_utils.look_at([0.85, 0.0, 0.65], [-0.05, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, np.pi/2, 0.01, 100)

    def _robot_init_pose(self) -> sapien.Pose:
        return sapien.Pose(p=self.robot_base_pos, q=self.robot_base_quat)

    def _load_agent(self, options: dict):
        super()._load_agent(options, self._robot_init_pose())

    def _build_open_box(self):
        """Build a kinematic open-top box (bottom + four walls)."""
        builder = self.scene.create_actor_builder()

        t = self.box_wall_thickness
        inner_x = self.box_inner_half_width
        inner_y = self.box_inner_half_length
        outer_x = inner_x + t
        outer_y = inner_y + t
        h = self.box_wall_half_height
        # Bottom plate sits under the walls; its top face is at z = t.
        bottom_half_h = t * 0.5
        wall_z = t + h

        mat = sapien.render.RenderMaterial(
            base_color=np.array([0, 100, 0, 255]) / 255,
            roughness=0.6,
            specular=0.3,
        )
        wall_mat = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba("#FFFFFF"),
            roughness=0.6,
            specular=0.3,
        )

        # Bottom (larger than inner opening by wall thickness on each side).
        builder.add_box_collision(
            sapien.Pose([0, 0, bottom_half_h]),
            half_size=[outer_x, outer_y, bottom_half_h],
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, bottom_half_h]),
            half_size=[outer_x, outer_y, bottom_half_h],
            material=mat,
        )

        # Four walls (+x, -x, +y, -y).
        wall_specs = [
            ([outer_x - t * 0.5, 0.0, wall_z], [t * 0.5, outer_y, h]),
            ([-outer_x + t * 0.5, 0.0, wall_z], [t * 0.5, outer_y, h]),
            ([0.0, outer_y - t * 0.5, wall_z], [inner_x, t * 0.5, h]),
            ([0.0, -outer_y + t * 0.5, wall_z], [inner_x, t * 0.5, h]),
        ]
        for pos, half_size in wall_specs:
            pose = sapien.Pose(pos)
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=wall_mat)

        builder.initial_pose = sapien.Pose(p=[0.1, 0, 0])
        return builder.build_kinematic(name="open_box")

    def _recolor_table_white(self):
        """Remove the wood texture and paint the tabletop white."""
        for part in self.table_scene.table._objs:
            render_body = part.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                mesh_parts = getattr(shape, "parts", None)
                if mesh_parts is None:
                    continue
                for triangle in mesh_parts:
                    triangle.material.set_base_color(
                        np.array([255, 255, 255, 255]) / 255
                    )
                    triangle.material.set_base_color_texture(None)
                    triangle.material.set_normal_texture(None)
                    triangle.material.set_emission_texture(None)
                    triangle.material.set_transmission_texture(None)
                    triangle.material.set_metallic_texture(None)
                    triangle.material.set_roughness_texture(None)

    def _build_tricolor_column(self):
        """Build an upright white/red/white cylinder with a small red tip on top.

        Sapien cylinders are aligned with local +X. After ``column_upright_quat``,
        local +X maps to world +Z, so the segment order below is bottom→top.
        """
        builder = self.scene.create_actor_builder()
        radius = self.column_radius
        half_len = self.column_half_length
        seg_half = half_len / 3.0
        tip_radius = self.column_tip_radius
        tip_half = self.column_tip_half_length
        # Tip sits on the top face of the main column (local +X end).
        tip_center_x = half_len + tip_half

        white = np.array([1.0, 1.0, 1.0, 1.0])
        red = np.array([220, 12, 12, 255]) / 255

        # Collision: main body + tip.
        builder.add_cylinder_collision(radius=radius, half_length=half_len)
        builder.add_cylinder_collision(
            pose=sapien.Pose(p=[tip_center_x, 0.0, 0.0]),
            radius=tip_radius,
            half_length=tip_half,
        )

        # Visual thirds along local X: white, red, white (bottom → top when upright).
        segment_specs = [
            (-2.0 * seg_half, white),  # first 1/3
            (0.0, red),  # middle 1/3
            (2.0 * seg_half, white),  # last 1/3
        ]
        for center_x, color in segment_specs:
            mat = sapien.render.RenderMaterial(base_color=color)
            builder.add_cylinder_visual(
                pose=sapien.Pose(p=[center_x, 0.0, 0.0]),
                radius=radius,
                half_length=seg_half,
                material=mat,
            )

        tip_mat = sapien.render.RenderMaterial(base_color=red)
        builder.add_cylinder_visual(
            pose=sapien.Pose(p=[tip_center_x, 0.0, 0.0]),
            radius=tip_radius,
            half_length=tip_half,
            material=tip_mat,
        )

        builder.initial_pose = sapien.Pose(
            p=[0, 0, half_len],
            q=self.column_upright_quat,
        )
        return builder.build(name="column")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._recolor_table_white()

        self.column = self._build_tricolor_column()
        self.box = self._build_open_box()

    def _reset_horizontal_panda(self, env_idx: torch.Tensor):
        """Reset joint state and force the sideways base pose.

        ``TableSceneBuilder.initialize`` places a standard upright Panda; we
        overwrite pose/qpos afterward for the horizontal mount.
        """
        b = len(env_idx)
        qpos = np.array(self.robot_init_qpos, dtype=np.float64)
        if self._enhanced_determinism:
            noise = self._batched_episode_rng[env_idx].normal(
                0, self.robot_init_qpos_noise, len(qpos)
            )
            qpos = qpos + noise
        else:
            noise = self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos))
            )
            qpos = qpos + noise
        if qpos.ndim == 1:
            qpos = np.tile(qpos, (b, 1))
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        self.agent.robot.set_pose(self._robot_init_pose())

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._reset_horizontal_panda(env_idx)

            # Column: sample an XY anchor, then add a small random perturbation.
            col_xy_anchors = torch.tensor(
                self.column_xy_list, dtype=torch.float32, device=self.device
            )
            column_xy_index = options.get("column_xy_index")
            if column_xy_index is None:
                col_xy_rand_id = torch.randint(
                    0, len(self.column_xy_list), (b,), device=self.device
                )
            else:
                col_xy_rand_id = torch.as_tensor(
                    column_xy_index, dtype=torch.long, device=self.device
                )
                if col_xy_rand_id.ndim == 0:
                    col_xy_rand_id = col_xy_rand_id.repeat(b)
                if col_xy_rand_id.shape != (b,):
                    raise ValueError(
                        "options['column_xy_index'] must be a scalar or have "
                        f"shape ({b},)"
                    )
                if torch.any(
                    (col_xy_rand_id < 0)
                    | (col_xy_rand_id >= len(self.column_xy_list))
                ):
                    raise ValueError(
                        "options['column_xy_index'] is outside column_xy_list"
                    )
            col_xy = col_xy_anchors[col_xy_rand_id]
            col_xy = col_xy + (
                torch.rand((b, 2), device=self.device) * 2 - 1
            ) * self.column_xy_noise

            col_xyz = torch.zeros((b, 3), device=self.device)
            col_xyz[:, :2] = col_xy
            col_xyz[:, 2] = self.column_half_length
            self.column.set_pose(
                Pose.create_from_pq(p=col_xyz, q=self.column_upright_quat)
            )

            # Open box on the right half of the workspace.
            box_xyz = torch.zeros((b, 3), device=self.device)
            box_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.10 - 0.15
            box_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.05 + 0.05
            box_xyz[:, 2] = 0.0
            # Small random yaw about world Z.
            box_yaw = (
                torch.rand((b,), device=self.device) * 2 - 1
            ) * self.box_yaw_noise
            half_yaw = box_yaw * 0.5
            box_quat = torch.zeros((b, 4), device=self.device)
            box_quat[:, 0] = torch.cos(half_yaw)  # w
            box_quat[:, 3] = torch.sin(half_yaw)  # z
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_quat))

    def _column_rest_z(self) -> float:
        """World-z of column center when sitting on the box floor."""
        # Box floor top is at box_pose.z + wall_thickness.
        return self.box_wall_thickness + self.column_half_length

    def evaluate(self):
        col_pos = self.column.pose.p
        box_pos = self.box.pose.p
        offset_world = col_pos - box_pos
        # Express offset in the box frame so yaw randomization is respected.
        # For yaw-only quats [w,0,0,z]: cos=w^2-z^2, sin=2wz.
        box_q = self.box.pose.q
        yaw_cos = box_q[:, 0] * box_q[:, 0] - box_q[:, 3] * box_q[:, 3]
        yaw_sin = 2.0 * box_q[:, 0] * box_q[:, 3]
        offset = torch.stack(
            [
                yaw_cos * offset_world[:, 0] + yaw_sin * offset_world[:, 1],
                -yaw_sin * offset_world[:, 0] + yaw_cos * offset_world[:, 1],
                offset_world[:, 2],
            ],
            dim=-1,
        )

        x_thresh = max(
            self.box_inner_half_width
            - self.column_radius
            + self.success_xy_tolerance,
            0.01,
        )
        y_thresh = max(
            self.box_inner_half_length
            - self.column_radius
            + self.success_xy_tolerance,
            0.01,
        )
        xy_flag = (torch.abs(offset[:, 0]) <= x_thresh) & (
            torch.abs(offset[:, 1]) <= y_thresh
        )

        rest_z = self._column_rest_z()
        height_error = offset[:, 2] - rest_z
        z_flag = (height_error >= -self.success_z_lower_tolerance) & (
            height_error <= self.success_z_upper_tolerance
        )

        is_inside = xy_flag & z_flag

        # Sapien cylinders align with local +X; upright means local +X ≈ world +Z.
        col_rot = self.column.pose.to_transformation_matrix()[..., :3, :3]
        axis_up = col_rot[..., 2, 0]  # world-z of local +X
        # Allow ~25deg tilt from vertical (cos(25deg) ≈ 0.906).
        is_upright = axis_up >= 0.9

        is_grasped = self.agent.is_grasping(self.column)
        is_static = self.column.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        linear_speed = torch.linalg.norm(self.column.linear_velocity, dim=-1)
        angular_speed = torch.linalg.norm(self.column.angular_velocity, dim=-1)
        success = is_inside & is_upright & is_static & (~is_grasped)
        return {
            "success": success,
            "is_inside": is_inside,
            "is_xy_inside": xy_flag,
            "is_height_valid": z_flag,
            "is_upright": is_upright,
            "is_grasped": is_grasped,
            "is_static": is_static,
            "x_offset": offset[:, 0],
            "y_offset": offset[:, 1],
            "height_error": height_error,
            "axis_up": axis_up,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            box_pos=self.box.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                column_pose=self.column.pose.raw_pose,
                tcp_to_column_pos=self.column.pose.p - self.agent.tcp_pose.p,
                column_to_box_pos=self.box.pose.p - self.column.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_col = torch.linalg.norm(
            self.column.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reward = 1 - torch.tanh(5 * tcp_to_col)

        is_grasped = info["is_grasped"]
        reward = reward + is_grasped.float()

        # While grasping, reward transporting the column over the box and
        # keeping it upright for insertion.
        place_dist = torch.linalg.norm(
            (self.column.pose.p - self.box.pose.p)[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * place_dist)
        col_rot = self.column.pose.to_transformation_matrix()[..., :3, :3]
        axis_up = col_rot[..., 2, 0]
        upright_reward = torch.clamp(axis_up, min=0.0)
        transport_reward = place_reward + upright_reward
        reward = reward + transport_reward * is_grasped.float()

        is_inside = info["is_inside"]
        is_upright = info["is_upright"]
        reward = reward + is_inside.float()

        # Releasing a correctly placed column should not lose the grasp and
        # transport rewards while the object settles. This stage has the same
        # maximum reward as holding the column inside the box.
        is_released_in_box = is_inside & is_upright & (~is_grasped)
        reward = reward + 3.0 * is_released_in_box.float()

        # Success is the unique maximum reward and matches the normalization
        # denominator below.
        reward = torch.where(info["success"], 6.0, reward)
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
