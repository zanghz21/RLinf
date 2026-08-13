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

"""Joint-target perturbation for widening expert state coverage."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class JointActionPerturbation(gym.Wrapper):
    """Perturb the arm targets sent to the simulator, not the reported action.

    Recording wrappers must sit *outside* this wrapper so datasets keep the
    expert's nominal action as the label for the state it was computed from.
    The simulator executes a perturbed controller-native action, which pushes
    the robot off the nominal path. The wrapper interprets ``std`` in radians
    for both absolute and normalized delta joint-position control.

    Args:
        env: Single-environment ManiSkill env using absolute or delta
            joint-position control.
        probability: Per-step probability of perturbing the executed action.
        std: Standard deviation, in radians, of the Gaussian noise added to the
            arm joint targets.
        num_arm_joints: Number of leading action dimensions that hold arm joint
            targets. Trailing dimensions (e.g. the gripper) are left untouched.
        seed: Optional seed for the perturbation RNG. ``reset(seed=...)``
            reseeds it so collection stays reproducible.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        probability: float,
        std: float,
        num_arm_joints: int = 7,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if std < 0.0:
            raise ValueError(f"std must be non-negative, got {std}")
        if num_arm_joints <= 0:
            raise ValueError(f"num_arm_joints must be positive, got {num_arm_joints}")

        self.probability = probability
        self.std = std
        self.num_arm_joints = num_arm_joints
        self._rng = np.random.default_rng(seed)
        self.num_steps = 0
        self.num_perturbed_steps = 0

        self.control_mode = getattr(self.env.unwrapped, "control_mode", None)
        if self.control_mode not in {"pd_joint_pos", "pd_joint_delta_pos"}:
            raise ValueError(
                "JointActionPerturbation requires pd_joint_pos or "
                f"pd_joint_delta_pos control, got {self.control_mode!r}"
            )
        if (
            self.action_space.shape is None
            or self.action_space.shape[-1] < num_arm_joints
        ):
            raise ValueError(
                "action space must contain at least "
                f"{num_arm_joints} arm dimensions"
            )

        self._arm_low = np.asarray(
            self.action_space.low[:num_arm_joints], dtype=np.float64
        )
        self._arm_high = np.asarray(
            self.action_space.high[:num_arm_joints], dtype=np.float64
        )
        self._noise_scale = np.ones(num_arm_joints, dtype=np.float64)
        if self.control_mode == "pd_joint_delta_pos":
            arm_controller = self.env.unwrapped.agent.controller.controllers[
                "arm"
            ]
            if not (
                arm_controller.config.use_delta
                and arm_controller.config.normalize_action
            ):
                raise ValueError(
                    "pd_joint_delta_pos perturbation requires a normalized "
                    "delta arm controller"
                )
            lower = np.broadcast_to(
                np.asarray(arm_controller.config.lower, dtype=np.float64),
                (num_arm_joints,),
            )
            upper = np.broadcast_to(
                np.asarray(arm_controller.config.upper, dtype=np.float64),
                (num_arm_joints,),
            )
            if np.any(upper <= lower):
                raise ValueError(
                    "delta controller upper bounds must exceed lower bounds"
                )
            self._noise_scale = 2.0 / (upper - lower)

    @property
    def enabled(self) -> bool:
        """Whether the wrapper can change the executed action."""
        return self.probability > 0.0 and self.std > 0.0

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        self.num_steps += 1
        if self.enabled and self._rng.random() < self.probability:
            self.num_perturbed_steps += 1
            action = self._perturb(action)
        return self.env.step(action)

    def _perturb(self, action: Any) -> Any:
        """Return ``action`` with Gaussian noise added to the arm targets."""
        noise = self._rng.normal(0.0, self.std, self.num_arm_joints)
        noise *= self._noise_scale
        if isinstance(action, torch.Tensor):
            if not action.is_floating_point():
                raise TypeError("joint actions must use a floating-point dtype")
            perturbed = action.clone()
            arm_action = perturbed[..., : self.num_arm_joints]
            arm_action += torch.as_tensor(
                noise, dtype=perturbed.dtype, device=perturbed.device
            )
            arm_low = torch.as_tensor(
                self._arm_low, dtype=perturbed.dtype, device=perturbed.device
            )
            arm_high = torch.as_tensor(
                self._arm_high, dtype=perturbed.dtype, device=perturbed.device
            )
            perturbed[..., : self.num_arm_joints] = torch.maximum(
                torch.minimum(arm_action, arm_high), arm_low
            )
            return perturbed
        perturbed = np.array(action, copy=True)
        if not np.issubdtype(perturbed.dtype, np.floating):
            raise TypeError("joint actions must use a floating-point dtype")
        arm_action = perturbed[..., : self.num_arm_joints]
        arm_action += noise
        perturbed[..., : self.num_arm_joints] = np.clip(
            arm_action, self._arm_low, self._arm_high
        )
        return perturbed
