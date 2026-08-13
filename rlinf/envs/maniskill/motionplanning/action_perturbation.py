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
    The simulator executes ``action + noise``, which pushes the robot off the
    nominal path; because the expert commands absolute joint targets, its later
    waypoints then act as recovery labels for those off-path states.

    Args:
        env: Single-environment ManiSkill env using absolute joint-position
            control.
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
        if isinstance(action, torch.Tensor):
            perturbed = action.clone()
            perturbed[..., : self.num_arm_joints] += torch.as_tensor(
                noise, dtype=perturbed.dtype, device=perturbed.device
            )
            return perturbed
        perturbed = np.array(action, copy=True)
        perturbed[..., : self.num_arm_joints] += noise
        return perturbed
