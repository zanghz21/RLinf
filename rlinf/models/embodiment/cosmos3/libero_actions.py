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

"""Convert normalized Cosmos3 policy actions to LIBERO controller commands."""

from __future__ import annotations

import json

import numpy as np
import torch


class Cosmos3LiberoActionConverter:
    """Denormalize translation and gripper and project rot6d to axis-angle."""

    def __init__(
        self,
        num_action_chunks: int,
        raw_action_dim: int,
        normalization: str,
        stats_path: str | None,
    ):
        self._num_action_chunks = num_action_chunks
        self._raw_action_dim = raw_action_dim
        self._action_normalization = normalization
        self._stats = self._load_stats(stats_path, normalization)

    @staticmethod
    def _load_stats(
        path: str | None, normalization: str
    ) -> dict[str, np.ndarray] | None:
        # Per-channel stats keys loaded from the LIBERO quantile stats JSON.
        stat_keys = ("mean", "std", "min", "max", "q01", "q99")
        if not path:
            return None
        with open(path, encoding="utf-8") as f:  # noqa: PTH123
            raw = json.load(f)
        key = "global_raw" if normalization == "quantile_rot" else "global"
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
        return {
            k: np.asarray(v, dtype=np.float32) for k, v in raw.items() if k in stat_keys
        }

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        """Invert the action normalization for the first ``raw_action_dim`` channels."""
        d = self._raw_action_dim
        norm = action[..., :d]
        if self._stats is None:
            return norm
        method = self._action_normalization
        if method in ("quantile", "quantile_rot"):
            q01 = self._stats["q01"]
            q99 = self._stats["q99"]
            return (norm + 1.0) / 2.0 * (q99 - q01) + q01
        if method == "meanstd":
            mean = self._stats["mean"]
            std = self._stats["std"]
            return norm * std + mean
        if method == "minmax":
            lo = self._stats["min"]
            hi = self._stats["max"]
            return (norm + 1.0) / 2.0 * (hi - lo) + lo
        return norm

    @staticmethod
    def _rot6d_to_axisangle(rot6d: np.ndarray) -> np.ndarray:
        """rot6d ``[N, 6]`` -> axis-angle (rotvec) ``[N, 3]``."""
        from scipy.spatial.transform import Rotation as R

        rot6d = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
        col0 = rot6d[:, :3]
        col1 = rot6d[:, 3:6]
        # Re-orthogonalize (the network output is only approximately rot6d).
        col0 = col0 / np.clip(np.linalg.norm(col0, axis=-1, keepdims=True), 1e-8, None)
        col1 = col1 - np.sum(col1 * col0, axis=-1, keepdims=True) * col0
        col1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
        col2 = np.cross(col0, col1)
        mat = np.stack([col0, col1, col2], axis=-1)  # [N, 3, 3] (columns)
        return R.from_matrix(mat).as_rotvec().astype(np.float32, copy=False)

    def native_actions_to_libero(
        self, native_batch: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """Convert normalized Cosmos actions to continuous LIBERO actions.

        Uses the configured normalization statistics. Padded action channels are
        discarded, and each horizon is clipped or padded to num_action_chunks.
        Gripper values remain continuous; environment preparation is separate.

        Args:
            native_batch: Actions with shape [batch, horizon, padded_action_dim].

        Returns:
            Float32 actions with shape [batch, num_action_chunks, 7].
        """
        if isinstance(native_batch, torch.Tensor):
            native_batch = native_batch.detach().to(dtype=torch.float32).cpu().numpy()
        actions = np.asarray(native_batch, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(
                "Cosmos3 native action batch must have shape [batch, horizon, D], "
                f"got {actions.shape}"
            )
        if actions.shape[1] == 0:
            raise ValueError("Cosmos3 returned an empty action array")
        if actions.shape[0] == 0:
            raise ValueError(
                "Cosmos3 native action batch must contain at least one env"
            )
        if actions.shape[-1] < self._raw_action_dim:
            raise ValueError(
                f"Cosmos3 native action width must be at least {self._raw_action_dim}, "
                f"got {actions.shape[-1]}"
            )
        return np.stack([self._action_to_env(chunk) for chunk in actions], axis=0)

    def _action_to_env(self, arr: np.ndarray) -> np.ndarray:
        """Convert a single [horizon, padded_width] action chunk."""
        arr = self._denormalize(arr)  # [T, 10]
        trans = arr[:, :3]
        rot6d = arr[:, 3:9]
        gripper = arr[:, 9:10]
        axisangle = self._rot6d_to_axisangle(rot6d)  # [T, 3]
        env = np.concatenate([trans, axisangle, gripper], axis=-1)  # [T, 7]

        if env.shape[0] < self._num_action_chunks:
            pad = np.repeat(env[-1:], self._num_action_chunks - env.shape[0], axis=0)
            env = np.concatenate([env, pad], axis=0)
        elif env.shape[0] > self._num_action_chunks:
            env = env[: self._num_action_chunks]
        return env
