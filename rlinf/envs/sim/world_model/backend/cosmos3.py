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

"""Native Cosmos dynamics for the standard world-model environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.world_model.backend import (
    FrameQueue,
    load_wovr_reward_model,
    wovr_reward_instructions,
)
from rlinf.envs.sim.world_model.registry import register_backend
from rlinf.models.embodiment.cosmos3.closed_loop_io import augment_prompt
from rlinf.models.embodiment.cosmos3.wovr_policy import Cosmos3WoVRPolicy


@register_backend("cosmos3")
class Cosmos3Backend:
    """Hold one frozen simulator with the policy's ``omni.*`` weight namespace.

    WorldModelEnv owns resets, rewards, and episode endings. A session keeps
    the last canvas and its instruction. KIR windows initialize that canvas
    from their final keyframe; Cosmos conditions on one image per chunk.
    """

    supports_kir = True
    requires_instructions = True

    def __init__(
        self,
        cfg: Any,
        device: torch.device,
        shared_model: Cosmos3WoVRPolicy | None = None,
    ) -> None:
        self.cfg, self.device = cfg, device
        self.chunk = int(cfg.chunk)
        self.condition_frame_length = int(cfg.condition_frame_length)
        self.image_size = tuple(cfg.image_size)
        if self.chunk <= 0 or self.condition_frame_length <= 0:
            raise ValueError("Cosmos chunk and condition_frame_length must be positive")
        if len(self.image_size) != 2 or any(int(size) <= 0 for size in self.image_size):
            raise ValueError("Cosmos image_size must contain positive [height, width]")
        if cfg.num_frames != self.chunk + 1:
            raise ValueError(
                "Cosmos num_frames must equal chunk + one conditioning image"
            )
        if cfg.get("action_dim", 10) != 10:
            raise ValueError("Cosmos dynamics require native normalized 10D actions")
        if type(cfg.get("share_weights", False)) is not bool:
            raise ValueError("Cosmos share_weights must be a boolean")
        model_cfg = OmegaConf.to_container(cfg.model_cfg, resolve=True)
        if model_cfg.get("num_action_chunks", self.chunk) != self.chunk:
            raise ValueError("Cosmos model_cfg.num_action_chunks must equal env.chunk")
        if "device" in model_cfg and torch.device(model_cfg["device"]) != device:
            raise ValueError(
                "Cosmos model_cfg.device must match the environment placement"
            )
        model_cfg.setdefault("device", str(device))
        if shared_model is not None and not cfg.get("share_weights", False):
            raise ValueError("A shared Cosmos model requires share_weights=true")
        self.model = (
            (
                Cosmos3WoVRPolicy.from_pretrained(model_cfg, for_simulation=True)
                if shared_model is None
                else shared_model
            )
            .eval()
            .requires_grad_(False)
        )
        self.weight_version = 0
        self._sessions: dict[int, dict[str, Any]] = {}

    @staticmethod
    def build_dataset(cfg: Any) -> Any:
        """Reuse curated LeRobot keyframes or the existing WoVR NPY trajectories."""
        if cfg.get("dataset") is not None:
            from rlinf.data.datasets.cosmos3_keyframes import Cosmos3KeyframeDataset

            return Cosmos3KeyframeDataset(cfg.dataset, cfg.image_size)
        from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper

        dataset = NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            enable_kir=cfg.get("enable_kir", True),
            action_key=cfg.get("dataset_action_key", "cosmos_action"),
            target_n_frames=max(0, cfg.condition_frame_length - 1),
        )
        if dataset.action_dim != 10:
            raise ValueError(
                "Cosmos NPY trajectories must contain native normalized 10D actions"
            )
        return dataset

    @staticmethod
    def load_reward_model(cfg: Any) -> torch.nn.Module:
        """Load an existing RLinf or WoVR success classifier without training it."""
        reward = cfg.reward_model
        if reward.type == "resnet":
            if not reward.get("model_path"):
                raise ValueError(
                    "Cosmos reward_model.model_path must name a trained checkpoint"
                )
            from rlinf.models.embodiment.reward import get_reward_model_class

            model_cfg = OmegaConf.merge(
                {"precision": "fp32", "pretrained": False}, reward
            )
            return get_reward_model_class("resnet")(model_cfg).requires_grad_(False)
        if reward.type in ("ResnetRewModel", "TaskEmbedResnetRewModel"):
            return load_wovr_reward_model(cfg).requires_grad_(False)
        raise ValueError(f"Unsupported Cosmos reward model type: {reward.type!r}")

    def reward_instructions(self, env: Any) -> list[str] | None:
        """Repeat each raw task instruction for its generated frames."""
        return wovr_reward_instructions(env)

    def reward_images(self, images: torch.Tensor) -> torch.Tensor:
        """Select the classifier's camera from normalized [B,3,H,W] canvases."""
        view = self.cfg.reward_model.get("camera_view", "canvas")
        if view == "canvas":
            return images
        if view not in ("external", "wrist"):
            raise ValueError(
                "reward_model.camera_view must be canvas, external, or wrist"
            )
        height, width = self.cfg.reward_model.get(
            "view_size", [images.shape[-2], images.shape[-1] // 2]
        )
        height, width = int(height), int(width)
        if (
            height <= 0
            or width <= 0
            or height > images.shape[-2]
            or 2 * width > images.shape[-1]
        ):
            raise ValueError(
                "reward_model.view_size must fit two cameras within the canvas"
            )
        start = 0 if view == "external" else width
        return images[..., :height, start : start + width]

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        seeds: Sequence[int],
        instructions: Sequence[str] | None = None,
    ) -> None:
        """Start each slot from the final frame of its KIR condition window."""
        if instructions is None or not (
            len(env_ids) == len(init_frames) == len(seeds) == len(instructions)
        ):
            raise ValueError(
                "Cosmos sessions require one condition window, seed, and instruction per slot"
            )
        sessions = {}
        for row, env_id in enumerate(env_ids):
            frame = init_frames[row][-1]
            expected = (3, 1, *self.image_size)
            if tuple(frame.shape) != expected or not torch.isfinite(frame).all():
                raise ValueError(
                    f"Expected a finite Cosmos condition frame {expected}, got {frame.shape}"
                )
            image = (
                ((frame[:, 0].float().cpu().clamp(-1, 1) + 1) * 127.5)
                .round()
                .to(torch.uint8)
            )
            sessions[int(env_id)] = {
                "image": image.permute(1, 2, 0).numpy(),
                "prompt": augment_prompt(instructions[row]),
                "seed": int(seeds[row]),
                "step": 0,
            }
        self._sessions.update(sessions)

    def close_session(self, env_ids: Sequence[int]) -> None:
        """Release only the requested slots."""
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    @torch.no_grad()
    def generate(self, env_ids: Sequence[int], actions: torch.Tensor) -> torch.Tensor:
        """Advance slots atomically and return new frames [B,3,chunk,H,W] in [-1,1]."""
        ids = [int(env_id) for env_id in env_ids]
        actions = torch.as_tensor(actions).detach().float().cpu()
        expected = (len(ids), self.chunk, 10)
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
            raise ValueError(
                f"Expected finite native actions {expected}, got {actions.shape}"
            )
        if len(set(ids)) != len(ids) or any(
            env_id not in self._sessions for env_id in ids
        ):
            raise ValueError(
                "Cosmos generation requires distinct slots with open sessions"
            )
        # Another train/eval environment can offload this shared module independently.
        self.onload()
        videos = []
        for row, env_id in enumerate(ids):
            session = self._sessions[env_id]
            seed = int(
                np.random.SeedSequence(
                    [
                        session["seed"],
                        env_id,
                        session["step"],
                    ]
                ).generate_state(1)[0]
            )
            frames = self.model.generate_dynamics(
                session["image"], session["prompt"], actions[row], seed
            )
            expected_frames = (self.chunk + 1, *self.image_size, 3)
            if frames.dtype != np.uint8 or frames.shape != expected_frames:
                raise ValueError(
                    f"Expected Cosmos RGB uint8 frames {expected_frames}, got {frames.shape}/{frames.dtype}"
                )
            videos.append(frames[1:].copy())
        for env_id, video in zip(ids, videos, strict=True):
            self._sessions[env_id]["image"] = video[-1].copy()
            self._sessions[env_id]["step"] += 1
        return (
            torch.from_numpy(np.stack(videos))
            .permute(0, 4, 1, 2, 3)
            .float()
            .div_(127.5)
            .sub_(1)
        )

    def on_weights_updated(self, version: int) -> None:
        """Record the published actor version; sessions store pixels, not model caches."""
        if type(version) is not int or version < self.weight_version:
            raise ValueError(
                "Cosmos simulator weight versions must be nondecreasing integers"
            )
        self.weight_version = version

    def offload(self) -> None:
        """Move the simulator off its execution device between environment phases."""
        self.model.to("cpu")

    def onload(self) -> None:
        """Restore the simulator to the environment worker's device."""
        self.model.to(self.device)
