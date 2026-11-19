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

"""Cosmos3 image-only dynamics driven by the standard environment worker."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import ListConfig
from PIL import Image

from rlinf.models.embodiment.cosmos3.closed_loop_client import (
    Cosmos3Client,
    dynamics_request,
)
from rlinf.models.embodiment.cosmos3.closed_loop_io import (
    augment_prompt,
    load_lerobot_observation,
    prepare_canvas,
    save_video,
    validate_native_actions,
)


class Cosmos3Env:
    """Advance LIBERO image fixtures with concurrent native-action requests.

    No reward model or proprioception is available. Zero reward tensors satisfy
    the worker interface; reported metrics contain only length and chunk count.
    Environments advance together only after every simulator response is valid.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("Cosmos3Env requires a positive integer num_envs")
        if total_num_processes != 1:
            raise ValueError("Cosmos3Env requires one process")
        self.num_envs = num_envs
        if cfg.get("auto_reset", False):
            raise ValueError("Cosmos3Env requires auto_reset=false")
        self.generation = dict(cfg.generation)
        self.horizon = int(self.generation["num_frames"]) - 1
        self.action_dim = int(self.generation["raw_action_dim"])
        self.max_episode_steps = int(cfg.max_episode_steps)
        if self.horizon <= 0 or self.max_episode_steps <= 0:
            raise ValueError(
                "Generation horizon and max_episode_steps must be positive"
            )
        if self.max_episode_steps % self.horizon:
            raise ValueError(
                "max_episode_steps must be a whole number of action chunks"
            )
        if self.action_dim != 10:
            raise ValueError("Cosmos3Env requires native 10-dimensional actions")
        self._episode_indices = self._dataset_indices("episode")
        self._frame_indices = self._dataset_indices("frame")
        self._save_artifacts = bool(cfg.get("debug", {}).get("save_artifacts", False))
        self._artifact_root: Path | None = None
        if self._save_artifacts:
            self._artifact_root = Path(cfg.artifact_dir).resolve()
            self._artifact_root.mkdir(parents=True, exist_ok=False)
        self._client: Cosmos3Client | None = None
        self._images: np.ndarray | None = None
        self._rollout_dirs: list[Path] = []
        self._frames: list[list[np.ndarray]] = []
        self._flushed_chunks = 0
        self._reset_index = 0
        self._steps = 0
        self._chunks = 0
        self._prompts: list[str] = []
        self.is_start = True

    def _dataset_indices(self, name: str) -> list[int]:
        """Resolve one fixture index per environment without silently truncating."""
        dataset = self.cfg.dataset
        values = dataset.get(f"{name}_indices")
        if values is None:
            values = [dataset.get(f"{name}_index")] * self.num_envs
        if (
            not isinstance(values, (list, ListConfig))
            or len(values) != self.num_envs
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise ValueError(
                f"dataset.{name}_indices must contain {self.num_envs} nonnegative "
                f"integers, or dataset.{name}_index must be a nonnegative integer"
            )
        return list(values)

    def set_sglang_server_urls(self, server_urls: list[str]) -> None:
        """Bind the same SGLang endpoint used by the rollout policy worker."""
        if len(server_urls) != 1:
            raise ValueError("Cosmos3Env requires exactly one shared SGLang endpoint")
        if self._client is not None:
            raise RuntimeError("Cosmos3Env server is already bound")
        self._client = Cosmos3Client(
            server_urls[0],
            timeout_s=float(self.cfg.server.request_timeout_s),
        )

    def _observation(self) -> dict[str, Any]:
        assert self._images is not None
        return {
            "main_images": torch.from_numpy(self._images.copy()),
            "wrist_images": None,
            "states": None,
            "task_descriptions": list(self._prompts),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load all initial views before replacing any environment's state."""
        if self._client is None:
            raise RuntimeError("Bind the shared SGLang endpoint before reset")
        images, prompts, records = [], [], []
        for episode_index, frame_index in zip(
            self._episode_indices, self._frame_indices, strict=True
        ):
            external, wrist, metadata = load_lerobot_observation(
                Path(self.cfg.dataset.root), episode_index, frame_index
            )
            canvas, layout = prepare_canvas(
                external,
                wrist,
                int(self.generation["width"]),
                int(self.generation["height"]),
            )
            images.append(np.array(canvas, copy=True))
            prompt = augment_prompt(metadata["instruction"])
            prompts.append(prompt)
            records.append({"dataset": metadata, "layout": layout, "prompt": prompt})
        next_images = np.stack(images)
        rollout_dirs = []
        if self._artifact_root is not None:
            episode_dir = self._artifact_root / f"episode_{self._reset_index:03d}"
            episode_dir.mkdir(exist_ok=False)
            for index, (image, record) in enumerate(zip(images, records, strict=True)):
                directory = (
                    episode_dir
                    if self.num_envs == 1
                    else episode_dir / f"env_{index:03d}"
                )
                if self.num_envs > 1:
                    directory.mkdir()
                Image.fromarray(image).save(directory / "initial_observation.png")
                self._write_json(directory / "initial_observation.json", record)
                rollout_dirs.append(directory)
        self.flush_video()
        self._images, self._prompts = next_images, prompts
        self._rollout_dirs = rollout_dirs
        self._steps = self._chunks = self._flushed_chunks = 0
        self._frames = (
            [[image.copy()] for image in images] if self._save_artifacts else []
        )
        self._reset_index += 1
        self.is_start = False
        return self._observation(), {}

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")

    def chunk_step(self, actions: Any) -> tuple:
        """Dispatch one request per environment concurrently and advance together."""
        if self._images is None or self._client is None:
            raise RuntimeError("Reset Cosmos3Env before stepping")
        if self._steps >= self.max_episode_steps:
            raise RuntimeError("The rollout horizon is complete; reset before stepping")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        batch = np.asarray(actions)
        expected_actions = (self.num_envs, self.horizon, self.action_dim)
        if batch.shape != expected_actions:
            raise ValueError(
                f"Expected native actions {expected_actions}, got {batch.shape}"
            )
        native = [
            validate_native_actions(chunk, self.horizon, self.action_dim)
            for chunk in batch
        ]
        payloads = [
            dynamics_request(prompt, image, self.generation, action)
            for prompt, image, action in zip(
                self._prompts, self._images, native, strict=True
            )
        ]
        cycle_dirs = []
        started = time.monotonic()
        try:
            for index, directory in enumerate(self._rollout_dirs):
                cycle_dir = directory / f"cycle_{self._chunks:03d}"
                cycle_dir.mkdir()
                cycle_dirs.append(cycle_dir)
                Image.fromarray(self._images[index]).save(cycle_dir / "observation.png")
                np.save(cycle_dir / "actions_native.npy", native[index])
                request_metadata = {
                    key: value
                    for key, value in payloads[index].items()
                    if key != "image_path"
                }
                request_metadata["image"] = {
                    "transport": "memory",
                    "shape": list(self._images[index].shape),
                    "dtype": str(self._images.dtype),
                }
                self._write_json(cycle_dir / "dynamics_request.json", request_metadata)
            results = self._client.generate_frames_batch(payloads)
            if len(results) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} Cosmos responses, got {len(results)}"
                )
            expected = (
                self.horizon + 1,
                int(self.generation["height"]),
                int(self.generation["width"]),
                3,
            )
            for index, (frames, _) in enumerate(results):
                if frames.shape != expected or frames.dtype != np.uint8:
                    raise ValueError(
                        f"Cosmos environment {index} returned {frames.shape}/{frames.dtype}, "
                        f"expected {expected}/uint8"
                    )
            next_images = np.stack([frames[-1] for frames, _ in results])
            next_frames = []
            for index, cycle_dir in enumerate(cycle_dirs):
                frames, response = results[index]
                response = {
                    **response,
                    "frames_sha256": hashlib.sha256(memoryview(frames)).hexdigest(),
                }
                self._write_json(cycle_dir / "dynamics_response.json", response)
                np.save(cycle_dir / "frames.npy", frames)
                save_video(
                    cycle_dir / "prediction.mp4", frames, int(self.generation["fps"])
                )
                Image.fromarray(next_images[index]).save(
                    cycle_dir / "next_observation.png"
                )
                next_frames.append(
                    self._frames[index] + [frame.copy() for frame in frames[1:]]
                )
        except BaseException as exc:
            for cycle_dir in cycle_dirs:
                self._write_json(
                    cycle_dir / "status.json",
                    {
                        "status": "failed",
                        "error": str(exc),
                        "seconds": time.monotonic() - started,
                    },
                )
            raise
        # Every response and debug frame is ready before replacing the batch state.
        self._images, self._frames = next_images, next_frames
        self._steps += self.horizon
        self._chunks += 1
        for cycle_dir in cycle_dirs:
            self._write_json(
                cycle_dir / "status.json",
                {"status": "completed", "seconds": time.monotonic() - started},
            )
        rewards = torch.zeros((self.num_envs, self.horizon), dtype=torch.float32)
        terminations = torch.zeros((self.num_envs, self.horizon), dtype=torch.bool)
        truncations = torch.zeros((self.num_envs, self.horizon), dtype=torch.bool)
        info: dict[str, Any] = {}
        if self._steps == self.max_episode_steps:
            truncations[:, -1] = True
            info["episode"] = {
                "episode_len": torch.full(
                    (self.num_envs,), self._steps, dtype=torch.int64
                ),
                "generated_chunks": torch.full(
                    (self.num_envs,), self._chunks, dtype=torch.int64
                ),
            }
            self.flush_video()
        return [self._observation()], rewards, terminations, truncations, [info]

    def flush_video(self) -> None:
        """Save each environment's debug preview once per completed chunk."""
        if self._rollout_dirs and self._chunks > self._flushed_chunks:
            for directory, frames in zip(self._rollout_dirs, self._frames, strict=True):
                save_video(
                    directory / "closed_loop.mp4",
                    np.stack(frames),
                    int(self.generation["fps"]),
                )
            self._flushed_chunks = self._chunks

    def update_reset_state_ids(self) -> None:
        """Keep the configured dataset fixtures for subsequent evaluations."""

    def close(self) -> None:
        """Release the client without stopping the shared model server."""
        try:
            self.flush_video()
        finally:
            if self._client is not None:
                self._client.close()
                self._client = None
