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

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rlinf.models.embodiment.cosmos3.closed_loop_client import (
    Cosmos3Client,
    dynamics_request,
)
from rlinf.models.embodiment.cosmos3.closed_loop_io import (
    augment_prompt,
    image_to_data_uri,
    load_lerobot_observation,
    prepare_canvas,
    read_video_frames,
    save_video,
    validate_native_actions,
)


class Cosmos3Env:
    """Advance a LIBERO image fixture with native Cosmos actions.

    No reward model or proprioception is available. Zero reward tensors satisfy
    the worker interface; reported metrics contain only length and chunk count.
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
        self.num_envs = num_envs
        if num_envs != 1 or total_num_processes != 1:
            raise ValueError("Cosmos3Env requires one environment and one process")
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
        self._artifact_root = Path(cfg.artifact_dir).resolve()
        self._artifact_root.mkdir(parents=True, exist_ok=False)
        self._client: Cosmos3Client | None = None
        self._image: Image.Image | None = None
        self._rollout_dir: Path | None = None
        self._frames: list[np.ndarray] = []
        self._reset_index = 0
        self._steps = 0
        self._chunks = 0
        self._prompt = ""
        self.is_start = True

    def set_sglang_server_urls(self, server_urls: list[str]) -> None:
        """Bind the same SGLang endpoint used by the rollout policy worker."""
        if len(server_urls) != 1:
            raise ValueError("Cosmos3Env requires exactly one shared SGLang endpoint")
        if self._client is not None:
            raise RuntimeError("Cosmos3Env server is already bound")
        self._client = Cosmos3Client(
            server_urls[0],
            timeout_s=float(self.cfg.server.request_timeout_s),
            poll_interval_s=float(self.cfg.server.poll_interval_s),
        )

    def _observation(self) -> dict[str, Any]:
        assert self._image is not None
        return {
            "main_images": torch.from_numpy(np.array(self._image, copy=True)).unsqueeze(
                0
            ),
            "wrist_images": None,
            "states": None,
            "task_descriptions": [self._prompt],
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load only the configured initial external and wrist images."""
        if self._client is None:
            raise RuntimeError("Bind the shared SGLang endpoint before reset")
        self.flush_video()
        dataset = self.cfg.dataset
        external, wrist, metadata = load_lerobot_observation(
            Path(dataset.root), int(dataset.episode_index), int(dataset.frame_index)
        )
        self._image, layout = prepare_canvas(
            external,
            wrist,
            int(self.generation["width"]),
            int(self.generation["height"]),
        )
        self._prompt = augment_prompt(metadata["instruction"])
        self._steps = self._chunks = 0
        self._frames = [np.array(self._image, copy=True)]
        self._rollout_dir = self._artifact_root / f"episode_{self._reset_index:03d}"
        self._rollout_dir.mkdir(exist_ok=False)
        self._reset_index += 1
        self._image.save(self._rollout_dir / "initial_observation.png")
        self._write_json(
            self._rollout_dir / "initial_observation.json",
            {"dataset": metadata, "layout": layout, "prompt": self._prompt},
        )
        self.is_start = False
        return self._observation(), {}

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")

    def chunk_step(self, actions: Any) -> tuple:
        """Condition forward dynamics on one unchanged native action chunk."""
        if self._image is None or self._client is None or self._rollout_dir is None:
            raise RuntimeError("Reset Cosmos3Env before stepping")
        if self._steps >= self.max_episode_steps:
            raise RuntimeError("The rollout horizon is complete; reset before stepping")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        batch = np.asarray(actions)
        if batch.shape != (1, self.horizon, self.action_dim):
            raise ValueError(
                f"Expected native actions [1, {self.horizon}, {self.action_dim}], got {batch.shape}"
            )
        native = validate_native_actions(batch[0], self.horizon, self.action_dim)
        cycle_dir = self._rollout_dir / f"cycle_{self._chunks:03d}"
        cycle_dir.mkdir()
        self._image.save(cycle_dir / "observation.png")
        np.save(cycle_dir / "actions_native.npy", native)
        payload = dynamics_request(
            self._prompt,
            image_to_data_uri(self._image),
            self.generation,
            native,
            cycle_dir / "server_output",
        )
        self._write_json(cycle_dir / "dynamics_request.json", payload)
        prediction_path = cycle_dir / "prediction.mp4"
        started = time.monotonic()
        try:
            response = self._client.generate_video(payload, prediction_path)
            self._write_json(cycle_dir / "dynamics_response.json", response)
            frames = read_video_frames(prediction_path)
            expected = (
                self.horizon + 1,
                int(self.generation["height"]),
                int(self.generation["width"]),
                3,
            )
            if frames.shape != expected or frames.dtype != np.uint8:
                raise ValueError(
                    f"Cosmos dynamics returned {frames.shape}/{frames.dtype}, expected {expected}/uint8"
                )
        except BaseException as exc:
            self._write_json(
                cycle_dir / "status.json",
                {
                    "status": "failed",
                    "error": str(exc),
                    "seconds": time.monotonic() - started,
                },
            )
            raise
        self._image = Image.fromarray(frames[-1].copy())
        self._image.save(cycle_dir / "next_observation.png")
        self._frames.extend(frame.copy() for frame in frames[1:])
        self._steps += self.horizon
        self._chunks += 1
        self._write_json(
            cycle_dir / "status.json",
            {"status": "completed", "seconds": time.monotonic() - started},
        )
        rewards = torch.zeros((1, self.horizon), dtype=torch.float32)
        terminations = torch.zeros((1, self.horizon), dtype=torch.bool)
        truncations = torch.zeros((1, self.horizon), dtype=torch.bool)
        info: dict[str, Any] = {}
        if self._steps == self.max_episode_steps:
            truncations[:, -1] = True
            info["episode"] = {
                "episode_len": torch.tensor([self._steps], dtype=torch.int64),
                "generated_chunks": torch.tensor([self._chunks], dtype=torch.int64),
            }
            self.flush_video()
        return [self._observation()], rewards, terminations, truncations, [info]

    def flush_video(self) -> None:
        """Save the trajectory without repeated conditioning frames."""
        if self._rollout_dir is not None and self._chunks:
            save_video(
                self._rollout_dir / "closed_loop.mp4",
                np.stack(self._frames),
                int(self.generation["fps"]),
            )

    def update_reset_state_ids(self) -> None:
        """Keep the configured dataset fixture for subsequent evaluations."""

    def close(self) -> None:
        """Release the client without stopping the shared model server."""
        try:
            self.flush_video()
        finally:
            if self._client is not None:
                self._client.close()
                self._client = None
