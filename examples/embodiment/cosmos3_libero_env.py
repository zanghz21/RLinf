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

"""Fixed-reset video comparisons using RLinf's vectorized LIBERO environment."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from rlinf.models.embodiment.cosmos3.closed_loop_io import prepare_canvas


class LiberoComparisonEnv:
    """Own a vector of LIBERO fixtures with repeatable controller resets."""

    def __init__(
        self,
        cfg: DictConfig,
        cases: list[tuple[int, int]],
        worker_info: Any = None,
    ):
        if not cases or any(
            len(case) != 2 or any(type(value) is not int or value < 0 for value in case)
            for case in cases
        ):
            raise ValueError("Expected nonempty nonnegative task/reset pairs")
        # Worker placement supplies EGL device IDs; children inherit that mapping.
        os.environ.setdefault("MUJOCO_GL", "egl")
        if os.environ["MUJOCO_GL"] == "egl":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from rlinf.envs.sim.libero.libero_env import LiberoEnv
        from rlinf.envs.sim.libero.utils import get_benchmark_overridden

        c, g = cfg.comparison, cfg.env.eval.generation
        suite = get_benchmark_overridden(c.suite)(
            task_order_index=int(c.task_order_index)
        )
        trial_counts = [
            len(suite.get_task_init_states(index))
            for index in range(suite.get_num_tasks())
        ]
        offsets = np.cumsum([0, *trial_counts])
        self._reset_ids = []
        for task_id, reset_id in cases:
            if task_id >= len(trial_counts) or reset_id >= trial_counts[task_id]:
                raise ValueError(
                    f"Invalid LIBERO task/reset pair {(task_id, reset_id)}"
                )
            self._reset_ids.append(int(offsets[task_id]) + reset_id)
        self.num_envs = len(cases)
        self._horizon = int(g.num_frames) - 1
        self._image_shape = (int(g.height), int(g.width), 3)
        self._seed = int(c.reset_seed)
        self._worker_info = worker_info
        self._env_cls = LiberoEnv
        self._env_cfg = OmegaConf.merge(
            OmegaConf.load(Path(__file__).parent / "config/env/libero_spatial.yaml"),
            {
                "task_suite_name": c.suite,
                "task_order_index": int(c.task_order_index),
                "seed": self._seed,
                "total_num_envs": self.num_envs,
                "is_eval": True,
                "auto_reset": False,
                "ignore_terminations": True,
                "skip_intermediate_renders": False,
                "use_rel_reward": False,
                "reset_gripper_open": True,
                "num_steps_wait": int(c.settle_steps),
                "specific_reset_id": self._reset_ids[0],
                "max_episode_steps": int(c.steps),
                "max_steps_per_rollout_epoch": int(c.steps),
                "video_cfg": {
                    "save_video": False,
                    "info_on_video": False,
                    "video_base_dir": "",
                },
                "init_params": {
                    "camera_heights": int(c.camera_size),
                    "camera_widths": int(c.camera_size),
                    "control_freq": int(g.fps),
                    "ignore_done": True,
                },
            },
        )
        self._env = None
        self._closed = False
        self._elapsed_steps = 0
        self._reference_images: np.ndarray | None = None
        self.init_states: list[np.ndarray] = []
        self.reference_states: list[np.ndarray] = []

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("LIBERO comparison environment is closed")

    def _canvases(self, observation: dict) -> np.ndarray:
        frames = []
        for external, wrist in zip(
            observation["main_images"], observation["wrist_images"], strict=True
        ):
            canvas, _ = prepare_canvas(
                Image.fromarray(np.asarray(external)),
                Image.fromarray(np.asarray(wrist)),
                self._image_shape[1],
                self._image_shape[0],
            )
            frames.append(np.asarray(canvas).copy())
        images = np.stack(frames)
        if (
            images.shape != (self.num_envs, *self._image_shape)
            or images.dtype != np.uint8
        ):
            raise ValueError("Expected one uint8 camera canvas per LIBERO environment")
        return images

    def _metadata(self, started: float) -> list[dict]:
        finished = time.monotonic()
        return [
            {
                "elapsed_s": finished - started,
                "worker_pid": worker.process.pid,
                "started_at_monotonic_s": started,
                "finished_at_monotonic_s": finished,
                "timing_scope": "vector_chunk",
            }
            for worker in self._env.env.workers
        ]

    def _discard_env(self, force: bool = False) -> None:
        env, self._env = self._env, None
        if env is not None:
            env.close(force=force)

    def reset(self) -> np.ndarray:
        """Rebuild physical simulators and verify the same settled state and pixels."""
        self._check_open()
        self._discard_env()
        try:
            self._env = self._env_cls(
                self._env_cfg,
                self.num_envs,
                seed_offset=0,
                total_num_processes=1,
                worker_info=self._worker_info,
            )
            observation, _ = self._env.reset(
                reset_state_ids=np.asarray(self._reset_ids),
                seed=[self._seed] * self.num_envs,
            )
            images = self._canvases(observation)
            init_states = self._env.get_reset_states()
            states = self._env.get_sim_states()
            if self._reference_images is not None and (
                not np.array_equal(images, self._reference_images)
                or any(
                    not np.array_equal(state, reference)
                    for state, reference in zip(
                        states, self.reference_states, strict=True
                    )
                )
            ):
                raise RuntimeError("LIBERO reset changed the initial state or pixels")
        except BaseException:
            self._discard_env(force=True)
            raise
        self.init_states, self.reference_states = init_states, states
        self._reference_images = images.copy()
        self._elapsed_steps = 0
        return images

    def step(
        self, native: np.ndarray, libero: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Execute each action in all environments and retain every rendered frame."""
        self._check_open()
        if self._env is None:
            raise RuntimeError("Reset LIBERO before stepping")
        native, libero = np.asarray(native), np.asarray(libero)
        if (
            native.shape != (self.num_envs, self._horizon, 10)
            or libero.shape != (self.num_envs, self._horizon, 7)
            or not np.isfinite(native).all()
            or not np.isfinite(libero).all()
        ):
            raise ValueError("Expected finite native[B,T,10] and LIBERO[B,T,7] actions")
        started = time.monotonic()
        try:
            observations, _, _, _, infos = self._env.chunk_step(libero)
            frames = np.stack([self._canvases(obs) for obs in observations], axis=1)
            if frames.shape != (self.num_envs, self._horizon, *self._image_shape):
                raise ValueError("Expected one frame for every LIBERO action")
            details = self._metadata(started)
            for index, detail in enumerate(details):
                detail["rewards"] = [
                    float(info["simulator_reward"][index]) for info in infos
                ]
                detail["success_steps"] = [
                    self._elapsed_steps + offset + 1
                    for offset, info in enumerate(infos)
                    if bool(info["episode"]["success_at_end"][index])
                ]
        except BaseException:
            self._discard_env(force=True)
            raise
        self._elapsed_steps += self._horizon
        return frames, details

    def close(self) -> None:
        """Release the vector environment owned by this comparison."""
        if not self._closed:
            self._closed = True
            self._discard_env()

    def __enter__(self) -> LiberoComparisonEnv:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
