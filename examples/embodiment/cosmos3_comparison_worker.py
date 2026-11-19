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

"""RLinf worker for paired Cosmos3/LIBERO rollouts and action replay."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models.embodiment.cosmos3.closed_loop_client import Cosmos3Client
from rlinf.models.embodiment.cosmos3.closed_loop_io import augment_prompt
from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter
from rlinf.workers.rollout.sglang.sglang_embodied_worker import SGLangEmbodiedWorker


class Cosmos3ComparisonWorker(SGLangEmbodiedWorker):
    """Own one fixed server connection and one local LIBERO environment batch."""

    async def init_worker(self) -> dict:
        await super().init_worker()
        from examples.embodiment.compare_cosmos3_libero import validate_policy_batching

        self.client = Cosmos3Client(self.sglang_server_url, self._http_timeout_s)
        info = self.client.server_info()
        if (
            Path(info.get("model_path", "")).resolve()
            != Path(self.model_cfg.model_path).resolve()
        ):
            raise ValueError(
                "Shared server does not advertise the configured checkpoint"
            )
        capabilities = validate_policy_batching(
            self.client, int(self.cfg.rollout.sglang.server.batching_max_size)
        )
        converter_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        converter_cfg.rollout.model.action_output = "libero"
        converter_cfg.rollout.model.action_dim = 7
        converter_cfg.rollout.model.action_stats_path = (
            self.cfg.comparison.action_stats_path
        )
        converter_cfg.rollout.model.action_normalization = "quantile_rot"
        self.libero_adapter = Cosmos3SGLangAdapter(converter_cfg, self._rank)
        self.prompts: list[str] = []
        self.last_batch_info: list[dict] = []
        self._request_durations: list[float] = []
        self.metadata = {
            "rank": self._rank,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "visible_accelerators": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "server_url": self.sglang_server_url,
            "server": info,
            "action_metadata": capabilities,
        }
        return self.metadata

    def _request_actions(self, env_obs, mode):
        started = time.perf_counter()
        result = super()._request_actions(env_obs, mode)
        self._request_durations.append(time.perf_counter() - started)
        return result

    def predict_batch(
        self, images: np.ndarray, prompts: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the existing rollout HTTP path and convert the same native actions."""
        from rlinf.envs.action_utils import prepare_actions_for_libero

        if prompts is not None:
            self.prompts = list(prompts)
        if (
            images.ndim != 4
            or images.shape[0] != len(self.prompts)
            or images.shape[-1] != 3
            or images.dtype != np.uint8
        ):
            raise ValueError("Expected one uint8 image [B,H,W,3] per prompt")
        observations = {
            "main_images": torch.from_numpy(images),
            "task_descriptions": self.prompts,
        }
        groups = self.sglang_adapter.request_groups(observations)
        self._request_durations = []
        native, _ = self.predict(observations, mode="eval")
        physical = prepare_actions_for_libero(
            self.libero_adapter.native_actions_to_libero(native), "cosmos3"
        )
        seed = self.cfg.rollout.sglang.get("seed", 1140)
        seed = 1140 if seed is None else int(seed)
        records = [{} for _ in self.prompts]
        for indices, elapsed in zip(groups, self._request_durations, strict=True):
            for position, index in enumerate(indices):
                records[index] = {
                    "indices": list(indices),
                    "position": position,
                    "batch_size": len(indices),
                    "seed": seed,
                    "elapsed_s": elapsed,
                    "worker_rank": self._rank,
                }
        if not np.isfinite(physical).all():
            raise ValueError("Expected finite LIBERO actions after conversion")
        self.last_batch_info = records
        return native.numpy(), physical

    def run_cases(self, cases: list[tuple[int, int]], output_dir: str) -> list[dict]:
        """Run all four trajectories for one shard and save only its case outputs."""
        from examples.embodiment.compare_cosmos3_libero import (
            CosmosSimulator,
            LiberoCase,
            run_batched_trajectories,
            save_comparison,
        )
        from examples.embodiment.cosmos3_libero_env import LiberoComparisonEnv

        if not cases or len(cases) > self.cfg.comparison.num_envs:
            raise ValueError("Case batch must fit comparison.num_envs")
        metadata = [LiberoCase(self.cfg, *case) for case in cases]
        self.prompts = [augment_prompt(case.instruction) for case in metadata]
        with LiberoComparisonEnv(self.cfg, cases, self.worker_info) as physical:
            initial = physical.reset()
            outputs = [
                Path(output_dir) / f"task_{task:03d}" / f"reset_{reset:03d}"
                for task, reset in cases
            ]
            for index, output in enumerate(outputs):
                output.mkdir(parents=True, exist_ok=False)
                Image.fromarray(initial[index]).save(output / "initial_observation.png")
                np.save(output / "libero_init_state.npy", physical.init_states[index])
                np.save(
                    output / "libero_settled_state.npy",
                    physical.reference_states[index],
                )
            learned = CosmosSimulator(
                self.client, initial, self.prompts, dict(self.cfg.env.eval.generation)
            )
            steps = self.cfg.comparison.steps
            horizon = self.cfg.rollout.model.num_action_chunks
            a = run_batched_trajectories(
                "cosmos_rollout", learned, self, steps, horizon
            )
            b = run_batched_trajectories(
                "libero_rollout", physical, self, steps, horizon
            )
            replay_a = run_batched_trajectories(
                "libero_replay_cosmos", physical, None, steps, horizon, a
            )
            replay_b = run_batched_trajectories(
                "cosmos_replay_libero", learned, None, steps, horizon, b
            )
            results = []
            for index, case in enumerate(metadata):
                save_comparison(
                    self.cfg,
                    case,
                    outputs[index],
                    initial[index],
                    a[index],
                    b[index],
                    replay_a[index],
                    replay_b[index],
                    worker=self.metadata,
                )
                results.append({"instruction": case.instruction})
            self.log_info(f"Saved {len(cases)} comparison cases on rank {self._rank}")
            return results

    def close(self) -> None:
        """Release this worker's HTTP sessions."""
        if getattr(self, "client", None) is not None:
            self.client.close()
            self.client = None
        if self.http_client is not None:
            self.http_client.close()
            self.http_client = None
