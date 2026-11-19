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

"""Compare Cosmos3 and LIBERO with independent rollouts and action replay."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw

from rlinf.models.embodiment.cosmos3.closed_loop_client import (
    Cosmos3Client,
    dynamics_request,
)
from rlinf.models.embodiment.cosmos3.closed_loop_io import (
    augment_prompt,
    save_video,
    validate_native_actions,
)
from rlinf.utils.logging import get_logger


@dataclass
class Trajectory:
    """Frames at t=0..steps and both representations of the same actions."""

    name: str
    frames: np.ndarray
    native_actions: np.ndarray
    libero_actions: np.ndarray
    chunks: list[dict[str, Any]]


def validate_policy_batching(client: Cosmos3Client, max_batch_size: int) -> dict:
    """Check the official action capability endpoint before batched inference."""
    with client.http_client.request(
        "GET", "/v1/actions/metadata", timeout_s=client.timeout_s
    ) as response:
        metadata = response.json()
    capabilities = metadata.get("capabilities", {})
    if metadata.get("policy_family") != "cosmos3":
        raise ValueError("Expected Cosmos3 action metadata from the shared server")
    if capabilities.get("max_batch_size", 1) < max_batch_size or (
        max_batch_size > 1
        and (
            capabilities.get("batch_inputs") is not True
            or "policy" not in capabilities.get("batched_action_modes", [])
        )
    ):
        raise ValueError(
            f"Server does not support the configured policy batch size {max_batch_size}; "
            "set its batching_max_size or lower the local request limit"
        )
    return metadata


class CosmosSimulator:
    """Advance independent images with concurrent, single-sample HTTP requests."""

    def __init__(
        self,
        client: Cosmos3Client,
        initial: np.ndarray,
        prompt: list[str],
        generation: dict,
    ):
        images = initial
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or images.shape[-1] != 3
            or images.dtype != np.uint8
        ):
            raise ValueError("Expected uint8 initial images [B,H,W,3]")
        self.prompts = list(prompt)
        self.num_envs = len(images)
        if len(self.prompts) != self.num_envs:
            raise ValueError("Expected one prompt per environment")
        self.client, self.initial = client, images.copy()
        self.generation = generation
        self.image = self.initial.copy()

    def reset(self) -> np.ndarray:
        """Restore each environment's exact initial pixels."""
        self.image = self.initial.copy()
        return self.image.copy()

    def step(
        self, native: np.ndarray, libero: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Dispatch all environments before waiting and commit feedback together."""
        del libero
        horizon = int(self.generation["num_frames"]) - 1
        batch = np.asarray(native)
        if batch.shape != (self.num_envs, horizon, 10):
            raise ValueError("Expected one native action chunk per environment")
        payloads = [
            dynamics_request(
                prompt,
                image,
                self.generation,
                validate_native_actions(actions, horizon, 10),
            )
            for prompt, image, actions in zip(
                self.prompts, self.image, batch, strict=True
            )
        ]
        results = self.client.generate_frames_batch(payloads)
        expected = (horizon + 1, *self.image.shape[1:])
        if len(results) != self.num_envs or any(
            frames.shape != expected or frames.dtype != np.uint8
            for frames, _ in results
        ):
            raise ValueError(
                "Expected one uint8 video with matching dimensions per environment"
            )
        frames = np.stack([frames for frames, _ in results])
        details = [info for _, info in results]
        self.image = frames[:, -1].copy()
        return frames[:, 1:], details


class LiberoCase:
    """Validate a task/reset pair and retain its output metadata."""

    def __init__(self, cfg: DictConfig, task_id: int, reset_id: int):
        from libero.libero import benchmark, get_libero_path

        c = cfg.comparison
        suites = benchmark.get_benchmark_dict()
        if c.suite not in suites:
            raise ValueError(f"Unknown LIBERO suite {c.suite!r}; choose {list(suites)}")
        suite = suites[c.suite](task_order_index=int(c.task_order_index))
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {suite.n_tasks}), got {task_id}")
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        if not 0 <= reset_id < len(states):
            raise ValueError(f"reset_id must be in [0, {len(states)}), got {reset_id}")
        self.task_id, self.reset_id = task_id, reset_id
        self.bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        self.instruction = task.language


def run_batched_trajectories(
    name: str,
    simulator: Any,
    policy: Any | None,
    steps: int,
    horizon: int,
    replay: list[Trajectory] | None = None,
) -> list[Trajectory]:
    """Advance a fixed environment group with batched policy or saved actions."""
    if steps <= 0 or horizon <= 0 or steps % horizon:
        raise ValueError("steps must be a positive whole number of action chunks")
    if (policy is None) == (replay is None):
        raise ValueError("Provide exactly one of policy or replay")
    count = simulator.num_envs
    if policy is not None and len(policy.prompts) != count:
        raise ValueError("Expected one policy prompt per environment")
    if replay is not None and (
        len(replay) != count
        or any(
            t.native_actions.shape != (steps, 10)
            or t.libero_actions.shape != (steps, 7)
            for t in replay
        )
    ):
        raise ValueError("Replay action arrays must cover every environment and step")
    images = simulator.reset()
    if images.ndim != 4 or images.shape[0] != count:
        raise ValueError("Batched trajectories require initial images [B,H,W,3]")
    videos = [[image.copy()] for image in images]
    natives, physicals, records = ([[] for _ in range(count)] for _ in range(3))
    for start in range(0, steps, horizon):
        if replay is None:
            native_batch, physical_batch = policy.predict_batch(images)
            policy_info = policy.last_batch_info
        else:
            native_batch = np.stack(
                [source.native_actions[start : start + horizon] for source in replay]
            )
            physical_batch = np.stack(
                [source.libero_actions[start : start + horizon] for source in replay]
            )
            policy_info = [None] * count
        before = time.perf_counter()
        frames, details = simulator.step(native_batch, physical_batch)
        batch_seconds = time.perf_counter() - before
        if (
            frames.shape != (count, horizon, *images.shape[1:])
            or frames.dtype != np.uint8
            or len(details) != count
        ):
            raise ValueError(
                "Simulator must return one uint8 frame per environment and action"
            )
        images = frames[:, -1].copy()
        for index in range(count):
            videos[index].extend(frames[index])
            natives[index].append(native_batch[index].copy())
            physicals[index].append(physical_batch[index].copy())
            records[index].append(
                {
                    "start_step": start,
                    "policy_s": policy_info[index]["elapsed_s"]
                    if policy_info[index]
                    else 0.0,
                    "policy_batch": policy_info[index],
                    "simulator_s": details[index]["elapsed_s"],
                    "concurrent_batch_s": batch_seconds,
                    "details": details[index],
                }
            )
        get_logger().info(
            "%s: %s/%s actions across %s environments",
            name,
            start + horizon,
            steps,
            count,
        )
    return [
        Trajectory(
            name,
            np.stack(videos[index]),
            np.concatenate(natives[index]),
            np.concatenate(physicals[index]),
            records[index],
        )
        for index in range(count)
    ]


def paired_frames(
    left: Trajectory, right: Trajectory, left_label: str, right_label: str
) -> np.ndarray:
    """Add labels above synchronized videos without modifying their pixels."""
    if left.frames.shape != right.frames.shape:
        raise ValueError(
            "Paired videos must have identical frame counts and dimensions"
        )
    count, height, width, _ = left.frames.shape
    header = Image.new("RGB", (width * 2, 32))
    draw = ImageDraw.Draw(header)
    draw.text((8, 8), left_label, fill="white")
    draw.text((width + 8, 8), right_label, fill="white")
    result = np.empty((count, height + 32, width * 2, 3), dtype=np.uint8)
    result[:, :32] = np.asarray(header)
    result[:, 32:, :width] = left.frames
    result[:, 32:, width:] = right.frames
    return result


def save_trajectory(
    root: Path, trajectory: Trajectory, fps: int, save_raw_frames: bool
) -> dict:
    """Save video, replayable action arrays, and separately measured request times."""
    directory = root / trajectory.name
    directory.mkdir()
    np.save(directory / "actions_native.npy", trajectory.native_actions)
    np.save(directory / "actions_libero.npy", trajectory.libero_actions)
    if save_raw_frames:
        np.save(directory / "frames.npy", trajectory.frames)
    save_video(directory / "video.mp4", trajectory.frames, fps)
    successes = [
        step
        for chunk in trajectory.chunks
        for step in chunk["details"].get("success_steps", [])
    ]
    metrics = {
        "steps": len(trajectory.native_actions),
        "frames": len(trajectory.frames),
        "policy_s": sum(chunk["policy_s"] for chunk in trajectory.chunks),
        "simulator_s": sum(chunk["simulator_s"] for chunk in trajectory.chunks),
        "first_success_step": min(successes) if successes else None,
        "chunks": trajectory.chunks,
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def comparison_cases(comparison: DictConfig) -> list[tuple[int, int]]:
    """Return task-major Cartesian pairs from nonempty, unique integer lists."""
    selections = []
    for name in ("task_ids", "reset_ids"):
        values = comparison.get(name)
        if not isinstance(values, (list, ListConfig)) or not values:
            raise ValueError(f"comparison.{name} must be a nonempty list")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"comparison.{name} must contain nonnegative integers")
        if len(set(values)) != len(values):
            raise ValueError(f"comparison.{name} must not contain duplicate IDs")
        selections.append(values)
    return [(task, reset) for task in selections[0] for reset in selections[1]]


def save_comparison(
    cfg: DictConfig,
    physical: LiberoCase,
    output: Path,
    initial: np.ndarray,
    a: Trajectory,
    b: Trajectory,
    replay_a: Trajectory,
    replay_b: Trajectory,
    worker: dict | None = None,
) -> dict:
    """Save one case after checking its initial frames and replay actions."""
    c, generation = cfg.comparison, cfg.env.eval.generation
    horizon = int(cfg.rollout.model.num_action_chunks)
    trajectories = (a, b, replay_a, replay_b)
    if any(not np.array_equal(t.frames[0], initial) for t in trajectories):
        raise RuntimeError("Every trajectory must start with the common initial pixels")
    for source, replay in ((a, replay_a), (b, replay_b)):
        if not (
            np.array_equal(source.native_actions, replay.native_actions)
            and np.array_equal(source.libero_actions, replay.libero_actions)
        ):
            raise RuntimeError("Replay changed the saved action sequence")
    metrics = {
        t.name: save_trajectory(output, t, generation.fps, c.save_raw_frames)
        for t in trajectories
    }
    pairs = [
        (
            "01_independent_rollouts.mp4",
            a,
            b,
            "Cosmos: policy rollout",
            "LIBERO: policy rollout",
        ),
        (
            "02_cosmos_actions.mp4",
            a,
            replay_a,
            "Cosmos: policy rollout",
            "LIBERO: replay Cosmos actions",
        ),
        (
            "03_libero_actions.mp4",
            replay_b,
            b,
            "Cosmos: replay LIBERO actions",
            "LIBERO: policy rollout",
        ),
    ]
    for filename, left, right, left_label, right_label in pairs:
        save_video(
            output / filename,
            paired_frames(left, right, left_label, right_label),
            generation.fps,
        )
    manifest = {
        "worker": worker,
        "suite": c.suite,
        "task_order_index": c.task_order_index,
        "task_id": physical.task_id,
        "reset_id": physical.reset_id,
        "instruction": physical.instruction,
        "prompt": augment_prompt(physical.instruction),
        "bddl": str(physical.bddl),
        "initial_image_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "action_stats_sha256": hashlib.sha256(
            Path(c.action_stats_path).read_bytes()
        ).hexdigest(),
        "same_initial_pixels": True,
        "same_replay_actions": True,
        "first_policy_chunk_equal": bool(
            np.array_equal(a.native_actions[:horizon], b.native_actions[:horizon])
        ),
        "videos": [item[0] for item in pairs],
        "trajectories": metrics,
        "timing_scope": "Policy request latency and Cosmos HTTP latency; LIBERO reports shared vector-chunk latency. concurrent_batch_s measures the simulator batch wall time. Shared policy and vector timings repeat across rows and must be counted once per batch. Excludes startup, reset and video writing.",
        "comparison_scope": "Fixed-length visual comparison; success is recorded only in LIBERO. Controller clipping and gripper binarization make physical actions a lossy projection of native10.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    get_logger().info("Comparison videos saved in %s", output)
    return manifest


def _write_summary(path: Path, summary: dict) -> None:
    """Atomically replace progress so completed case records survive interruption."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(path)


def run(cfg: DictConfig) -> None:
    """Distribute fixed case batches across RLinf workers and model replicas."""
    from examples.embodiment.cosmos3_comparison_worker import (
        Cosmos3ComparisonWorker,
    )
    from examples.embodiment.eval_cosmos3_closed_loop import (
        runtime_metadata,
        validate_closed_loop_cfg,
    )
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

    validate_closed_loop_cfg(cfg, single_server=False)
    c, generation = cfg.comparison, cfg.env.eval.generation
    horizon = int(cfg.rollout.model.num_action_chunks)
    for name, minimum in (
        ("steps", 1),
        ("num_envs", 1),
        ("camera_size", 1),
        ("settle_steps", 0),
        ("reset_seed", 0),
        ("task_order_index", 0),
    ):
        if type(c[name]) is not int or c[name] < minimum:
            raise ValueError(f"comparison.{name} must be an integer >= {minimum}")
    if c.steps % horizon:
        raise ValueError("comparison.steps must be divisible by the action horizon")
    if type(generation.fps) is not int or generation.fps <= 0:
        raise ValueError("generation.fps must be a positive integer")
    if generation.fps != generation.action_fps:
        raise ValueError("Comparison requires one action per video frame")
    if not Path(c.action_stats_path).is_file():
        raise FileNotFoundError(
            f"Training action statistics not found: {c.action_stats_path}"
        )
    cases = comparison_cases(c)
    for task_id, reset_id in cases:
        LiberoCase(cfg, task_id, reset_id)
    output = Path(c.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    (output / "runtime.json").write_text(
        json.dumps(runtime_metadata(), indent=2) + "\n"
    )
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    count = placement.get_world_size("env")
    if count != placement.get_world_size("rollout"):
        raise ValueError("Comparison requires equal env and rollout worker counts")
    for component in ("env", "rollout"):
        assignments = placement.get_strategy(component).get_placement(cluster)
        allowed = (
            (0, 1)
            if component == "env" or not cfg.rollout.sglang.launch_server
            else (1,)
        )
        if any(len(item.visible_accelerators) not in allowed for item in assignments):
            raise ValueError(
                f"Each {component} worker must use at most one accelerator"
            )
    summary = {
        "suite": c.suite,
        "task_order_index": c.task_order_index,
        "steps_per_case": c.steps,
        "num_envs_per_worker": c.num_envs,
        "num_workers": count,
        "max_concurrent_envs": count * c.num_envs,
        "status": "running",
        "completed_cases": 0,
        "cases": [
            {
                "task_id": task,
                "reset_id": reset,
                "directory": f"task_{task:03d}/reset_{reset:03d}",
                "status": "pending",
            }
            for task, reset in cases
        ],
    }
    summary_path = output / "summary.json"
    _write_summary(summary_path, summary)
    servers = router = workers = None
    try:
        serving = cfg.rollout.sglang
        if serving.launch_server:
            servers, router = launch_sglang_router_and_server(
                cfg,
                cluster,
                rollout_hardware_ranks=None,
                router_server_args=serving,
                placement_strategy=placement.get_strategy("rollout"),
            )
            urls = list(servers.get_server_url().wait())
        else:
            urls = list(serving.server_urls)
        if len(urls) != count or len(set(urls)) != count:
            raise ValueError(f"Expected {count} distinct server URLs, got {urls}")
        workers = Cosmos3ComparisonWorker.create_group(cfg, placement).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        workers.set_sglang_server_urls(urls).wait()
        worker_info = workers.init_worker().wait()
        (output / "workers.json").write_text(json.dumps(worker_info, indent=2) + "\n")
        batches = [
            summary["cases"][i : i + c.num_envs]
            for i in range(0, len(cases), c.num_envs)
        ]
        for start in range(0, len(batches), count):
            pending = []
            for rank, entries in enumerate(batches[start : start + count]):
                for entry in entries:
                    entry.update(status="running", worker_rank=rank)
                task_pairs = [(e["task_id"], e["reset_id"]) for e in entries]
                pending.append(
                    (
                        entries,
                        workers.execute_on(rank).run_cases(task_pairs, str(output)),
                    )
                )
            _write_summary(summary_path, summary)
            failure = None
            for entries, handle in pending:
                try:
                    results = handle.wait()[0]
                    for entry, result in zip(entries, results, strict=True):
                        entry.update(
                            status="completed",
                            instruction=result["instruction"],
                            manifest=f"{entry['directory']}/manifest.json",
                        )
                        summary["completed_cases"] += 1
                except Exception as exc:
                    failure = failure or exc
                    for entry in entries:
                        manifest_path = output / entry["directory"] / "manifest.json"
                        if entry["status"] == "completed":
                            continue
                        if manifest_path.is_file():
                            entry.update(
                                status="completed",
                                manifest=f"{entry['directory']}/manifest.json",
                            )
                            summary["completed_cases"] += 1
                        else:
                            entry.update(status="failed", error=str(exc))
                _write_summary(summary_path, summary)
            if failure is not None:
                raise failure
        summary["status"] = "completed"
        get_logger().info("All %s comparison cases saved in %s", len(cases), output)
    except BaseException as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        try:
            _write_summary(summary_path, summary)
        finally:
            for group, method in (
                (workers, "close"),
                (router, "stop"),
                (servers, "stop"),
            ):
                if group is not None:
                    try:
                        getattr(group, method)().wait()
                    except Exception:
                        get_logger().exception("Failed to close comparison %s", method)


@hydra.main(
    version_base="1.1", config_path="config", config_name="cosmos3_libero_compare"
)
def main(cfg: DictConfig) -> None:
    """Run a reproducible comparison with Hydra configuration overrides."""
    run(cfg)


if __name__ == "__main__":
    main()
