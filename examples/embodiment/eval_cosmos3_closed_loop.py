# Copyright 2025 The RLinf Authors.
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

"""Evaluate a Cosmos3 policy against Cosmos3 dynamics through RLinf workers."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import hydra
from omegaconf import DictConfig, OmegaConf

from rlinf.config import validate_cfg
from rlinf.models.embodiment.cosmos3.closed_loop_client import Cosmos3Client
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.sglang.sglang_embodied_worker import SGLangEmbodiedWorker
from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server


def validate_closed_loop_cfg(cfg: DictConfig) -> None:
    """Reject unsupported modes and mismatched interfaces before starting Ray."""
    expected = {
        "runner.task_type": "embodied_eval",
        "runner.only_eval": True,
        "env.eval.env_type": "cosmos3",
        "env.eval.is_eval": True,
        "env.eval.total_num_envs": 1,
        "env.eval.group_size": 1,
        "env.eval.rollout_epoch": 1,
        "env.eval.auto_reset": False,
        "env.eval.enable_offload": False,
        "rollout.pipeline_stage_num": 1,
        "rollout.rollout_backend": "sglang",
        "rollout.return_logprobs": False,
        "rollout.enable_offload": False,
        "rollout.sglang.serving_mode": "embodied",
        "rollout.sglang.server_type": "embodied",
        "rollout.sglang.launch_router": False,
        "rollout.sglang.tensor_parallel_size": 1,
        "rollout.sglang.pipeline_parallel_size": 1,
        "rollout.sglang.http_max_retries": 0,
        "rollout.sglang.server.pipeline_class_name": "Cosmos3OmniDiffusersPipeline",
        "rollout.sglang.server.num_gpus": 1,
        "rollout.sglang.server.tp_size": 1,
        "rollout.sglang.server.cfg_parallel_degree": 1,
        "rollout.sglang.server.sp_degree": 1,
        "rollout.sglang.server.batching_max_size": 1,
        "rollout.model.model_type": "cosmos3",
        "rollout.model.action_output": "native",
        "rollout.model.action_dim": 10,
        "rollout.model.raw_action_dim": 10,
        "rollout.model.action_stats_path": None,
        "rollout.model.domain_name": "libero",
        "rollout.model.action_mode": "policy",
        "env.eval.generation.domain_name": "libero",
        "env.eval.generation.raw_action_dim": 10,
    }
    for field, value in expected.items():
        actual = OmegaConf.select(cfg, field)
        if actual != value:
            raise ValueError(f"{field} must be {value!r}, got {actual!r}")
    model, env, serving = cfg.rollout.model, cfg.env.eval, cfg.rollout.sglang
    for field in (
        "num_frames",
        "fps",
        "action_fps",
        "action_view_point",
        "num_inference_steps",
    ):
        if model[field] != env.generation[field]:
            raise ValueError(f"Policy and dynamics {field} must match")
    if model.get("domain_id", 5) != 5:
        raise ValueError("Native LIBERO actions require domain_id=5")
    if serving.seed != env.generation.seed:
        raise ValueError("Policy and dynamics seeds must match")
    horizon = model.num_action_chunks
    if not isinstance(horizon, int) or horizon <= 0 or model.num_frames != horizon + 1:
        raise ValueError("num_frames must equal positive num_action_chunks + 1")
    steps = env.max_steps_per_rollout_epoch
    if (
        not isinstance(steps, int)
        or steps <= 0
        or steps % horizon
        or env.max_episode_steps != steps
    ):
        raise ValueError(
            "Episode and rollout steps must match and contain whole action chunks"
        )
    if model.size != f"{env.generation.width}x{env.generation.height}":
        raise ValueError("Policy size must match dynamics width and height")
    for field in ("width", "height", "fps", "action_fps", "num_inference_steps"):
        if env.generation[field] <= 0:
            raise ValueError(f"Generation {field} must be positive")
    if env.server.request_timeout_s <= 0 or env.server.poll_interval_s <= 0:
        raise ValueError("Server timeouts must be positive")
    if Path(serving.server.model_path).resolve() != Path(model.model_path).resolve():
        raise ValueError("Policy and dynamics must use the same checkpoint")
    urls = list(serving.server_urls)
    if serving.launch_server:
        if urls:
            raise ValueError("Disable launch_server when supplying an external URL")
    elif (
        len(urls) != 1
        or urlparse(urls[0]).scheme not in ("http", "https")
        or not urlparse(urls[0]).netloc
    ):
        raise ValueError(
            "External mode requires exactly one absolute HTTP(S) server URL"
        )


def runtime_metadata() -> dict:
    """Record the driver runtime and checkout used by the worker entry."""
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "sglang", "ray", "numpy", "pyarrow")
    }
    distribution = importlib.metadata.distribution("sglang")
    direct_url = distribution.read_text("direct_url.json")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "packages": versions,
        "sglang_source": json.loads(direct_url) if direct_url else None,
        "git_revision": revision,
    }


def run(cfg: DictConfig) -> None:
    """Run the normal evaluation runner with one server shared by both workers."""
    validate_closed_loop_cfg(cfg)
    artifact_dir = Path(cfg.env.eval.artifact_dir).resolve()
    if artifact_dir.exists():
        raise FileExistsError(f"Choose a new artifact directory: {artifact_dir}")
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    for name in ("env", "rollout"):
        if placement.get_world_size(name) != 1:
            raise ValueError(f"Cosmos3 initial loop requires one {name} worker")
    server_group = router_group = env_group = None
    try:
        serving = cfg.rollout.sglang
        if serving.launch_server:
            server_group, router_group = launch_sglang_router_and_server(
                cfg,
                cluster,
                rollout_hardware_ranks=None,
                router_server_args=serving,
                placement_strategy=placement.get_strategy("rollout"),
            )
            urls = list(server_group.get_server_url().wait())
        else:
            urls = list(serving.server_urls)
        if len(urls) != 1:
            raise ValueError(f"Expected one shared Cosmos3 server, got {urls}")
        client = Cosmos3Client(urls[0], timeout_s=serving.http_timeout_s)
        try:
            server_info = client.server_info()
        finally:
            client.close()
        advertised = server_info.get("model_path")
        if (
            not advertised
            or Path(advertised).resolve()
            != Path(cfg.rollout.model.model_path).resolve()
        ):
            raise ValueError(
                f"Server checkpoint differs from rollout.model.model_path: {advertised}"
            )
        rollout_group = SGLangEmbodiedWorker.create_group(cfg, placement).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env_group = EnvWorker.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        rollout_group.set_sglang_server_urls(urls).wait()
        env_group.set_sglang_server_urls(urls).wait()
        runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout_group, env=env_group)
        runner.init_workers()
        # Environment initialization owns creation of the new artifact directory.
        (artifact_dir / "config.json").write_text(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + "\n"
        )
        (artifact_dir / "server.json").write_text(
            json.dumps(server_info, indent=2) + "\n"
        )
        (artifact_dir / "runtime.json").write_text(
            json.dumps(runtime_metadata(), indent=2) + "\n"
        )
        runner.run()
    finally:
        for group, method in (
            (env_group, "close_cosmos3_envs"),
            (router_group, "stop"),
            (server_group, "stop"),
        ):
            if group is not None:
                try:
                    getattr(group, method)().wait()
                except Exception:
                    get_logger().exception("Failed to close %s", method)


@hydra.main(version_base="1.1", config_path="config", config_name="cosmos3_closed_loop")
def main(cfg: DictConfig) -> None:
    """Launch the inference-only Cosmos3 closed-loop example."""
    run(cfg)


if __name__ == "__main__":
    main()
