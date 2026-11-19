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
import math
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
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
from rlinf.workers.rollout.sglang_server import SGLangServerWorker


def _validate_serving_cfg(
    serving: DictConfig, model_path: str, role: str, *, single_server: bool
) -> None:
    """Validate one role's server before allocating workers."""
    expected = {
        "serving_mode": "embodied",
        "server_type": "embodied",
        "launch_router": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "http_max_retries": 0,
        "server.num_gpus": 1,
        "server.tp_size": 1,
        "server.cfg_parallel_degree": 1,
        "server.sp_degree": 1,
    }
    for field, value in expected.items():
        actual = OmegaConf.select(serving, field)
        if actual != value:
            raise ValueError(f"{role} {field} must be {value!r}, got {actual!r}")
    if type(serving.launch_server) is not bool:
        raise ValueError(f"{role} launch_server must be a boolean")
    if (
        type(serving.server.batching_max_size) is not int
        or serving.server.batching_max_size <= 0
    ):
        raise ValueError(f"{role} server.batching_max_size must be a positive integer")
    if not math.isfinite(serving.http_timeout_s) or serving.http_timeout_s <= 0:
        raise ValueError(f"{role} http_timeout_s must be finite and positive")
    pipeline_name = serving.server.pipeline_class_name
    if pipeline_name not in ("Cosmos3OmniDiffusersPipeline", "RLinfCosmos3Pipeline"):
        raise ValueError(f"Unsupported {role} Cosmos3 pipeline: {pipeline_name!r}")
    if pipeline_name == "RLinfCosmos3Pipeline" and serving.launch_server:
        expected_package = "rlinf.models.embodiment.cosmos3.sglang_extension"
        if serving.server.get("external_model_package") != expected_package:
            raise ValueError(
                f"{role} external_model_package must be {expected_package!r}"
            )
    advertised_path = serving.server.get("model_path")
    if (
        not advertised_path
        or Path(advertised_path).resolve() != Path(model_path).resolve()
    ):
        raise ValueError(
            f"{role} server.model_path must match its configured checkpoint"
        )
    urls = list(serving.server_urls)
    if serving.launch_server:
        if urls:
            raise ValueError(
                f"Disable {role} launch_server when supplying an external URL"
            )
    elif (
        not urls
        or (single_server and len(urls) != 1)
        or any(
            not isinstance(url, str)
            or urlparse(url).scheme not in ("http", "https")
            or not urlparse(url).netloc
            for url in urls
        )
    ):
        raise ValueError(
            f"{role} external mode requires absolute HTTP(S) server URLs "
            "(one in closed-loop eval)"
        )


def get_cosmos_server_configs(
    cfg: DictConfig, *, single_server: bool = True
) -> dict[str, DictConfig]:
    """Resolve shared or independent role servers without changing user config.

    Simulator serving options inherit the policy defaults. Its checkpoint,
    launch mode, URLs and worker names are independently configured.
    """
    share_weights = OmegaConf.select(cfg, "cosmos.share_weights", default=True)
    if type(share_weights) is not bool:
        raise ValueError("cosmos.share_weights must be a boolean")
    policy_path = cfg.rollout.model.model_path
    if not isinstance(policy_path, str) or not policy_path.strip():
        raise ValueError("rollout.model.model_path must be a nonempty checkpoint path")
    policy = OmegaConf.create(OmegaConf.to_container(cfg.rollout.sglang, resolve=True))
    _validate_serving_cfg(policy, policy_path, "policy", single_server=single_server)
    simulator_path = OmegaConf.select(cfg, "cosmos.simulator.model_path")
    overrides = OmegaConf.select(cfg, "cosmos.simulator.sglang", default={})
    if share_weights:
        if simulator_path is not None or overrides.get("server_urls"):
            raise ValueError(
                "Shared weights use rollout.model.model_path and rollout.sglang; "
                "clear cosmos.simulator.model_path and simulator server_urls"
            )
        return {"policy": policy, "simulator": policy}
    if not single_server:
        raise ValueError("Cosmos3 comparison requires cosmos.share_weights=true")
    if not isinstance(simulator_path, str) or not simulator_path.strip():
        raise ValueError("Separate weights require cosmos.simulator.model_path")
    simulator = OmegaConf.merge(
        policy,
        {
            "launch_server": True,
            "server_urls": [],
            "group_name": "Cosmos3SimulatorServerGroup",
            "router_group_name": "Cosmos3SimulatorRouterGroup",
            "server": {"model_path": simulator_path},
        },
        OmegaConf.to_container(overrides, resolve=True)
        if isinstance(overrides, DictConfig)
        else overrides,
    )
    _validate_serving_cfg(
        simulator, simulator_path, "simulator", single_server=single_server
    )
    if policy.launch_server and simulator.launch_server:
        if policy.group_name == simulator.group_name:
            raise ValueError("Separate servers require distinct group_name values")
    if not policy.launch_server and not simulator.launch_server:
        if {url.rstrip("/") for url in policy.server_urls} & {
            url.rstrip("/") for url in simulator.server_urls
        }:
            raise ValueError(
                "Separate weights require distinct policy and simulator URLs"
            )
    return {"policy": policy, "simulator": simulator}


def validate_server_checkpoint(url: str, serving: DictConfig, role: str) -> dict:
    """Check a server's advertised checkpoint before either role sends actions."""
    client = Cosmos3Client(url, timeout_s=serving.http_timeout_s)
    try:
        info = client.server_info()
    finally:
        client.close()
    advertised = info.get("model_path")
    expected = serving.server.model_path
    if not advertised or Path(advertised).resolve() != Path(expected).resolve():
        raise ValueError(
            f"{role} server checkpoint differs from configured model_path "
            f"{expected!r}: {advertised!r}"
        )
    return info


def _stop_server(group: object) -> None:
    try:
        group.stop().wait()
    except Exception:
        get_logger().exception("Failed to stop Cosmos3 server")


@contextmanager
def cosmos_servers(
    cfg: DictConfig, cluster: Cluster, placement: HybridComponentPlacement
) -> Iterator[tuple[dict[str, list[str]], dict]]:
    """Own locally launched servers; leave externally supplied servers running.

    Shared weights allocate one model instance. Separate weights allocate one
    instance per role, with simulator placement selected by ``simulator``.
    Cleanup is registered before model initialization so failed warmup is safe.
    """
    serving_configs = get_cosmos_server_configs(cfg)
    shared = serving_configs["policy"] is serving_configs["simulator"]
    urls, metadata = {}, {"share_weights": shared}
    with ExitStack() as cleanup:
        for role in ("policy", "simulator"):
            if role == "simulator" and shared:
                urls[role] = urls["policy"]
                metadata[role] = metadata["policy"]
                continue
            serving = serving_configs[role]
            if serving.launch_server:
                component = "rollout" if role == "policy" else "simulator"
                if placement.get_world_size(component) != 1:
                    raise ValueError(f"Expected one {component} server worker")
                strategy = placement.get_strategy(component)
                assignments = strategy.get_placement(cluster)
                if any(len(item.visible_accelerators) != 1 for item in assignments):
                    raise ValueError(f"Each {role} server requires one accelerator")
                group = SGLangServerWorker.create_group(
                    config=cfg,
                    sglang_cfg=serving.server,
                    server_type=serving.server_type,
                ).launch(
                    cluster=cluster,
                    name=serving.group_name,
                    placement_strategy=strategy,
                )
                cleanup.callback(_stop_server, group)
                group.init_server().wait()
                urls[role] = list(group.get_server_url().wait())
            else:
                urls[role] = list(serving.server_urls)
            if len(urls[role]) != 1:
                raise ValueError(f"Expected one {role} server URL, got {urls[role]}")
            metadata[role] = validate_server_checkpoint(urls[role][0], serving, role)
        yield urls, metadata


def validate_closed_loop_cfg(cfg: DictConfig, *, single_server: bool = True) -> None:
    """Reject unsupported modes and mismatched interfaces before starting Ray."""
    expected = {
        "runner.task_type": "embodied_eval",
        "runner.only_eval": True,
        "env.eval.env_type": "cosmos3",
        "env.eval.is_eval": True,
        "env.eval.group_size": 1,
        "env.eval.rollout_epoch": 1,
        "env.eval.auto_reset": False,
        "env.eval.enable_offload": False,
        "env.eval.video_cfg.save_video": False,
        "rollout.pipeline_stage_num": 1,
        "rollout.rollout_backend": "sglang",
        "rollout.return_logprobs": False,
        "rollout.enable_offload": False,
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
    get_cosmos_server_configs(cfg, single_server=single_server)
    if type(env.total_num_envs) is not int or env.total_num_envs <= 0:
        raise ValueError("env.eval.total_num_envs must be a positive integer")
    pipeline_name = serving.server.pipeline_class_name
    if (
        model.get("guidance_scale") is not None
        and pipeline_name != "RLinfCosmos3Pipeline"
    ):
        raise ValueError("Configurable policy guidance requires RLinfCosmos3Pipeline")
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
    if env.server.request_timeout_s <= 0:
        raise ValueError("Server request timeout must be positive")


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
    """Evaluate Cosmos3 using shared or independent policy and dynamics weights."""
    validate_closed_loop_cfg(cfg)
    artifact_dir = Path(cfg.env.eval.artifact_dir).resolve()
    save_artifacts = bool(cfg.env.eval.get("debug", {}).get("save_artifacts", False))
    if save_artifacts and artifact_dir.exists():
        raise FileExistsError(f"Choose a new artifact directory: {artifact_dir}")
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    for name in ("env", "rollout"):
        if placement.get_world_size(name) != 1:
            raise ValueError(f"Cosmos3 initial loop requires one {name} worker")
    with cosmos_servers(cfg, cluster, placement) as (urls, server_info):
        _run_workers(
            cfg, cluster, placement, urls, server_info, artifact_dir, save_artifacts
        )


def _run_workers(
    cfg: DictConfig,
    cluster: Cluster,
    placement: HybridComponentPlacement,
    urls: dict[str, list[str]],
    server_info: dict,
    artifact_dir: Path,
    save_artifacts: bool,
) -> None:
    from rlinf.workers.rollout.sglang.sglang_embodied_worker import SGLangEmbodiedWorker

    env_group = None
    try:
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
        rollout_group.set_sglang_server_urls(urls["policy"]).wait()
        env_group.set_sglang_server_urls(urls["simulator"]).wait()
        runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout_group, env=env_group)
        runner.init_workers()
        if save_artifacts:
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
        if env_group is not None:
            try:
                env_group.close_cosmos3_envs().wait()
            except Exception:
                get_logger().exception("Failed to close Cosmos3 environments")


@hydra.main(version_base="1.1", config_path="config", config_name="cosmos3_closed_loop")
def main(cfg: DictConfig) -> None:
    """Launch the inference-only Cosmos3 closed-loop example."""
    run(cfg)


if __name__ == "__main__":
    main()
