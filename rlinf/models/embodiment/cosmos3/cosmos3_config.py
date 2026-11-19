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

"""Cosmos3 native simulator routing and SFT model configuration.

The SFT model is built from cosmos-framework's own experiment recipe
(``make_config`` + ``override(experiment=<SKU>)`` + ``instantiate``).

Model defaults (model_type, precision, load_to_device, ema_enabled,
disable_cosmos_parallelism, keys_to_skip_loading, keys_to_select,
lr_multipliers, action_normalization, ...) live in the Hydra config group
``examples/sft/config/model/cosmos3.yaml`` -- pull them in via
``defaults: - model/cosmos3@actor.model``. Recipe yamls override only
deployment-specific fields (``model_path``, ``wan_vae_path``,
``max_samples_per_batch``).

Dataset-recipe knobs (``image_size``, ``chunk_length``, ``action_space``,
``fps``, ...) are owned by the cosmos experiment recipe (resolved by cosmos);
rlinf does not read them from ``actor.model``, so they are deliberately NOT
duplicated here or in ``model/cosmos3.yaml``.
"""

from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.utils.logging import get_logger

logger = get_logger()


def configure_native_rl_envs(cfg: DictConfig, policy: DictConfig) -> None:
    """Validate native action routing and resolve any Cosmos dynamics replicas."""
    if cfg.runner.get("resume_dir") and cfg.weight_syncer.type == "patch":
        # Rollout starts from the base export. Establish the restored actor as
        # its first patch baseline before sending incremental updates.
        cfg.weight_syncer.patch.init_sync.enabled = True
        cfg.weight_syncer.patch.init_sync.prefixes = None
    simulators = []
    for mode in ("train", "eval"):
        env = cfg.env.get(mode)
        if env is None:
            continue
        if env.env_type == "libero":
            if policy.action_dim != 10 or policy.get("raw_action_dim", 10) != 10:
                raise ValueError("Native Cosmos3 GRPO requires 10D policy actions")
            if not env.get("action_stats_path"):
                raise ValueError(
                    f"Native Cosmos3 requires env.{mode}.action_stats_path"
                )
        elif env.env_type == "world_model" and env.get("backend") == "cosmos3":
            simulators.append(env)
    if not simulators:
        return
    shared = cfg.cosmos.share_weights
    if type(shared) is not bool:
        raise ValueError("cosmos.share_weights must be a boolean")
    simulator = cfg.cosmos.simulator
    simulator_path = simulator.get("model_path")
    native_config = simulator.get("native_config_path")
    if shared and (simulator_path or native_config):
        raise ValueError("Shared weights use actor.model. Clear cosmos.simulator paths")
    if not shared and (
        not isinstance(simulator_path, str) or not simulator_path.strip()
    ):
        raise ValueError("Separate weights require cosmos.simulator.model_path")
    if shared and OmegaConf.select(cfg, "weight_syncer.type") == "patch":
        init_sync = cfg.weight_syncer.patch.init_sync
        if not init_sync.enabled or init_sync.prefixes is not None:
            raise ValueError("Shared Cosmos weights require a full initial patch sync")

    model = OmegaConf.create(OmegaConf.to_container(policy, resolve=True))
    if not shared:
        model.model_path = simulator_path
        # A separate export can supply its own config.json next to the weights.
        model.native_config_path = native_config
    for env in simulators:
        with open_dict(env):
            env.model_cfg = model
            env.share_weights = shared


def validate_cosmos3_sft_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate a Cosmos3 SFT cfg."""

    with open_dict(model_cfg):
        model_path = model_cfg.get("model_path", None)
        assert model_path, "Cosmos3 SFT requires actor.model.model_path"

    logger.info(
        "Cosmos3 SFT model cfg validated: model_path=%s "
        "ema_enabled=%s disable_cosmos_parallelism=%s (keys_to_select=%d, "
        "lr_multipliers=%d, skip=%d)",
        model_path,
        model_cfg.get("ema_enabled"),
        model_cfg.get("disable_cosmos_parallelism"),
        len(model_cfg.get("keys_to_select", [])),
        len(model_cfg.get("lr_multipliers", {})),
        len(model_cfg.get("keys_to_skip_loading", [])),
    )
    return model_cfg
