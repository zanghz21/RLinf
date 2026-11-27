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

from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir

from rlinf.config import validate_cfg


@pytest.fixture
def wovr_recipe(monkeypatch):
    # Exercise the normal configuration entry without starting a Ray cluster.
    monkeypatch.setattr("rlinf.config.Cluster", Mock())
    placement = Mock()
    placement.return_value.get_world_size.return_value = 1
    monkeypatch.setattr("rlinf.config.HybridComponentPlacement", placement)
    directory = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):

        def recipe(*overrides, config_name="cosmos3_wovr"):
            return compose(config_name=config_name, overrides=list(overrides))

        yield recipe


@pytest.mark.parametrize("shared", [True, False])
def test_embodied_config_routes_cosmos_weights(wovr_recipe, shared):
    overrides = [
        "actor.model.model_path=policy",
        "actor.model.native_config_path=policy.json",
        "actor.model.wovr.lora_enabled=true",
    ]
    if not shared:
        overrides += [
            "cosmos.share_weights=false",
            "cosmos.simulator.model_path=dynamics",
        ]
    cfg = validate_cfg(wovr_recipe(*overrides))
    assert cfg.runner.task_type == "embodied"
    assert cfg.algorithm.adv_type == "grpo"
    assert cfg.algorithm.loss_type == "actor"
    assert cfg.actor.training_backend == "fsdp"
    assert cfg.actor.model.train_dit_backbone is True
    assert cfg.actor.model.train_action_input_projection is False
    assert cfg.actor.model.train_action_output_projection is False
    assert cfg.actor.model.train_action_modality_embedding is False
    assert cfg.actor.model.train_time_embedder is False
    assert cfg.env.train.backend == "cosmos3"
    assert cfg.env.train.share_weights is shared
    assert cfg.env.train.model_cfg.wovr.lora_enabled is shared
    assert cfg.rollout.model.model_path == "policy"
    assert cfg.env.train.model_cfg.model_path == ("policy" if shared else "dynamics")
    assert cfg.env.train.model_cfg.native_config_path == (
        "policy.json" if shared else None
    )
    assert cfg.env.train.model_cfg.raw_action_dim == cfg.actor.model.action_dim == 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["cosmos.share_weights=false"], "require cosmos.simulator.model_path"),
        (["cosmos.simulator.model_path=other"], "Shared weights"),
        (
            [
                "weight_syncer=patch_syncer",
                "weight_syncer.patch.init_sync.enabled=false",
            ],
            "initial patch sync",
        ),
    ],
)
def test_invalid_weight_routing_fails_before_workers_start(
    wovr_recipe, overrides, message
):
    with pytest.raises(ValueError, match=message):
        validate_cfg(wovr_recipe(*overrides))


def test_native_libero_grpo_recipe(wovr_recipe):
    cfg = validate_cfg(
        wovr_recipe(
            "actor.model.model_path=policy",
            "env.train.action_stats_path=stats.json",
            "runner.val_check_interval=1",
            "actor.model.wovr.lora_enabled=false",
            "actor.model.train_action_output_projection=true",
            config_name="libero_spatial_grpo_cosmos3",
        )
    )
    assert cfg.actor.model.wovr.enabled and cfg.actor.model.action_dim == 10
    assert not cfg.actor.model.wovr.lora_enabled
    assert cfg.actor.model.train_dit_backbone is True
    assert cfg.actor.model.train_action_output_projection is True
    assert cfg.rollout.model.model_path == "policy"
    assert cfg.algorithm.adv_type == "grpo"
    assert cfg.env.train.use_fixed_reset_state_ids
    assert cfg.env.train.group_size == cfg.algorithm.group_size > 1
    assert not cfg.env.train.ignore_terminations
    assert cfg.env.eval.is_eval and cfg.env.eval.group_size == 1
    assert "cosmos" not in cfg and not cfg.reward.use_reward_model
    for env in (cfg.env.train, cfg.env.eval):
        assert env.env_type == "libero" and env.task_suite_name == "libero_spatial"
        assert env.action_stats_path == "stats.json"
        assert "model_cfg" not in env and "share_weights" not in env
    cfg.env.train.task_suite_name = "libero_goal"
    assert cfg.env.eval.task_suite_name == "libero_goal"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("actor.model.action_dim=7", "10D policy actions"),
        ("env.train.action_stats_path=null", "env.train.action_stats_path"),
        (
            "actor.model.wovr.lora_enabled=invalid",
            "wovr.lora_enabled must be a boolean",
        ),
    ],
)
def test_invalid_native_libero_config(wovr_recipe, override, message):
    with pytest.raises(ValueError, match=message):
        validate_cfg(wovr_recipe(override, config_name="libero_spatial_grpo_cosmos3"))


@pytest.mark.parametrize(
    "override",
    [
        "runner.resume_dir=/run/checkpoints/global_step_3",
        "actor.model.wovr.lora_enabled=true",
    ],
)
def test_native_libero_rollout_starts_from_the_full_actor(wovr_recipe, override):
    cfg = validate_cfg(
        wovr_recipe(
            override,
            "weight_syncer=patch_syncer",
            "weight_syncer.patch.init_sync.enabled=false",
            "weight_syncer.patch.init_sync.prefixes=[value_head]",
            config_name="libero_spatial_grpo_cosmos3",
        )
    )
    assert cfg.weight_syncer.patch.init_sync.enabled
    assert cfg.weight_syncer.patch.init_sync.prefixes is None
