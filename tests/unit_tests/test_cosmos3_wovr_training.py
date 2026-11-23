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

"""CPU integration across simulated environments, GRPO and the FSDP actor."""

import contextlib
import inspect
import json
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from test_cosmos3_closed_loop import comparison_vendor as comparison_vendor
from test_cosmos3_wovr_config import wovr_recipe as wovr_recipe
from test_cosmos3_wovr_env import config as config
from test_cosmos3_wovr_env import external_runtime as external_runtime
from test_cosmos3_wovr_env import make_env
from test_cosmos3_wovr_env import simulator as simulator
from test_cosmos3_wovr_policy import native_sdk as native_sdk
from test_cosmos3_wovr_policy import observations
from test_cosmos3_wovr_policy import policy as policy
from test_cosmos3_wovr_policy import prefix_policy as prefix_policy
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from rlinf.config import validate_cfg
from rlinf.data.schema.embodied_types import Trajectory, TrajectoryStep
from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.scheduler import Worker
from rlinf.utils.nested_dict_process import process_nested_dict_for_train
from rlinf.utils.utils import get_loss_agg_func
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(str(tmp_path / "gloo_store"), 1),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_world_model_rollout_trains_standard_fsdp_actor(
    request, single_rank_gloo, monkeypatch, tmp_path
):
    env_cfg = request.getfixturevalue("config")
    simulator_model = request.getfixturevalue("simulator")
    policy_model = request.getfixturevalue("policy")
    recipe = request.getfixturevalue("wovr_recipe")
    cfg = validate_cfg(
        recipe(
            "actor.model.action_horizon=4",
            "actor.model.num_action_chunks=4",
            "actor.micro_batch_size=8",
            "actor.global_batch_size=8",
            "env.train.num_frames=5",
            "env.train.max_episode_steps=8",
        )
    )
    # Keep production worker/loss config and use the existing on-disk toy dataset.
    env_cfg.group_size = cfg.algorithm.group_size
    env = make_env(env_cfg, simulator_model, num_envs=4)
    obs, _ = env.reset()
    # Broad exploration makes the external pixel reward distinguish this group.
    policy_model.noise_std = 100
    torch.manual_seed(41)
    records, rewards = [], []
    dones = [torch.zeros(4, 4, dtype=torch.bool)]
    for _ in range(2):
        actions, record = policy_model.predict_action_batch(obs)
        observations, reward, terminated, truncated, _ = env.chunk_step(actions)
        records.append(record)
        rewards.append(reward)
        dones.append(terminated | truncated)
        obs = observations[-1]
    assert torch.stack(rewards).sum((0, 2)).std() > 0

    actor = train_actor(cfg, policy_model, records, rewards, dones, monkeypatch)
    assert all(parameter.grad is None for parameter in simulator_model.parameters())

    actor.lr_scheduler.step()
    saved_weight = policy_model.omni.net.weight.detach().clone()
    optimizer = actor.optimizer
    saved_moment = optimizer.state[policy_model.omni.net.weight]["exp_avg"].clone()
    saved_epoch = actor.lr_scheduler.last_epoch
    checkpoint = str(tmp_path / "actor_checkpoint")
    actor.save_checkpoint(checkpoint, step=1)
    assert (tmp_path / "actor_checkpoint/dcp_checkpoint/.metadata").is_file()
    with torch.no_grad():
        policy_model.omni.net.weight.add_(1)
        optimizer.state[policy_model.omni.net.weight]["exp_avg"].zero_()
    actor.lr_scheduler.step()
    actor.load_checkpoint(checkpoint)
    torch.testing.assert_close(policy_model.omni.net.weight, saved_weight)
    torch.testing.assert_close(
        optimizer.state[policy_model.omni.net.weight]["exp_avg"], saved_moment
    )
    assert actor.lr_scheduler.last_epoch == saved_epoch


def train_actor(cfg, policy_model, records, rewards, dones, monkeypatch):
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = cfg
    actor._cfg = cfg.actor
    actor.device = torch.device("cpu")
    actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
    actor._strategy = FSDPStrategy(cfg.actor, world_size=1)
    actor.model = FullyShardedDataParallel(
        policy_model,
        device_id=torch.device("cpu"),
        use_orig_params=cfg.actor.fsdp_config.use_orig_params,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    actor.amp_context = contextlib.nullcontext()
    actor.optimizer_steps = actor.critic_warmup_steps = 0
    actor.enable_sft_co_train = False
    actor.gradient_accumulation = 1
    actor.grad_scaler = actor.build_grad_scaler(enabled=False)
    actor.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
    monkeypatch.setattr(
        Worker.torch_platform, "current_device", lambda: "cpu", raising=False
    )
    for name in ("synchronize", "ipc_collect"):
        monkeypatch.setattr(Worker.torch_platform, name, lambda: None, raising=False)
    monkeypatch.setattr(
        Worker.torch_platform, "is_available", lambda: False, raising=False
    )
    steps = [
        TrajectoryStep(
            rewards=reward,
            dones=done,
            prev_logprobs=record["prev_logprobs"],
            prev_values=record["prev_values"],
            forward_inputs=record["forward_inputs"],
        )
        for record, reward, done in zip(records, rewards, dones[1:], strict=True)
    ]
    steps[0].initial_dones = dones[0]
    batch = Trajectory.to_batch([Trajectory.from_steps(steps)])
    actor.rollout_batch = actor._process_received_rollout_batch(batch)
    rollout_metrics = inspect.unwrap(actor.compute_advantages_and_returns)(actor)
    assert torch.count_nonzero(actor.rollout_batch["advantages"]) > 0
    assert not actor.rollout_batch["loss_mask"].all()
    assert 0 < rollout_metrics["loss_mask_fraction"] < 1
    micro_batch = process_nested_dict_for_train(actor.rollout_batch, torch.arange(8))
    optimizer = FP32MasterAdamW(actor.model.parameters(), lr=0.01)
    actor.optimizer = optimizer
    actor.lr_scheduler = actor.build_lr_scheduler(optimizer, cfg.actor.optim)
    before = policy_model.omni.net.weight.detach().clone()
    metrics = {}
    actor.model.train()
    actor.train_micro_batch(micro_batch, metrics, is_last=True)
    assert torch.isfinite(policy_model.omni.net.weight.grad).all()
    assert policy_model.omni.net.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, policy_model.omni.net.weight)
    assert all(
        torch.isfinite(torch.as_tensor(values)).all() for values in metrics.values()
    )
    return actor


def make_rollout(cfg, policy_model):
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.cfg = cfg
    rollout.hf_model = policy_model
    rollout.model_cfg = cfg.actor.model
    rollout.algorithm_cfg = cfg.algorithm
    rollout._train_sampling_params = {}
    rollout._eval_sampling_params = {}
    rollout.enable_dagger = False
    rollout.expert_model = None
    return rollout


@pytest.mark.parametrize("world_model_mode", ["train", "eval"])
def test_rollout_sends_complete_plan_to_learned_simulator(world_model_mode, request):
    cfg = request.getfixturevalue("wovr_recipe")("actor.model.num_action_chunks=5")
    libero_mode = "eval" if world_model_mode == "train" else "train"
    OmegaConf.update(
        cfg.env,
        world_model_mode,
        {"env_type": "world_model", "backend": "cosmos3"},
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg.env, libero_mode, {"env_type": "libero"}, merge=False, force_add=True
    )
    rollout = make_rollout(cfg, request.getfixturevalue("prefix_policy"))
    obs = observations(np.zeros((16, 32, 3), dtype=np.uint8), batch_size=1)
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    torch.manual_seed(41)
    plan, learned = predict(rollout, obs, world_model_mode)
    torch.manual_seed(41)
    executed, physical = predict(rollout, obs, libero_mode)
    assert plan.shape == (1, 16, 10) and executed.shape == (1, 5, 10)
    torch.testing.assert_close(executed, plan[:, :5])
    torch.testing.assert_close(learned["forward_inputs"]["action"], executed.flatten(1))
    torch.testing.assert_close(learned["prev_logprobs"], physical["prev_logprobs"])


def test_libero_rollout_trains_standard_fsdp_actor(
    request, single_rank_gloo, monkeypatch, tmp_path
):
    policy_model = request.getfixturevalue("prefix_policy")
    stats = tmp_path / "action_stats.json"
    stats.write_text(json.dumps({"global_raw": {"q01": [-1] * 10, "q99": [1] * 10}}))
    cfg = validate_cfg(
        request.getfixturevalue("wovr_recipe")(
            "algorithm.group_size=4",
            "env.train.total_num_envs=4",
            "env.train.rollout_epoch=1",
            "actor.model.action_horizon=16",
            "actor.model.num_action_chunks=5",
            "actor.micro_batch_size=8",
            "actor.global_batch_size=8",
            "env.train.max_episode_steps=10",
            "env.train.max_steps_per_rollout_epoch=10",
            "+env.train.num_steps_wait=0",
            f"env.train.action_stats_path={json.dumps(stats.as_posix())}",
            config_name="libero_spatial_grpo_cosmos3",
        )
    )
    vendor, _ = request.getfixturevalue("comparison_vendor")
    monkeypatch.syspath_prepend(str(vendor))
    monkeypatch.setenv("LIBERO_TYPE", "standard")
    # Isolate the vendor fixture's temporary package from later LIBERO tests.
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "libero" or name.startswith(
                ("libero.", "rlinf.envs.sim.libero.")
            ):
                del sys.modules[name]
        from libero.libero.envs import OffScreenRenderEnv

        from rlinf.envs.sim.libero.libero_env import LiberoEnv

        frame = OffScreenRenderEnv("task_0.bddl", 16, 16).observation()
        frame["robot0_eye_in_hand_image"].fill(190)
        frames = [frame] * 4
        vector_env = Mock()
        vector_env.reset.return_value = vector_env.set_init_state.return_value = frames
        # Replace only the external simulation. Exercise LIBERO's reset, camera
        # extraction, task success reward, truncation and chunking in RLinf.
        with patch(
            "rlinf.envs.sim.libero.libero_env.ReconfigureSubprocEnv",
            return_value=vector_env,
        ):
            cfg.env.train.task_suite_name = "fake"
            env = LiberoEnv(cfg.env.train, 4, 0, 1, None)
            obs, _ = env.reset()
    successes = np.zeros(4, dtype=bool)
    steps = 0

    def step(controls):
        nonlocal steps
        assert controls.shape == (4, 7)
        assert np.isfinite(controls).all()
        assert set(controls[:, -1]) <= {-1, 1}
        # Score the first executed translation; keep successes terminal.
        if steps == 0:
            successes[:] = controls[:, 0] > 0
        steps += 1
        return frames, successes.astype(float), successes.copy(), [{} for _ in frames]

    vector_env.step.side_effect = step
    rollout = make_rollout(cfg, policy_model)
    worker = object.__new__(EnvWorker)
    worker.cfg = cfg
    worker.model_cfg = cfg.actor.model
    worker.env_list = [env]
    worker.enable_online_lerobot = worker.use_external_reward_model = False
    records, rewards = [], []
    dones = [torch.zeros(4, 5, dtype=torch.bool)]
    torch.manual_seed(41)
    for _ in range(2):
        actions, record = inspect.unwrap(MultiStepRolloutWorker.predict)(rollout, obs)
        canvas = record["forward_inputs"]["image"]
        torch.testing.assert_close(canvas[:, 0, 0], obs["main_images"][:, 0, 0])
        torch.testing.assert_close(canvas[:, 0, 16], obs["wrist_images"][:, 0, 0])
        sampled = actions.clone()
        output, _, payload = inspect.unwrap(EnvWorker.env_interact_step)(
            worker, actions, 0
        )
        assert payload["chunk_actions"].shape == (4, 5, 7)
        torch.testing.assert_close(
            record["forward_inputs"]["action"], sampled.flatten(1)
        )
        torch.testing.assert_close(actions, sampled)
        torch.testing.assert_close(
            policy_model(forward_inputs=record["forward_inputs"])["logprobs"],
            record["prev_logprobs"],
        )
        records.append(record)
        rewards.append(output.transition.rewards)
        dones.append(output.transition.dones)
        obs = output.obs
    assert steps == 10
    np.testing.assert_allclose(torch.stack(rewards).sum((0, 2)), successes)
    train_actor(cfg, policy_model, records, rewards, dones, monkeypatch)
