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

"""Exercise Cosmos through WorldModelEnv with external native SDK doubles."""

import contextlib
import io
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from test_cosmos3_wovr_policy import TinyOmni
from test_cosmos3_wovr_policy import native_sdk as native_sdk

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.world_model.backend.cosmos3 import Cosmos3Backend
from rlinf.envs.sim.world_model.env import WorldModelEnv
from rlinf.envs.sim.world_model.registry import get_backend
from rlinf.models.embodiment.cosmos3.wovr_policy import Cosmos3WoVRPolicy
from rlinf.scheduler import Worker


class PixelSuccessModel(torch.nn.Module):
    """External WoVR reward classifier with known image-dependent probabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def predict_rew(self, images, instructions=None):
        self.calls.append((images.clone(), instructions))
        return (images[:, 0, 0, 0] + 1) / 2 * self.scale


class PixelOmni(TinyOmni):
    """External generator: action zero specifies a frame's uint8 red channel."""

    def generate_samples_from_batch(self, batch, **kwargs):
        self.batches.append(batch)
        video = batch["video"][0][0].float() / 127.5 - 1
        for index, action in enumerate(batch["action"][0][0]):
            video[:, index + 1] = action[0] / 127.5 - 1
        return {"vision": [video]}


class ProbabilityOmni(PixelOmni):
    """External dynamics with continuous action-dependent success probabilities."""

    def generate_samples_from_batch(self, batch, **kwargs):
        self.batches.append(batch)
        video = batch["video"][0][0].float() / 127.5 - 1
        for index, action in enumerate(batch["action"][0][0]):
            video[:, index + 1] = action[0].tanh()
        return {"vision": [video]}


@pytest.fixture(autouse=True)
def external_runtime(monkeypatch):
    monkeypatch.setattr(Worker, "torch_device_type", None)
    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(
            device=lambda _: contextlib.nullcontext(), empty_cache=lambda: None
        ),
    )
    module = ModuleType("diffsynth.models.reward_model")
    module.ResnetRewModel = PixelSuccessModel
    module.TaskEmbedResnetRewModel = PixelSuccessModel
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def config(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (tmp_path / "meta").mkdir()
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "fps": 20,
            }
        )
    )
    (tmp_path / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "move the bowl"}) + "\n"
    )

    def encoded(red):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (red, 0, 0)).save(buffer, format="PNG")
        return {"bytes": buffer.getvalue()}

    rows = [
        {
            "image": encoded(red),
            "wrist_image": encoded(min(red + 7, 255)),
            "task_index": 0,
            "episode_index": 1,
            "frame_index": frame,
        }
        for frame, red in [(0, 10), (10, 40), (20, 255)]
    ]
    pq.write_table(
        pa.Table.from_pylist(rows), tmp_path / "data/chunk-000/episode_000001.parquet"
    )
    return OmegaConf.create(
        {
            "env_type": "world_model",
            "backend": "cosmos3",
            "wm_env_type": "libero",
            "seed": 17,
            "group_size": 2,
            "max_episode_steps": 8,
            "num_frames": 5,
            "chunk": 4,
            "action_dim": 10,
            "image_size": [16, 32],
            "condition_frame_length": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "terminate_at_first_success": True,
            "reject_successful_reset": True,
            "use_fixed_reset_state_ids": True,
            "reset_gripper_open": False,
            "enable_kir": True,
            "video_cfg": {},
            "reward_coef": 1,
            "use_rel_reward": False,
            "success_reward_threshold": 0.5,
            "share_weights": True,
            "model_cfg": {"num_action_chunks": 4},
            "dataset": {
                "root": str(tmp_path),
                "keyframes": [{"episode_index": 1, "frame_index": 0}],
            },
            "reward_model": {
                "type": "ResnetRewModel",
                "from_pretrained": "external-test-classifier",
                "camera_view": "external",
                "view_size": [16, 16],
            },
        }
    )


@pytest.fixture
def simulator(request):
    request.getfixturevalue("native_sdk")
    return Cosmos3WoVRPolicy(PixelOmni(), num_action_chunks=4, num_inference_steps=3)


def make_env(config, simulator, num_envs=2):
    return WorldModelEnv(config, num_envs, 0, 1, shared_model=simulator)


def actions(values):
    result = torch.zeros((len(values), 4, 10), dtype=torch.float32)
    result[..., 0] = torch.tensor(values)
    result[..., 1:] = torch.arange(9) - 4
    return result


def test_grouped_keyframes_feed_native_canvas_and_raw_task(config, simulator):
    env = make_env(config, simulator)
    for _ in range(2):
        obs, _ = env.reset()
        assert obs["main_images"].shape == (2, 16, 32, 3)
        torch.testing.assert_close(obs["main_images"][0], obs["main_images"][1])
        assert obs["wrist_images"] is None
        assert obs["task_descriptions"] == ["move the bowl"] * 2
        assert obs["main_images"][0, 0, 0, 0].item() == 10
        assert obs["main_images"][0, 0, 16, 0].item() == 17
    assert all(
        not parameter.requires_grad for parameter in env.backend.model.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in env.reward_model.parameters()
    )
    assert get_backend("cosmos3") is Cosmos3Backend


def test_reset_seeds_are_reproducible_and_distinct_per_slot_epoch_rank(
    config, simulator
):
    env = make_env(config, simulator)
    replay = make_env(config, simulator)
    other_rank = WorldModelEnv(config, 2, 1, 2, shared_model=simulator)
    for item in (env, replay, other_rank):
        item.reset()
    seeds = [session["seed"] for session in env.backend._sessions.values()]
    assert len(set(seeds)) == 2
    assert seeds == [session["seed"] for session in replay.backend._sessions.values()]
    assert seeds != [
        session["seed"] for session in other_rank.backend._sessions.values()
    ]
    env.reset()
    assert seeds != [session["seed"] for session in env.backend._sessions.values()]


def test_standard_rl_path_uses_generated_rewards_for_real_policy_update(
    config, request
):
    request.getfixturevalue("native_sdk")
    from rlinf.algorithms.advantages import compute_grpo_advantages
    from rlinf.algorithms.registry import policy_loss
    from rlinf.utils.metric_utils import compute_loss_mask

    config.group_size = 4
    actor = Cosmos3WoVRPolicy(TinyOmni(), num_action_chunks=4, num_inference_steps=3)
    simulator = Cosmos3WoVRPolicy(
        ProbabilityOmni(), num_action_chunks=4, num_inference_steps=3
    )
    env = make_env(config, simulator, num_envs=4)
    observations, _ = env.reset()
    torch.manual_seed(41)
    raw_actions, rollout = actor.predict_action_batch(observations)
    prepared = prepare_actions(
        raw_actions, "world_model", "cosmos3", 4, 10, env_cfg=config
    )
    _, rewards, terminations, truncations, _ = env.chunk_step(prepared)
    dones = terminations | truncations
    mask, _ = compute_loss_mask(torch.stack([torch.zeros_like(dones), dones]))
    assert torch.unique(rewards.sum(-1)).numel() > 1
    assert mask.any() and not mask.all()
    grouped_advantages, _ = compute_grpo_advantages(
        rewards.sum(-1), torch.ones(1, 4, dtype=torch.bool), group_size=4
    )
    output = actor(forward_inputs=rollout["forward_inputs"])
    output["logprobs"].retain_grad()
    loss, _ = policy_loss(
        loss_type="actor",
        task_type="embodied",
        logprob_type="action_level",
        reward_type="action_level",
        single_action_dim=10,
        logprobs=output["logprobs"],
        old_logprobs=rollout["prev_logprobs"],
        advantages=grouped_advantages.T.expand(4, 4),
        loss_mask=mask[0],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    before = actor.omni.net.weight.detach().clone()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    assert torch.count_nonzero(output["logprobs"].grad[~mask[0]]) == 0
    assert actor.omni.net.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, actor.omni.net.weight)
    simulator.load_state_dict(actor.state_dict(), strict=True)
    env.backend.on_weights_updated(1)
    torch.testing.assert_close(actor.omni.net.weight, simulator.omni.net.weight)
    assert all(parameter.grad is None for parameter in simulator.parameters())


def test_first_success_masks_suffix_and_stops_future_simulation(config, simulator):
    env = make_env(config, simulator)
    env.reset()
    obs, reward, term, trunc, infos = env.chunk_step(
        actions([[30, 160, 250, 0], [30, 40, 50, 60]])
    )
    assert term.tolist() == [[False, True, True, True], [False] * 4]
    assert not trunc.any()
    torch.testing.assert_close(
        reward, torch.tensor([[30, 160, 0, 0], [30, 40, 50, 60]]) / 255
    )
    assert obs[0]["main_images"][0, 0, 0, 0].item() == pytest.approx(160, abs=1)
    assert infos[0]["episode"]["episode_len"].tolist() == [2, 4]
    assert len(simulator.omni.batches) == 2
    _, reward, term, trunc, infos = env.chunk_step(actions([[250] * 4, [70] * 4]))
    assert len(simulator.omni.batches) == 3
    assert term.tolist() == [[True] * 4, [False] * 4]
    assert trunc.tolist() == [[False] * 4, [False, False, False, True]]
    assert reward[0].sum() == 0
    assert infos[0]["episode"]["episode_len"].tolist() == [2, 8]
    _, reward, term, trunc, _ = env.chunk_step(actions([[250] * 4] * 2))
    assert len(simulator.omni.batches) == 3
    assert reward.sum() == 0
    assert term.tolist() == [[True] * 4, [False] * 4]
    assert trunc[1, -1]


def test_relative_rewards_end_at_first_success(config, simulator):
    config.use_rel_reward = True
    env = make_env(config, simulator)
    env.reset()
    _, rewards, _, _, _ = env.chunk_step(actions([[20, 60, 160, 250]] * 2))
    torch.testing.assert_close(rewards[0], torch.tensor([20, 40, 100, 0]) / 255)


def test_legacy_chunk_termination_remains_opt_in(config, simulator):
    config.terminate_at_first_success = False
    env = make_env(config, simulator)
    env.reset()
    obs, reward, term, _, _ = env.chunk_step(actions([[30, 160, 250, 0]] * 2))
    assert term.tolist() == [[False, False, False, True]] * 2
    torch.testing.assert_close(reward[0], torch.tensor([30, 160, 250, 0]) / 255)
    assert obs[0]["main_images"][0].sum() == 0
    env.chunk_step(actions([[10] * 4] * 2))
    assert len(simulator.omni.batches) == 4


def test_rejects_already_successful_keyframes(config, simulator):
    config.dataset.keyframes[0].frame_index = 20
    env = make_env(config, simulator)
    with pytest.raises(ValueError, match="already successful"):
        env.reset()


@pytest.mark.parametrize("view,red", [("external", 10), ("wrist", 17)])
def test_reward_selects_camera_on_reset_and_step(config, simulator, view, red):
    config.reward_model.camera_view = view
    env = make_env(config, simulator)
    env.reset()
    initial_images = env.reward_model.calls[0][0]
    assert initial_images.shape == (2, 3, 16, 16)
    assert ((initial_images[0, 0, 0, 0] + 1) * 127.5).item() == pytest.approx(
        red, abs=0.01
    )
    env.chunk_step(actions([[20] * 4] * 2))
    assert env.reward_model.calls[-1][0].shape == (8, 3, 16, 16)


def test_task_conditioned_reward_receives_unaugmented_instructions(config, simulator):
    config.reward_model.type = "TaskEmbedResnetRewModel"
    config.task_suite_name = "libero_spatial"
    env = make_env(config, simulator)
    env.reset()
    env.chunk_step(actions([[20] * 4] * 2))
    assert env.reward_model.calls[0][1] == ["move the bowl"] * 2
    assert env.reward_model.calls[-1][1] == ["move the bowl"] * 8


def test_action_preparation_preserves_native_10d(config):
    raw = actions([[1, -2, 3, -4]])
    expected = raw.clone()
    result = prepare_actions(
        raw, "world_model", "cosmos3", 4, 10, wm_env_type="libero", env_cfg=config
    )
    np.testing.assert_array_equal(result, expected.numpy())
    with pytest.raises(ValueError, match="10-D"):
        prepare_actions(raw, "world_model", "cosmos3", 4, 7, env_cfg=config)


def test_registered_world_model_keeps_existing_closed_loop_selector(config):
    from rlinf.envs.sim.cosmos3.env import Cosmos3Env

    assert get_env_cls("cosmos3", config) is Cosmos3Env
    assert get_env_cls("world_model", config) is WorldModelEnv
    assert get_backend("cosmos3") is Cosmos3Backend


def test_shared_backend_model_has_independent_sessions_and_version(config, simulator):
    env = make_env(config, simulator)
    other = make_env(config, simulator)
    assert env.backend.model is other.backend.model
    env.reset()
    other.reset()
    env.chunk_step(actions([[30] * 4] * 2))
    assert other.backend._sessions[0]["step"] == 0
    env.backend.on_weights_updated(2)
    assert env.backend.weight_version == 2
    with pytest.raises(ValueError, match="nondecreasing"):
        env.backend.on_weights_updated(1)
    env.offload()
    env.onload()
    assert env.backend.model.device.type == "cpu"


def test_npy_kir_uses_final_native_keyframe(config, simulator, tmp_path):
    dataset_dir = tmp_path / "npy"
    dataset_dir.mkdir()
    frames = [
        {
            "image": np.full((16, 32, 3), pixel, dtype=np.uint8),
            "cosmos_action": np.arange(10, dtype=np.float32),
            "instruction": "move the bowl",
        }
        for pixel in [10, 20, 30, 40, 50]
    ]
    np.save(dataset_dir / "episode_kir.npy", frames)
    config.dataset = None
    config.initial_image_path = str(dataset_dir)
    config.condition_frame_length = 5
    env = make_env(config, simulator)
    obs, _ = env.reset()
    assert obs["main_images"][0, 0, 0, 0].item() == pytest.approx(50, abs=1)
    env.chunk_step(actions([[30] * 4] * 2))
    assert simulator.omni.batches[0]["video"][0][0][:, 0].float().mean().item() == 50


def test_existing_rlinf_resnet_checkpoint_contract(config, simulator, tmp_path):
    from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel

    reward_cfg = OmegaConf.create(
        {
            "precision": "fp32",
            "pretrained": False,
            "image_size": [3, 16, 16],
        }
    )
    original = ResNetRewardModel(reward_cfg).eval()
    with torch.no_grad():
        original.backbone.fc.weight.zero_()
        original.backbone.fc.bias.fill_(-2)
    checkpoint = tmp_path / "reward.pth"
    torch.save(original.state_dict(), checkpoint)
    config.reward_model = OmegaConf.merge(
        reward_cfg,
        {
            "type": "resnet",
            "model_path": str(checkpoint),
            "camera_view": "external",
            "view_size": [16, 16],
        },
    )
    env = make_env(config, simulator)
    env.reset()
    _, reward, term, _, _ = env.chunk_step(actions([[20] * 4] * 2))
    torch.testing.assert_close(
        reward, torch.full((2, 4), torch.sigmoid(torch.tensor(-2.0)).item())
    )
    assert not term.any()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("action_dim", 7, "10D"),
        ("num_frames", 4, "chunk"),
        ("share_weights", False, "shared Cosmos"),
    ],
)
def test_backend_rejects_incompatible_config(config, simulator, field, value, match):
    config[field] = value
    with pytest.raises(ValueError, match=match):
        make_env(config, simulator)
