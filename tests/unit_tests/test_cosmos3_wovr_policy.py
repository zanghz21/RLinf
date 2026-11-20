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

"""CPU likelihood and optimizer checks with a small native Cosmos SDK double."""

import contextlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rlinf.models.embodiment.cosmos3.wovr_policy import (
    Cosmos3WoVRPolicy,
)


class TinyOmni(nn.Module):
    """Implement the external Cosmos velocity interface with a trainable network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 10)
        with torch.no_grad():
            self.net.weight.copy_(torch.eye(10) * 0.1)
            self.net.bias.zero_()
        self.config = SimpleNamespace(
            max_action_dim=12,
            action_gen=True,
            video_temporal_causal=False,
            rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
        )
        self.input_video_key = "video"
        self.input_caption_key = "ai_caption"
        self.batches = []

    def _prepare_inference_data(self, data_batch, seed, has_negative_prompt):
        self.batches.append(data_batch)
        assert not has_negative_prompt
        assert data_batch["video"][0][0].shape[1] == 1
        actions = data_batch["action"][0][0]
        vision = torch.tensor([data_batch["video"][0][0].float().mean(), 0.25])
        generator = torch.Generator().manual_seed(seed[0])
        initial_action = torch.randn(actions.shape, generator=generator)
        initial_action[:, 10:] = 0
        initial = torch.cat([vision, initial_action.flatten()])
        return (
            data_batch["sequence_plan"],
            SimpleNamespace(x0_tokens_action=[actions]),
            [[1]],
            [[0]],
            [initial],
            [torch.cat([vision, torch.zeros(actions.numel())])],
            [torch.cat([torch.ones(2), torch.zeros(actions.numel())])],
            True,
        )

    def _get_velocity(
        self,
        *,
        net,
        noise_x,
        timestep,
        text_tokens,
        sequence_plans,
        gen_data_clean,
        has_noisy_actions,
    ):
        assert has_noisy_actions
        assert timestep.shape == (1, 1)
        assert sequence_plans[0].mode == "policy"
        action = noise_x[0][2:].reshape_as(gen_data_clean.x0_tokens_action[0])
        assert torch.count_nonzero(action[:, 10:]) == 0
        velocity = net(action[:, :10].to(net.weight.dtype)) + text_tokens[0][0] * 0.03
        padded = torch.nn.functional.pad(velocity, (0, 2))
        return [torch.cat([torch.zeros(2), padded.flatten()])]

    def generate_samples_from_batch(self, batch, **kwargs):
        self.batches.append(batch)
        assert batch["mode"] == ["forward_dynamics"]
        video = batch["video"][0][0].float() / 127.5 - 1
        video = video + batch["action"][0][0].mean() * 0.01 + self.net.bias[0] * 5
        return {"vision": [video]}

    def decode(self, latent):
        return latent


@pytest.fixture
def native_sdk(monkeypatch):
    """Supply only the external SDK data builders used by the adapter."""
    prefix = "cosmos_framework.data.generator.action.utils"
    modules = {
        "action_processing": {
            "ActionProcessingRecord": lambda **kwargs: SimpleNamespace(**kwargs),
            "make_batched_action_processing_fields": lambda record, size: {
                "raw_action_dim": [torch.tensor(record.raw_action_dim)] * size
            },
            "pad_action_to_max_dim": lambda action, width: torch.nn.functional.pad(
                action, (0, width - action.shape[-1])
            ),
        },
        "domain_utils": {"get_domain_id": lambda name: 5},
        "transforms": {
            "build_sequence_plan_from_mode": lambda **kwargs: SimpleNamespace(**kwargs)
        },
    }

    class Formatter:
        caption_key = "ai_caption"

        def __call__(self, value):
            assert value["video_num_frames"] == 5
            return value

    modules["json_formatter"] = {"ActionPromptJsonFormatter": Formatter}
    for name, attributes in modules.items():
        module = ModuleType(f"{prefix}.{name}")
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def policy(native_sdk):
    return Cosmos3WoVRPolicy(
        TinyOmni(), num_action_chunks=4, num_inference_steps=3, noise_std=0.2
    )


@pytest.fixture
def image():
    return np.full((16, 32, 3), 120, dtype=np.uint8)


def observations(image, batch_size=2):
    return {
        "main_images": torch.from_numpy(np.stack([image] * batch_size)),
        "wrist_images": None,
        "task_descriptions": ["move the cube \u7ea2\u8272"] * batch_size,
    }


def test_rollout_tensor_records_replay_with_live_gradients(policy, image):
    from rlinf.models.embodiment.base_policy import BasePolicy
    from rlinf.models.embodiment.cosmos3.closed_loop_io import augment_prompt

    assert isinstance(policy, BasePolicy)
    obs = observations(image)
    actions, result = policy.predict_action_batch(obs)
    inputs = result["forward_inputs"]
    assert actions.shape == (2, 4, 10)
    assert result["prev_logprobs"].shape == (2, 4, 10)
    assert result["prev_values"] is None
    assert all(isinstance(value, torch.Tensor) for value in inputs.values())
    assert all(not value.requires_grad for value in inputs.values())
    assert not policy.training
    torch.testing.assert_close(inputs["action"], actions.flatten(1))
    output = policy(forward_inputs=inputs, compute_entropy=True)
    torch.testing.assert_close(output["logprobs"], result["prev_logprobs"])
    assert output["entropy"].shape == (2, 4, 10)
    assert torch.isfinite(output["entropy"]).all()
    (-output["logprobs"].mean()).backward()
    assert policy.omni.net.weight.grad.abs().sum() > 0
    assert torch.isfinite(policy.omni.net.weight.grad).all()
    assert policy.omni.batches[-1]["prompt"] == [
        augment_prompt(obs["task_descriptions"][1])
    ]
    obs["main_images"].fill_(255)
    assert torch.all(inputs["image"] == 120)


def test_libero_both_cameras_are_used_for_sampling_and_replay(native_sdk):
    policy = Cosmos3WoVRPolicy(
        TinyOmni(), num_action_chunks=4, num_inference_steps=3, image_size=(16, 32)
    )
    views = torch.arange(2 * 16 * 16 * 3, dtype=torch.int32).reshape(2, 16, 16, 3)
    main = (views % 128).to(torch.uint8)
    wrist = 255 - main
    expected = torch.cat([main, wrist], dim=2)
    obs = {
        "main_images": main,
        "wrist_images": wrist,
        "task_descriptions": ["stack the cubes", "pick up the bowl"],
    }
    _, result = policy.predict_action_batch(obs)
    inputs = result["forward_inputs"]
    torch.testing.assert_close(inputs["image"], expected)
    for batch, canvas in zip(policy.omni.batches, expected, strict=True):
        torch.testing.assert_close(batch["video"][0][0][:, 0], canvas.permute(2, 0, 1))

    main.zero_()
    wrist.zero_()
    replay = policy(forward_inputs=inputs)
    torch.testing.assert_close(replay["logprobs"], result["prev_logprobs"])
    torch.testing.assert_close(inputs["image"], expected)

    obs["wrist_images"] = wrist[:1]
    with pytest.raises(ValueError, match="matching uint8 LIBERO cameras"):
        policy.predict_action_batch(obs)


def test_standard_grpo_update_and_weight_publication(policy, image):
    from rlinf.algorithms.advantages import compute_grpo_advantages
    from rlinf.algorithms.registry import policy_loss
    from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW

    shared_simulator = Cosmos3WoVRPolicy(
        TinyOmni(), num_action_chunks=4, num_inference_steps=3, noise_std=0.2
    ).requires_grad_(False)
    independent_simulator = Cosmos3WoVRPolicy(
        TinyOmni(), num_action_chunks=4, num_inference_steps=3, noise_std=0.2
    ).requires_grad_(False)
    policy.bfloat16()
    shared_simulator.bfloat16()
    independent_simulator.bfloat16()
    torch.manual_seed(41)
    actions, result = policy.predict_action_batch(observations(image, 4))
    before = shared_simulator.generate_dynamics(image, "move the cube", actions[0], 8)
    independent_before = independent_simulator.generate_dynamics(
        image, "move the cube", actions[0], 8
    )
    grouped_advantages, _ = compute_grpo_advantages(
        rewards=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        loss_mask=torch.ones(1, 4, dtype=torch.bool),
        group_size=4,
    )
    advantages = grouped_advantages.T.expand(4, 4)
    mask = torch.tensor([[True, True, False, False]] * 4)
    output = policy(forward_inputs=result["forward_inputs"])
    output["logprobs"].retain_grad()
    loss, metrics = policy_loss(
        loss_type="actor",
        task_type="embodied",
        logprob_type="action_level",
        reward_type="action_level",
        single_action_dim=10,
        logprobs=output["logprobs"],
        old_logprobs=result["prev_logprobs"],
        advantages=advantages,
        loss_mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert torch.isfinite(loss)
    optimizer = FP32MasterAdamW(policy.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    assert torch.count_nonzero(output["logprobs"].grad[:, 2:]) == 0
    assert torch.count_nonzero(output["logprobs"].grad[:, :2]) > 0
    assert policy.omni.net.weight.grad.abs().sum() > 0
    assert all(np.isfinite(value) for value in metrics.values())
    optimizer.step()
    assert (
        optimizer.state[policy.omni.net.weight]["fp32_master_param"].dtype
        == torch.float32
    )
    after_logprob = policy(forward_inputs=result["forward_inputs"])["logprobs"]
    assert not torch.allclose(after_logprob, result["prev_logprobs"])
    # Standard state_dict publication updates the shared simulator copy only.
    shared_simulator.load_state_dict(policy.state_dict(), strict=True)
    after = shared_simulator.generate_dynamics(image, "move the cube", actions[0], 8)
    independent_after = independent_simulator.generate_dynamics(
        image, "move the cube", actions[0], 8
    )
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(independent_before, independent_after)
    assert all(parameter.grad is None for parameter in shared_simulator.parameters())
    assert all(
        parameter.grad is None for parameter in independent_simulator.parameters()
    )
    torch.testing.assert_close(
        shared_simulator(forward_inputs=result["forward_inputs"])["logprobs"],
        after_logprob,
    )
    assert after.shape == (5, 16, 32, 3)
    assert after.dtype == np.uint8


def test_rollout_seed_repeatability_and_distinct_exploration(policy, image):
    torch.manual_seed(18)
    actions, result = policy.predict_action_batch(observations(image))
    torch.manual_seed(18)
    again, repeated = policy.predict_action_batch(observations(image))
    torch.testing.assert_close(actions, again)
    for key, value in result["forward_inputs"].items():
        torch.testing.assert_close(value, repeated["forward_inputs"][key])
    torch.manual_seed(19)
    different, _ = policy.predict_action_batch(observations(image))
    assert not torch.equal(actions, different)


def test_actor_rejects_cast_states_and_missing_critic(policy, image):
    _, result = policy.predict_action_batch(observations(image))
    with pytest.raises(ValueError, match="no value head"):
        policy(forward_inputs=result["forward_inputs"], compute_values=True)
    result["forward_inputs"]["state_before"] = result["forward_inputs"][
        "state_before"
    ].bfloat16()
    with pytest.raises(ValueError, match="retain float32"):
        policy(forward_inputs=result["forward_inputs"])
    policy.max_prompt_bytes = 4
    with pytest.raises(ValueError, match="exceeds max_prompt_bytes"):
        policy.predict_action_batch(observations(image))


def test_to_device_moves_native_unregistered_vae(policy):
    vae_network = nn.Linear(2, 2).requires_grad_(False)
    vae = SimpleNamespace(
        model=vae_network, device="cuda:3", scale=(torch.ones(2), torch.ones(2))
    )
    policy.omni.tokenizer_vision_gen = SimpleNamespace(model=vae)
    policy.omni.tensor_kwargs = {"device": "cuda:3", "dtype": torch.bfloat16}
    policy.omni.tensor_kwargs_fp32 = {"device": "cuda:3", "dtype": torch.float32}
    policy.to("cpu")
    assert vae.device == torch.device("cpu")
    assert policy.omni.tensor_kwargs["device"] == torch.device("cpu")
    assert policy.omni.tensor_kwargs_fp32["dtype"] == torch.float32
    assert not any(name.startswith("omni.tokenizer") for name in policy.state_dict())


@pytest.mark.parametrize(
    "field,value", [("noise_std", 0), ("shift", -1), ("num_inference_steps", 0)]
)
def test_invalid_sampling_config(field, value):
    with pytest.raises(ValueError, match=field):
        Cosmos3WoVRPolicy(TinyOmni(), **{field: value})


def test_invalid_actions_fail_before_simulator(policy, image):
    with pytest.raises(ValueError, match="Expected actions"):
        policy.generate_dynamics(image, "move the cube", torch.zeros(4, 7), 0)
    with pytest.raises(ValueError, match="finite"):
        policy.generate_dynamics(
            image, "move the cube", torch.full((4, 10), float("nan")), 0
        )


def test_checkpoint_vae_and_patch_geometry_are_validated(native_sdk, image):
    omni = TinyOmni()
    omni.tokenizer_vision_gen = SimpleNamespace(
        temporal_compression_factor=4, spatial_compression_factor=16
    )
    omni.config.diffusion_expert_config = SimpleNamespace(patch_spatial=2)
    with pytest.raises(ValueError, match="temporal factor 4"):
        Cosmos3WoVRPolicy(omni, num_action_chunks=2)
    policy = Cosmos3WoVRPolicy(omni, num_action_chunks=4)
    with pytest.raises(ValueError, match="multiples of 32"):
        policy.predict_action_batch(observations(image))


def test_loader_preserves_weights_and_selects_trainable_heads(
    monkeypatch, tmp_path, native_sdk, image
):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": {"config": {}}}))
    loaded = TinyOmni()
    with torch.no_grad():
        loaded.net.weight.fill_(0.4)
        loaded.net.bias.fill_(0.7)
    calls = []

    class NativeConfig:
        @classmethod
        def from_json_file(cls, path):
            return SimpleNamespace(**json.loads(config_file.read_text()))

    class NativeModel:
        @classmethod
        def from_pretrained_dcp(cls, path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(model=loaded)

    native_modules = {
        "cosmos_framework.configs.base.defaults.compile": {
            "CompileConfig": lambda **kwargs: SimpleNamespace(**kwargs)
        },
        "cosmos_framework.configs.base.defaults.parallelism": {
            "ParallelismConfig": lambda **kwargs: SimpleNamespace(**kwargs)
        },
        "cosmos_framework.inference.model": {
            "Cosmos3OmniConfig": NativeConfig,
            "Cosmos3OmniModel": NativeModel,
        },
    }
    for name, attributes in native_modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    # The SDK double stays on CPU while the loader selects its accelerator context.
    from rlinf.scheduler import Worker

    contexts = []

    def device_context(device):
        contexts.append(device)
        return contextlib.nullcontext()

    monkeypatch.setattr(Worker.torch_platform, "device", device_context)
    monkeypatch.setattr(Worker.torch_platform, "current_device", lambda: 3)
    from omegaconf import OmegaConf

    from rlinf.models import get_model

    model_cfg = OmegaConf.create(
        {
            "model_type": "cosmos3",
            "precision": "bf16",
            "is_lora": False,
            "load_to_device": False,
            "wovr": {"enabled": True},
            "model_path": str(tmp_path),
            "keys_to_select": ["weight"],
        }
    )
    policy = get_model(model_cfg)
    assert policy.omni.net.weight.requires_grad
    assert not policy.omni.net.bias.requires_grad
    assert torch.all(policy.omni.net.weight == 0.4)
    assert torch.all(policy.omni.net.bias == 0.7)
    assert calls[0][0] == tmp_path.resolve()
    assert contexts[0] == torch.device(Worker.torch_device_type, 3)
    assert calls[0][1]["parallelism_config"].data_parallel_shard_degree == 1
    assert calls[0][1]["parallelism_config"].data_parallel_replicate_degree == -1
    assert model_cfg.wovr.enabled
    assert not calls[0][1]["compile_config"].enabled
    with pytest.raises(ValueError, match="No Cosmos network parameters"):
        Cosmos3WoVRPolicy.from_pretrained(
            {
                "model_path": str(tmp_path),
                "device": Worker.torch_device_type,
                "keys_to_select": ["missing"],
            }
        )
    loaded.config.video_temporal_causal = True
    simulator = Cosmos3WoVRPolicy.from_pretrained(
        {
            "model_path": str(tmp_path),
            "keys_to_select": ["missing"],
            "num_action_chunks": 4,
        },
        for_simulation=True,
    )
    assert all(not parameter.requires_grad for parameter in simulator.parameters())
    video = simulator.generate_dynamics(image, "move the cube", torch.zeros(4, 10), 0)
    assert video.shape == (5, 16, 32, 3)
    with pytest.raises(ValueError, match="simulator-only"):
        simulator.predict_action_batch(observations(image))
    with pytest.raises(ValueError, match="bidirectional"):
        Cosmos3WoVRPolicy.from_pretrained(
            {"model_path": str(tmp_path), "keys_to_select": ["weight"]}
        )
    loaded.config.action_gen = False
    with pytest.raises(ValueError, match="action-conditioned simulation"):
        Cosmos3WoVRPolicy.from_pretrained(
            {"model_path": str(tmp_path)}, for_simulation=True
        )
    (tmp_path / "hf_quant_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unquantized"):
        Cosmos3WoVRPolicy.from_pretrained({"model_path": str(tmp_path)})
