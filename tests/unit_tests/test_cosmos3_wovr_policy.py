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
        self.velocity_batch_sizes = []
        self.prompt_ids = {}

    def _prepare_inference_data(self, data_batch, seed, has_negative_prompt):
        self.batches.append(data_batch)
        assert not has_negative_prompt
        plans = data_batch["sequence_plan"]
        assert (
            data_batch["mode"] == [plan.mode for plan in plans] == ["wam"] * len(seed)
        )
        assert data_batch["image_size"].shape == (len(seed), 4)
        actions = [action for [action] in data_batch["action"]]
        initial, clean, masks = [], [], []
        vision_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        for [video], action, plan, row_seed in zip(
            data_batch["video"], actions, plans, seed, strict=True
        ):
            assert plan.video_length == video.shape[1] == action.shape[0] + 1
            assert plan.action_length == action.shape[0]
            # Two channels, each with a conditioned observation and a future latent.
            vision = torch.tensor([video[:, 0].float().mean(), 0.0, 0.25, 0.0])
            generator = torch.Generator().manual_seed(row_seed)
            initial_vision = vision + (1 - vision_mask) * torch.randn(
                4, generator=generator
            )
            initial_action = torch.randn(action.shape, generator=generator)
            initial_action[:, 10:] = 0
            initial.append(torch.cat([initial_vision, initial_action.flatten()]))
            clean.append(torch.cat([vision, torch.zeros(action.numel())]))
            masks.append(torch.cat([vision_mask, torch.zeros(action.numel())]))
        return (
            plans,
            SimpleNamespace(x0_tokens_action=actions),
            [
                [self.prompt_ids.setdefault(prompt, len(self.prompt_ids) + 1)]
                for prompt in data_batch["prompt"]
            ],
            [[0] for _ in seed],
            initial,
            clean,
            masks,
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
        batch_size = len(noise_x)
        assert timestep.numel() == batch_size
        assert [plan.mode for plan in sequence_plans] == ["wam"] * batch_size
        self.velocity_batch_sizes.append(batch_size)
        state = torch.stack(noise_x)
        visions = state[:, :4].reshape(batch_size, 2, 2)
        actions = state[:, 4:].reshape_as(torch.stack(gen_data_clean.x0_tokens_action))
        assert torch.count_nonzero(actions[:, :, 10:]) == 0
        native_actions = actions[:, :, :10]
        predictions = net(native_actions.to(net.weight.dtype))
        velocities = []
        for prediction, vision, action, tokens in zip(
            predictions, visions, native_actions, text_tokens, strict=True
        ):
            velocity = prediction + vision[:, 1].mean() * 0.3 + tokens[0] * 0.03
            padded = torch.nn.functional.pad(velocity, (0, 2))
            video_velocity = torch.zeros_like(vision)
            video_velocity[:, 1] = vision[:, 1] + action.mean()
            velocities.append(torch.cat([video_velocity.flatten(), padded.flatten()]))
        return velocities

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
            assert value["video_num_frames"] == value["video"].shape[1]
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
def prefix_policy(native_sdk):
    omni = TinyOmni()
    omni.tokenizer_vision_gen = SimpleNamespace(temporal_compression_factor=4)
    return Cosmos3WoVRPolicy(
        omni, action_horizon=16, num_action_chunks=5, num_inference_steps=3
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


@pytest.fixture
def distinct_observations(image):
    obs = observations(image, batch_size=5)
    for row, frame in enumerate(obs["main_images"]):
        frame.fill_(row * 10)
        obs["task_descriptions"][row] = f"move object {row}"
    return obs


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
        augment_prompt(task) for task in obs["task_descriptions"]
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
    videos = [video for batch in policy.omni.batches for [video] in batch["video"]]
    for video, canvas in zip(videos, expected, strict=True):
        torch.testing.assert_close(video[:, 0], canvas.permute(2, 0, 1))

    main.zero_()
    wrist.zero_()
    replay = policy(forward_inputs=inputs)
    torch.testing.assert_close(replay["logprobs"], result["prev_logprobs"])
    torch.testing.assert_close(inputs["image"], expected)

    obs["wrist_images"] = wrist[:1]
    with pytest.raises(ValueError, match="matching uint8 LIBERO cameras"):
        policy.predict_action_batch(obs)


def test_execution_prefix_matches_full_plan_and_replays(prefix_policy, image):
    full_policy = Cosmos3WoVRPolicy(
        prefix_policy.omni,
        action_horizon=16,
        num_action_chunks=16,
        num_inference_steps=3,
    )
    obs = observations(image, batch_size=1)
    torch.manual_seed(41)
    full_actions, full_record = full_policy.predict_action_batch(obs)
    torch.manual_seed(41)
    actions, record = prefix_policy.predict_action_batch(obs)

    assert full_actions.shape == (1, 16, 10)
    assert prefix_policy.omni.batches[-1]["sequence_plan"][0].video_length == 17
    torch.testing.assert_close(actions, full_actions[:, :5])
    torch.testing.assert_close(record["forward_inputs"]["action"], actions.flatten(1))
    torch.testing.assert_close(
        record["prev_logprobs"], full_record["prev_logprobs"][:, :5]
    )
    torch.testing.assert_close(
        record["forward_inputs"]["state_before"],
        full_record["forward_inputs"]["state_before"],
    )
    replay = prefix_policy(
        forward_inputs=record["forward_inputs"], compute_entropy=True
    )
    torch.testing.assert_close(replay["logprobs"], record["prev_logprobs"])
    assert replay["entropy"].shape == (1, 5, 10)


@pytest.mark.parametrize("guidance_scale", [1.0, 1.5])
def test_actor_replays_mixed_timesteps_in_one_batch(
    prefix_policy, distinct_observations, monkeypatch, guidance_scale
):
    prefix_policy.guidance_scale = guidance_scale
    calls = []
    original_velocity = prefix_policy.omni._get_velocity
    train_steps = (
        prefix_policy.omni.config.rectified_flow_inference_config.num_train_timesteps
    )

    def time_dependent_velocity(*, timestep, **kwargs):
        calls.append(timestep.flatten().clone())
        return [
            velocity * (1 + t / train_steps)
            for velocity, t in zip(
                original_velocity(timestep=timestep, **kwargs),
                timestep.flatten(),
                strict=True,
            )
        ]

    monkeypatch.setattr(prefix_policy.omni, "_get_velocity", time_dependent_velocity)
    torch.manual_seed(1140)
    _, rollout = prefix_policy.predict_action_batch(distinct_observations)
    order = torch.tensor([4, 2, 0, 3, 1])
    inputs = {key: value[order] for key, value in rollout["forward_inputs"].items()}
    steps = inputs["denoise_index"]
    assert steps.unique().numel() == prefix_policy.num_inference_steps
    calls.clear()
    # Replay with updated weights, as happens after the first optimizer step.
    with torch.no_grad():
        prefix_policy.omni.net.bias.add_(0.02)
    batched = prefix_policy(forward_inputs=inputs, compute_entropy=True)
    assert len(calls) == (1 if guidance_scale == 1 else 2)
    for times in calls:
        torch.testing.assert_close(
            times, prefix_policy._schedule()[steps] * train_steps
        )
    assert not torch.allclose(batched["logprobs"], rollout["prev_logprobs"][order])
    calls.clear()
    entropy_only = prefix_policy(
        forward_inputs=inputs, compute_entropy=True, compute_logprobs=False
    )
    assert not calls
    torch.testing.assert_close(entropy_only, {"entropy": batched["entropy"]})
    parameters = tuple(prefix_policy.omni.net.parameters())
    gradients = torch.autograd.grad(batched["logprobs"].mean(), parameters)

    singletons = [
        prefix_policy(
            forward_inputs={key: value[i : i + 1] for key, value in inputs.items()},
            compute_entropy=True,
        )
        for i in range(len(order))
    ]
    for key in ("logprobs", "entropy"):
        torch.testing.assert_close(
            batched[key], torch.cat([row[key] for row in singletons])
        )
    reference = torch.cat([row["logprobs"] for row in singletons]).mean()
    for gradient, expected in zip(
        gradients, torch.autograd.grad(reference, parameters), strict=True
    ):
        assert gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize(
    ("rollout_batch_size", "guidance_scale", "return_full_horizon"),
    [(2, 1.0, False), (4, 1.5, True)],
)
def test_batched_rollout_matches_serial_and_replays(
    prefix_policy,
    distinct_observations,
    rollout_batch_size,
    guidance_scale,
    return_full_horizon,
):
    prefix_policy.guidance_scale = guidance_scale
    prefix_policy.rollout_batch_size = 1
    torch.manual_seed(1140)
    serial_actions, serial = prefix_policy.predict_action_batch(
        distinct_observations, return_full_horizon=return_full_horizon
    )

    prefix_policy.rollout_batch_size = rollout_batch_size
    prefix_policy.omni.velocity_batch_sizes.clear()
    torch.manual_seed(1140)
    actions, batched = prefix_policy.predict_action_batch(
        distinct_observations, return_full_horizon=return_full_horizon
    )
    inputs = batched["forward_inputs"]
    branches = 1 if guidance_scale == 1 else 2
    assert prefix_policy.omni.velocity_batch_sizes == [
        len(indices)
        for indices in inputs["denoise_index"].split(rollout_batch_size)
        for _ in range(prefix_policy.num_inference_steps * branches)
    ]
    torch.testing.assert_close(actions, serial_actions)
    torch.testing.assert_close(batched["prev_logprobs"], serial["prev_logprobs"])
    assert inputs["seed"].unique().numel() == 5
    assert inputs["denoise_index"].unique().numel() == prefix_policy.num_inference_steps
    torch.testing.assert_close(inputs, serial["forward_inputs"])
    replay = prefix_policy(forward_inputs=inputs)
    torch.testing.assert_close(replay["logprobs"], batched["prev_logprobs"])


def test_singleton_sdk_padding_contributes_only_the_real_row(
    prefix_policy, image, monkeypatch
):
    prefix_policy.guidance_scale = 1
    _, rollout = prefix_policy.predict_action_batch(observations(image, batch_size=1))
    inputs = rollout["forward_inputs"]
    params = tuple(prefix_policy.omni.net.parameters())
    baseline = prefix_policy(forward_inputs=inputs)["logprobs"]
    expected_grads = torch.autograd.grad(baseline.sum(), params)
    sdk_velocity = prefix_policy.omni._get_velocity
    padding = torch.tensor(9.0, requires_grad=True)

    def duplicate(clean):
        assert len(clean.x0_tokens_action) == 1
        return SimpleNamespace(x0_tokens_action=clean.x0_tokens_action * 2)

    def padded_velocity(**kwargs):
        real, extra = sdk_velocity(**kwargs)
        torch.testing.assert_close(kwargs["noise_x"][0], kwargs["noise_x"][1])
        return [real, extra * padding]

    monkeypatch.setattr(
        prefix_policy.omni, "_duplicate_gen_data_clean", duplicate, raising=False
    )
    monkeypatch.setattr(prefix_policy.omni, "_get_velocity", padded_velocity)
    actual = prefix_policy(forward_inputs=inputs)["logprobs"]
    torch.testing.assert_close(actual, baseline)
    torch.testing.assert_close(actual, rollout["prev_logprobs"])
    *param_grads, padding_grad = torch.autograd.grad(
        actual.sum(), (*params, padding), allow_unused=True
    )
    torch.testing.assert_close(param_grads, list(expected_grads))
    assert padding_grad is None


def test_wam_denoises_future_video_and_replays_its_selected_state(
    policy, image, monkeypatch
):
    policy.guidance_scale = 1
    states = []
    get_velocity = policy.omni._get_velocity

    def record_velocity(**kwargs):
        states.append(kwargs["noise_x"][0].detach().clone())
        return get_velocity(**kwargs)

    monkeypatch.setattr(policy.omni, "_get_velocity", record_velocity)
    _, rollout = policy.predict_action_batch(observations(image, batch_size=1))
    assert len(states) == policy.num_inference_steps
    vision = torch.stack(states)[:, :4].reshape(-1, 2, 2)
    observed = vision[:, :, 0]
    torch.testing.assert_close(observed, observed[0].expand_as(observed))
    assert not torch.equal(vision[0, :, 1], vision[-1, :, 1])

    inputs = rollout["forward_inputs"]
    replay = policy(forward_inputs=inputs)
    torch.testing.assert_close(states[-1], states[inputs["denoise_index"].item()])
    torch.testing.assert_close(replay["logprobs"], rollout["prev_logprobs"])
    # Changing only the saved future video must change action likelihoods.
    inputs["state_before"][0, 1] += 1
    changed = policy(forward_inputs=inputs)
    assert not torch.allclose(changed["logprobs"], rollout["prev_logprobs"])


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
    "field,value",
    [
        ("noise_std", 0),
        ("shift", -1),
        ("num_inference_steps", 0),
        ("rollout_batch_size", 0),
    ],
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
    with pytest.raises(ValueError, match="action_horizon"):
        Cosmos3WoVRPolicy(omni, action_horizon=4, num_action_chunks=5)
    with pytest.raises(ValueError, match="temporal factor 4"):
        Cosmos3WoVRPolicy(omni, action_horizon=5, num_action_chunks=4)
    policy = Cosmos3WoVRPolicy(omni, num_action_chunks=4)
    with pytest.raises(ValueError, match="multiples of 32"):
        policy.predict_action_batch(observations(image))


@pytest.fixture
def native_loader(monkeypatch, tmp_path):
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
    return loaded, calls, contexts


def test_loader_preserves_weights_and_selects_trainable_heads(
    native_loader, tmp_path, native_sdk, image
):
    from omegaconf import OmegaConf

    from rlinf.models import get_model
    from rlinf.scheduler import Worker

    loaded, calls, contexts = native_loader
    model_cfg = OmegaConf.create(
        {
            "model_type": "cosmos3",
            "precision": "bf16",
            "is_lora": False,
            "load_to_device": False,
            "wovr": {"enabled": True},
            "model_path": str(tmp_path),
            "keys_to_select": ["weight"],
            "train_dit_backbone": False,
            "rollout_batch_size": 3,
        }
    )
    policy = get_model(model_cfg)
    assert policy.rollout_batch_size == 3
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
    with pytest.raises(ValueError, match="DiT"):
        Cosmos3WoVRPolicy.from_pretrained({**model_cfg, "train_dit_backbone": True})
    with pytest.raises(ValueError, match="No Cosmos network parameters"):
        Cosmos3WoVRPolicy.from_pretrained(
            {
                "model_path": str(tmp_path),
                "device": Worker.torch_device_type,
                "keys_to_select": ["missing"],
                "train_dit_backbone": False,
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
        Cosmos3WoVRPolicy.from_pretrained(model_cfg)
    loaded.config.action_gen = False
    with pytest.raises(ValueError, match="action-conditioned simulation"):
        Cosmos3WoVRPolicy.from_pretrained(
            {"model_path": str(tmp_path)}, for_simulation=True
        )
    (tmp_path / "hf_quant_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unquantized"):
        Cosmos3WoVRPolicy.from_pretrained({"model_path": str(tmp_path)})


@pytest.fixture
def native_parameter_groups(native_loader):
    omni, _, _ = native_loader
    net = omni.net
    net.action2llm = nn.Linear(2, 4)
    net.llm2action = nn.Linear(4, 2)
    net.action_modality_embed = nn.Parameter(torch.zeros(4))
    net.time_embedder = nn.Linear(1, 4)
    net.vae2llm = nn.Linear(2, 4)
    net.llm2vae = nn.Linear(4, 2)
    attention = nn.ModuleDict(
        {
            "q_proj": nn.Linear(4, 4),
            "q_proj_moe_gen": nn.Linear(4, 4),
            "k_norm_und_for_gen": nn.LayerNorm(4, bias=False),
        }
    )
    net.language_model = nn.Module()
    net.language_model.model = nn.Module()
    net.language_model.model.layers = nn.ModuleList(
        [nn.ModuleDict({"self_attn": attention})]
    )
    omni.tokenizer_vision_gen = nn.Linear(2, 2)
    layer = "net.language_model.model.layers.0.self_attn"
    return {
        "dit": {
            f"{layer}.q_proj_moe_gen.weight",
            f"{layer}.q_proj_moe_gen.bias",
            f"{layer}.k_norm_und_for_gen.weight",
        },
        "input": {"net.action2llm.weight", "net.action2llm.bias"},
        "output": {"net.llm2action.weight", "net.llm2action.bias"},
        "modality": {"net.action_modality_embed"},
        "time": {"net.time_embedder.weight", "net.time_embedder.bias"},
    }


@pytest.mark.parametrize(
    ("overrides", "expected_groups"),
    [
        ({}, ["dit"]),
        ({"train_action_input_projection": True}, ["dit", "input"]),
        ({"train_action_output_projection": True}, ["dit", "output"]),
        ({"train_action_modality_embedding": True}, ["dit", "modality"]),
        ({"train_time_embedder": True}, ["dit", "time"]),
        (
            {
                "train_dit_backbone": False,
                "train_action_input_projection": True,
                "train_action_output_projection": True,
                "train_action_modality_embedding": True,
            },
            ["input", "output", "modality"],
        ),
        (
            {
                "keys_to_select": ["action2llm", "llm2action", "moe_gen"],
                "train_dit_backbone": False,
                "train_action_input_projection": False,
            },
            ["output"],
        ),
    ],
)
def test_loader_selects_only_requested_policy_parameters(
    native_parameter_groups, tmp_path, overrides, expected_groups
):
    policy = Cosmos3WoVRPolicy.from_pretrained(
        {"model_path": str(tmp_path), **overrides}
    )
    expected = set().union(*(native_parameter_groups[name] for name in expected_groups))
    trainable = {name for name, p in policy.omni.named_parameters() if p.requires_grad}
    assert trainable == expected


def test_loader_rejects_empty_actor_selection_but_freezes_simulator(
    native_parameter_groups, tmp_path
):
    cfg = {"model_path": str(tmp_path), "train_dit_backbone": False}
    with pytest.raises(ValueError, match="No trainable Cosmos parameters"):
        Cosmos3WoVRPolicy.from_pretrained(cfg)
    with pytest.raises(ValueError, match="keys_to_select"):
        Cosmos3WoVRPolicy.from_pretrained(
            {"model_path": str(tmp_path), "keys_to_select": ["missing"]}
        )
    simulator = Cosmos3WoVRPolicy.from_pretrained(
        {**cfg, "train_dit_backbone": True, "train_action_output_projection": True},
        for_simulation=True,
    )
    assert all(not p.requires_grad for p in simulator.parameters())


def test_native_time_embedder_keeps_scalar_values_and_trainable_parameters(
    native_parameter_groups, native_loader, tmp_path, monkeypatch
):
    omni, _, _ = native_loader
    embedder = omni.net.time_embedder
    original_parameters = dict(omni.named_parameters())
    linear = embedder.forward
    calls = []

    def native_embed(times):
        calls.append(times.tolist())
        return linear(times[:, None])

    monkeypatch.setattr(embedder, "forward", native_embed)
    policy = Cosmos3WoVRPolicy.from_pretrained(
        {
            "model_path": str(tmp_path),
            "train_dit_backbone": False,
            "train_time_embedder": True,
        }
    )
    parameters = dict(policy.omni.named_parameters())
    assert parameters.keys() == original_parameters.keys()
    assert all(param is original_parameters[name] for name, param in parameters.items())
    assert {
        name for name, param in parameters.items() if param.requires_grad
    } == native_parameter_groups["time"]

    times = torch.tensor([850.0, 245.0, 850.0, 0.0, 245.0])
    actual = policy.omni.net.time_embedder(times)
    assert sorted(calls) == [[0.0], [245.0], [850.0]]
    expected = torch.cat([linear(time.reshape(1, 1)) for time in times])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual)
    grads = torch.autograd.grad((actual * weights).sum(), tuple(embedder.parameters()))
    reference = torch.autograd.grad(
        (expected * weights).sum(), tuple(embedder.parameters())
    )
    torch.testing.assert_close(grads, reference)
    assert all(torch.count_nonzero(grad) for grad in grads)


def test_loader_rejects_missing_requested_action_projection(
    native_parameter_groups, native_loader, tmp_path
):
    omni, _, _ = native_loader
    del omni.net.llm2action
    with pytest.raises(
        ValueError, match="Missing requested Cosmos modules.*llm2action"
    ):
        Cosmos3WoVRPolicy.from_pretrained(
            {"model_path": str(tmp_path), "train_action_output_projection": True}
        )
