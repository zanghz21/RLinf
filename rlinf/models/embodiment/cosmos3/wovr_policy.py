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

"""Native Cosmos action diffusion with replayable Gaussian transition densities."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class Cosmos3WoVRPolicy(nn.Module, BasePolicy):
    """Train a Cosmos action denoiser and optionally use it for forward dynamics.

    The policy is a Gaussian Markov chain around Euler flow updates. Every
    denoising transition, including the last, has positive variance. Replaying
    stored states therefore produces actual transition likelihoods for PPO.
    Only action tokens are sampled; the observation is fully conditioned.
    Model evaluation mode is used for both sampling and gradient recomputation
    so dropout does not change the behavior distribution.
    """

    _no_split_modules = ["MoTDecoderLayer", "PackedAttentionMoT", "Cosmos3VFMNetwork"]
    # DomainAwareLinear accesses its embedding weights without calling forward.
    _fsdp_wrap_embeddings = False

    def __init__(
        self,
        omni: nn.Module,
        *,
        num_action_chunks: int = 16,
        num_inference_steps: int = 30,
        raw_action_dim: int = 10,
        fps: int = 20,
        action_fps: float = 20.0,
        guidance_scale: float = 1.5,
        noise_std: float = 0.1,
        shift: float = 5.0,
        domain_name: str = "libero",
        action_view_point: str = "ego_view",
        image_size: tuple[int, int] = (192, 320),
        max_prompt_bytes: int = 4096,
        for_simulation: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_action_chunks", num_action_chunks),
            ("num_inference_steps", num_inference_steps),
            ("fps", fps),
            ("max_prompt_bytes", max_prompt_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name, value in (
            ("noise_std", noise_std),
            ("shift", shift),
            ("action_fps", action_fps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if domain_name != "libero" or raw_action_dim != 10:
            raise ValueError("Cosmos WoVR requires native LIBERO 10D actions")
        if int(omni.config.max_action_dim) < raw_action_dim:
            raise ValueError("Checkpoint action width is smaller than raw_action_dim")
        if not omni.config.action_gen:
            raise ValueError(
                "Checkpoint must enable action_gen: native Cosmos needs its action encoder "
                "for both policy generation and action-conditioned simulation"
            )
        if for_simulation and not getattr(omni.config, "vision_gen", True):
            raise ValueError("Simulator checkpoint must enable vision_gen")
        if not for_simulation and getattr(omni.config, "video_temporal_causal", False):
            raise ValueError("Cosmos WoVR requires a bidirectional action-policy model")
        temporal_factor = int(
            getattr(
                getattr(omni, "tokenizer_vision_gen", None),
                "temporal_compression_factor",
                1,
            )
        )
        if temporal_factor <= 0 or num_action_chunks % temporal_factor:
            raise ValueError(
                f"num_action_chunks must be divisible by the VAE temporal factor {temporal_factor}"
            )
        spatial_factor = int(
            getattr(
                getattr(omni, "tokenizer_vision_gen", None),
                "spatial_compression_factor",
                16,
            )
        )
        patch_size = int(
            getattr(
                getattr(omni.config, "diffusion_expert_config", None),
                "patch_spatial",
                1,
            )
        )
        if spatial_factor <= 0 or patch_size <= 0:
            raise ValueError(
                "Checkpoint VAE spatial factor and patch size must be positive"
            )
        self.spatial_multiple = spatial_factor * patch_size
        self.omni = omni
        self.num_action_chunks = num_action_chunks
        self.num_inference_steps = num_inference_steps
        self.raw_action_dim = raw_action_dim
        self.fps = fps
        self.action_fps = action_fps
        self.guidance_scale = guidance_scale
        self.noise_std = noise_std
        self.shift = shift
        self.domain_name = domain_name
        self.action_view_point = action_view_point
        self.image_size = tuple(image_size)
        if len(self.image_size) != 2 or any(
            type(dim) is not int or dim <= 0 for dim in self.image_size
        ):
            raise ValueError("Expected positive Cosmos image_size [height,width]")
        self.max_prompt_bytes = max_prompt_bytes
        self.for_simulation = for_simulation
        if for_simulation:
            self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        """Device on which the Cosmos network is materialized."""
        return next(self.omni.net.parameters()).device

    def _device_context(self):
        # Native Cosmos allocates some packed metadata on the current device.
        if self.device.type == "cpu":
            return contextlib.nullcontext()
        from rlinf.scheduler import Worker

        return Worker.torch_platform.device(self.device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Native inference allocates from these dictionaries after offload/onload.
        for module in self.omni.modules():
            for name in ("tensor_kwargs", "tensor_kwargs_fp32"):
                options = getattr(module, name, None)
                if isinstance(options, dict):
                    options["device"] = self.device
        # The Wan tokenizer is an ABC wrapper, outside nn.Module's recursion.
        tokenizer = getattr(self.omni, "tokenizer_vision_gen", None)
        vae = getattr(tokenizer, "model", None)
        vae_network = getattr(vae, "model", None)
        if not isinstance(tokenizer, nn.Module) and isinstance(vae_network, nn.Module):
            vae_network.to(device=self.device)
            vae.device = self.device
            vae.scale = tuple(value.to(self.device) for value in vae.scale)
            decoder = getattr(tokenizer, "_decoder_override", None)
            if isinstance(decoder, nn.Module):
                decoder.to(device=self.device)
        return result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        raise NotImplementedError(
            "Cosmos WoVR currently requires actor.fsdp_config.gradient_checkpointing=false"
        )

    @classmethod
    def from_pretrained(
        cls, model_cfg: Mapping[str, Any], *, for_simulation: bool = False
    ) -> Cosmos3WoVRPolicy:
        """Load native exported Cosmos weights without resetting action heads.

        ``model_path`` is a local native/HF or diffusers checkpoint directory.
        It must include Cosmos's native ``config.json``; ``native_config_path``
        can point at that file separately for diffusers-only weight exports.
        ``keys_to_select`` chooses trainable network parameter substrings.
        ``for_simulation`` freezes all weights and also accepts causal dynamics
        models; native action conditioning still requires ``action_gen=true``.
        """
        from cosmos_framework.configs.base.defaults.compile import CompileConfig
        from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
        from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

        from rlinf.scheduler import Worker

        checkpoint = Path(str(model_cfg["model_path"])).expanduser().resolve()
        config_path = (
            Path(str(model_cfg.get("native_config_path") or checkpoint / "config.json"))
            .expanduser()
            .resolve()
        )
        if not checkpoint.is_dir() or not config_path.is_file():
            raise FileNotFoundError(
                f"Expected checkpoint directory {checkpoint} and native config {config_path}"
            )
        if (checkpoint / "hf_quant_config.json").exists():
            raise ValueError("Cosmos WoVR requires unquantized trainable weights")
        config = Cosmos3OmniConfig.from_json_file(str(config_path))
        if not config.model:
            raise ValueError(f"Expected a native Cosmos model config in {config_path}")
        keys = (
            []
            if for_simulation
            else list(
                model_cfg.get(
                    "keys_to_select",
                    ["action2llm", "llm2action", "action_modality_embed"],
                )
            )
        )
        if not for_simulation and (
            not keys or any(not isinstance(key, str) or not key for key in keys)
        ):
            raise ValueError(
                "keys_to_select must contain nonempty parameter substrings"
            )
        requested_device = model_cfg.get("device")
        device = (
            torch.device(requested_device)
            if requested_device is not None
            else torch.device(
                Worker.torch_device_type, Worker.torch_platform.current_device()
            )
        )
        if device.type != Worker.torch_device_type:
            raise ValueError(
                f"Requested device {device} differs from accelerator {Worker.torch_device_type}"
            )
        with Worker.torch_platform.device(device):
            loaded = Cosmos3OmniModel.from_pretrained_dcp(
                checkpoint,
                config=config,
                parallelism_config=ParallelismConfig(
                    enable_inference_mode=True,
                    data_parallel_shard_degree=1,
                    data_parallel_replicate_degree=-1,
                    context_parallel_shard_degree=1,
                    cfg_parallel_shard_degree=1,
                ),
                compile_config=CompileConfig(enabled=False, use_cuda_graphs=False),
            )
        omni = loaded.model
        from rlinf.models.embodiment.cosmos3 import _promote_scalar_params_to_1d

        _promote_scalar_params_to_1d(omni)
        omni.requires_grad_(False)
        selected = []
        for name, parameter in omni.net.named_parameters():
            if any(key in name for key in keys):
                parameter.requires_grad_(True)
                selected.append(name)
        if not for_simulation and not selected:
            raise ValueError(
                f"No Cosmos network parameters match keys_to_select={keys}"
            )
        options = {
            key: model_cfg[key]
            for key in (
                "num_action_chunks",
                "num_inference_steps",
                "raw_action_dim",
                "fps",
                "action_fps",
                "guidance_scale",
                "noise_std",
                "shift",
                "domain_name",
                "action_view_point",
                "image_size",
                "max_prompt_bytes",
            )
            if key in model_cfg
        }
        return cls(omni, for_simulation=for_simulation, **options)

    def _build_batch(
        self, image: np.ndarray, prompt: str, actions: torch.Tensor | None = None
    ) -> dict[str, Any]:
        from cosmos_framework.data.generator.action.utils.action_processing import (
            ActionProcessingRecord,
            make_batched_action_processing_fields,
            pad_action_to_max_dim,
        )
        from cosmos_framework.data.generator.action.utils.domain_utils import (
            get_domain_id,
        )
        from cosmos_framework.data.generator.action.utils.json_formatter import (
            ActionPromptJsonFormatter,
        )
        from cosmos_framework.data.generator.action.utils.transforms import (
            build_sequence_plan_from_mode,
        )

        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected an RGB uint8 observation [height,width,3]")
        height, width = image.shape[:2]
        if (
            min(height, width) <= 0
            or height % self.spatial_multiple
            or width % self.spatial_multiple
        ):
            raise ValueError(
                f"Cosmos canvas dimensions must be positive multiples of {self.spatial_multiple}"
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Expected a nonempty task prompt")
        dynamics = actions is not None
        mode = "forward_dynamics" if dynamics else "policy"
        frames = self.num_action_chunks + 1 if dynamics else 1
        video = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(1)
        video = video.repeat(1, frames, 1, 1)
        if actions is None:
            actions = torch.zeros(self.num_action_chunks, self.raw_action_dim)
        actions = torch.as_tensor(actions).detach().float().cpu()
        if actions.shape != (self.num_action_chunks, self.raw_action_dim):
            raise ValueError(
                f"Expected actions {(self.num_action_chunks, self.raw_action_dim)}, got {actions.shape}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite")
        padded_actions = pad_action_to_max_dim(actions, self.omni.config.max_action_dim)
        size = torch.tensor([height, width, height, width], dtype=torch.long)
        formatter = ActionPromptJsonFormatter()
        caption = formatter(
            {
                "viewpoint": self.action_view_point,
                "ai_caption": prompt,
                "video": video,
                "video_num_frames": self.num_action_chunks + 1,
                "action": padded_actions,
                "conditioning_fps": torch.tensor(self.fps),
                "image_size": size,
            }
        )[formatter.caption_key]
        record = ActionProcessingRecord(
            raw_action_dim=self.raw_action_dim, action_normalizer=None
        )
        return {
            self.omni.input_video_key: [[video]],
            self.omni.input_caption_key: [
                json.dumps(caption) if isinstance(caption, dict) else caption
            ],
            "action": [[padded_actions]],
            **make_batched_action_processing_fields(record, 1),
            "mode": [mode],
            "prompt": [prompt],
            "conditioning_fps": [torch.tensor(self.fps)],
            "conditioning_fps_action": [torch.tensor(self.action_fps)],
            "image_size": size.unsqueeze(0).to(self.device),
            "domain_id": [torch.tensor(get_domain_id(self.domain_name))],
            "sequence_plan": [
                build_sequence_plan_from_mode(
                    mode=mode,
                    video_length=frames,
                    action_length=self.num_action_chunks,
                    has_text=True,
                )
            ],
        }

    @torch.no_grad()
    def _prepare(self, image: np.ndarray, prompt: str, seed: int) -> tuple:
        if self.for_simulation:
            raise ValueError(
                "A simulator-only Cosmos model cannot sample or score policy actions"
            )
        with self._device_context():
            prepared = self.omni._prepare_inference_data(
                self._build_batch(image, prompt), seed=[seed], has_negative_prompt=False
            )
        plans, clean, _, _, noise, _, masks, has_actions = prepared
        if len(plans) != 1 or not has_actions or len(noise) != 1:
            raise ValueError("Expected a single action-policy sample from Cosmos")
        if clean.x0_tokens_action is None or len(clean.x0_tokens_action) != 1:
            raise ValueError("Cosmos policy sample is missing action tokens")
        shape = tuple(clean.x0_tokens_action[0].shape)
        if shape != (self.num_action_chunks, self.omni.config.max_action_dim):
            raise ValueError(f"Unexpected Cosmos action token shape {shape}")
        action_size = math.prod(shape)
        offset = noise[0].numel() - action_size
        if offset <= 0 or not torch.all(masks[0][:offset] == 1):
            raise ValueError("Cosmos WoVR policy must condition all vision tokens")
        if not torch.all(masks[0][offset:] == 0):
            raise ValueError("Cosmos WoVR policy must predict every action token")
        return prepared

    def _schedule(self) -> torch.Tensor:
        time = torch.linspace(1.0, 0.0, self.num_inference_steps + 1)
        return self.shift * time / (1 + (self.shift - 1) * time)

    def _velocity(
        self, prepared: tuple, action: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        plans, clean, tokens, uncond_tokens, initial, _, _, has_actions = prepared
        padded = torch.nn.functional.pad(
            action.to(self.device),
            (0, self.omni.config.max_action_dim - self.raw_action_dim),
        )
        offset = initial[0].numel() - padded.numel()
        state = torch.cat([initial[0][:offset], padded.flatten()])
        timestep = torch.tensor(
            [
                [
                    sigma
                    * self.omni.config.rectified_flow_inference_config.num_train_timesteps
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )

        def predict(text: list[list[int]]) -> torch.Tensor:
            with self._device_context():
                result = self.omni._get_velocity(
                    net=self.omni.net,
                    noise_x=[state],
                    timestep=timestep,
                    text_tokens=text,
                    sequence_plans=plans,
                    gen_data_clean=clean,
                    has_noisy_actions=has_actions,
                )[0]
            return result[offset:].reshape_as(padded)[:, : self.raw_action_dim].float()

        conditional = predict(tokens)
        if self.guidance_scale == 1:
            return conditional
        unconditional = predict(uncond_tokens)
        return unconditional + self.guidance_scale * (conditional - unconditional)

    @torch.no_grad()
    def _sample_transition(
        self, image: np.ndarray, prompt: str, seed: int, selected_step: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Generate native actions and retain one selected Gaussian transition."""
        self.eval()
        prepared = self._prepare(image, prompt, seed)
        padded_size = self.num_action_chunks * self.omni.config.max_action_dim
        action = (
            prepared[4][0][-padded_size:]
            .reshape(self.num_action_chunks, self.omni.config.max_action_dim)[
                :, : self.raw_action_dim
            ]
            .float()
        )
        # An independent stream avoids reusing the initial state's Gaussian draw.
        generator = torch.Generator(device=self.device).manual_seed(seed + 1)
        schedule = self._schedule().tolist()
        transition = {}
        for step, (sigma, next_sigma) in enumerate(
            zip(schedule[:-1], schedule[1:], strict=True)
        ):
            if step == selected_step:
                transition["state_before"] = action.cpu().clone()
            dt = sigma - next_sigma
            mean = action - dt * self._velocity(prepared, action, sigma)
            std = self.noise_std * math.sqrt(dt)
            action = mean + std * torch.randn(
                mean.shape, device=self.device, dtype=torch.float32, generator=generator
            )
            if not torch.isfinite(action).all():
                raise FloatingPointError("Cosmos policy generated nonfinite actions")
            if step == selected_step:
                transition["state_after"] = action.cpu().clone()
                transition["logprobs"] = (
                    torch.distributions.Normal(mean, std).log_prob(action).cpu()
                )
        return action.detach().cpu().contiguous(), transition

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Replay one stored Gaussian transition per sample for RLinf's RL loss.

        Log densities are [batch,horizon,10]. Standard ``action_level`` loss
        preprocessing sums channels and applies the environment's horizon mask.
        These are transition densities, not marginal final-action densities.
        """
        if compute_values:
            raise ValueError("Cosmos WoVR has no value head; use GRPO advantages")
        self.eval()
        required = {
            "image",
            "prompt_bytes",
            "prompt_length",
            "seed",
            "denoise_index",
            "state_before",
            "state_after",
            "action",
        }
        if required - forward_inputs.keys():
            raise ValueError(
                f"Missing Cosmos forward inputs: {required - forward_inputs.keys()}"
            )
        if any(not isinstance(forward_inputs[key], torch.Tensor) for key in required):
            raise TypeError("Cosmos forward inputs must contain only tensors")
        batch_size = forward_inputs["image"].shape[0]
        if batch_size == 0 or any(
            forward_inputs[key].shape[0] != batch_size for key in required
        ):
            raise ValueError(
                "Cosmos forward inputs must have the same nonempty batch axis"
            )
        expected = (batch_size, self.num_action_chunks, self.raw_action_dim)
        for key in ("state_before", "state_after"):
            if tuple(forward_inputs[key].shape) != expected:
                raise ValueError(
                    f"Expected {key} shape {expected}, got {forward_inputs[key].shape}"
                )
            if forward_inputs[key].dtype != torch.float32:
                raise ValueError(
                    "Denoising states must retain float32; disable FSDP cast_forward_inputs"
                )
        if forward_inputs["prompt_bytes"].shape != (batch_size, self.max_prompt_bytes):
            raise ValueError("Cosmos prompt byte width differs from max_prompt_bytes")
        schedule = self._schedule().tolist()
        logprobs, entropies = [], []
        for row in range(batch_size):
            step = int(forward_inputs["denoise_index"][row].item())
            if not 0 <= step < self.num_inference_steps:
                raise ValueError("denoise_index must address a stored transition")
            length = int(forward_inputs["prompt_length"][row].item())
            if not 0 < length <= self.max_prompt_bytes:
                raise ValueError("Invalid Cosmos prompt length")
            dt = schedule[step] - schedule[step + 1]
            std = self.noise_std * math.sqrt(dt)
            if compute_logprobs:
                prompt = bytes(
                    forward_inputs["prompt_bytes"][row, :length].cpu().tolist()
                ).decode("utf-8")
                prepared = self._prepare(
                    forward_inputs["image"][row].detach().cpu().numpy(),
                    prompt,
                    int(forward_inputs["seed"][row].item()),
                )
                before = forward_inputs["state_before"][row].detach().to(self.device)
                after = forward_inputs["state_after"][row].detach().to(self.device)
                mean = before - dt * self._velocity(prepared, before, schedule[step])
                logprobs.append(torch.distributions.Normal(mean, std).log_prob(after))
            if compute_entropy:
                entropies.append(
                    torch.full(
                        expected[1:],
                        float(torch.distributions.Normal(0.0, std).entropy()),
                        device=self.device,
                    )
                )
        result = {}
        if compute_logprobs:
            result["logprobs"] = torch.stack(logprobs)
        if compute_entropy:
            result["entropy"] = torch.stack(entropies)
        return result

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "train", **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions and tensor-only transition records for rollout workers.

        Uniformly selecting a denoising step gives an unbiased estimate of the
        mean transition-factor surrogate used by the standard clipped RL loss.
        Seeds and selected steps are carried to the actor for exact replay.
        """
        from rlinf.models.embodiment.cosmos3.closed_loop_io import (
            augment_prompt,
            prepare_libero_canvases,
        )

        if mode not in ("train", "eval"):
            raise ValueError(f"Expected train or eval mode, got {mode!r}")
        images = torch.as_tensor(env_obs["main_images"]).detach().cpu()
        tasks = env_obs["task_descriptions"]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != torch.uint8:
            raise ValueError("Expected main_images uint8 [batch,height,width,3]")
        batch_size = images.shape[0]
        if batch_size == 0 or len(tasks) != batch_size:
            raise ValueError(
                "Expected one task description for each nonempty observation"
            )
        # WorldModelEnv already supplies a composed canvas. LiberoEnv supplies
        # separate cameras; save their composed pixels for exact actor replay.
        if env_obs.get("wrist_images") is not None:
            wrist = torch.as_tensor(env_obs["wrist_images"]).detach().cpu()
            height, width = self.image_size
            images = torch.from_numpy(
                prepare_libero_canvases(images.numpy(), wrist.numpy(), width, height)
            )
        prompts = [augment_prompt(task) for task in tasks]
        encoded = [prompt.encode("utf-8") for prompt in prompts]
        if any(len(value) > self.max_prompt_bytes for value in encoded):
            raise ValueError(
                "Task prompt exceeds max_prompt_bytes; increase it on actor and rollout"
            )
        prompt_bytes = torch.zeros(
            (batch_size, self.max_prompt_bytes), dtype=torch.uint8
        )
        prompt_length = torch.tensor(
            [len(value) for value in encoded], dtype=torch.long
        )
        for row, value in enumerate(encoded):
            prompt_bytes[row, : len(value)] = torch.tensor(
                list(value), dtype=torch.uint8
            )
        seeds = torch.randint(0, 2**31 - 1, (batch_size,), dtype=torch.long)
        indices = torch.randint(
            self.num_inference_steps, (batch_size,), dtype=torch.long
        )
        actions, before, after, logprobs = [], [], [], []
        for row in range(batch_size):
            step = int(indices[row])
            action, transition = self._sample_transition(
                images[row].numpy(), prompts[row], int(seeds[row]), step
            )
            actions.append(action)
            before.append(transition["state_before"])
            after.append(transition["state_after"])
            logprobs.append(transition["logprobs"])
        actions = torch.stack(actions)
        return actions, {
            "prev_logprobs": torch.stack(logprobs),
            "prev_values": None,
            "forward_inputs": {
                "image": images.clone(),
                "prompt_bytes": prompt_bytes,
                "prompt_length": prompt_length,
                "seed": seeds,
                "denoise_index": indices,
                "state_before": torch.stack(before),
                "state_after": torch.stack(after),
                "action": actions.flatten(1),
            },
        }

    @torch.no_grad()
    def generate_dynamics(
        self,
        image: np.ndarray,
        prompt: str,
        actions: torch.Tensor | np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Generate [horizon+1,height,width,3] uint8 frames conditioned on actions."""
        self.eval()
        batch = self._build_batch(image, prompt, torch.as_tensor(actions))
        with self._device_context():
            generated = self.omni.generate_samples_from_batch(
                batch,
                net=self.omni.net,
                guidance=self.guidance_scale,
                seed=[seed],
                num_steps=self.num_inference_steps,
                shift=self.shift,
            )
        latents = generated.get("vision", [])
        if len(latents) != 1:
            raise ValueError("Expected one generated Cosmos video latent")
        latent = latents[0]
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        with self._device_context():
            video = self.omni.decode(latent).float()
        expected = (1, 3, self.num_action_chunks + 1, *image.shape[:2])
        if tuple(video.shape) != expected or not torch.isfinite(video).all():
            raise ValueError(
                f"Expected finite decoded video {expected}, got {video.shape}"
            )
        return (
            ((video[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
