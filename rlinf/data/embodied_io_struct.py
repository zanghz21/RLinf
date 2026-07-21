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

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    pass

from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)


def get_model_weights_id(versions: torch.Tensor) -> str:
    """
    Get the model weights id from the tensor.

    Args:
        versions (torch.Tensor): The tensor to get the model weights id from.

    Returns:
        str: The model weights id.
    """

    name_bytes = versions.cpu().numpy().tobytes()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_bytes.hex()))


@dataclass(kw_only=True)
class EnvOutput:
    """Environment output for a single chunk step."""

    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]
    env_infos: Optional[dict[str, Any]] = None

    intervene_actions: Optional[torch.Tensor] = None  # [B]
    intervene_flags: Optional[torch.Tensor] = None  # [B]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        self.env_infos = (
            put_tensor_device(self.env_infos, "cpu")
            if self.env_infos is not None
            else None
        )
        self.intervene_actions = (
            self.intervene_actions.cpu().contiguous()
            if self.intervene_actions is not None
            else None
        )
        self.intervene_flags = (
            self.intervene_flags.cpu().contiguous()
            if self.intervene_flags is not None
            else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    @staticmethod
    def merge_env_outputs(env_outputs: list[dict]) -> dict[str, Any]:
        """Merge multiple env output dicts into one batch-aligned env output.

        Merge strategy:

        - Tensor fields: concatenate on batch dimension.
        - List fields: flatten in source order.
        - ``None`` fields: keep ``None``.
        - ``final_obs`` supports partial ``None`` across shards. For shards
            without ``final_obs``, use the corresponding ``obs`` as fallback to
            keep batch alignment.

        Args:
            env_outputs: Per-source env output dicts that share the same schema.

        Returns:
            A merged env output dict produced via ``EnvOutput(...).to_dict()``.
        """

        def _get_batch_size(env_output: dict[str, Any]) -> int:
            dones = env_output.get("dones")
            if isinstance(dones, torch.Tensor):
                return dones.shape[0]

            obs = env_output["obs"]
            for key in ("states", "main_images", "task_descriptions"):
                value = obs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
                if isinstance(value, list):
                    return len(value)
            raise ValueError("Cannot infer batch size from env output.")

        def _merge_obs_dicts(obs_dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged_obs = {}
            for key in obs_dicts[0].keys():
                obs_elements = [obs_dict[key] for obs_dict in obs_dicts]
                first_non_none = next(
                    (element for element in obs_elements if element is not None), None
                )
                if first_non_none is None:
                    merged_obs[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged_obs[key] = torch.cat(obs_elements, dim=0)
                elif isinstance(first_non_none, list):
                    merged_obs[key] = [
                        item for sublist in obs_elements for item in sublist
                    ]
                else:
                    merged_obs[key] = obs_elements
            return merged_obs

        def _merge_optional_tensor_field(
            field_name: str,
            *,
            allow_partial_none: bool = False,
            fill_value: float | bool = 0,
        ) -> torch.Tensor | None:
            values = [env_output[field_name] for env_output in env_outputs]
            if all(value is None for value in values):
                return None

            if any(value is None for value in values):
                if not allow_partial_none:
                    raise ValueError(
                        f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                    )

                ref_tensor = next(value for value in values if value is not None)
                filled_values = []
                for env_output, value in zip(env_outputs, values):
                    if value is None:
                        batch_size = _get_batch_size(env_output)
                        fill_shape = (batch_size, *ref_tensor.shape[1:])
                        filled_values.append(
                            torch.full(
                                fill_shape,
                                fill_value=fill_value,
                                dtype=ref_tensor.dtype,
                            )
                        )
                    else:
                        filled_values.append(value)
                values = filled_values

            return torch.cat(values, dim=0)

        merged_obs = _merge_obs_dicts([env_output["obs"] for env_output in env_outputs])

        merged_final_obs = None
        final_obs_list = [env_output["final_obs"] for env_output in env_outputs]
        if any(final_obs is not None for final_obs in final_obs_list):
            # Some shards may not have done episodes in this step, so their final_obs
            # is None. Use obs as fallback to keep merged batch shape aligned.
            final_obs_or_obs = [
                final_obs if final_obs is not None else env_output["obs"]
                for env_output, final_obs in zip(env_outputs, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        merged_dones = _merge_optional_tensor_field("dones")
        merged_terminations = _merge_optional_tensor_field("terminations")
        merged_truncations = _merge_optional_tensor_field("truncations")
        merged_rewards = _merge_optional_tensor_field("rewards")
        merged_intervene_actions = _merge_optional_tensor_field(
            "intervene_actions",
            allow_partial_none=True,
            fill_value=0.0,
        )
        merged_intervene_flags = _merge_optional_tensor_field(
            "intervene_flags",
            allow_partial_none=True,
            fill_value=False,
        )
        # turn to EnvOutput and turn to dict to call post init for tensor processing
        return EnvOutput(
            obs=merged_obs,
            final_obs=merged_final_obs,
            dones=merged_dones,
            terminations=merged_terminations,
            truncations=merged_truncations,
            rewards=merged_rewards,
            intervene_actions=merged_intervene_actions,
            intervene_flags=merged_intervene_flags,
        ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["env_infos"] = self.env_infos
        env_output_dict["intervene_actions"] = self.intervene_actions
        env_output_dict["intervene_flags"] = self.intervene_flags

        return env_output_dict


@dataclass(kw_only=True)
class RolloutResult:
    """Rollout result for a single chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]

    bootstrap_values: torch.Tensor = None  # [B, 1]
    save_flags: torch.Tensor = None  # [B, num_action_chunks]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.save_flags is not None:
            self.save_flags = self.save_flags.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()

    @staticmethod
    def merge_rollout_results(
        rollout_results: list["RolloutResult"],
    ) -> "RolloutResult":
        def _merge_optional_tensor(field_name: str) -> torch.Tensor | None:
            values = [
                getattr(rollout_result, field_name)
                for rollout_result in rollout_results
            ]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                )
            return torch.cat(values, dim=0)

        merged_actions = _merge_optional_tensor("actions")
        merged_prev_logprobs = _merge_optional_tensor("prev_logprobs")
        merged_prev_values = _merge_optional_tensor("prev_values")
        merged_bootstrap_values = _merge_optional_tensor("bootstrap_values")
        merged_save_flags = _merge_optional_tensor("save_flags")
        merged_versions = _merge_optional_tensor("versions")

        forward_inputs_list = [
            rollout_result.forward_inputs for rollout_result in rollout_results
        ]
        if all(not forward_inputs for forward_inputs in forward_inputs_list):
            merged_forward_inputs = {}
        else:
            merged_forward_inputs = cat_list_of_dict_tensor(forward_inputs_list)
        return RolloutResult(
            actions=merged_actions,
            prev_logprobs=merged_prev_logprobs,
            prev_values=merged_prev_values,
            bootstrap_values=merged_bootstrap_values,
            save_flags=merged_save_flags,
            forward_inputs=merged_forward_inputs,
            versions=merged_versions,
        )


@dataclass(kw_only=True)
class ChunkStepResult:
    """Model outputs, env outputs (without observations), and training forward inputs for a chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.dones is not None:
            self.dones = self.dones.cpu().contiguous()
        if self.terminations is not None:
            self.terminations = self.terminations.cpu().contiguous()
        if self.truncations is not None:
            self.truncations = self.truncations.cpu().contiguous()
        if self.rewards is not None:
            self.rewards = self.rewards.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()


@dataclass
class Trajectory:
    """
    trajectory contains multiple episodes.
    """

    max_episode_length: int = 0  # max episode length
    model_weights_id: str = ""  # str(uuid(versions))
    actions: torch.Tensor = None
    intervene_flags: torch.Tensor = None
    rewards: torch.Tensor = None
    terminations: torch.Tensor = None
    truncations: torch.Tensor = None
    dones: torch.Tensor = None
    prev_logprobs: torch.Tensor = None
    prev_values: torch.Tensor = None
    versions: torch.Tensor = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)

    curr_obs: dict[str, Any] = field(default_factory=dict)
    next_obs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _generate_field_mask(
        ref_tensor: torch.Tensor, mask: torch.Tensor, traj_len: int
    ) -> torch.Tensor:
        """
        Generate a mask for terminations/truncations/dones based on their original shape.
        """
        assert mask.dim() == 1, f"Expected 1D mask, got {mask.shape=}"
        if ref_tensor.shape[0] == traj_len:
            return mask
        elif ref_tensor.shape[0] > traj_len:
            extra = int(ref_tensor.shape[0] - traj_len)
            assert traj_len % extra == 0, (
                f"Trajectory length {traj_len} is not divisible by extra {extra} for terminations/truncations/dones"
            )
            epoch_len = traj_len // extra

            field_mask = torch.zeros(
                ref_tensor.shape[0], dtype=torch.bool, device=mask.device
            )
            original_indices = torch.arange(ref_tensor.shape[0], device=mask.device)
            epoch_idx = original_indices // (epoch_len + 1)
            step_idx = original_indices % (epoch_len + 1)

            # Keep the first position of each epoch (step_idx == 0)
            field_mask[step_idx == 0] = True

            # Map positions with step_idx >= 1 to mask
            valid_mask = step_idx >= 1
            mask_idx = epoch_idx[valid_mask] * epoch_len + (step_idx[valid_mask] - 1)
            valid_original_indices = original_indices[valid_mask]
            valid_mask_idx = mask_idx < len(mask)
            field_mask[valid_original_indices[valid_mask_idx]] = mask[
                mask_idx[valid_mask_idx]
            ].to(dtype=torch.bool)

            return field_mask
        else:
            raise ValueError(
                f"Reference tensor length {ref_tensor.shape[0]} < traj_len {traj_len}"
            )

    def extract_intervene_traj(self, mode="any"):
        if self.intervene_flags is None or (~self.intervene_flags).all():
            return None

        if mode == "any":
            mask = self.intervene_flags.any(dim=-1)
        elif mode == "all":
            mask = self.intervene_flags.all(dim=-1)
        else:
            raise NotImplementedError(
                f"Unsupported extract_intervene_traj mode: {mode}"
            )
        assert mask.dim() == 2, (
            f"Expected 2D mask after processing (traj len, bsz), got {mask.shape=}"
        )
        traj_len = int(mask.shape[0])

        def apply_mask(tensor, i):
            return tensor[:, i][mask[:, i]].unsqueeze(1) if tensor is not None else None

        def apply_mask_to_dict(d, i):
            return (
                {k: v[:, i][mask[:, i]].unsqueeze(1) for k, v in d.items()} if d else {}
            )

        filtered_trajectories = []
        for i in range(mask.shape[1]):
            if not mask[:, i].any():
                continue

            actions = apply_mask(self.actions, i)
            rewards = apply_mask(self.rewards, i)
            prev_logprobs = apply_mask(self.prev_logprobs, i)
            prev_values = apply_mask(self.prev_values, i)
            intervene_flags = apply_mask(self.intervene_flags, i)

            forward_inputs = apply_mask_to_dict(self.forward_inputs, i)
            curr_obs = apply_mask_to_dict(self.curr_obs, i)
            next_obs = apply_mask_to_dict(self.next_obs, i)

            terminations = truncations = dones = None
            if self.terminations is not None:
                field_mask = self._generate_field_mask(
                    self.terminations[:, i : i + 1], mask[:, i], traj_len
                )
                terminations = self.terminations[:, i : i + 1][field_mask]
                truncations = self.truncations[:, i : i + 1][field_mask]
                dones = self.dones[:, i : i + 1][field_mask]

            filtered_trajectories.append(
                Trajectory(
                    max_episode_length=self.max_episode_length,
                    model_weights_id=self.model_weights_id,
                    actions=actions,
                    intervene_flags=intervene_flags,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    dones=dones,
                    prev_logprobs=prev_logprobs,
                    prev_values=prev_values,
                    forward_inputs=forward_inputs,
                    curr_obs=curr_obs,
                    next_obs=next_obs,
                )
            )

        return filtered_trajectories if filtered_trajectories else None


def _find_batch_size(data: dict[str, Any]) -> int:
    """Infer the batch size (dim-0) from the first tensor found in a (possibly
    nested) dict whose values are batch-major tensors."""
    for value in data.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        if isinstance(value, dict):
            bsz = _find_batch_size(value)
            if bsz is not None:
                return bsz
    return None


def _unbind_dict_along_batch(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a batch-major dict ``{key: [B, ...]}`` into ``B`` per-sample dicts
    ``{key: [...]}`` (the batch dim is stripped). Nested dicts are handled
    recursively; non-tensor values are replicated across samples."""
    bsz = _find_batch_size(data)
    assert bsz is not None, f"No tensor found to infer batch size: {data.keys()=}"

    per_sample: list[dict[str, Any]] = [{} for _ in range(bsz)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            parts = torch.unbind(value, dim=0)
            for b in range(bsz):
                per_sample[b][key] = parts[b].contiguous()
        elif isinstance(value, dict):
            sub = _unbind_dict_along_batch(value)
            for b in range(bsz):
                per_sample[b][key] = sub[b]
        else:
            for b in range(bsz):
                per_sample[b][key] = value
    return per_sample


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    """
    Collect chunk-step results and transitions during rollout,
    and convert them into trajectory tensors.

    Storage layout: every member is ``list[list[...]]`` (or, for dict members,
    ``list[list[dict]]``) where the **outer** list indexes the batch dim ``B``
    and the **inner** list indexes the chunk-step dim ``T``. Each stored
    element is a single-sample tensor with the batch dim stripped (e.g.
    ``actions[b][t]`` has shape ``[action_dim]``). The outer (batch) lists are
    created lazily on the first ``append_*`` call.
    """

    max_episode_length: int = 0

    # list[list[Tensor]]: [B][T] -> per-sample tensor (batch dim stripped)
    actions: list[list[torch.Tensor]] = field(default_factory=list)  # T
    intervene_flags: list[list[torch.Tensor]] = field(default_factory=list)  # T
    rewards: list[list[torch.Tensor]] = field(default_factory=list)  # T
    terminations: list[list[torch.Tensor]] = field(
        default_factory=list
    )
    truncations: list[list[torch.Tensor]] = field(
        default_factory=list
    )
    dones: list[list[torch.Tensor]] = field(
        default_factory=list
    )
    prev_logprobs: list[list[torch.Tensor]] = field(default_factory=list)  # T
    prev_values: list[list[torch.Tensor]] = field(
        default_factory=list
    )
    versions: list[list[torch.Tensor]] = field(default_factory=list)  # T

    # list[list[dict]]: [B][T] -> per-sample dict (batch dim stripped)
    forward_inputs: list[list[dict[str, Any]]] = field(default_factory=list)  # T

    curr_obs: list[list[dict[str, Any]]] = field(default_factory=list)  # T
    next_obs: list[list[dict[str, Any]]] = field(default_factory=list)  # T

    @staticmethod
    def _append_tensor_field(
        field_list: list[list[torch.Tensor]], tensor: torch.Tensor
    ) -> None:
        """Append a batch-major ``[B, ...]`` tensor as one stripped sample per
        outer (batch) list. Lazily creates the ``B`` outer lists."""
        parts = torch.unbind(tensor, dim=0)
        if not field_list:
            field_list.extend([] for _ in range(len(parts)))
        for b, part in enumerate(parts):
            field_list[b].append(part.contiguous())

    @staticmethod
    def _append_dict_field(
        field_list: list[list[dict[str, Any]]], data: dict[str, Any]
    ) -> None:
        """Append a batch-major dict ``{key: [B, ...]}`` as one stripped
        per-sample dict per outer (batch) list."""
        per_sample = _unbind_dict_along_batch(data)
        if not field_list:
            field_list.extend([] for _ in range(len(per_sample)))
        for b, item in enumerate(per_sample):
            field_list[b].append(item)

    @staticmethod
    def _stack_tensor_field(
        field_list: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Stack ``list[list[Tensor]]`` ([B][T]) into a ``[B, T, ...]`` tensor."""
        per_b = [torch.stack(steps, dim=0) for steps in field_list]  # each [T, ...]
        return torch.stack(per_b, dim=0)  # [B, T, ...]

    @staticmethod
    def _stack_dict_field(
        field_list: list[list[dict[str, Any]]],
    ) -> dict[str, torch.Tensor]:
        """Stack ``list[list[dict]]`` ([B][T]) into ``{key: [B, T, ...]}``."""
        per_b = [
            stack_list_of_dict_tensor(steps, dim=0) for steps in field_list
        ]  # each {key: [T, ...]}
        return stack_list_of_dict_tensor(per_b, dim=0)  # {key: [B, T, ...]}

    def append_step_result(self, result: ChunkStepResult):
        if result.actions is not None:
            self._append_tensor_field(self.actions, result.actions)
            self._append_tensor_field(
                self.intervene_flags,
                torch.zeros_like(result.actions, dtype=torch.bool),
            )
        if result.rewards is not None:
            self._append_tensor_field(self.rewards, result.rewards)
        if result.terminations is not None:
            self._append_tensor_field(self.terminations, result.terminations)
        if result.truncations is not None:
            self._append_tensor_field(self.truncations, result.truncations)
        if result.dones is not None:
            self._append_tensor_field(self.dones, result.dones)
        if result.prev_logprobs is not None:
            self._append_tensor_field(self.prev_logprobs, result.prev_logprobs)
        if result.prev_values is not None:
            self._append_tensor_field(self.prev_values, result.prev_values)
        if result.versions is not None:
            self._append_tensor_field(self.versions, result.versions)
        if result.forward_inputs:
            self._append_dict_field(self.forward_inputs, result.forward_inputs)

    def mark_last_step_with_flags(self, save_flags: torch.Tensor):
        if not self.intervene_flags:
            return

        if save_flags.dim() == 1:
            save_flags = save_flags[:, None]
        assert save_flags.dim() == 2, f"Expected 2D tensor, got {save_flags.shape=}"

        bsz, num_action_chunks = save_flags.shape
        for b in range(bsz):
            last_action = self.actions[b][-1]  # [num_action_chunks x action_dim]
            action_dim = last_action.numel() // num_action_chunks
            expanded_flags = (
                save_flags[b]
                .reshape(num_action_chunks, 1)
                .expand(num_action_chunks, action_dim)
                .reshape(-1)
            )
            self.intervene_flags[b][-1] = expanded_flags.to(torch.bool)

    def update_last_actions(
        self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor
    ):
        # action: [bsz, num-chunk-size x action-dim]
        # intervene_actions: [bsz, num-chunk-size x action-dim]
        # intervene_flags: [bsz, num-chunk-size]

        if not self.actions or len(self.actions) == 0:
            return

        assert intervene_actions.dim() == 2, (
            f"Expected 2D tensor, got {intervene_actions.shape=}"
        )

        # Normalize intervene_flags dimensions
        if intervene_flags.dim() == 1:
            intervene_flags = intervene_flags[:, None]
        assert intervene_flags.dim() == 2, (
            f"Expected 2D tensor, got {intervene_flags.shape=}"
        )

        bsz, num_action_chunks = intervene_flags.shape[:2]
        for b in range(bsz):
            last_action = self.actions[b][-1]  # [num_action_chunks x action_dim]
            assert last_action.dim() == 1, (
                f"Expected 1D tensor, got {last_action.shape=}"
            )
            action_dim = last_action.numel() // num_action_chunks
            flags = intervene_flags[b].reshape(num_action_chunks, 1)

            # Combine intervene_actions and last_action based on flags
            last_full_action = intervene_actions[b].reshape(
                num_action_chunks, action_dim
            ) * flags + last_action.reshape(num_action_chunks, action_dim) * (~flags)
            self.actions[b][-1] = last_full_action.reshape(-1)

            full_flags = flags.expand(num_action_chunks, action_dim).reshape(-1)
            self.intervene_flags[b][-1] = full_flags

            if self.forward_inputs and self.forward_inputs[b]:
                last_fi = self.forward_inputs[b][-1]
                if "action" in last_fi:
                    last_fi["action"] = (
                        last_full_action.reshape(-1).cpu().contiguous()
                    )
                last_fi.pop("model_action", None)

    def append_transitions(self, curr_obs=None, next_obs=None):
        assert curr_obs is not None and next_obs is not None
        if "task_descriptions" in curr_obs:
            curr_obs.pop("task_descriptions")
        if "task_descriptions" in next_obs:
            next_obs.pop("task_descriptions")
        self._append_dict_field(self.curr_obs, curr_obs)
        self._append_dict_field(self.next_obs, next_obs)

    def clear(self):
        self.actions.clear()
        self.intervene_flags.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.dones.clear()
        self.prev_logprobs.clear()
        self.prev_values.clear()
        self.versions.clear()
        self.forward_inputs.clear()
        self.curr_obs.clear()
        self.next_obs.clear()

    # Inner-stored fields, grouped by element type, used when slicing episodes.
    _TENSOR_FIELDS = (
        "actions",
        "intervene_flags",
        "rewards",
        "prev_logprobs",
        "versions",
        "terminations",
        "truncations",
        "dones",
        "prev_values",
    )
    _N_STEP_TENSOR_FIELDS = (
        "actions",
        "intervene_flags",
        "rewards",
        "prev_logprobs",
        "versions",
    )
    _N_1_STEP_TENSOR_FIELDS = (
        "terminations",
        "truncations",
        "dones",
        "prev_values",
    )

    _DICT_FIELDS = ("forward_inputs", "curr_obs", "next_obs")

    @staticmethod
    def _detect_episode_spans(
        done_steps: list[torch.Tensor],
    ) -> list[tuple[int, int]]:
        """Given a per-chunk-step ``done`` sequence for a single sample, return
        the ``[start, end]`` spans of each *complete* episode.
        """
        spans: list[tuple[int, int]] = []
        start = 0
        for t, done in enumerate(done_steps):
            if t > 0 and bool(done.any()):
                spans.append((start, t))
                start = t + 1
        return spans

    def _build_episode_trajectory(
        self, batch_idx: int, start: int, end: int
    ) -> Trajectory:
        """Build a single ``bsz=1`` :class:`Trajectory` for the episode covering
        chunk-step range ``[start, end]`` of sample ``batch_idx``.

        The time dim keeps its true (ragged) length: tensors are
        ``[1, episode_len, ...]`` and dict values are ``{key: [1, episode_len, ...]}``.
        """
        trajectory = Trajectory(max_episode_length=self.max_episode_length)

        for name in self._TENSOR_FIELDS:
            field_list = getattr(self, name)
            if not field_list or batch_idx >= len(field_list) or not field_list[batch_idx]:
                continue
            steps = field_list[batch_idx][start : end + 1]
            if name in self._N_1_STEP_TENSOR_FIELDS:
                extra_step = torch.zeros_like(steps[0])
                steps = [extra_step, *steps]
                
            stacked = torch.stack(steps, dim=0).unsqueeze(0)  # [1, episode_len, ...]
            setattr(trajectory, name, stacked.cpu().contiguous())


        for name in self._DICT_FIELDS:
            field_list = getattr(self, name)
            if not field_list or batch_idx >= len(field_list) or not field_list[batch_idx]:
                continue
            steps = field_list[batch_idx][start : end + 1]
            stacked = stack_list_of_dict_tensor(steps, dim=0)  # {key: [episode_len, ...]}
            setattr(
                trajectory,
                name,
                {
                    key: value.unsqueeze(0).cpu().contiguous()
                    for key, value in stacked.items()
                },
            )
        trajectory.model_weights_id = get_model_weights_id(
            trajectory.versions
            if trajectory.versions is not None else torch.zeros(1, dtype=torch.float32)
        )
        return trajectory

    def _assert_aligned_inner_lengths(self, batch_idx: int, ref_len: int) -> None:
        for name in self._TENSOR_FIELDS + self._DICT_FIELDS:
            field_list = getattr(self, name)
            if not field_list or batch_idx >= len(field_list) or not field_list[batch_idx]:
                continue
            field_len = len(field_list[batch_idx])
            if field_len == ref_len:
                continue
            raise NotImplementedError(
                "Per-episode splitting with misaligned inner lengths is not "
                f"implemented yet: field '{name}' has length {field_len} for "
                f"batch index {batch_idx}, but the boundary axis 'dones' has "
                f"length {ref_len}. This happens when the rollout carries "
                "extra bootstrap entries (rollout_epoch trailing steps)."
            )

    def to_trajectory(self) -> list[Trajectory]:
        if not self.dones:
            return []

        # All inner-stored fields consumed in lock-step with ``dones``.
        consumable_fields = self._TENSOR_FIELDS + self._DICT_FIELDS

        trajectories: list[Trajectory] = []
        for batch_idx in range(len(self.dones)):
            done_steps = self.dones[batch_idx]
            if not done_steps:
                continue
            self._assert_aligned_inner_lengths(batch_idx, len(done_steps))

            spans = self._detect_episode_spans(done_steps)
            if not spans:
                # No complete episode for this sample yet; keep everything.
                continue

            for start, end in spans:
                trajectories.append(
                    self._build_episode_trajectory(batch_idx, start, end)
                )

            for name in consumable_fields:
                field_list = getattr(self, name)
                if not field_list or batch_idx >= len(field_list):
                    continue
                field_list[batch_idx] = field_list[batch_idx][end + 1:]

        return trajectories

    def to_splited_trajectories(self, split_size: int) -> list[list[Trajectory]]:
        """Build per-episode trajectories (see :meth:`to_trajectory`) and
        distribute them round-robin into ``split_size`` groups (e.g. one group
        per actor split)."""
        all_trajectories = self.to_trajectory()

        # breakpoint()
        splited_trajectories: list[list[Trajectory]] = [
            [] for _ in range(split_size)
        ]
        for idx, trajectory in enumerate(all_trajectories):
            splited_trajectories[idx % split_size].append(trajectory)
        return splited_trajectories

    def to_splited_trajectories_by_sizes(
        self, split_sizes: list[int]
    ) -> list[Trajectory]:
        trajectory = self.to_trajectory()
        trajectories = [Trajectory() for _ in split_sizes]

        for field_name in trajectory.__dataclass_fields__:
            value = getattr(trajectory, field_name)
            if value is None:
                continue
            if isinstance(value, (int, str)):
                for split_trajectory in trajectories:
                    setattr(split_trajectory, field_name, value)
            elif isinstance(value, torch.Tensor):
                for split_trajectory, split_value in zip(
                    trajectories, torch.split(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value.contiguous())
            elif isinstance(value, dict):
                for split_trajectory, split_value in zip(
                    trajectories, split_dict(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value)
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        return trajectories


def convert_trajectories_to_batch(
    trajectories: list[Trajectory],
) -> dict[str, torch.Tensor]:
    """
    convert a list of trajectories to a batch dict, the shape of the batch is [T, B, ...].
    """
    if not trajectories:
        return {}

    batch: dict[str, torch.Tensor] = {}

    # -------- obs / forward_inputs: dict[str, Tensor] --------
    if trajectories[0].curr_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.curr_obs.keys())
        batch["curr_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.curr_obs[key] for traj in trajectories if key in traj.curr_obs
            ]
            if tensors:
                batch["curr_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].next_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.next_obs.keys())
        batch["next_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.next_obs[key] for traj in trajectories if key in traj.next_obs
            ]
            if tensors:
                batch["next_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].forward_inputs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.forward_inputs.keys())
        batch["forward_inputs"] = {}
        for key in all_keys:
            tensors = [
                traj.forward_inputs[key]
                for traj in trajectories
                if key in traj.forward_inputs
            ]
            if tensors:
                batch["forward_inputs"][key] = torch.cat(tensors, dim=1)

    # -------- tensor fields --------
    n_1_step_tensor_fields = ["dones", "terminations", "truncations", "prev_values"]
    reference_trajectory = trajectories[0]
    for field_name in reference_trajectory.__dataclass_fields__.keys():
        if not isinstance(getattr(reference_trajectory, field_name), torch.Tensor):
            continue

        if field_name in n_1_step_tensor_fields:
            field_list = [
                getattr(traj, field_name)[:, 1:] if traj_idx > 0 else getattr(traj, field_name)
                for traj_idx, traj in enumerate(trajectories)
                if getattr(traj, field_name) is not None
            ]
        else:
            field_list = [
                getattr(traj, field_name)
                for traj in trajectories
                if getattr(traj, field_name) is not None
            ]
        if field_list:
            batch[field_name] = torch.cat(field_list, dim=1)

    return batch
