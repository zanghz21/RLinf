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

import asyncio
import os
import queue
import threading
from typing import Any, Optional

import numpy as np
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import expand_to_target_dim
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.priority_store import PriorityStore
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import apply_rebalance_weight, clear_memory, masked_mean, reshape_entropy
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor, process_nested_dict_for_train


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Embodied FSDP actor worker for async PPO / decoupled actor-critic training."""

    should_stop = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_store_size = self.cfg.algorithm.get(
            "rollout_store_size_per_rank", 1
        )
        self.rollout_store = PriorityStore(maxsize=self.rollout_store_size)

    async def recv_rollout_trajectories(self, input_channel):
        # drain channel
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        while not self.should_stop:
            trajectories: list[Trajectory] = input_channel.get()
            for trajectory in trajectories:
                self.log_info(
                    f"recv trajectory versions.shape={trajectory.versions.shape} "
                    f"input_channel.qsize={input_channel.qsize()}"
                )
                if trajectory.versions.min() < self.version - self.cfg.algorithm.get("staleness_threshold", None):
                    continue
                self._recv_queue.put(trajectory)

    @Worker.timer("drain_received_trajectories")
    def _drain_received_trajectories(self):
        while True:
            try:
                traj: Trajectory = self._recv_queue.get_nowait()
                self.log_info(
                    f"drain traj versions.shape={traj.versions.shape} "
                    f"versions.min={traj.versions.min()} version={self.version} "
                    f"recv_queue.size={self._recv_queue.qsize()}"
                )
                if traj.versions.min() < self.version - self.cfg.algorithm.get(
                    "staleness_threshold", None
                ):
                    continue
                min_v = float(traj.versions.min().item())
                mean_v = float(traj.versions.float().mean().item())
                self.rollout_store.add((min_v, mean_v), traj)
                self.log_info(f"rollout_store size={len(self.rollout_store)}")
            except queue.Empty:
                break

    @Worker.timer("wait_for_rollout_store_ready")
    async def _wait_for_rollout_store_ready(self):
        while getattr(self, "_recv_queue", None) is None:
            await asyncio.sleep(1)

        on_policy_min_ratio = self.cfg.algorithm.get("on_policy_min_ratio", 0.0)
        while True:
            self._drain_received_trajectories()
            with self.worker_timer("remove_below"):
                self.rollout_store.remove_below(
                    self.version - self.cfg.algorithm.get("staleness_threshold", None)
                )

            local_ready = len(self.rollout_store) >= self.rollout_store_size
            if local_ready and on_policy_min_ratio > 0.0:
                metrics_data = self.rollout_store.get_metric()
                on_policy_ratio = metrics_data.get(int(self.version), {}).get(
                    "ratio", 0.0
                )
                self.log_info(
                    f"rollout store metrics={metrics_data} "
                    f"on_policy_ratio={on_policy_ratio:.4f} "
                    f"on_policy_min_ratio={on_policy_min_ratio}"
                )
                local_ready = on_policy_ratio >= on_policy_min_ratio

            ready_flag = torch.tensor(
                [1 if local_ready else 0], dtype=torch.int32, device=self.device
            )
            torch.distributed.all_reduce(
                ready_flag, op=torch.distributed.ReduceOp.MIN
            )
            if bool(ready_flag.item()):
                break
            await asyncio.sleep(1)

    @Worker.timer("construct_rollout_batch")
    async def construct_rollout_batch(self, max_trajectories: int | None = None):
        # from _recv_queue to rollout_batch
        await self._wait_for_rollout_store_ready()

        version_metrics = self.rollout_store.get_metric()
        rollout_batch = self.rollout_store.topn(self.rollout_store_size)
        self.log_info(f"rollout store version metrics={version_metrics}")

        staleness_metrics: dict = {}
        for version_val, stats in version_metrics.items():
            if version_val == "discarded_unused":
                staleness_metrics["discarded_unused_trajs"] = stats
                continue
            diff = int(self.version) - int(version_val)
            staleness_metrics[f"data_staleness_{diff}/ratio"] = stats["ratio"]

        self.full_rollout_batch = convert_trajectories_to_batch(rollout_batch)
        self.full_rollout_batch = self._process_received_rollout_batch(self.full_rollout_batch)
        self.log_info(f"staleness metrics={staleness_metrics}")
        return staleness_metrics

    @torch.inference_mode()
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        proximal_values = self.full_rollout_batch.get("proximal_values", None)
        prev_values = self.full_rollout_batch.get("prev_values", None)

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.full_rollout_batch["rewards"],
            "dones": self.full_rollout_batch["dones"],
            "values": proximal_values if proximal_values is not None else prev_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.full_rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.full_rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.normalize_advantages,
        }

        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.full_rollout_batch.update(adv_and_ret)

        if kwargs["loss_mask"] is not None:
            self.full_rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            self.full_rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]

        # self._debug_save_rollout_batch()

        rollout_metrics = compute_rollout_metrics(self.full_rollout_batch)
        return rollout_metrics

    def _debug_save_rollout_batch(self) -> None:
        import os

        save_dir = self.cfg.runner.logger.log_path
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"rollout_batch_step{int(self.version)}_rank{int(self._rank)}.pt",
        )

        torch.save(self.full_rollout_batch, path)
        self.log_info(f"[debug] saved rollout_batch → {path}")

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing proximal logprobs."
        )

        t_dim = self.full_rollout_batch["prev_logprobs"].shape[0]
        b_dim = self.full_rollout_batch["prev_logprobs"].shape[1]

        flat = process_nested_dict_for_train(self.full_rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size

        iterator = split_dict_to_chunk(flat, num_splits)

        self.model.eval()
        proximal_logprobs_list = []

        for micro_batch in iterator:
            micro_batch = put_tensor_device(micro_batch, self.device)
            forward_inputs = micro_batch.get("forward_inputs", None)
            if forward_inputs is None:
                raise ValueError(
                    "Missing forward_inputs in compute_proximal_logprobs. "
                    "This usually means batch splitting dropped nested dict fields."
                )

            model_kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                model_kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
                model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
            elif SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T,
                SupportedModel.ABOT_M0,
            ]:
                model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

            out = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **model_kwargs,
            )
            proximal_logprobs_list.append(out["logprobs"].cpu())

        proximal_logprobs = torch.cat(proximal_logprobs_list, dim=0).view(
            t_dim,
            b_dim,
            *self.full_rollout_batch["prev_logprobs"].shape[2:],
        )
        self.full_rollout_batch["proximal_logprobs"] = proximal_logprobs

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self._rebalance_full_rollout_batch_across_ranks()

        rollout_size = (
            self.full_rollout_batch["prev_logprobs"].shape[0]
            * self.full_rollout_batch["prev_logprobs"].shape[1]
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.actor.seed) + int(self._rank))


        if self.cfg.algorithm.normalize_advantages:
            self.full_rollout_batch["advantages"] = masked_normalization(
                self.full_rollout_batch["advantages"],
                self.full_rollout_batch.get("loss_mask", None),
            )

        self.model.train()

        world_size = int(self._world_size)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)

        assert global_batch_size % (micro_batch_size * world_size) == 0, (
            f"global_batch_size {global_batch_size} must be divisible by "
            f"micro_batch_size {micro_batch_size} * world_size {world_size}"
        )

        batch_size_per_rank = global_batch_size // world_size
        micro_per_rank = batch_size_per_rank // micro_batch_size
        self.gradient_accumulation = micro_per_rank


        metrics: dict[str, list] = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))

        for _ in range(update_epoch):
            batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
            real_rollout_size = rollout_size - rollout_size % batch_size_per_rank
            shuffle_id = torch.randperm(rollout_size, generator=generator)[:real_rollout_size]
            assert real_rollout_size % batch_size_per_rank == 0, (
                f"Flattened rollout size {real_rollout_size} must be divisible by "
                f"per-rank batch size {batch_size_per_rank}"
            )

            with torch.no_grad():
                self.rollout_batch = process_nested_dict_for_train(
                    self.full_rollout_batch, shuffle_id
                )

            global_batch_iter = split_dict_to_chunk(
                self.rollout_batch,
                real_rollout_size // batch_size_per_rank,
            )

            for train_global_batch in global_batch_iter:
                train_global_batch_size = int(
                    train_global_batch["prev_logprobs"].shape[0]
                )
                assert train_global_batch_size == batch_size_per_rank, (
                    f"Expected per-rank global batch size {batch_size_per_rank}, "
                    f"got {train_global_batch_size}"
                )
                assert train_global_batch_size % micro_batch_size == 0

                micro_batch_iter = split_dict_to_chunk(
                    train_global_batch,
                    micro_per_rank,
                )

                self.optimizer.zero_grad()

                for mb_idx, data in enumerate(micro_batch_iter):
                    data = put_tensor_device(
                        data,
                        f"cuda:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
                    )

                    advantages = data["advantages"]
                    old_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)
                    rebalance_weight = data.get("rebalance_weight", None)

                    versions = data.get("versions", None)
                    proximal_logprobs = data.get("proximal_logprobs", None)
                    proximal_values = data.get("proximal_values", None)
                    current_version = int(self.version) + 1

                    forward_inputs = data.get("forward_inputs", None)
                    if forward_inputs is None:
                        raise ValueError(
                            "Missing forward_inputs in run_training. "
                            "This usually means batch splitting dropped nested dict fields."
                        )

                    model_kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        model_kwargs["temperature"] = (
                            self.cfg.rollout.sampling_params.temperature_train
                        )
                        model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
                    elif SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        model_kwargs["prev_logprobs"] = old_logprobs

                    compute_values = self.cfg.algorithm.adv_type == "gae"

                    with self.amp_context:
                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                            compute_values=compute_values,
                            use_cache=False,
                            **model_kwargs,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        old_logprobs = out["prev_logprobs"]

                    loss_kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": out["logprobs"],
                        "values": out.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": proximal_values
                        if proximal_values is not None
                        else prev_values,
                        "proximal_logprobs": proximal_logprobs,
                        "versions": versions,
                        "current_version": current_version,
                        "behave_weight_threshold": self.cfg.algorithm.get(
                            "behave_weight_threshold", None
                        ),
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "rebalance_weight": rebalance_weight,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }

                    loss, metrics_data = policy_loss(**loss_kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not loss_kwargs["critic_warmup"]
                    ):
                        entropy = out["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=out["logprobs"].shape[0],
                        )
                        entropy_mask = loss_mask
                        if rebalance_weight is not None:
                            entropy_mask = apply_rebalance_weight(
                                loss_mask,
                                expand_to_target_dim(rebalance_weight, entropy.shape),
                            )
                        entropy_loss = masked_mean(entropy, mask=entropy_mask)
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                    loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/entropy_loss"] = float(
                        entropy_loss.detach().item()
                    )
                    metrics_data["actor/total_loss"] = float(loss.detach().item())
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                extra_metrics = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    extra_metrics["critic/lr"] = lr_list[1]
                append_to_dict(metrics, extra_metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict,
            op=torch.distributed.ReduceOp.AVG,
        )
        return mean_metric_dict
