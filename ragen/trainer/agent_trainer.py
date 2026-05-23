"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import os
import uuid
import ray
import torch
import numpy as np
import collections
from collections import defaultdict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from ragen.trainer.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


from verl.trainer.ppo.ray_trainer import ResourcePoolManager, compute_response_mask, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger
from ragen.trainer.rollout_filter import build_rollout_filter
from ragen.trainer.collapse_metrics import CollapseDetector
from ragen.trainer.gradient_reporter import run_gradient_analysis

from tensordict import TensorDict


def adjust_batch(batch: DataProto, size_divisor: int, mode: str = "copy") -> DataProto:
    """
    Adjust batch size to be divisible by size_divisor.

    Args:
        batch: The DataProto batch to adjust
        size_divisor: The number that batch size should be divisible by
        mode: "copy" to duplicate samples, "delete" to remove samples

    Returns:
        Adjusted DataProto with batch size divisible by size_divisor
    """
    bs = len(batch.batch) if hasattr(batch.batch, '__len__') else batch.batch.batch_size[0]
    remainder = bs % size_divisor

    if remainder == 0:
        return batch

    if mode == "delete":
        # Remove samples to make it divisible
        remove_indices = np.random.choice(bs, remainder, replace=False)
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool)
        if batch.batch is not None:
            tensor_data = batch.batch[keep_mask_tensor]
        else:
            tensor_data = None

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    non_tensor_data[key] = val[keep_mask]
                elif isinstance(val, list):
                    non_tensor_data[key] = [v for v, m in zip(val, keep_mask) if m]
                else:
                    non_tensor_data[key] = val

        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=batch.meta_info)

    elif mode == "copy":
        # Duplicate samples to make it divisible
        to_add = size_divisor - remainder
        if to_add > bs:
            dup_indices = np.random.choice(bs, to_add, replace=True)
        else:
            dup_indices = np.random.choice(bs, to_add, replace=False)

        # Create duplicated batch using TensorDict concat
        dup_indices_tensor = torch.tensor(dup_indices, dtype=torch.long)
        if batch.batch is not None:
            dup_tensor_data = batch.batch[dup_indices_tensor]
            # Use TensorDict's cat method
            tensor_data = TensorDict.cat([batch.batch, dup_tensor_data], dim=0)
        else:
            tensor_data = None

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    dup_val = val[dup_indices]
                    non_tensor_data[key] = np.concatenate([val, dup_val], axis=0)
                elif isinstance(val, list):
                    dup_val = [val[i] for i in dup_indices]
                    non_tensor_data[key] = val + dup_val
                else:
                    non_tensor_data[key] = val

        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=batch.meta_info)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'copy' or 'delete'.")

    return adjusted_batch


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0, soft_advantage_reweight=False):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                loss_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
                high_level_gamma=high_level_gamma,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # Pass episode_ids for deduplication in single_turn/limited_multi_turn mode
        episode_ids = data.non_tensor_batch.get("episode_ids", None)
        result = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            episode_ids=episode_ids,
            return_group_std=soft_advantage_reweight,
        )
        if soft_advantage_reweight:
            advantages, returns, group_std = result
            data.batch["group_std"] = group_std
        else:
            advantages, returns = result
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


class RayAgentTrainer(VerlRayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        # do not use the original val logger, but use this here
        self.generations_logger = GenerationsLogger()
        
        # Early stopping state
        self.first_10_steps_variances = []
        self.base_variance = None
        self.consecutive_variances = collections.deque(maxlen=10)
        self.consecutive_empty_filtered_steps = 0
        self.consecutive_low_success = collections.defaultdict(int)
        self.early_stopped = False
        self.early_stop_type = None
        self.group_rv_table = None
        self.gradient_analysis_proxy = None
        self.gradient_analysis_rollout_filter = None
        self.gradient_analysis_config = None

    def _early_stop_metric_key(self) -> str:
        stop_type = self.early_stop_type if self.early_stop_type else "unknown"
        return f"early_stopped/{stop_type}"

        
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
        # val_start = 100000
        # self.train_seeds = [seed for seed in range(0, self.config.trainer.total_training_steps * 1000, 1000)]
        # self.val_seeds = [seed for seed in range(val_start, val_start + self.config.trainer.validation_steps)]

    def _get_gradient_analysis_batch_shape(self):
        train_env_groups = int(self.config.es_manager.train.env_groups)
        train_group_size = int(self.config.es_manager.train.group_size)
        analysis_env_groups = self.config.trainer.get("gradient_analysis_env_groups", None)
        analysis_group_size = self.config.trainer.get("gradient_analysis_group_size", None)

        env_groups = train_env_groups if analysis_env_groups is None else int(analysis_env_groups)
        group_size = train_group_size if analysis_group_size is None else int(analysis_group_size)

        if env_groups <= 0 or group_size <= 0:
            raise ValueError(
                f"Gradient analysis batch shape must be positive. Got env_groups={env_groups}, group_size={group_size}."
            )

        if env_groups == train_env_groups and group_size == train_group_size:
            return None

        return env_groups, group_size

    @staticmethod
    def _scale_env_config_group_counts(base_n_groups, target_total: int):
        base_n_groups = [int(n) for n in base_n_groups]
        base_total = sum(base_n_groups)
        if base_total <= 0:
            raise ValueError("Sum of base env_config n_groups must be positive.")
        if target_total <= 0:
            raise ValueError("Target env_groups must be positive.")

        raw_scaled = [target_total * n / base_total for n in base_n_groups]
        scaled = [int(x) for x in raw_scaled]
        remainder = target_total - sum(scaled)
        if remainder > 0:
            order = sorted(
                range(len(base_n_groups)),
                key=lambda i: (raw_scaled[i] - scaled[i], base_n_groups[i]),
                reverse=True,
            )
            for idx in order[:remainder]:
                scaled[idx] += 1
        elif remainder < 0:
            order = sorted(
                range(len(base_n_groups)),
                key=lambda i: (raw_scaled[i] - scaled[i], base_n_groups[i]),
            )
            for idx in order[: -remainder]:
                if scaled[idx] <= 0:
                    continue
                scaled[idx] -= 1

        return scaled

    def _build_gradient_analysis_config(self):
        batch_shape = self._get_gradient_analysis_batch_shape()
        if batch_shape is None:
            return None

        env_groups, group_size = batch_shape
        analysis_config = deepcopy(self.config)
        scaled_n_groups = self._scale_env_config_group_counts(
            analysis_config.es_manager.train.env_configs.n_groups,
            env_groups,
        )

        with open_dict(analysis_config):
            analysis_config.es_manager.train.env_groups = env_groups
            analysis_config.es_manager.train.group_size = group_size
            analysis_config.es_manager.train.env_configs.n_groups = scaled_n_groups

        print(
            f"[Gradient Analysis] Using separate analysis batch: env_groups={env_groups}, "
            f"group_size={group_size}, env_configs.n_groups={scaled_n_groups}"
        )
        return analysis_config

    def init_agent_proxy(self):
        if self.gradient_analysis_config is None:
            self.gradient_analysis_config = self._build_gradient_analysis_config()
        self.agent_proxy = LLMAgentProxy(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer
        )
        if self.gradient_analysis_config is not None:
            self.gradient_analysis_proxy = LLMAgentProxy(
                config=self.gradient_analysis_config,
                actor_rollout_wg=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
            )
        else:
            self.gradient_analysis_proxy = None

    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.generations_to_log_to_wandb[_type]

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _build_group_rv_wandb_table(self, group_ids, reward_std_values, selected_group_ids):
        """Append per-step group RV values to a cumulative W&B table."""
        if not self.config.trainer.get("log_group_rv_table", False):
            return None
        if "wandb" not in self.config.trainer.logger:
            return None

        try:
            import wandb
        except ImportError:
            return None

        if torch.is_tensor(group_ids):
            group_ids = group_ids.detach().cpu().tolist()
        elif isinstance(group_ids, np.ndarray):
            group_ids = group_ids.tolist()
        if torch.is_tensor(reward_std_values):
            reward_std_values = reward_std_values.detach().cpu().tolist()
        elif isinstance(reward_std_values, np.ndarray):
            reward_std_values = reward_std_values.tolist()
        if torch.is_tensor(selected_group_ids):
            selected_group_ids = selected_group_ids.detach().cpu().tolist()
        elif isinstance(selected_group_ids, np.ndarray):
            selected_group_ids = selected_group_ids.tolist()

        if selected_group_ids is None:
            selected_group_ids = []

        selected_group_id_set = {int(group_id) for group_id in selected_group_ids}
        columns = ["step", "group_id", "reward_std", "selected"]

        if self.group_rv_table is None:
            self.group_rv_table = wandb.Table(columns=columns)

        # Work around W&B incremental table logging by re-creating the table with prior rows.
        new_table = wandb.Table(columns=columns, data=self.group_rv_table.data)
        for group_id, reward_std in zip(group_ids, reward_std_values):
            new_table.add_data(
                int(self.global_steps),
                int(group_id),
                float(reward_std),
                int(group_id) in selected_group_id_set,
            )

        self.group_rv_table = new_table
        return new_table

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            import time
            start_time = time.time()
            test_batch = self.agent_proxy.rollout(test_gen_batch, val=True)
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            for key, value in test_batch.meta_info["metrics"].items():
                if "val-env/" + key not in env_metric_dict:
                    env_metric_dict["val-env/" + key] = []
                env_metric_dict["val-env/" + key].append(value)

            # Store original inputs and outputs
            batch_size = test_batch.batch["input_ids"].shape[0]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            # Handle single_turn/limited_multi_turn mode: group messages by episode
            context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")
            is_turn_level_mode = context_window_mode in ("single_turn", "limited_multi_turn")
            if is_turn_level_mode and "messages_list" in test_batch.non_tensor_batch:
                # Group samples by episode_id to reconstruct episodes
                episode_ids = test_batch.non_tensor_batch["episode_ids"]
                messages_list = test_batch.non_tensor_batch["messages_list"]

                # Find unique episodes and their samples
                unique_groups = []
                group_to_indices = {}
                for i, eid in enumerate(episode_ids):
                    if eid not in group_to_indices:
                        unique_groups.append(eid)
                        group_to_indices[eid] = []
                    group_to_indices[eid].append(i)

                # Create grouped outputs
                grouped_inputs = []
                grouped_outputs = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Combine all messages from this episode
                    episode_output = ""
                    for idx in indices:
                        msgs = messages_list[idx]
                        for msg in msgs:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            episode_output += f"[{role}]\n{content}\n\n"
                    grouped_inputs.append("")
                    grouped_outputs.append(episode_output.strip())

                sample_inputs.extend(grouped_inputs)
                sample_outputs.extend(grouped_outputs)
            else:
                input_texts = ["" for _ in range(batch_size)]
                sample_inputs.extend(input_texts)
                sample_outputs.extend(output_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            # Group scores by episode if turn-level mode
            if is_turn_level_mode and "messages_list" in test_batch.non_tensor_batch:
                grouped_scores = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Use the first turn's score (all turns have same episode reward)
                    episode_score = scores[indices[0]]
                    grouped_scores.append(episode_score)
                sample_scores.extend(grouped_scores)
                reward_extra_infos_dict["reward"].extend(grouped_scores)
            else:
                sample_scores.extend(scores)
                reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Get data sources and group if needed
            data_sources_batch = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            if is_turn_level_mode and "messages_list" in test_batch.non_tensor_batch:
                # Group data sources by episode
                grouped_data_sources = [data_sources_batch[group_to_indices[gid][0]] for gid in unique_groups]
                data_source_lst.append(grouped_data_sources)
            else:
                data_source_lst.append(data_sources_batch)

        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        super().init_workers()
        if self.gradient_analysis_config is None:
            self.gradient_analysis_config = self._build_gradient_analysis_config()

        # create rollout filter
        rollout_cfg = self.config.actor_rollout_ref.rollout
        rollout_metric = getattr(rollout_cfg, "rollout_filter_metric", "reward_variance")
        self.rollout_filter = build_rollout_filter(
            value=getattr(rollout_cfg, "rollout_filter_value", getattr(rollout_cfg, "rollout_filter_ratio", 0.25)),
            filter_type=rollout_cfg.rollout_filter_type,
            num_groups=self.config.es_manager.train.env_groups,
            group_size=self.config.es_manager.train.group_size,
            metric=rollout_metric,
            compute_log_prob=self.actor_rollout_wg.compute_log_prob,
            include_zero=getattr(rollout_cfg, "rollout_filter_include_zero", True),
            strategy=getattr(rollout_cfg, "rollout_filter_strategy", "top_p"),
            top_p_prob_mode=getattr(rollout_cfg, "rollout_filter_top_p_prob_mode", "linear"),
            selection_eps=getattr(rollout_cfg, "rollout_filter_selection_eps", 0.01),
            bucket_count=getattr(rollout_cfg, "gradient_analysis_num_buckets", 6),
            bucket_mode=getattr(rollout_cfg, "gradient_analysis_bucket_mode", "quantile"),
        )
        if self.gradient_analysis_config is not None:
            analysis_rollout_cfg = self.gradient_analysis_config.actor_rollout_ref.rollout
            self.gradient_analysis_rollout_filter = build_rollout_filter(
                value=getattr(
                    analysis_rollout_cfg,
                    "rollout_filter_value",
                    getattr(analysis_rollout_cfg, "rollout_filter_ratio", 0.25),
                ),
                filter_type=analysis_rollout_cfg.rollout_filter_type,
                num_groups=self.gradient_analysis_config.es_manager.train.env_groups,
                group_size=self.gradient_analysis_config.es_manager.train.group_size,
                metric=rollout_metric,
                compute_log_prob=self.actor_rollout_wg.compute_log_prob,
                include_zero=getattr(analysis_rollout_cfg, "rollout_filter_include_zero", True),
                strategy=getattr(analysis_rollout_cfg, "rollout_filter_strategy", "top_p"),
                top_p_prob_mode=getattr(analysis_rollout_cfg, "rollout_filter_top_p_prob_mode", "linear"),
                selection_eps=getattr(analysis_rollout_cfg, "rollout_filter_selection_eps", 0.01),
                bucket_count=getattr(analysis_rollout_cfg, "gradient_analysis_num_buckets", 6),
                bucket_mode=getattr(analysis_rollout_cfg, "gradient_analysis_bucket_mode", "quantile"),
            )
        else:
            self.gradient_analysis_rollout_filter = self.rollout_filter

        # create collapse detector
        collapse_cfg = self.config.get("collapse_detection", {})
        context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")
        enable_think = bool(getattr(self.config.agent_proxy, "enable_think", True))
        # num_samples: int for specific count, "all" for using all samples
        num_samples_cfg = collapse_cfg.get("num_samples", "all")
        if num_samples_cfg is None:
            raise ValueError("collapse_detection.num_samples must be an int or 'all'")
        if isinstance(num_samples_cfg, str):
            if num_samples_cfg.lower() != "all":
                raise ValueError("collapse_detection.num_samples must be an int or 'all'")
            num_samples = None
        else:
            try:
                num_samples = int(num_samples_cfg)
            except (TypeError, ValueError) as exc:
                raise ValueError("collapse_detection.num_samples must be an int or 'all'") from exc
            if num_samples <= 0:
                raise ValueError("collapse_detection.num_samples must be a positive int or 'all'")
        collapse_first = collapse_cfg.get("first_turn_enabled", False)
        collapse_multi = collapse_cfg.get("multi_turn_enabled", False)
        if not enable_think:
            collapse_first = False
            collapse_multi = False
        trainer_nnodes = int(self.config.trainer.get("nnodes", 1) or 1)
        trainer_n_gpus = int(self.config.trainer.get("n_gpus_per_node", 1) or 1)

        self.collapse_detector = CollapseDetector(
            compute_freq=collapse_cfg.get("compute_freq", 10),
            micro_batch_size=collapse_cfg.get("micro_batch_size", 16),
            context_window_mode=context_window_mode,
            multi_turn_enabled=collapse_multi,
            first_turn_enabled=collapse_first,
            num_samples=num_samples,
            std_eps=collapse_cfg.get("std_eps", 1e-3),
            ema_decay=collapse_cfg.get("ema_decay", 0.9),
            log_prob_world_size=trainer_nnodes * trainer_n_gpus,
        )

    @staticmethod
    def _to_python_scalar(value):
        if torch.is_tensor(value):
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _attach_group_reward_std(self, batch, reward_std_values, group_ids_for_metrics):
        if reward_std_values is None or batch.batch is None:
            return

        device = batch.batch["input_ids"].device if "input_ids" in batch.batch.keys() else torch.device("cpu")
        reward_std_values = torch.as_tensor(reward_std_values, dtype=torch.float32, device=device)

        if (
            batch.non_tensor_batch is not None
            and "group_ids" in batch.non_tensor_batch
            and group_ids_for_metrics is not None
        ):
            batch_group_ids = batch.non_tensor_batch["group_ids"]
            if torch.is_tensor(batch_group_ids):
                batch_group_ids = batch_group_ids.detach().cpu().tolist()
            else:
                batch_group_ids = np.asarray(batch_group_ids).tolist()

            if torch.is_tensor(group_ids_for_metrics):
                group_ids_for_metrics = group_ids_for_metrics.detach().cpu().tolist()
            else:
                group_ids_for_metrics = np.asarray(group_ids_for_metrics).tolist()

            reward_std_map = {
                self._to_python_scalar(group_id): reward_std_values[idx].item()
                for idx, group_id in enumerate(group_ids_for_metrics)
            }
            sample_reward_std = [
                reward_std_map[self._to_python_scalar(group_id)]
                for group_id in batch_group_ids
            ]
            batch.batch["reward_std"] = torch.tensor(
                sample_reward_std,
                dtype=reward_std_values.dtype,
                device=device,
            )
            return

        group_size = (
            int(self.gradient_analysis_config.es_manager.train.group_size)
            if self.gradient_analysis_config is not None
            else int(self.config.es_manager.train.group_size)
        )
        batch.batch["reward_std"] = reward_std_values.repeat_interleave(group_size)

    def _prepare_gradient_analysis_batch(self, batch, source_env_groups, metrics=None, metrics_prefix=None):
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        n_gpus = self.config.trainer.n_gpus_per_node
        size_divisor = np.lcm.reduce([source_env_groups, ppo_mini_batch_size, n_gpus])
        adjust_mode = getattr(self.config.agent_proxy, "batch_adjust_mode", "copy")
        batch = adjust_batch(batch, size_divisor, mode=adjust_mode)
        if metrics is not None and metrics_prefix is not None:
            metrics[f"{metrics_prefix}/adjusted_batch_size"] = batch.batch["input_ids"].shape[0]

        batch.meta_info = dict(batch.meta_info or {})
        batch.non_tensor_batch = batch.non_tensor_batch or {}
        if "group_ids" in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]
        batch.batch["response_mask"] = batch.batch["loss_mask"]

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics={})

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        if self.use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)

        if self.config.reward_model.launch_reward_fn_async:
            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)

        if self.config.reward_model.launch_reward_fn_async:
            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

        batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        if self.config.algorithm.use_kl_in_reward:
            batch, _ = apply_kl_penalty(
                batch,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
                multi_turn=True,
            )
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        soft_advantage_reweight = self.config.algorithm.get("soft_advantage_reweight", False)
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            multi_turn=True,
            high_level_gamma=self.config.algorithm.high_level_gamma,
            bi_level_gae=self.config.algorithm.bi_level_gae,
            soft_advantage_reweight=soft_advantage_reweight,
        )

        if soft_advantage_reweight and "group_std" in batch.batch:
            group_std = batch.batch["group_std"]
            index = batch.non_tensor_batch["uid"]
            unique_idx, inverse = np.unique(index, return_inverse=True)
            prompt_std = torch.zeros(len(unique_idx), device=group_std.device)
            for i, _ in enumerate(unique_idx):
                mask = inverse == i
                prompt_std[i] = group_std[mask][0]
            max_std = prompt_std.max()
            epsilon = 1e-6
            prompt_weight = prompt_std / (max_std + epsilon)
            sample_weight = prompt_weight[torch.from_numpy(inverse).to(group_std.device)]
            batch.batch["advantages"] = batch.batch["advantages"] * sample_weight.unsqueeze(-1)

        filter_loss_scaling = getattr(self.config.actor_rollout_ref.actor, "filter_loss_scaling", "none")
        filter_kept_ratio = batch.meta_info.get("filter_kept_ratio", 1.0)
        if filter_loss_scaling == "linear":
            batch.batch["advantages"] *= filter_kept_ratio
        elif filter_loss_scaling == "sqrt":
            batch.batch["advantages"] *= (filter_kept_ratio ** 0.5)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
            response_mask = batch.batch["response_mask"]
            advantages = batch.batch["advantages"]
            response_relative_lengths = (
                (torch.sum(response_mask, dim=-1) + 1e-6)
                / torch.sum(response_mask, dim=-1).float().mean()
            )
            batch.batch["advantages"] = advantages / response_relative_lengths.unsqueeze(-1)

        if self.config.algorithm.get("zero_task_advantage", False):
            batch.batch["advantages"] = torch.zeros_like(batch.batch["advantages"])

        return batch

    def _build_separate_gradient_analysis_batches(self, metrics):
        if self.gradient_analysis_proxy is None or self.gradient_analysis_rollout_filter is None:
            return None, None

        analysis_env_groups = int(self.gradient_analysis_config.es_manager.train.env_groups)
        analysis_group_size = int(self.gradient_analysis_config.es_manager.train.group_size)
        metrics["grad_analysis/source_env_groups"] = analysis_env_groups
        metrics["grad_analysis/source_group_size"] = analysis_group_size

        batch = DataProto()
        batch.meta_info = {"compute_collapse": False}
        batch = self.gradient_analysis_proxy.rollout(batch, val=False)
        raw_batch_size = batch.batch["input_ids"].shape[0]
        metrics["grad_analysis/source_batch_size"] = raw_batch_size

        prefilter_source_batch = None
        if self.config.trainer.get("gradient_analysis_log_prefilter", False):
            prefilter_source_batch = deepcopy(batch)
            metrics["grad_analysis_prefilter/source_batch_size"] = raw_batch_size

        batch, filter_metrics = self.gradient_analysis_rollout_filter.filter(batch)
        metrics["grad_analysis/empty_after_filter"] = float(len(batch) == 0)
        for key, value in filter_metrics.items():
            if key.startswith("rollout/_"):
                continue
            metrics[f"grad_analysis/{key.split('/', 1)[-1]}"] = value

        prefilter_batch = None
        if prefilter_source_batch is not None:
            group_reward_std = filter_metrics.get("rollout/_group_reward_std", None)
            group_ids_for_metrics = filter_metrics.get("rollout/_group_ids", None)
            self._attach_group_reward_std(prefilter_source_batch, group_reward_std, group_ids_for_metrics)
            prefilter_batch = self._prepare_gradient_analysis_batch(
                prefilter_source_batch,
                source_env_groups=analysis_env_groups,
                metrics=metrics,
                metrics_prefix="grad_analysis_prefilter",
            )

        if len(batch) == 0:
            return batch, prefilter_batch

        filter_kept_ratio = filter_metrics.get("rollout/filter_kept_ratio", None)
        if filter_kept_ratio is not None:
            batch.meta_info = dict(batch.meta_info or {})
            batch.meta_info["filter_kept_ratio"] = float(self._to_python_scalar(filter_kept_ratio))

        batch = self._prepare_gradient_analysis_batch(
            batch,
            source_env_groups=analysis_env_groups,
            metrics=metrics,
            metrics_prefix="grad_analysis",
        )
        return batch, prefilter_batch

    def get_gradient_analysis_batches(self, batch, metrics):
        analysis_batch, prefilter_batch = self._build_separate_gradient_analysis_batches(metrics)
        if analysis_batch is None:
            metrics["grad_analysis/uses_training_batch"] = 1.0
            return batch, None, self.rollout_filter

        metrics["grad_analysis/uses_training_batch"] = 0.0
        return analysis_batch, prefilter_batch, self.gradient_analysis_rollout_filter

    def get_gradient_analysis_batch_and_filter(self, batch, metrics):
        analysis_batch, _, analysis_rollout_filter = self.get_gradient_analysis_batches(batch, metrics)
        return analysis_batch, analysis_rollout_filter


    def _save_checkpoint(self):
        """ 
        Different from VerlRayPPOTrainer, we have no dataloader so we won"t save it. Other logic is the same.
        """
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
         to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        def _finish_logger():
            if hasattr(logger, "finish"):
                logger.finish()

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                _finish_logger()
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # For a resume-based gradient-analysis-only probe, keep the checkpoint
        # step number in logs instead of advancing to a synthetic next step.
        analysis_probe_from_resume = bool(
            self.config.trainer.get("gradient_analysis_only", False) and self.global_steps > 0
        )
        if not analysis_probe_from_resume:
            # we start from step 1
            self.global_steps += 1
        else:
            print(f"[Gradient Analysis] Probing resumed checkpoint at step {self.global_steps}")
        last_val_metrics = None

        def _process_batch_for_logging(batch):
            inputs_raw = batch.batch["input_ids"]
            inputs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in inputs_raw]
            outputs = [""] * len(inputs)
            scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()

            # Group by episode if turn-level mode
            context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")
            is_turn_level_mode = context_window_mode in ("single_turn", "limited_multi_turn")
            if is_turn_level_mode and "messages_list" in batch.non_tensor_batch:
                episode_ids = batch.non_tensor_batch["episode_ids"]
                messages_list = batch.non_tensor_batch["messages_list"]

                # Find unique episodes and their samples
                unique_groups = []
                group_to_indices = {}
                for i, eid in enumerate(episode_ids):
                    if eid not in group_to_indices:
                        unique_groups.append(eid)
                        group_to_indices[eid] = []
                    group_to_indices[eid].append(i)

                # Create grouped outputs
                grouped_inputs = []
                grouped_outputs = []
                grouped_scores = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Combine all messages from this episode
                    episode_output = ""
                    for idx in indices:
                        msgs = messages_list[idx]
                        for msg in msgs:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            episode_output += f"[{role}]\n{content}\n\n"
                    grouped_inputs.append("")
                    grouped_outputs.append(episode_output.strip())
                    grouped_scores.append(scores[indices[0]])

                return grouped_inputs, grouped_outputs, grouped_scores

            return inputs, outputs, scores

        import time
        self.start_time = time.time()
        self.train_time_total = 0.0
        self.eval_time_total = 0.0
        self.collapse_time_total = 0.0
        self.collapse_first_turn_time_total = 0.0
        self.collapse_multi_turn_time_total = 0.0
        for step in range(self.total_training_steps):
            # metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # Generate and filter exactly once per training step. If the
                # filtered batch is empty, treat this step as empty_after_filter
                # and move to the next step instead of retrying rollout.
                attempts = 1
                batch = DataProto()
                batch.meta_info = batch.meta_info or {}
                batch.meta_info["compute_collapse"] = self.collapse_detector.should_compute(
                    self.global_steps
                )

                with marked_timer("gen", timing_raw):
                    batch = self.agent_proxy.rollout(batch, val=False)

                metrics = {}

                # Compute collapse detection metrics before filtering (for fair comparison)
                with marked_timer("collapse_metrics", timing_raw, color="cyan"):
                    collapse_metrics = self.collapse_detector.compute_collapse_metrics(
                        batch=batch,
                        actor_compute_log_prob_fn=self.actor_rollout_wg.compute_log_prob,
                        global_step=self.global_steps,
                    )
                    metrics.update(collapse_metrics)

                with marked_timer("filter", timing_raw):
                    # Filter first, then adjust batch size
                    batch, filter_metrics = self.rollout_filter.filter(batch)

                    reward_matrix = filter_metrics.pop("rollout/_reward_matrix", None)
                    group_reward_std = filter_metrics.pop("rollout/_group_reward_std", None)
                    group_ids = filter_metrics.pop("rollout/_group_ids", None)
                    selected_group_ids = filter_metrics.pop("rollout/_selected_group_ids", None)
                    if reward_matrix is not None and "wandb" in self.config.trainer.logger:
                        try:
                            import wandb

                            num_groups, group_size = reward_matrix.shape
                            columns = [f"group_{i}" for i in range(num_groups)]
                            table_data = [
                                [reward_matrix[g, s].item() for g in range(num_groups)]
                                for s in range(group_size)
                            ]
                            filter_metrics["rollout/reward_table"] = wandb.Table(
                                columns=columns,
                                data=table_data,
                            )
                        except ImportError:
                            pass
                    if group_reward_std is not None and group_ids is not None:
                        group_rv_table = self._build_group_rv_wandb_table(
                            group_ids=group_ids,
                            reward_std_values=group_reward_std,
                            selected_group_ids=selected_group_ids,
                        )
                        if group_rv_table is not None:
                            filter_metrics["rollout/group_rv_table"] = group_rv_table

                    metrics.update(filter_metrics)

                    # Add kept ratio to meta_info for loss scaling
                    if "rollout/filter_kept_ratio" in metrics:
                        batch.meta_info["filter_kept_ratio"] = metrics["rollout/filter_kept_ratio"]
                    else:
                        batch.meta_info["filter_kept_ratio"] = 1.0

                if self.early_stopped:
                    print("[Early Stopping] Stopping training.")
                    # Ensure we log the metric before finishing
                    metrics.update({self._early_stop_metric_key(): 1.0})
                    logger.log(data=metrics, step=self.global_steps)
                    break

                if len(batch) == 0:
                    self.consecutive_empty_filtered_steps += 1
                    empty_stop_steps = getattr(
                        self.config.actor_rollout_ref.rollout,
                        "rollout_filter_empty_stop_steps",
                        10,
                    )
                    metrics.update(
                        {
                            "train/rollout_attempts": attempts,
                            "rollout/empty_after_filter": 1.0,
                            "rollout/consecutive_empty_after_filter_steps": float(
                                self.consecutive_empty_filtered_steps
                            ),
                        }
                    )

                    if (
                        empty_stop_steps > 0
                        and self.consecutive_empty_filtered_steps >= empty_stop_steps
                    ):
                        print(
                            f"[Early Stopping] No samples kept after filtering for "
                            f"{self.consecutive_empty_filtered_steps} consecutive steps."
                        )
                        self.early_stopped = True
                        self.early_stop_type = "empty_filtered_steps"
                        metrics.update({self._early_stop_metric_key(): 1.0})
                    else:
                        print(
                            f"[Warning] No samples kept after filtering for this step. "
                            f"Consecutive empty steps: {self.consecutive_empty_filtered_steps}."
                        )

                    gradient_analysis_every = self.config.trainer.get("gradient_analysis_every", 0)
                    gradient_analysis_only = bool(self.config.trainer.get("gradient_analysis_only", False))
                    if (
                        gradient_analysis_only
                        and self.config.trainer.get("gradient_analysis_mode", False)
                        and gradient_analysis_every > 0
                        and (self.global_steps - 1) % gradient_analysis_every == 0
                    ):
                        print(
                            "[Gradient Analysis] Main training batch is empty after filtering; "
                            "falling back to separate analysis batch."
                        )
                        with marked_timer("gradient_analysis", timing_raw):
                            run_gradient_analysis(self, batch, metrics)
                        if self.config.trainer.get("exit_after_gradient_analysis", False):
                            metrics["trainer/exited_after_gradient_analysis"] = 1.0

                    logger.log(data=metrics, step=self.global_steps)

                    if self.early_stopped or is_last_step:
                        _finish_logger()
                        progress_bar.close()
                        return

                    progress_bar.update(1)
                    self.global_steps += 1
                    continue

                self.consecutive_empty_filtered_steps = 0

                # Adjust batch size to be divisible by num_groups, ppo_mini_batch_size, and n_gpus
                num_groups = self.config.es_manager.train.env_groups
                ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                n_gpus = self.config.trainer.n_gpus_per_node
                size_divisor = np.lcm.reduce([num_groups, ppo_mini_batch_size, n_gpus])
                adjust_mode = getattr(self.config.agent_proxy, "batch_adjust_mode", "copy")
                batch = adjust_batch(batch, size_divisor, mode=adjust_mode)

                # Record batch and mini-batch statistics
                batch_size = batch.batch["input_ids"].shape[0]
                num_mini_batches = batch_size // ppo_mini_batch_size
                metrics.update({
                    "train/batch_size": batch_size,
                    "train/num_mini_batches": num_mini_batches,
                    "train/rollout_attempts": attempts,
                    "rollout/empty_after_filter": 0.0,
                    "rollout/consecutive_empty_after_filter_steps": 0.0,
                })
                metrics.update({"train/" + key: value for key, value in batch.meta_info["metrics"].items()})

                # Record successful step variance for base variance calculation
                # and run step-level reward-variance early stopping.
                if "rollout/in_group_reward_std" in metrics and len(self.first_10_steps_variances) < 10:
                    current_var = metrics["rollout/in_group_reward_std"]
                    if isinstance(current_var, torch.Tensor):
                        current_var = current_var.item()
                    self.first_10_steps_variances.append(current_var)
                    if len(self.first_10_steps_variances) == 10:
                        self.base_variance = sum(self.first_10_steps_variances) / 10
                        # Start counting collapse window only after baseline is ready.
                        self.consecutive_variances.clear()
                        print(f"\n[Early Stopping] Base variance calculated from first 10 steps: {self.base_variance:.6f}")
                elif "rollout/in_group_reward_std" in metrics and self.base_variance is not None:
                    current_var = metrics["rollout/in_group_reward_std"]
                    if isinstance(current_var, torch.Tensor):
                        current_var = current_var.item()
                    self.consecutive_variances.append(current_var)

                    if len(self.consecutive_variances) == 10:
                        threshold = 0.1 * self.base_variance
                        if all(v < threshold for v in self.consecutive_variances):
                            print(f"\n[Early Stopping] Reward variance collapsed!")
                            print(f"Base variance (mean of first 10 successful steps): {self.base_variance:.6f}")
                            print(f"Recent step variances: {[f'{v:.6f}' for v in self.consecutive_variances]}")
                            self.early_stopped = True
                            self.early_stop_type = "reward_variance_collapse"

                if self.early_stopped:
                    print("[Early Stopping] Stopping training.")
                    metrics.update({self._early_stop_metric_key(): 1.0})
                    logger.log(data=metrics, step=self.global_steps)
                    break

                inputs, outputs, scores = _process_batch_for_logging(batch)
                # self._maybe_log_generations(inputs=inputs, outputs=outputs, scores=scores, _type="train")

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yer
                    logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()
                    with marked_timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch, val=False)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)

                # NOTE reward normalization already done in ctx_manager, so set group size = 1 here
                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                
                # NOTE: do not do reward normalization in ctx_manager, so we need to do it here
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]

                # batch.batch["response_mask"] = compute_response_mask(batch)
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if self.use_rm:
                    with marked_timer("reward", timing_raw):
                    # compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    print(f"{list(reward_extra_infos_dict.keys())=}")
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                    soft_advantage_reweight = self.config.algorithm.get("soft_advantage_reweight", False)

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=True,
                        high_level_gamma=self.config.algorithm.high_level_gamma,
                        bi_level_gae=self.config.algorithm.bi_level_gae,
                        soft_advantage_reweight=soft_advantage_reweight,
                    )

                    # Apply soft advantage reweighting based on reward variance (group std)
                    # This scales advantages by (group_std / max_group_std) to down-weight low-variance prompts
                    if soft_advantage_reweight and "group_std" in batch.batch:
                        group_std = batch.batch["group_std"]  # (batch_size,)
                        index = batch.non_tensor_batch["uid"]

                        # Compute per-prompt std (take first occurrence per group)
                        unique_idx, inverse = np.unique(index, return_inverse=True)
                        prompt_std = torch.zeros(len(unique_idx), device=group_std.device)
                        for i, idx in enumerate(unique_idx):
                            mask = (inverse == i)
                            prompt_std[i] = group_std[mask][0]

                        # Compute soft weight: weight = prompt_std / (max_std + epsilon)
                        max_std = prompt_std.max()
                        epsilon = 1e-6
                        prompt_weight = prompt_std / (max_std + epsilon)

                        # Broadcast back to batch
                        sample_weight = prompt_weight[torch.from_numpy(inverse).to(group_std.device)]

                        # Apply to advantages (expand to match token dimension)
                        batch.batch["advantages"] = batch.batch["advantages"] * sample_weight.unsqueeze(-1)

                        # Log soft reweight metrics
                        metrics["train/soft_reweight_min"] = prompt_weight.min().item()
                        metrics["train/soft_reweight_max"] = prompt_weight.max().item()
                        metrics["train/soft_reweight_mean"] = prompt_weight.mean().item()

                    # Apply filter loss scaling by scaling advantages
                    # This avoids modifying the actor implementation in the submodule
                    filter_loss_scaling = getattr(self.config.actor_rollout_ref.actor, "filter_loss_scaling", "none")
                    filter_kept_ratio = batch.meta_info.get("filter_kept_ratio", 1.0)
                    if filter_loss_scaling == "linear":
                        batch.batch["advantages"] *= filter_kept_ratio
                    elif filter_loss_scaling == "sqrt":
                        batch.batch["advantages"] *= (filter_kept_ratio ** 0.5)

                ##### A very different setting, just here for testing: Can I normalize the advantages to have a mean of 0?
                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    response_mask = batch.batch["response_mask"]
                    advantages = batch.batch["advantages"]
                    response_relative_lengths = (torch.sum(response_mask, dim=-1) + 1e-6) / torch.sum(response_mask, dim=-1).float().mean()
                    advantages = advantages / response_relative_lengths.unsqueeze(-1) 
                    batch.batch["advantages"] = advantages

                # Task-agnostic ablation: remove task-driven policy gradient by zeroing advantages
                if self.config.algorithm.get("zero_task_advantage", False):
                    batch.batch["advantages"] = torch.zeros_like(batch.batch["advantages"])

                gradient_analysis_only = bool(self.config.trainer.get("gradient_analysis_only", False))
                metrics["trainer/gradient_analysis_only"] = float(gradient_analysis_only)

                # update critic
                if self.use_critic and not gradient_analysis_only:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                elif self.use_critic and gradient_analysis_only:
                    metrics["critic/skipped_for_gradient_analysis_only"] = 1.0

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    gradient_analysis_every = self.config.trainer.get("gradient_analysis_every", 0)
                    if self.config.trainer.get("gradient_analysis_mode", False) and gradient_analysis_every > 0:
                        if (self.global_steps - 1) % gradient_analysis_every == 0:
                            with marked_timer("gradient_analysis", timing_raw):
                                run_gradient_analysis(self, batch, metrics)
                            if self.config.trainer.get("exit_after_gradient_analysis", False):
                                metrics["trainer/exited_after_gradient_analysis"] = 1.0
                                logger.log(data=metrics, step=self.global_steps)
                                _finish_logger()
                                progress_bar.close()
                                return

                    # update actor
                    if not gradient_analysis_only:
                        with marked_timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = True
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                    else:
                        metrics["actor/skipped_for_gradient_analysis_only"] = 1.0

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw):
                        inputs = self.tokenizer.batch_decode(batch.batch["input_ids"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            gts=[""] * len(inputs),
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                    # Success-based early stopping logic
                    for key, value in val_metrics.items():
                        if key.startswith("val-env/") and key.endswith("/success"):
                            if value < 0.01:
                                self.consecutive_low_success[key] += 1
                            else:
                                self.consecutive_low_success[key] = 0
                            
                            if self.consecutive_low_success[key] >= 5:
                                print(f"\n[Early Stopping] Model failed to reach 1% success on {key} for 5 consecutive steps.")
                                self.early_stopped = True
                                self.early_stop_type = "low_validation_success"
                                break
                    
                    if self.early_stopped:
                        metrics.update({self._early_stop_metric_key(): 1.0})
                        logger.log(data=metrics, step=self.global_steps)
                        break

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            eval_time = timing_raw.get("testing", 0.0)
            save_time = timing_raw.get("save_checkpoint", 0.0)
            collapse_time = timing_raw.get("collapse_metrics", 0.0)
            step_time = timing_raw.get("step", 0.0)
            train_time = step_time - eval_time - save_time - collapse_time
            if train_time < 0:
                train_time = 0.0
            self.train_time_total += train_time
            self.eval_time_total += eval_time
            self.collapse_time_total += collapse_time
            collapse_first_turn_step = metrics.get("timing_s/collapse_first_turn_step", 0.0)
            collapse_multi_turn_step = metrics.get("timing_s/collapse_multi_turn_step", 0.0)
            self.collapse_first_turn_time_total += collapse_first_turn_step
            self.collapse_multi_turn_time_total += collapse_multi_turn_step

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            metrics.update({
                "timing_s/train_step": train_time,
                "timing_s/eval_step": eval_time,
                "timing_s/train_total": self.train_time_total,
                "timing_s/eval_total": self.eval_time_total,
                "timing_s/collapse_total": self.collapse_time_total,
                "timing_s/collapse_first_turn_total": self.collapse_first_turn_time_total,
                "timing_s/collapse_multi_turn_total": self.collapse_multi_turn_time_total,
            })
            # add another timing metric: total time
            metrics.update({"timing_s/total": time.time() - self.start_time})
            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                _finish_logger()
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
