"""
Archer Algorithm Trainer for RAGEN
Implements the off-policy Actor-Critic RL algorithm with replay buffer
"""

import json
import os
import uuid
import time
import copy
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from ragen.trainer import core_algos
from ragen.trainer.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]

from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, compute_response_mask, _timer, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger
from ragen.trainer.agent_trainer import RayAgentTrainer, compute_advantage


class RayArcherTrainer(RayAgentTrainer):
    """
    Archer Algorithm Trainer implementing off-policy actor-critic RL with replay buffer.
    Follows RAGEN's data flow but maintains a replay buffer for off-policy learning.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, 
                        ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        
        # Copy attributes from parent that might be needed
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        
        # Define KL control if needed (exactly like RAGEN)
        if config.algorithm.use_kl_in_reward:
            from ragen.trainer import core_algos
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        
        # Archer-specific parameters
        self.replay_buffer_capacity = config.algorithm.get("replay_buffer_capacity", 100000)
        self.critic_epochs = config.algorithm.get("critic_epochs", 3)
        self.actor_epochs = config.algorithm.get("actor_epochs", 3)
        self.warmup_iterations = config.algorithm.get("warmup_iterations", 20)
        self.tau = config.algorithm.get("tau", 0.1)  # soft update parameter
        self.batch_size = config.algorithm.get("batch_size", 32)
        
        # Initialize replay buffer to store DataProto batches
        self.replay_buffer = []
        self.buffer_position = 0
        
        print(f"Initialized Archer Trainer with:")
        print(f"  - Replay buffer capacity: {self.replay_buffer_capacity}")
        print(f"  - Critic epochs: {self.critic_epochs}")
        print(f"  - Actor epochs: {self.actor_epochs}")
        print(f"  - Warmup iterations: {self.warmup_iterations}")
        print(f"  - Tau (soft update): {self.tau}")
        print(f"  - Batch size: {self.batch_size}")

    def _store_batch_in_replay_buffer(self, batch: DataProto):
        """Store a complete batch in the replay buffer"""
        # Create a copy of the batch to store
        batch_copy = copy.deepcopy(batch)
        
        if len(self.replay_buffer) < self.replay_buffer_capacity:
            self.replay_buffer.append(batch_copy)
        else:
            # Replace oldest batch
            self.replay_buffer[self.buffer_position] = batch_copy
            self.buffer_position = (self.buffer_position + 1) % self.replay_buffer_capacity
    
    def _sample_from_replay_buffer(self):
        """Sample a batch from the replay buffer"""
        if len(self.replay_buffer) == 0:
            return None
        
        # For simplicity, sample one random batch from the buffer
        # In a full implementation, you might want to sample individual transitions
        idx = np.random.randint(0, len(self.replay_buffer))
        return self.replay_buffer[idx]
    
    def _compute_archer_advantages(self, batch: DataProto):
        """Compute Archer-style advantages using Q(s,a) - V(s)"""
        if not self.use_critic:
            # Fallback: use token-level rewards as advantages
            return batch.batch["token_level_rewards"]
        
        # Use RAGEN's compute_advantage but with custom logic
        # For now, use the existing advantage computation as a baseline
        # In a full implementation, you would have separate Q and V networks
        
        # Compute values using the critic
        values_output = self.critic_wg.compute_values(batch)
        batch_with_values = batch.union(values_output)
        
        # For Archer, we want Q(s,a) - V(s)
        # Approximate Q(s,a) as V(s) + r (this is a simplification)
        values = batch_with_values.batch["values"]
        rewards = batch.batch["token_level_rewards"]
        
        # Simple Q-value approximation: Q ≈ V + r
        q_values = values + rewards.sum(dim=-1, keepdim=True)
        
        # Advantages = Q(s,a) - V(s)
        advantages = q_values - values
        
        return advantages

    def _process_trajectories_for_buffer(self, batch: DataProto):
        """Convert batch data to trajectory format for replay buffer"""
        trajectories = []
        
        # Get batch data
        batch_size = len(batch.batch["input_ids"])
        
        for i in range(batch_size):
            # Create trajectory for this sample
            trajectory = []
            
            # For simplicity, treat each token as a transition
            # In practice, you might want to use episode-level or turn-level transitions
            input_ids = batch.batch["input_ids"][i]
            response_ids = batch.batch["responses"][i]
            attention_mask = batch.batch["attention_mask"][i]
            response_mask = batch.batch["response_mask"][i]
            rewards = batch.batch["token_level_rewards"][i]
            
            # Create transitions for each token in the response
            response_length = response_mask.sum().item()
            
            for t in range(int(response_length)):
                if response_mask[t] > 0:  # Only for actual response tokens
                    transition = {
                        "observation": input_ids,  # Full input context
                        "action": response_ids,    # Full response
                        "reward": rewards[t].item(),  # Token-level reward
                        "next_observation": input_ids,  # Same for now
                        "done": t == (response_length - 1),  # Last token
                        "token_position": t,
                        "attention_mask": attention_mask,
                        "response_mask": response_mask,
                    }
                    trajectory.append(transition)
            
            if trajectory:  # Only add non-empty trajectories
                trajectories.append(trajectory)
        
        return advantages

    def set_archer_agent(self, archer_agent):
        """Set an ArCHer agent for direct integration with ArCHer components"""
        self.archer_agent = archer_agent
        print("ArCHer agent integration enabled - will use direct Q/V computation")

    def _update_critic(self, batch_data):
        """Update critic networks"""
        if not self.use_critic:
            return {}
        
        # Use the existing critic update mechanism
        # Convert batch_data to proper DataProto format with TensorDict
        batch_dict = {}
        
        # Convert list format back to tensor format for critic
        for key, value in batch_data.items():
            if key in ["observation", "action"] and isinstance(value, list):
                # Stack tensors if they're in list format
                if isinstance(value[0], torch.Tensor):
                    batch_dict[key] = torch.stack(value)
                else:
                    batch_dict[key] = torch.tensor(value)
            else:
                batch_dict[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        
        # Map to expected keys
        if "observation" in batch_dict:
            batch_dict["input_ids"] = batch_dict["observation"]
        if "action" in batch_dict:
            batch_dict["responses"] = batch_dict["action"]
        
        # Add required fields
        batch_dict["attention_mask"] = batch_data.get("attention_mask", torch.ones_like(batch_dict["input_ids"]))
        batch_dict["response_mask"] = batch_data.get("response_mask", torch.ones_like(batch_dict["responses"]))
        batch_dict["token_level_rewards"] = torch.tensor(batch_data.get("reward", [0.0] * len(batch_data["observation"])))
        
        # Create TensorDict and DataProto
        tensor_dict = TensorDict(batch_dict, batch_size=[len(batch_data["observation"])])
        proto_batch = DataProto(batch=tensor_dict)
        
        critic_output = self.critic_wg.update_critic(proto_batch)
        return reduce_metrics(critic_output.meta_info["metrics"])

    def _update_actor(self, batch_data, advantages):
        """Update actor network with computed advantages"""
        # Convert to DataProto format with TensorDict
        batch_dict = {}
        
        # Convert batch data
        for key, value in batch_data.items():
            if key in ["observation", "action"] and isinstance(value, list):
                if isinstance(value[0], torch.Tensor):
                    batch_dict[key] = torch.stack(value)
                else:
                    batch_dict[key] = torch.tensor(value)
            else:
                batch_dict[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        
        # Map to expected keys
        if "observation" in batch_dict:
            batch_dict["input_ids"] = batch_dict["observation"]
        if "action" in batch_dict:
            batch_dict["responses"] = batch_dict["action"]
        
        # Add computed advantages
        batch_dict["advantages"] = advantages.unsqueeze(-1).expand(-1, batch_dict["responses"].size(1))
        batch_dict["returns"] = batch_dict["advantages"]  # For compatibility
        
        # Add required fields
        batch_dict["attention_mask"] = batch_data.get("attention_mask", torch.ones_like(batch_dict["input_ids"]))
        batch_dict["response_mask"] = batch_data.get("response_mask", torch.ones_like(batch_dict["responses"]))
        
        # Create TensorDict and DataProto
        tensor_dict = TensorDict(batch_dict, batch_size=[len(batch_data["observation"])])
        proto_batch = DataProto(batch=tensor_dict, meta_info={"multi_turn": True})
        
        actor_output = self.actor_rollout_wg.update_actor(proto_batch)
        return reduce_metrics(actor_output.meta_info["metrics"])

    def fit(self):
        """
        Main training loop for Archer algorithm.
        Follows RAGEN's agent_proxy.rollout() pattern but uses replay buffer for off-policy learning.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Add tqdm progress bar
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Archer Training Progress")

        # We start from step 1
        self.global_steps += 1
        last_val_metrics = None

        import time
        self.start_time = time.time()
        
        # Define the rollout filtering function (exactly like RAGEN)
        def _filter_rollout(batch):
            """filter rollout based on in-group max - in-group mean. We want those groups to have high-quality rollouts that deviates significantly from the mean"""
            rollout_filter_ratio = self.config.actor_rollout_ref.rollout.rollout_filter_ratio
            num_groups, group_size = self.config.es_manager.train.env_groups, self.config.es_manager.train.group_size

            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
            in_group_std = rm_scores.std(dim=-1)
            in_group_max = rm_scores.max(dim=-1).values
            in_group_mean = rm_scores.mean(dim=-1)
            if rollout_filter_ratio == 1:
                return batch, {"rollout/in_group_std": in_group_std.mean(), "rollout/in_group_max": in_group_max.mean(), "rollout/in_group_mean": in_group_mean.mean(), "rollout/chosen_in_group_std": in_group_std.mean(), "rollout/chosen_in_group_max": in_group_max.mean(), "rollout/chosen_in_group_mean": in_group_mean.mean()}

            if self.config.actor_rollout_ref.rollout.rollout_filter_type == "std_rev":
                top_groups = (-in_group_std).topk(int(rollout_filter_ratio * num_groups)).indices
            elif self.config.actor_rollout_ref.rollout.rollout_filter_type == "std":
                top_groups = in_group_std.topk(int(rollout_filter_ratio * num_groups)).indices
            else:
                raise ValueError(f"Invalid rollout filter type: {self.config.actor_rollout_ref.rollout.rollout_filter_type}")

            mask = torch.zeros(num_groups, dtype=torch.bool)
            mask[top_groups] = True
            mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

            batch.batch = batch.batch[mask]

            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]

            metrics = {
                "rollout/in_group_std": in_group_std.mean(),
                "rollout/in_group_max": in_group_max.mean(),
                "rollout/in_group_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_std": in_group_std[top_groups].mean(),
                "rollout/chosen_in_group_max": in_group_max[top_groups].mean(),
                "rollout/chosen_in_group_mean": in_group_mean[top_groups].mean()
            }
            return batch, metrics
        
        # Follow RAGEN's exact step-based training loop
        for step in range(self.total_training_steps):
            timing_raw = {}

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw):
                # Generate a batch using agent_proxy (exactly like RAGEN)
                with _timer("gen", timing_raw):
                    batch = self.agent_proxy.rollout(batch, val=False)
                    # Apply RAGEN's rollout filtering (exactly like RAGEN)
                    batch, metrics = _filter_rollout(batch)
                    metrics.update({"train/" + key: value for key, value in batch.meta_info["metrics"].items()})

                # Follow ALL RAGEN preprocessing steps exactly
                
                # Handle REMAX if needed (exactly like RAGEN)
                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yet
                    logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()
                    with _timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch, val=False)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # Add UIDs (exactly like RAGEN)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                
                # Set response mask (exactly like RAGEN)
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                
                # Balance batch if needed (exactly like RAGEN)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # Compute global valid tokens (exactly like RAGEN)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Compute reward model score if needed (exactly like RAGEN)
                if self.use_rm:
                    with _timer("reward", timing_raw):
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                # Compute rewards (exactly like RAGEN)
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # Recompute old log probs (exactly like RAGEN)
                with _timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)
                    avg_old_log_prob = masked_mean(old_log_prob.batch["old_log_probs"], batch.batch["response_mask"])
                    metrics.update({"rollout/old_log_prob": avg_old_log_prob})

                # Compute reference log prob if needed (exactly like RAGEN)
                if self.use_reference_policy:
                    with _timer("ref", timing_raw):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                # Compute values (exactly like RAGEN)
                if self.use_critic:
                    with _timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                # Process rewards and advantages (exactly like RAGEN)
                with _timer("adv", timing_raw):
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
                    )

                ##### A very different setting, just here for testing: Can I normalize the advantages to have a mean of 0?
                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    response_mask = batch.batch["response_mask"]
                    advantages = batch.batch["advantages"]
                    response_relative_lengths = (torch.sum(response_mask, dim=-1) + 1e-6) / torch.sum(response_mask, dim=-1).float().mean()
                    advantages = advantages / response_relative_lengths.unsqueeze(-1) 
                    batch.batch["advantages"] = advantages
                # ARCHER SPECIFIC: Store batch in replay buffer
                with _timer("store_replay", timing_raw):
                    self._store_batch_in_replay_buffer(batch)
                    metrics["replay_buffer/size"] = len(self.replay_buffer)

                # ARCHER SPECIFIC: Train from replay buffer (only after we have some data)
                if len(self.replay_buffer) > 0:
                    # Update critic multiple times
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_metrics_list = []
                            for epoch_idx in range(self.critic_epochs):
                                replay_batch = self._sample_from_replay_buffer()
                                if replay_batch is not None:
                                    # Ensure replay batch has values computed for critic update
                                    if "values" not in replay_batch.batch:
                                        values_output = self.critic_wg.compute_values(replay_batch)
                                        replay_batch = replay_batch.union(values_output)
                                    
                                    critic_output = self.critic_wg.update_critic(replay_batch)
                                    critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                    critic_metrics_list.append(critic_metrics)
                            
                            # Average critic metrics
                            if critic_metrics_list:
                                for key in critic_metrics_list[0].keys():
                                    metrics[f"critic/{key}"] = np.mean([m[key] for m in critic_metrics_list])

                    # Update actor (only after warmup)
                    if self.global_steps > self.warmup_iterations:
                        with _timer("update_actor", timing_raw):
                            actor_metrics_list = []
                            for epoch_idx in range(self.actor_epochs):
                                replay_batch = self._sample_from_replay_buffer()
                                if replay_batch is not None:
                                    # Ensure replay batch has all required fields for actor update
                                    if "values" not in replay_batch.batch and self.use_critic:
                                        values_output = self.critic_wg.compute_values(replay_batch)
                                        replay_batch = replay_batch.union(values_output)
                                    
                                    # Compute Archer advantages
                                    archer_advantages = self._compute_archer_advantages(replay_batch)
                                    replay_batch.batch["advantages"] = archer_advantages
                                    replay_batch.batch["returns"] = archer_advantages  # For compatibility
                                    
                                    # Update actor
                                    replay_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    actor_output = self.actor_rollout_wg.update_actor(replay_batch)
                                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                    actor_metrics_list.append(actor_metrics)
                            
                            # Average actor metrics
                            if actor_metrics_list:
                                for key in actor_metrics_list[0].keys():
                                    metrics[f"actor/{key}"] = np.mean([m[key] for m in actor_metrics_list])

                # Validate (exactly like RAGEN)
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                # Save checkpoint (exactly like RAGEN)
                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # Collect metrics (like RAGEN)
            metrics.update({"training/global_step": self.global_steps, "training/epoch": step})
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # Add total time metric
            metrics.update({"timing_s/total": time.time() - self.start_time})
            
            # Log metrics
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

    def _save_checkpoint(self):
        """Save checkpoint including replay buffer"""
        # Call parent's checkpoint saving logic
        super()._save_checkpoint()
        
        # Save Archer-specific data (replay buffer)
        if hasattr(self, 'global_steps'):
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
            replay_buffer_path = os.path.join(local_global_step_folder, "archer_replay_buffer.pt")
            
            # Create directory if it doesn't exist
            os.makedirs(local_global_step_folder, exist_ok=True)
            
            torch.save(self.replay_buffer, replay_buffer_path)
            print(f"Saved Archer replay buffer to {replay_buffer_path}")

    def _load_checkpoint(self):
        """Load checkpoint including replay buffer"""
        # Call parent's checkpoint loading logic
        super()._load_checkpoint()
        
        # Try to load Archer-specific data (replay buffer)
        if hasattr(self, 'global_steps') and self.global_steps > 0:
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
            replay_buffer_path = os.path.join(local_global_step_folder, "archer_replay_buffer.pt")
            
            if os.path.exists(replay_buffer_path):
                self.replay_buffer = torch.load(replay_buffer_path, weights_only=False)
                print(f"Loaded Archer replay buffer with {len(self.replay_buffer)} batches")
