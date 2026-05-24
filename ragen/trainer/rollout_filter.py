"""Utilities for filtering rollout trajectories before PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from verl import DataProto


@dataclass
class RolloutFilterConfig:
    """Configuration container for rollout filtering."""

    value: float
    filter_type: str
    group_size: int
    num_groups: int
    metric: str = "reward_variance"
    include_zero: bool = True
    strategy: str = "top_p"
    top_p_prob_mode: str = "linear"
    selection_eps: float = 0.01
    bucket_count: int = 6
    bucket_mode: str = "quantile"


class RolloutFilter:
    """Base class for rollout filters."""

    def __init__(self, config: RolloutFilterConfig):
        self.config = config

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @property
    def value(self) -> float:
        return self.config.value

    @property
    def strategy(self) -> str:
        return self.config.strategy

    @property
    def filter_type(self) -> str:
        return self.config.filter_type

    @property
    def group_size(self) -> int:
        return self.config.group_size

    @property
    def num_groups(self) -> int:
        return self.config.num_groups

    @property
    def bucket_count(self) -> int:
        return self.config.bucket_count

    @property
    def bucket_mode(self) -> str:
        return self.config.bucket_mode

    def _select_top_groups(self, scores: torch.Tensor) -> torch.Tensor:
        # Convert to float for safety with epsilon
        scores = scores.float()
        indices = torch.arange(self.num_groups, device=scores.device)

        # Handle zero exclusion
        if not self.config.include_zero:
            non_zero_mask = (torch.abs(scores) > 1e-10)
            scores = scores[non_zero_mask]
            indices = indices[non_zero_mask]
            
            if indices.numel() == 0:
                return torch.tensor([], dtype=torch.long, device=scores.device)

        # Regular ratio logic

        # Regular ratio logic
        # Strategy dispatch
        if self.strategy == "top_p":
            if self.value >= 1.0:
                return indices

            # Nucleus-like filtering with two score aggregation modes.
            if self.config.top_p_prob_mode == "softmax":
                if self.filter_type == "largest":
                    logits = scores
                elif self.filter_type == "smallest":
                    logits = -scores
                else:
                    raise ValueError(f"Invalid rollout filter type: {self.filter_type}")
                probs = torch.softmax(logits, dim=0)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff_index = torch.searchsorted(cumulative_probs, self.value).item()
                k = cutoff_index + 1
                k = min(k, indices.numel())
                top_groups_local_indices = sorted_indices[:k]
                return indices[top_groups_local_indices]

            elif self.config.top_p_prob_mode == "linear":
                if self.filter_type == "largest":
                    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                elif self.filter_type == "smallest":
                    sorted_scores, sorted_indices = torch.sort(scores, descending=False)
                else:
                    raise ValueError(f"Invalid rollout filter type: {self.filter_type}")

                threshold = self.value * scores.sum() - self.config.selection_eps
                cumulative_score = 0.0
                selected_count = 0

                for score in sorted_scores:
                    if cumulative_score >= threshold:
                        break
                    if score.item() <= 0:
                        break
                    cumulative_score += score.item()
                    selected_count += 1

                if cumulative_score >= threshold:
                    top_groups_local_indices = sorted_indices[:selected_count]
                    return indices[top_groups_local_indices]
                return torch.empty(0, dtype=torch.long, device=indices.device)
            else:
                raise ValueError(
                    f"Unknown top_p_prob_mode: {self.config.top_p_prob_mode}. "
                    "Expected one of {'linear', 'softmax'}."
                )

        elif self.strategy == "top_k":
            # top-k: choose top k fraction of the groups
            k = int(self.config.value * self.num_groups)
            k = min(k, indices.numel())
            k = max(k, 1) # Ensure at least 1
            
            if self.filter_type == "smallest":
                top_groups_local_indices = (-scores).topk(k).indices
            elif self.filter_type == "largest":
                top_groups_local_indices = scores.topk(k).indices
            else:
                raise ValueError(f"Invalid rollout filter type: {self.filter_type}")
            
            return indices[top_groups_local_indices]

        elif self.strategy == "top_k_abs":
            k = int(self.config.value)
            k = min(k, indices.numel())
            k = max(k, 1) # Ensure at least 1
            
            if self.filter_type == "smallest":
                top_groups_local_indices = (-scores).topk(k).indices
            elif self.filter_type == "largest":
                top_groups_local_indices = scores.topk(k).indices
            else:
                raise ValueError(f"Invalid rollout filter type: {self.filter_type}")
            
            return indices[top_groups_local_indices]

        elif self.strategy == "min_p":
            if self.filter_type == "largest":
                max_score = scores.max()
                threshold = max_score * self.config.value
                mask = scores >= (threshold - 1e-10)
            elif self.filter_type == "smallest":
                min_score = scores.min()
                # For smallest, we keep groups whose score is "close enough" to the minimum.
                # If value=0.5 and min=0.1, threshold=0.2. We keep scores <= 0.2.
                threshold = min_score / (self.config.value + 1e-10)
                mask = scores <= (threshold + 1e-10)
            else:
                raise ValueError(f"Invalid rollout filter type: {self.filter_type}")
            
            return indices[mask]

            return indices[mask]
            
        else:
             raise ValueError(f"Unknown strategy: {self.strategy}")

    def _groups_to_mask(self, top_groups: torch.Tensor, group_size: int = None) -> torch.Tensor:
        device = top_groups.device
        if group_size is None:
            group_size = self.group_size
        mask = torch.zeros(self.num_groups, dtype=torch.bool, device=device)
        if top_groups.numel() > 0:
            mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, group_size).reshape(-1).cpu()
        return mask

    def _apply_mask(self, batch: DataProto, mask: torch.Tensor) -> DataProto:
        batch.batch = batch.batch[mask]

        if batch.non_tensor_batch is not None:
            np_mask = mask.cpu().numpy()
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[np_mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, np_mask) if m]

        return batch

    def _selected_mean(self, values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if selected.numel() == 0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return values[selected].mean()

    def _build_base_metrics(
        self,
        in_group_std: torch.Tensor,
        in_group_max: torch.Tensor,
        in_group_mean: torch.Tensor,
        top_groups: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        metrics = {
            "rollout/in_group_std": in_group_std.mean(),
            "rollout/in_group_max": in_group_max.mean(),
            "rollout/in_group_mean": in_group_mean.mean(),
        }

        chosen = top_groups
        metrics.update(
            {
                "rollout/chosen_in_group_std": self._selected_mean(in_group_std, chosen),
                "rollout/chosen_in_group_max": self._selected_mean(in_group_max, chosen),
                "rollout/chosen_in_group_mean": self._selected_mean(in_group_mean, chosen),
            }
        )
        return metrics


class LengthRolloutFilter(RolloutFilter):
    """Filters rollouts based on response length within groups."""

    _METRIC_OPTIONS = {"length"}

    def __init__(self, config: RolloutFilterConfig) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"LengthRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        num_groups = self.num_groups
        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            response_mask = batch.batch.get("loss_mask")
        if response_mask is None:
            raise ValueError("LengthRolloutFilter requires response_mask or loss_mask in the batch")

        # Calculate length per trajectory
        length_per_traj = response_mask.sum(dim=-1).float()

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]
            unique_episodes = []
            episode_to_indices = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_indices:
                    unique_episodes.append(eid)
                    episode_to_indices[eid] = []
                episode_to_indices[eid].append(i)

            # Get episode-level length (sum of all turns)
            num_episodes = len(unique_episodes)
            episode_length = torch.zeros(num_episodes, device=length_per_traj.device)
            for i, eid in enumerate(unique_episodes):
                indices = episode_to_indices[eid]
                episode_length[i] = length_per_traj[indices].sum()

            group_size = num_episodes // num_groups
            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )

            length_per_group = episode_length.view(num_groups, group_size)
        else:
            actual_batch_size = length_per_traj.shape[0]
            group_size = actual_batch_size // num_groups
            length_per_group = length_per_traj.view(num_groups, group_size)

        in_group_std = length_per_group.std(dim=-1)
        in_group_max = length_per_group.max(dim=-1).values
        in_group_mean = length_per_group.mean(dim=-1)

        # For length, we usually use mean length for filtering
        selection_scores = in_group_mean
        top_groups = self._select_top_groups(selection_scores)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_length_std": in_group_std.mean(),
                "rollout/in_group_length_max": in_group_max.mean(),
                "rollout/in_group_length_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_length_std": self._selected_mean(in_group_std, top_groups),
                "rollout/chosen_in_group_length_max": self._selected_mean(in_group_max, top_groups),
                "rollout/chosen_in_group_length_mean": self._selected_mean(in_group_mean, top_groups),
                "rollout/filter_kept_count": torch.tensor(float(top_groups.numel())),
                "rollout/filter_kept_ratio": torch.tensor(top_groups.numel() / self.num_groups),
                "rollout/filter_zero_count": (torch.abs(selection_scores) <= 1e-10).sum(),
            }
        )

        if self.strategy == "top_p" and self.config.value >= 1 and self.config.include_zero:
            return batch, metrics

        if has_episode_ids:
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)
        return batch, metrics


class RewardRolloutFilter(RolloutFilter):
    """Filters rollouts based on reward statistics within groups."""

    _METRIC_OPTIONS = {"reward", "reward_sum", "reward_variance"}

    def __init__(self, config: RolloutFilterConfig) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"RewardRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )

    def _selection_scores(
        self,
        in_group_std: torch.Tensor,
        in_group_mean: torch.Tensor,
        in_group_sum: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.metric == "reward":
            return in_group_mean
        if self.config.metric == "reward_sum":
            return in_group_sum
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        num_groups = self.num_groups

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]
            group_ids = batch.non_tensor_batch["group_ids"]
            all_scores = batch.batch["original_rm_scores"].sum(dim=-1)

            # Get unique episodes and their rewards
            unique_episodes = []
            episode_to_first_idx = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_first_idx:
                    unique_episodes.append(eid)
                    episode_to_first_idx[eid] = i

            # Get episode-level rewards and group_ids
            num_episodes = len(unique_episodes)
            episode_rewards = torch.zeros(num_episodes, device=all_scores.device)
            episode_group_ids = []
            for i, eid in enumerate(unique_episodes):
                idx = episode_to_first_idx[eid]
                episode_rewards[i] = all_scores[idx]
                episode_group_ids.append(group_ids[idx])

            # Calculate group_size as episodes per group
            group_size = num_episodes // num_groups
            
            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )
            
            # Reshape to (num_groups, group_size)
            rm_scores = episode_rewards.view(num_groups, group_size)
            group_ids_for_metrics = torch.as_tensor(
                episode_group_ids,
                device=rm_scores.device,
            ).view(num_groups, group_size)[:, 0]
        else:
            # Original mode: each sample is an episode
            actual_batch_size = batch.batch["original_rm_scores"].shape[0]
            group_size = actual_batch_size // num_groups
            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
            if batch.non_tensor_batch is not None and "group_ids" in batch.non_tensor_batch:
                raw_group_ids = batch.non_tensor_batch["group_ids"]
                if not torch.is_tensor(raw_group_ids):
                    raw_group_ids = torch.as_tensor(raw_group_ids, device=rm_scores.device)
                else:
                    raw_group_ids = raw_group_ids.to(rm_scores.device)
                group_ids_for_metrics = raw_group_ids.view(num_groups, group_size)[:, 0]
            else:
                group_ids_for_metrics = torch.arange(num_groups, device=rm_scores.device)

        in_group_std = rm_scores.std(dim=-1)
        in_group_max = rm_scores.max(dim=-1).values
        in_group_mean = rm_scores.mean(dim=-1)
        in_group_sum = rm_scores.sum(dim=-1)

        selection_scores = self._selection_scores(in_group_std, in_group_mean, in_group_sum)
        top_groups = self._select_top_groups(selection_scores)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_reward_std": in_group_std.mean(),
                "rollout/in_group_reward_max": in_group_max.mean(),
                "rollout/in_group_reward_mean": in_group_mean.mean(),
                "rollout/in_group_reward_sum": in_group_sum.mean(),
                "rollout/chosen_in_group_reward_std": self._selected_mean(in_group_std, top_groups),
                "rollout/chosen_in_group_reward_max": self._selected_mean(in_group_max, top_groups),
                "rollout/chosen_in_group_reward_mean": self._selected_mean(in_group_mean, top_groups),
                "rollout/chosen_in_group_reward_sum": self._selected_mean(in_group_sum, top_groups),
                "rollout/filter_kept_count": torch.tensor(float(top_groups.numel())),
                "rollout/filter_kept_ratio": torch.tensor(top_groups.numel() / self.num_groups),
                "rollout/filter_zero_count": (torch.abs(selection_scores) <= 1e-10).sum(),
            }
        )

        metrics["rollout/_reward_matrix"] = rm_scores.detach().cpu()
        metrics["rollout/_group_reward_std"] = in_group_std.detach().cpu()
        metrics["rollout/_group_ids"] = group_ids_for_metrics.detach().cpu()
        metrics["rollout/_selected_group_ids"] = group_ids_for_metrics[top_groups].detach().cpu()

        if self.strategy == "top_p" and self.config.value >= 1 and self.config.include_zero:
            # Attach reward std to batch even if not filtering.
            reward_std_per_sample = in_group_std.unsqueeze(1).expand(-1, group_size).reshape(-1)
            if has_episode_ids:
                # Expand episode-level std to turn-level.
                episode_id_to_idx = {eid: idx for idx, eid in enumerate(unique_episodes)}
                turn_ep_indices = torch.tensor(
                    [episode_id_to_idx[eid] for eid in episode_ids],
                    dtype=torch.long, device=reward_std_per_sample.device
                )
                batch.batch["reward_std"] = reward_std_per_sample[turn_ep_indices]
            else:
                batch.batch["reward_std"] = reward_std_per_sample
            return batch, metrics

        if has_episode_ids:
            # Build mask for turn-level samples based on selected groups
            # First, find which episodes belong to selected groups
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            # Build turn-level mask
            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)

        # Re-compute reward std for kept samples to ensure alignment.
        # reward_std_per_sample is episode-level: shape [num_episodes].
        reward_std_per_sample = in_group_std.unsqueeze(1).expand(-1, group_size).reshape(-1)

        if has_episode_ids:
            # mask is turn-level [num_turns]; reward_std_per_sample is episode-level [num_episodes].
            # Expand reward_std to turn-level by mapping each turn's episode_id to its episode index.
            episode_id_to_idx = {eid: idx for idx, eid in enumerate(unique_episodes)}
            turn_ep_indices = torch.tensor(
                [episode_id_to_idx[eid] for eid in episode_ids],
                dtype=torch.long, device=reward_std_per_sample.device
            )
            reward_std_filtered = reward_std_per_sample[turn_ep_indices][mask]
        else:
            reward_std_filtered = reward_std_per_sample[mask]
        
        batch.batch["reward_std"] = reward_std_filtered

        return batch, metrics

    def split_into_buckets(self, batch: DataProto) -> Dict[str, DataProto]:
        """Splits the batch into variance buckets for gradient analysis."""
        if "reward_std" not in batch.batch:
             raise ValueError("Batch must have 'reward_std' to split into buckets.")
        
        reward_std = batch.batch["reward_std"]

        total_samples = reward_std.numel()
        if batch.non_tensor_batch is not None and "group_ids" in batch.non_tensor_batch:
            group_ids = batch.non_tensor_batch["group_ids"]
            if not torch.is_tensor(group_ids):
                group_ids = torch.tensor(group_ids, device=reward_std.device)
            else:
                group_ids = group_ids.to(reward_std.device)

            unique_group_ids, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
            num_groups = unique_group_ids.numel()
            reward_std_per_group = torch.zeros(num_groups, device=reward_std.device)
            counts = torch.zeros(num_groups, device=reward_std.device)
            reward_std_per_group.scatter_add_(0, inverse, reward_std)
            counts.scatter_add_(0, inverse, torch.ones_like(reward_std))
            reward_std_per_group = reward_std_per_group / counts.clamp_min(1)
        else:
            group_size = self.group_size
            if total_samples % group_size != 0:
                raise ValueError(
                    f"Batch size ({total_samples}) must be divisible by group_size ({group_size}) for bucketization."
                )
            num_groups = total_samples // group_size
            reward_std_per_group = reward_std.view(num_groups, group_size).mean(dim=1)

        buckets_masks = {"all": torch.ones_like(reward_std, dtype=torch.bool)}
        bucket_group_masks = {"all": torch.ones(num_groups, dtype=torch.bool, device=reward_std.device)}
        bucket_group_indices = {"all": torch.arange(num_groups, device=reward_std.device)}

        if self.bucket_mode == "fixed_rv":
            # Fixed RV gaps: [0,1), [1,2), [2,3), [3,4), [4,5), [5, +inf)
            rv_edges = [0, 1, 2, 3, 4, 5]
            for i, low in enumerate(rv_edges):
                high = rv_edges[i + 1] if i + 1 < len(rv_edges) else None
                name = f"bucket_{i + 1}"
                if high is None:
                    group_mask = reward_std_per_group >= low
                else:
                    group_mask = (reward_std_per_group >= low) & (reward_std_per_group < high)
                if batch.non_tensor_batch is not None and "group_ids" in batch.non_tensor_batch:
                    sample_mask = group_mask[inverse]
                else:
                    sample_mask = group_mask.unsqueeze(1).expand(-1, group_size).reshape(-1)
                buckets_masks[name] = sample_mask
                bucket_group_masks[name] = group_mask
                bucket_group_indices[name] = torch.where(group_mask)[0]
        else:
            # Equal-percentage buckets by groups. If there are fewer kept groups than the
            # requested bucket count, reduce the effective bucket count so each bucket has
            # at least one group. Remainder is still assigned to the last bucket.
            num_buckets = min(self.bucket_count, num_groups) if num_groups > 0 else 0
            groups_per_bucket = num_groups // num_buckets if num_buckets > 0 else 0
            remainder = num_groups % num_buckets if num_buckets > 0 else 0
            sorted_group_ids = torch.argsort(reward_std_per_group)

            if 0 < num_buckets < self.bucket_count:
                print(
                    f"[Gradient Analysis] Reducing quantile bucket count from {self.bucket_count} "
                    f"to {num_buckets} because only {num_groups} groups were kept."
                )

            start = 0
            for i in range(num_buckets):
                size = groups_per_bucket + (remainder if i == num_buckets - 1 else 0)
                end = start + size
                name = f"bucket_{i + 1}"

                if size == 0:
                    buckets_masks[name] = torch.zeros_like(reward_std, dtype=torch.bool)
                    bucket_group_masks[name] = torch.zeros(num_groups, dtype=torch.bool, device=reward_std.device)
                    continue

                group_ids = sorted_group_ids[start:end]
                group_mask = torch.zeros(num_groups, dtype=torch.bool, device=reward_std.device)
                group_mask[group_ids] = True
                if batch.non_tensor_batch is not None and "group_ids" in batch.non_tensor_batch:
                    sample_mask = group_mask[inverse]
                else:
                    sample_mask = group_mask.unsqueeze(1).expand(-1, group_size).reshape(-1)
                buckets_masks[name] = sample_mask
                bucket_group_masks[name] = group_mask
                bucket_group_indices[name] = group_ids
                start = end

        result = {}
        
        print(f"\n[Gradient Analysis] Bucket Distribution (Total Samples: {total_samples})")
        print("-" * 80)
        print(f"{'Bucket':<20} | {'Count':<10} | {'Percentage':<12} | {'Avg Reward Std':<15}")
        print("-" * 80)
        
        for name, mask in buckets_masks.items():
            count = mask.sum().item()
            if count > 0:
                percentage = (count / total_samples) * 100
                group_mask = bucket_group_masks[name]
                avg_std = reward_std_per_group[group_mask].mean().item()
                group_ids_for_bucket = bucket_group_indices[name]
                bucket_rv_values = reward_std_per_group[group_ids_for_bucket].detach().cpu().tolist()
                if batch.non_tensor_batch is not None and "group_ids" in batch.non_tensor_batch:
                    bucket_group_ids = unique_group_ids[group_ids_for_bucket].detach().cpu().tolist()
                else:
                    bucket_group_ids = group_ids_for_bucket.detach().cpu().tolist()
                print(f"{name:<20} | {count:<10} | {percentage:>10.2f}% | {avg_std:>12.4f}")
                
                try:
                    subset = batch[mask]
                except Exception:
                    subset = batch[mask]
                subset.meta_info = dict(subset.meta_info or {})
                subset.meta_info["bucket_reward_std_mean"] = avg_std
                subset.meta_info["bucket_reward_std_values"] = bucket_rv_values
                subset.meta_info["bucket_group_ids"] = bucket_group_ids
                result[name] = subset
            else:
                if name == "all":
                    print(f"{name:<20} | {count:<10} | {0:>10.2f}% | {'N/A':>12}")
        
        print("-" * 80 + "\n")
        
        return result


class EntropyRolloutFilter(RolloutFilter):
    """Filters rollouts based on policy entropy statistics within groups."""

    _METRIC_OPTIONS = {"entropy", "entropy_variance"}

    def __init__(
        self,
        config: RolloutFilterConfig,
        compute_log_prob: Callable[[DataProto], DataProto],
    ) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"EntropyRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )
        self._compute_log_prob = compute_log_prob

    def _selection_scores(
        self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor
    ) -> torch.Tensor:
        if self.config.metric == "entropy":
            return in_group_mean
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        num_groups = self.num_groups

        if "entropys" not in batch.batch:
            log_prob = self._compute_log_prob(batch)
            # Only keep entropys to avoid conflicts with later logprob recomputation in agent_trainer.py
            batch.batch["entropys"] = log_prob.batch["entropys"]

        entropys = batch.batch["entropys"]
        loss_mask = batch.batch.get("loss_mask")
        if loss_mask is None:
            loss_mask = batch.batch.get("response_mask")
        if loss_mask is None:
            raise ValueError("EntropyRolloutFilter requires loss_mask or response_mask in the batch")

        loss_mask = loss_mask.to(entropys.device)
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        entropy_per_traj = (entropys * loss_mask).sum(dim=-1) / token_counts

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]

            # Get unique episodes and their entropy (average across turns)
            unique_episodes = []
            episode_to_indices = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_indices:
                    unique_episodes.append(eid)
                    episode_to_indices[eid] = []
                episode_to_indices[eid].append(i)

            # Get episode-level entropy (mean of all turns)
            num_episodes = len(unique_episodes)
            episode_entropy = torch.zeros(num_episodes, device=entropy_per_traj.device)
            for i, eid in enumerate(unique_episodes):
                indices = episode_to_indices[eid]
                episode_entropy[i] = entropy_per_traj[indices].mean()

            # Calculate group_size as episodes per group
            group_size = num_episodes // num_groups

            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )

            # Reshape to (num_groups, group_size)
            entropy_per_group = episode_entropy.view(num_groups, group_size)
        else:
            # Original mode: each sample is an episode
            actual_batch_size = entropy_per_traj.shape[0]
            group_size = actual_batch_size // num_groups
            entropy_per_group = entropy_per_traj.view(num_groups, group_size)

        in_group_std = entropy_per_group.std(dim=-1)
        in_group_max = entropy_per_group.max(dim=-1).values
        in_group_mean = entropy_per_group.mean(dim=-1)

        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_entropy_std": in_group_std.mean(),
                "rollout/in_group_entropy_max": in_group_max.mean(),
                "rollout/in_group_entropy_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_entropy_std": self._selected_mean(in_group_std, top_groups),
                "rollout/chosen_in_group_entropy_max": self._selected_mean(in_group_max, top_groups),
                "rollout/chosen_in_group_entropy_mean": self._selected_mean(in_group_mean, top_groups),
                "rollout/filter_kept_count": torch.tensor(float(top_groups.numel())),
                "rollout/filter_kept_ratio": torch.tensor(top_groups.numel() / self.num_groups),
                "rollout/filter_zero_count": (torch.abs(selection_scores) <= 1e-10).sum(),
            }
        )

        if self.strategy == "top_p" and self.config.value >= 1 and self.config.include_zero:
            return batch, metrics

        if has_episode_ids:
            # Build mask for turn-level samples based on selected groups
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            # Build turn-level mask
            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)

        return batch, metrics


# Backwards compatibility: preserve older class names.
RewardVarianceRolloutFilter = RewardRolloutFilter
EntropyVarianceRolloutFilter = EntropyRolloutFilter


def build_rollout_filter(
    value: float,
    filter_type: str,
    num_groups: int,
    group_size: int,
    metric: Optional[str],
    compute_log_prob: Optional[Callable[[DataProto], DataProto]] = None,
    include_zero: bool = True,
    strategy: str = "top_p",
    top_p_prob_mode: str = "linear",
    selection_eps: float = 0.01,
    bucket_count: int = 6,
    bucket_mode: str = "quantile",
) -> RolloutFilter:
    metric = (metric or "reward_variance").lower()
    metric = {
        "reward_std": "reward_variance",
        "entropy_std": "entropy_variance",
    }.get(metric, metric)

    config = RolloutFilterConfig(
        value=value,
        filter_type=filter_type,
        num_groups=num_groups,
        group_size=group_size,
        metric=metric,
        include_zero=include_zero,
        strategy=strategy,
        top_p_prob_mode=top_p_prob_mode,
        selection_eps=selection_eps,
        bucket_count=bucket_count,
        bucket_mode=bucket_mode,
    )

    if metric == "length":
        return LengthRolloutFilter(config)
    if metric in {"reward", "reward_sum", "reward_variance"}:
        return RewardRolloutFilter(config)
    if metric in {"entropy", "entropy_variance"}:
        if compute_log_prob is None:
            raise ValueError("Entropy filtering requires a compute_log_prob callable")
        return EntropyRolloutFilter(config, compute_log_prob=compute_log_prob)

    raise ValueError(f"Unsupported rollout filter metric: {metric}")
