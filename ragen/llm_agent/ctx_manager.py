"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest
import logging

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
from ragen.llm_agent.memory.factory import create_memory
from ragen.llm_agent.memory.base import BaseMemory
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    if "qwen" in tokenizer.name_or_path.lower():
        special_token = tokenizer.encode("<|im_start|>")[0]
        reward_token = tokenizer.encode("<|im_end|>")[0]
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1].float() # remove the last token
    response_mask = response_mask[:, :-1].float() # remove the last token

    return score_tensor, loss_mask, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
        self._env_type_counts = self._compute_env_type_counts()

        # Initialize memory managers per environment type
        self._memory_managers: Dict[str, BaseMemory] = {}

    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            observation_format = env_config_new.get("observation_format", "grid")
            if observation_format == "grid" and env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if observation_format == "coord":
                coord_hint = (
                    "\nStates are provided as coordinate lists using zero-based indexing "
                    "with the format (row, col) for each entity type: Walls, Targets, Boxes, "
                    "Boxes on target, Player, and Player on target when applicable."
                )
                env_instruction += coord_hint
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {
                'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length),
                'action_lookup': env_config_new.get("action_lookup", {}),
                'env_type': env_config.env_type,
            }

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _compute_env_type_counts(self) -> Dict[str, int]:
        env_type_counts: Dict[str, int] = {}
        for env_config in self.env_config_lookup.values():
            env_type = env_config.get("env_type", "default")
            env_type_counts[env_type] = env_type_counts.get(env_type, 0) + 1
        return env_type_counts

    def reset_memory_managers(self) -> None:
        """Reset memory managers for a new batch of environments."""
        for env_type, count in self._env_type_counts.items():
            if env_type not in self._memory_managers:
                self._memory_managers[env_type] = create_memory(env_type)
            memory_manager = self._memory_managers[env_type]
            if count > 0:
                memory_manager.reset(count)

    def _get_memory_manager(self, env_id: int) -> BaseMemory:
        """Get or create memory manager for environment."""
        env_config = self.env_config_lookup.get(env_id, {})
        env_type = env_config.get("env_type", "default")

        if env_type not in self._memory_managers:
            self._memory_managers[env_type] = create_memory(env_type)
        memory_manager = self._memory_managers[env_type]
        env_type_count = self._env_type_counts.get(env_type, 0)
        if getattr(memory_manager, "batch_size", 0) == 0 and env_type_count > 0:
            memory_manager.reset(env_type_count)
        return memory_manager

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        think_coord_fmt_ok = False
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
            think_len = -1  # no valid format, don't penalize think separately
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)


            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
            think_len = len(think_content)
            # Check if think starts with coordinate format: "Player: (r, c) | Box: (r, c) | Target: (r, c)"
            think_coord_fmt_ok = bool(re.match(r'\s*Player:\s*\(\d+,\s*\d+\)', think_content))
        return llm_response, actions, think_len, think_coord_fmt_ok
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor

    def _resolve_max_context_window(self, max_context_window: Optional[int] = None) -> Optional[int]:
        if max_context_window is None:
            max_context_window = getattr(self.config.agent_proxy, "max_context_window", 1)
        if max_context_window is None:
            return None
        if max_context_window < 0:
            return None
        return int(max_context_window)

    def _get_history_start(self, turn_idx: int, max_context_window: Optional[int]) -> int:
        if max_context_window is None:
            return 0
        return max(0, turn_idx - max_context_window + 1)

    # ==================== Shared Infrastructure Functions ====================

    def _extract_history(self, env_output: Dict, prepare_for_update: bool) -> List[Dict]:
        """Extract history from env_output, removing trailing state-only entry for update."""
        history = env_output['history']
        if prepare_for_update and history and 'state' in history[-1] and 'llm_response' not in history[-1]:
            return history[:-1]
        return history

    def _normalize_episode_rewards(
        self,
        env_outputs: List[Dict],
        episode_raw_rewards: List[float],
        episode_penalties: List[float],
    ) -> torch.Tensor:
        """
        Normalize episode rewards based on grouping and method config.

        Returns:
            Tensor of normalized rewards (one per episode)
        """
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method

        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")

        if method == "mean_std":
            norm_func = lambda x: (x - x.mean()) / (x.std() + 1e-6) if x.std().abs() > 1e-6 else torch.zeros_like(x)
        elif method == "mean":
            norm_func = lambda x: (x - x.mean())
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean()) / (x.std() + 1e-6) if x.std().abs() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        episode_rewards_tensor = torch.tensor(episode_raw_rewards, dtype=torch.float32)
        episode_penalties_tensor = torch.tensor(episode_penalties, dtype=torch.float32)
        normalized = episode_rewards_tensor + episode_penalties_tensor

        group2index = {}
        for i, tag in enumerate(group_tags):
            if tag not in group2index:
                group2index[tag] = []
            group2index[tag].append(i)

        if len(group2index) < len(env_outputs):
            for group, indices in group2index.items():
                indices_tensor = torch.tensor(indices)
                normalized[indices_tensor] = norm_func(normalized[indices_tensor])

        return normalized

    def _tokenize_and_build_tensors(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize texts and build input tensors.

        Returns:
            (input_ids, attention_mask, position_ids)
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        return input_ids, attention_mask, position_ids

    def _build_dataproto(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        score_tensor: torch.Tensor,
        env_ids: List[int],
        group_ids: List[int],
        messages_list: List[List[Dict]],
        episode_ids: Optional[List[int]] = None,
        uid_list: Optional[List[Any]] = None,
    ) -> DataProto:
        """Build DataProto with common structure for all modes."""
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],
            "loss_mask": loss_mask,
            "rm_scores": score_tensor,
            "original_rm_scores": score_tensor.clone(),
        }, batch_size=input_ids.shape[0])

        non_tensor = {
            "env_ids": np.array(env_ids, dtype=int),
            "group_ids": np.array(group_ids, dtype=int),
            "messages_list": np.array(messages_list, dtype=object),
        }
        if episode_ids is not None:
            non_tensor["episode_ids"] = np.array(episode_ids, dtype=int)
        if uid_list is not None:
            non_tensor["uid"] = np.array(uid_list, dtype=object)

        llm_inputs.non_tensor_batch = non_tensor
        return llm_inputs

    def _compute_metrics(
        self,
        env_outputs: List[Dict],
        response_length: float,
    ) -> Dict[str, float]:
        """Compute aggregated metrics from env_outputs."""
        metrics = {}
        for env_output in env_outputs:
            for key, value in env_output.get("metrics", {}).items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        mean_metrics = {
            key: np.sum(value) / self.env_nums.get(key.split("/")[0], 1)
            for key, value in metrics.items()
        }
        try:
            # charts/avg_episode_return: average across all 2048 env groups
            ep_keys = [k for k in metrics.keys() if k.endswith('/episodic_return')]
            if len(ep_keys) > 0:
                ep_vals = []
                for k in ep_keys:
                    tag = k.split('/')[0]
                    denom = self.env_nums.get(tag, max(1, len(metrics[k])))
                    ep_vals.append(float(np.sum(metrics[k]) / denom))
                mean_metrics["charts/avg_episode_return"] = float(np.mean(ep_vals))
        except Exception:
            pass
        try:
            # rollout/max_tile: max over all envs in this batch
            tile_keys = [k for k in metrics.keys() if k.endswith('/max_tile')]
            if len(tile_keys) > 0:
                tile_vals = []
                for k in tile_keys:
                    tile_vals.extend(metrics[k])
                if len(tile_vals) > 0:
                    mean_metrics["rollout/max_tile"] = int(np.max(tile_vals))
        except Exception:
            pass
        for key, values in metrics.items():
            if not isinstance(values, list):
                continue
            prefix, suffix = key.split("/", 1)
            non_zero_values = [v for v in values if v != 0]
            if non_zero_values:
                non_zero_key = f"{prefix}/non-zero/{suffix}"
                mean_metrics[non_zero_key] = np.mean(non_zero_values)

        mean_metrics["response_length"] = response_length
        return mean_metrics

    def _apply_max_length(
        self,
        messages: List[Dict],
        add_generation_prompt: bool = False
    ) -> List[Dict]:
        """
        Token hard truncation, preserving system prompt and current turn.
        Truncates history from left (oldest turns first).

        Args:
            messages: Message list [system, user, assistant, user, assistant, ...]
            add_generation_prompt: Whether to account for generation prompt length

        Returns:
            Truncated message list
        """
        max_length = getattr(self.config.actor_rollout_ref.rollout, "max_model_len", None)
        if max_length is None:
            return messages

        # Calculate current length
        full_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
        token_len = len(self.tokenizer(full_text, add_special_tokens=False)["input_ids"])

        if token_len <= max_length:
            return messages

        # Protect system prompt
        system_msg = messages[0]
        conversation = messages[1:]  # [user, assistant, user, assistant, ...]

        while token_len > max_length and len(conversation) > 2:
            # Detect and remove oldest complete turn
            if len(conversation) >= 3:
                # Check for user-assistant-user(reward) pattern
                if (conversation[0]["role"] == "user" and
                    conversation[1]["role"] == "assistant" and
                    len(conversation) > 2 and
                    conversation[2]["role"] == "user" and
                    "Reward" in conversation[2].get("content", "")):
                    # Remove complete turn with reward
                    conversation = conversation[3:]
                elif (conversation[0]["role"] == "user" and
                      conversation[1]["role"] == "assistant"):
                    # Remove user-assistant pair
                    conversation = conversation[2:]
                else:
                    # Single message removal
                    conversation = conversation[1:]
            else:
                break

            # Recalculate length
            truncated = [system_msg] + conversation
            full_text = self.tokenizer.apply_chat_template(
                truncated,
                add_generation_prompt=add_generation_prompt,
                tokenize=False
            )
            token_len = len(self.tokenizer(full_text, add_special_tokens=False)["input_ids"])

        if token_len > max_length:
            logging.warning(
                f"Cannot truncate to {max_length} tokens (current: {token_len}). "
                "Single turn may exceed max length."
            )

        return [system_msg] + conversation

    # ==================== Message Building Functions ====================

    def _build_format_prompt(self, env_id: int) -> Tuple[str, str]:
        """Build FORMAT_PROMPT and LENGTH_PROMPT for an environment."""
        FORMAT_PROMPT = (
            "<think>...</think><answer>...</answer>"
            if self.config.agent_proxy.enable_think
            else "<answer>...</answer>"
        )
        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_id]['max_tokens']} words (tokens)."
        return FORMAT_PROMPT, LENGTH_PROMPT

    def _build_system_content(self, env_id: int) -> str:
        env_instruction = self.prefix_lookup.get(env_id, "")
        if env_instruction:
            return "You're a helpful assistant. " + env_instruction
        return "You're a helpful assistant. "

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _build_single_turn_messages(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_idx: int,
        history_start: int,
        turn_offset: int,
        include_warning: bool,
        include_assistant: bool,
        old_history_summary: Optional[str] = None,
    ) -> List[Dict]:
        messages = [
            {"role": "system", "content": self._build_system_content(env_output["env_id"])},
            {"role": "user", "content": ""},
        ]
        messages[-1]["content"] = self._build_single_turn_user_content(
            env_output=env_output,
            history=history,
            turn_idx=turn_idx,
            history_start=history_start,
            turn_offset=turn_offset,
            include_warning=include_warning,
            old_history_summary=old_history_summary,
        )
        if include_assistant:
            messages.append({"role": "assistant", "content": history[turn_idx]["llm_response"]})
        return messages

    def _fit_single_turn_history_start_to_max_len(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_idx: int,
        history_start: int,
        turn_offset: int,
        include_warning: bool,
        include_assistant: bool,
        add_generation_prompt: bool,
    ) -> int:
        max_length = getattr(self.config.actor_rollout_ref.rollout, "max_model_len", None)
        if max_length is None:
            return history_start

        while True:
            messages = self._build_single_turn_messages(
                env_output=env_output,
                history=history,
                turn_idx=turn_idx,
                history_start=history_start,
                turn_offset=turn_offset,
                include_warning=include_warning,
                include_assistant=include_assistant,
            )
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            if add_generation_prompt and not include_assistant:
                if self.config.agent_proxy.enable_think:
                    text += "<think>"
                else:
                    text += "<answer>"
            token_len = self._count_tokens(text)
            if token_len <= max_length or history_start >= turn_idx:
                return history_start
            history_start += 1

    def _build_turn_state_content(
        self,
        turn: Dict,
        turn_number: int,
        env_id: int,
        include_warning: bool = False,
    ) -> str:
        """Build state content for a single turn."""
        FORMAT_PROMPT, LENGTH_PROMPT = self._build_format_prompt(env_id)
        warning = ""
        if include_warning and turn.get('manager_invalid_action'):
            warning = "No valid action provided previously. Environment state remains the same. Please try again.\n"

        content = f"\nTurn {turn_number}:\n"
        content += (
            f"State:\n{turn['state']}\n{warning}"
            f"You have {turn['actions_left']} actions left. Always output: {FORMAT_PROMPT} "
            f"with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
        )
        return content

    def _build_first_turn_prompt_and_reasoning_ids(
        self,
        env_output: Dict,
        history: List[Dict],
    ) -> Tuple[List[int], List[int]]:
        messages = [
            {"role": "system", "content": self._build_system_content(env_output["env_id"])},
            {"role": "user", "content": ""},
        ]
        if history:
            first_turn = history[0]
            if "state" in first_turn:
                messages[-1]["content"] += self._build_turn_state_content(
                    first_turn, 1, env_output["env_id"]
                )

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self.config.agent_proxy.enable_think:
            prompt_text += "<think>"
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        reasoning_text = ""
        if self.config.agent_proxy.enable_think and history:
            first_turn = history[0]
            if "llm_response" in first_turn:
                match = re.search(r"<think>(.*?)</think>", first_turn["llm_response"], re.DOTALL)
                if match:
                    reasoning_text = match.group(1)
        for special_token in self.special_token_list:
            reasoning_text = reasoning_text.replace(special_token, "")
        reasoning_text = reasoning_text.strip()
        reasoning_ids = self.tokenizer(reasoning_text, add_special_tokens=False)["input_ids"]
        return prompt_ids, reasoning_ids

    def _build_turn_prompt_and_reasoning_ids(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_idx: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Build prompt_ids and reasoning_ids for a specific turn.

        Args:
            env_output: Environment output dictionary
            history: Complete history list
            turn_idx: Target turn index (0-based)

        Returns:
            (prompt_ids, reasoning_ids):
            - prompt_ids: All context tokens before generating this turn's reasoning
            - reasoning_ids: This turn's reasoning tokens (from <think> tag)
        """
        if turn_idx >= len(history):
            return [], []

        # Build messages up to turn_idx (including the user message for turn_idx)
        messages = [
            {"role": "system", "content": self._build_system_content(env_output["env_id"])},
            {"role": "user", "content": ""},
        ]

        for idx in range(turn_idx + 1):
            content = history[idx]
            actual_turn = idx + 1

            # Add state to user message
            if "state" in content:
                messages[-1]["content"] += self._build_turn_state_content(
                    content, actual_turn, env_output["env_id"]
                )

            # For turns before turn_idx, add complete assistant response and reward
            if idx < turn_idx:
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content:
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

        # Build prompt text (everything before the reasoning)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self.config.agent_proxy.enable_think:
            prompt_text += "<think>"
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Extract reasoning from the target turn
        reasoning_text = ""
        if self.config.agent_proxy.enable_think:
            target_turn = history[turn_idx]
            if "llm_response" in target_turn:
                match = re.search(r"<think>(.*?)</think>", target_turn["llm_response"], re.DOTALL)
                if match:
                    reasoning_text = match.group(1)

        # Clean special tokens
        for special_token in self.special_token_list:
            reasoning_text = reasoning_text.replace(special_token, "")
        reasoning_text = reasoning_text.strip()
        reasoning_ids = self.tokenizer(reasoning_text, add_special_tokens=False)["input_ids"]

        return prompt_ids, reasoning_ids

    def _build_all_turns_prompt_and_reasoning_ids(
        self,
        env_output: Dict,
        history: List[Dict],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Build prompt_ids and reasoning_ids for all turns with valid llm_response.

        Returns:
            (all_prompt_ids, all_reasoning_ids):
            - all_prompt_ids: List of prompt_ids for each turn
            - all_reasoning_ids: List of reasoning_ids for each turn
        """
        all_prompt_ids = []
        all_reasoning_ids = []

        for turn_idx, turn in enumerate(history):
            # Only include turns with llm_response (valid reasoning)
            if "llm_response" not in turn:
                continue

            prompt_ids, reasoning_ids = self._build_turn_prompt_and_reasoning_ids(
                env_output, history, turn_idx
            )

            # Skip turns with empty reasoning
            if not reasoning_ids:
                continue

            all_prompt_ids.append(prompt_ids)
            all_reasoning_ids.append(reasoning_ids)

        return all_prompt_ids, all_reasoning_ids

    # ==================== Mask Computation ====================

    def _compute_loss_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified loss mask computation for all modes.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            mode: "single_turn", "limited_multi_turn", or "full"

        Returns:
            (loss_mask, response_mask): Both (batch_size, seq_len-1)
        """
        batch_size, seq_len = input_ids.shape
        special_token, _ = get_special_tokens(self.tokenizer)

        if mode == "single_turn":
            # For single_turn, each sample has [system, user, assistant]
            # Mask only the assistant (last) turn
            loss_mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.float32)
            response_mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.float32)

            for i in range(batch_size):
                turn_starts = (input_ids[i] == special_token).nonzero(as_tuple=True)[0]
                if len(turn_starts) >= 3:  # system, user, assistant
                    assistant_start = turn_starts[2].item()
                    if assistant_start > 0:
                        loss_mask[i, assistant_start-1:] = 1
                        response_mask[i, assistant_start-1:] = 1

            # Remove padding from masks
            for i in range(batch_size):
                valid_len = attention_mask[i].sum().item()
                if valid_len < seq_len:
                    pad_len = seq_len - valid_len
                    loss_mask[i, :pad_len-1] = 0
                    response_mask[i, :pad_len-1] = 0

        elif mode == "limited_multi_turn":
            # Only train the last assistant turn
            loss_mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.float32)
            response_mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.float32)

            for i in range(batch_size):
                turn_starts = (input_ids[i] == special_token).nonzero(as_tuple=True)[0]
                if len(turn_starts) == 0:
                    continue
                # Last assistant is at the last special token position
                assistant_start = turn_starts[-1].item()

                # For left-padded sequences, valid content ends at seq_len
                # (not valid_len which is a count, not a position)
                assistant_end = seq_len

                if assistant_start > 0:
                    loss_mask[i, assistant_start-1:assistant_end-1] = 1
                    response_mask[i, assistant_start-1:assistant_end-1] = 1

            # Remove padding from masks (for left-padded sequences)
            for i in range(batch_size):
                valid_len = attention_mask[i].sum().item()
                if valid_len < seq_len:
                    pad_len = seq_len - valid_len
                    loss_mask[i, :pad_len-1] = 0
                    response_mask[i, :pad_len-1] = 0

        else:  # mode == "full"
            # Use existing logic - all assistant turns are trained
            turn_starts = torch.where(input_ids == special_token, 1, 0)
            turn_indicators = torch.cumsum(turn_starts, dim=-1)
            if self.config.enable_response_mask:
                loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
            else:
                loss_mask = (turn_indicators > 1)
            response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
            loss_mask = loss_mask[:, :-1].float()
            response_mask = response_mask[:, :-1].float()

        return loss_mask, response_mask

    # ==================== End Mask Computation ====================

    # ==================== Summary Memory ====================

    def _build_episode_summary(self, old_history: List[Dict], env_id=None) -> str:
        """
        Build a compact template-based summary of turns that fall outside the
        recent context window.  Called when enable_summary_memory=True and
        there is history before history_start.

        Args:
            old_history: history entries (each a dict) for turns BEFORE the
                         recent window, i.e. history[:history_start].

        Returns:
            A short multi-line string (~60-100 tokens) summarising what
            happened in those turns, or "" if old_history is empty.
        """
        if not old_history:
            return ""

        n = len(old_history)

        # --- action lookup for human-readable names ---
        action_lookup = {}
        if env_id is not None:
            action_lookup = self.env_config_lookup.get(env_id, {}).get("action_lookup", {})

        # --- collect actions ---
        all_actions: List[str] = []
        for h in old_history:
            all_actions.extend(h.get("actions", []))

        # --- effectiveness ---
        effective_count = sum(
            1 for h in old_history
            if h.get("info", {}).get("action_is_effective", False)
        )

        # --- reward ---
        total_reward = sum(h.get("reward", 0.0) for h in old_history)

        # --- success flag (any turn solved the puzzle) ---
        any_success = any(
            h.get("info", {}).get("success", False) for h in old_history
        )

        # --- invalid-format action count ---
        invalid_count = sum(
            1 for h in old_history if h.get("manager_invalid_action", False)
        )

        def _action_name(a):
            return action_lookup.get(a, str(a))

        # --- last known state coordinates ---
        last_state_coords = ""
        last_state = old_history[-1].get("state", "") if old_history else ""
        if last_state:
            coord_lines = [
                line.strip() for line in last_state.splitlines()
                if any(key in line for key in ("Targets:", "Boxes:", "Player:"))
                and re.search(r'\(\d+', line)  # must contain an actual coordinate like (2,
            ]
            if coord_lines:
                last_state_coords = "State at end of summary: " + " | ".join(coord_lines)

        lines = [
            f"[Memory: {n} earlier step(s) summarized, not shown in full]",
            f"Actions tried: {', '.join(_action_name(a) for a in all_actions) if all_actions else 'none'}",
            f"Effective moves: {effective_count}/{len(all_actions)}",
            f"Cumulative reward from these steps: {total_reward:.2f}",
        ]
        if last_state_coords:
            lines.append(last_state_coords)
        if invalid_count > 0:
            lines.append(f"Invalid action format: {invalid_count} time(s). Valid actions are: Up, Down, Left, Right (separated by ' || ').")
        if any_success:
            lines.append("Note: puzzle was already solved during these steps.")
        return "\n".join(lines)

    # ==================== End Summary Memory ====================

    def _build_single_turn_user_content(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_idx: int,
        history_start: int,
        turn_offset: int = 0,
        include_warning: bool = False,
        old_history_summary: Optional[str] = None,
    ) -> str:
        content = ""

        if history_start < turn_idx:
            completed_steps = turn_idx + turn_offset
            history_length = turn_idx - history_start
            if old_history_summary is not None:
                # Summary memory mode: show compact summary of old turns, then recent window in full
                content += old_history_summary + "\n\n"
                content += (
                    f"Recent {history_length} turn(s) (full detail):\n---\n"
                )
            else:
                # Default: simple counter + recent window in full
                content += (
                    f"Summary: {completed_steps} step(s) completed so far. "
                    f"Showing the last {history_length} turn(s) with state/action/reward details. "
                )
                content += "Recent turns:\n---\n"
            for h_idx in range(history_start, turn_idx):
                actual_turn = h_idx + 1 + turn_offset
                h_turn = history[h_idx]
                content += f"Turn {actual_turn} State:\n{h_turn.get('state', '')}\n"
                content += f"Turn {actual_turn} Action: {h_turn.get('llm_response', '')}\n"
                if 'reward' in h_turn:
                    displayed_reward = h_turn['reward'] + (self.config.es_manager.format_penalty if h_turn.get('manager_invalid_action', False) else 0.0)
                    content += f"Turn {actual_turn} Reward: {displayed_reward}\n"
                content += "---\n"

        actual_turn = turn_idx + 1 + turn_offset
        current_turn_prefix = "Current Turn " if history_start < turn_idx else ""
        FORMAT_PROMPT = "<think>...</think><answer>...</answer>" if self.config.agent_proxy.enable_think else "<answer>...</answer>"
        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
        warning = ""
        if include_warning and history[turn_idx].get('manager_invalid_action'):
            warning = "No valid action provided previously. Environment state remains the same. Please try again.\n"
        """Build user content for single-turn format using memory manager."""
        env_id = env_output["env_id"]

        # Get format prompts
        FORMAT_PROMPT = (
            "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
            if self.config.agent_proxy.enable_think
            else "<answer> [your answer] </answer>"
        )
        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_id]['max_tokens']} words (tokens)."

        # Use memory manager to build content
        memory_manager = self._get_memory_manager(env_id)
        return memory_manager.build_user_content(
            env_output=env_output,
            history=history,
            turn_idx=turn_idx,
            history_start=history_start,
            turn_offset=turn_offset,
            include_warning=include_warning,
            format_prompt=FORMAT_PROMPT,
            length_prompt=LENGTH_PROMPT,
        )

    def _build_single_turn_samples(self, env_outputs: List[Dict]) -> DataProto:
        """
        Build single-turn samples with optional history context in user prompt.
        Format: [system, user(with previous k-1 turns history), assistant]

        When max_context_window=1, no previous history is shown.
        When max_context_window>1, previous turns are included as text in user prompt.
        """
        llm_input_texts = []
        messages_list = []
        env_ids = []
        group_ids = []
        episode_ids = []
        episode_rewards = []
        uid_list = []

        max_context_window = self._resolve_max_context_window()

        # Collect episode-level rewards for normalization
        episode_raw_rewards = []
        episode_penalties = []
        for env_output in env_outputs:
            history = self._extract_history(env_output, prepare_for_update=True)
            total_reward = sum(turn.get('reward', 0.0) for turn in history)
            episode_raw_rewards.append(total_reward)
            episode_penalties.append(env_output.get("penalty", 0))

        # Apply reward normalization
        normalized_episode_rewards = self._normalize_episode_rewards(
            env_outputs, episode_raw_rewards, episode_penalties
        )

        # Build turn samples
        for episode_idx, env_output in enumerate(env_outputs):
            history = self._extract_history(env_output, prepare_for_update=True)
            normalized_reward = normalized_episode_rewards[episode_idx].item()

            for turn_idx, turn in enumerate(history):
                if 'llm_response' not in turn:
                    continue

                messages = [
                    {"role": "system", "content": self._build_system_content(env_output["env_id"])},
                    {"role": "user", "content": ""}
                ]

                history_start = self._get_history_start(turn_idx, max_context_window)
                history_start = self._fit_single_turn_history_start_to_max_len(
                    env_output=env_output,
                    history=history,
                    turn_idx=turn_idx,
                    history_start=history_start,
                    turn_offset=0,
                    include_warning=False,
                    include_assistant=True,
                    add_generation_prompt=False,
                )

                # Summary memory: summarise turns outside the recent window
                enable_summary = getattr(self.config.agent_proxy, "enable_summary_memory", False)
                old_history_summary = None
                if enable_summary and history_start > 0:
                    old_history_summary = self._build_episode_summary(history[:history_start], env_id=env_output["env_id"])

                messages = self._build_single_turn_messages(
                    env_output=env_output,
                    history=history,
                    turn_idx=turn_idx,
                    history_start=history_start,
                    turn_offset=0,
                    include_warning=False,
                    include_assistant=True,
                    old_history_summary=old_history_summary,
                )

                text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

                llm_input_texts.append(text)
                messages_list.append(messages)
                env_ids.append(env_output["env_id"])
                group_ids.append(env_output["group_id"])
                episode_ids.append(episode_idx)
                episode_rewards.append(normalized_reward)
                uid_list.append(env_output.get("uid", env_output["env_id"]))

        # Tokenize
        input_ids, attention_mask, position_ids = self._tokenize_and_build_tensors(llm_input_texts)

        # Build score tensor
        score_tensor = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32)
        for i, reward in enumerate(episode_rewards):
            score_tensor[i, -1] = reward

        # Compute loss mask
        loss_mask, response_mask = self._compute_loss_mask(
            input_ids, attention_mask, mode="single_turn"
        )

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # Build DataProto
        llm_inputs = self._build_dataproto(
            input_ids, attention_mask, position_ids, loss_mask, score_tensor,
            env_ids, group_ids, messages_list, episode_ids, uid_list
        )

        llm_inputs.meta_info = {"metrics": self._compute_metrics(env_outputs, response_length)}
        return llm_inputs

    def _build_limited_multi_turn_samples(self, env_outputs: List[Dict]) -> DataProto:
        """
        Build multi-turn samples with limited context window.
        Format: [system, user, assistant] × k, but only train the last assistant.

        Each sample contains up to k turns of conversation, but only the last
        assistant response is included in loss computation.
        """
        llm_input_texts = []
        messages_list = []
        env_ids = []
        group_ids = []
        episode_ids = []
        episode_rewards = []
        uid_list = []

        max_context_window = self._resolve_max_context_window()

        # Collect episode-level rewards for normalization
        episode_raw_rewards = []
        episode_penalties = []
        for env_output in env_outputs:
            history = self._extract_history(env_output, prepare_for_update=True)
            total_reward = sum(turn.get('reward', 0.0) for turn in history)
            episode_raw_rewards.append(total_reward)
            episode_penalties.append(env_output.get("penalty", 0))

        # Apply reward normalization
        normalized_episode_rewards = self._normalize_episode_rewards(
            env_outputs, episode_raw_rewards, episode_penalties
        )

        # Build multi-turn samples
        for episode_idx, env_output in enumerate(env_outputs):
            history = self._extract_history(env_output, prepare_for_update=True)
            normalized_reward = normalized_episode_rewards[episode_idx].item()

            for turn_idx, turn in enumerate(history):
                if 'llm_response' not in turn:
                    continue

                messages = [
                    {"role": "system", "content": self._build_system_content(env_output["env_id"])},
                    {"role": "user", "content": ""}
                ]

                history_start = self._get_history_start(turn_idx, max_context_window)

                # Summary memory for limited_multi_turn
                enable_summary = getattr(self.config.agent_proxy, "enable_summary_memory", False)
                if enable_summary and history_start > 0:
                    old_summary = self._build_episode_summary(history[:history_start], env_id=env_output["env_id"])
                    messages[1]["content"] = old_summary + "\n\n"

                # Add all turns from history_start to turn_idx (inclusive)
                for h_idx in range(history_start, turn_idx + 1):
                    h_turn = history[h_idx]
                    actual_turn = h_idx + 1

                    # Add turn state using helper
                    messages[-1]["content"] += self._build_turn_state_content(
                        h_turn, actual_turn, env_output["env_id"]
                    )

                    # Add assistant response
                    messages.append({"role": "assistant", "content": h_turn["llm_response"]})

                    # Add reward for non-final turns
                    if h_idx < turn_idx and 'reward' in h_turn:
                        displayed_reward = h_turn['reward'] + (self.config.es_manager.format_penalty if h_turn.get('manager_invalid_action', False) else 0.0)
                        messages.append({"role": "user", "content": f"Reward:\n{displayed_reward}\n"})

                # Apply max length truncation
                messages = self._apply_max_length(messages, add_generation_prompt=False)

                text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

                llm_input_texts.append(text)
                messages_list.append(messages)
                env_ids.append(env_output["env_id"])
                group_ids.append(env_output["group_id"])
                episode_ids.append(episode_idx)
                episode_rewards.append(normalized_reward)
                uid_list.append(env_output.get("uid", env_output["env_id"]))

        # Tokenize
        input_ids, attention_mask, position_ids = self._tokenize_and_build_tensors(llm_input_texts)

        # Build score tensor
        score_tensor = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32)
        for i, reward in enumerate(episode_rewards):
            score_tensor[i, -1] = reward

        # Compute loss mask (only last assistant)
        loss_mask, response_mask = self._compute_loss_mask(
            input_ids, attention_mask, mode="limited_multi_turn"
        )

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # Build DataProto
        llm_inputs = self._build_dataproto(
            input_ids, attention_mask, position_ids, loss_mask, score_tensor,
            env_ids, group_ids, messages_list, episode_ids, uid_list
        )

        llm_inputs.meta_info = {"metrics": self._compute_metrics(env_outputs, response_length)}
        return llm_inputs

    def _build_samples_full(
        self, env_outputs: List[Dict], include_collapse_data: bool = True
    ) -> DataProto:
        """
        Build full multi-turn samples for update (original behavior).
        All assistant turns are trained.
        """
        llm_input_texts = []
        messages_list = []
        collapse_cfg = getattr(self.config, "collapse_detection", None)
        first_turn_enabled = bool(getattr(collapse_cfg, "first_turn_enabled", False)) if collapse_cfg else False
        multi_turn_enabled = bool(getattr(collapse_cfg, "multi_turn_enabled", False)) if collapse_cfg else False
        collapse_enabled = first_turn_enabled or multi_turn_enabled

        include_collapse_data = include_collapse_data and collapse_enabled
        include_first_turn = include_collapse_data and first_turn_enabled
        include_all_turns = include_collapse_data and multi_turn_enabled

        first_turn_prompt_ids_list = [] if include_first_turn else None
        first_turn_reasoning_ids_list = [] if include_first_turn else None
        # Multi-turn data for collapse detection
        all_turns_prompt_ids_list = [] if include_all_turns else None
        all_turns_reasoning_ids_list = [] if include_all_turns else None
        turn_counts_list = [] if include_all_turns else None
        turn_counts_total_list = [] if include_all_turns else None

        for env_output in env_outputs:
            history = self._extract_history(env_output, prepare_for_update=True)
            if include_first_turn:
                first_prompt_ids, first_reasoning_ids = self._build_first_turn_prompt_and_reasoning_ids(
                    env_output, history
                )
                first_turn_prompt_ids_list.append(first_prompt_ids)
                first_turn_reasoning_ids_list.append(first_reasoning_ids)

            # Build multi-turn prompt/reasoning for collapse detection
            if include_all_turns:
                all_prompt_ids, all_reasoning_ids = self._build_all_turns_prompt_and_reasoning_ids(
                    env_output, history
                )
                all_turns_prompt_ids_list.append(all_prompt_ids)
                all_turns_reasoning_ids_list.append(all_reasoning_ids)
                turn_counts_list.append(len(all_prompt_ids))
                total_turns = sum(1 for turn in history if "llm_response" in turn)
                turn_counts_total_list.append(total_turns)
            messages = [
                {"role": "system", "content": self._build_system_content(env_output["env_id"])},
                {"role": "user", "content": ""}
            ]

            for idx, content in enumerate(history):
                actual_turn = idx + 1
                if "state" in content:
                    messages[-1]["content"] += self._build_turn_state_content(
                        content, actual_turn, env_output["env_id"]
                    )
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and idx < len(history) - 1:
                    displayed_reward = content['reward'] + (self.config.es_manager.format_penalty if content.get('manager_invalid_action', False) else 0.0)
                    messages.append({"role": "user", "content": f"Reward:\n{displayed_reward}\n"})

            # Apply max length truncation
            messages = self._apply_max_length(messages, add_generation_prompt=False)

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            llm_input_texts.append(text)
            messages_list.append(messages)

        # Tokenize
        input_ids, attention_mask, position_ids = self._tokenize_and_build_tensors(llm_input_texts)

        # Build scores using existing logic
        scores = [[i.get('reward', 0.0) for i in self._extract_history(env_output, True)] for env_output in env_outputs]
        score_tensor, loss_mask, response_mask = get_masks_and_scores(
            input_ids, self.tokenizer, scores,
            use_turn_scores=self.config.agent_proxy.use_turn_scores,
            enable_response_mask=self.config.enable_response_mask
        )

        # Normalize scores
        if not self.config.agent_proxy.use_turn_scores:
            score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # Build DataProto
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],
            "loss_mask": loss_mask,
            "rm_scores": score_tensor,
            "original_rm_scores": score_tensor.clone(),
        }, batch_size=input_ids.shape[0])

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=int),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=int),
            "messages_list": np.array(messages_list, dtype=object),
        }
        if include_first_turn:
            llm_inputs.non_tensor_batch["first_turn_prompt_ids"] = np.array(
                first_turn_prompt_ids_list, dtype=object
            )
            llm_inputs.non_tensor_batch["first_turn_reasoning_ids"] = np.array(
                first_turn_reasoning_ids_list, dtype=object
            )
        if include_all_turns:
            llm_inputs.non_tensor_batch["all_turns_prompt_ids"] = np.array(
                all_turns_prompt_ids_list, dtype=object
            )
            llm_inputs.non_tensor_batch["all_turns_reasoning_ids"] = np.array(
                all_turns_reasoning_ids_list, dtype=object
            )
            llm_inputs.non_tensor_batch["turn_counts"] = np.array(
                turn_counts_list, dtype=int
            )
            llm_inputs.non_tensor_batch["turn_counts_total"] = np.array(
                turn_counts_total_list, dtype=int
            )

        llm_inputs.meta_info = {"metrics": self._compute_metrics(env_outputs, response_length)}
        return llm_inputs

    def _build_infer_samples(self, env_outputs: List[Dict]) -> DataProto:
        """Build samples for inference (generation)."""
        context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")
        resolved_max_context_window = self._resolve_max_context_window()

        llm_input_texts = []
        messages_list = []

        for env_output in env_outputs:
            history = env_output['history']

            # Apply context window for non-full modes
            turn_offset = 0
            old_history_for_summary: List[Dict] = []
            if context_window_mode in ("single_turn", "limited_multi_turn"):
                if resolved_max_context_window is not None and resolved_max_context_window > 0:
                    if len(history) > resolved_max_context_window:
                        turn_offset = len(history) - resolved_max_context_window
                        old_history_for_summary = history[:turn_offset]  # save before truncation
                        history = history[-resolved_max_context_window:]

            if context_window_mode == "single_turn":
                enable_summary = getattr(self.config.agent_proxy, "enable_summary_memory", False)
                old_history_summary = (
                    self._build_episode_summary(old_history_for_summary, env_id=env_output["env_id"])
                    if enable_summary and old_history_for_summary
                    else None
                )
                messages = self._build_infer_single_turn_messages(
                    env_output, history, turn_offset, resolved_max_context_window,
                    old_history_summary=old_history_summary,
                )
            elif context_window_mode == "limited_multi_turn":
                enable_summary = getattr(self.config.agent_proxy, "enable_summary_memory", False)
                old_history_summary = (
                    self._build_episode_summary(old_history_for_summary, env_id=env_output["env_id"])
                    if enable_summary and old_history_for_summary
                    else None
                )
                messages = self._build_infer_multi_turn_messages(
                    env_output, history, turn_offset, old_history_summary=old_history_summary
                )
            else:  # full
                messages = self._build_infer_multi_turn_messages(
                    env_output, history, turn_offset
                )

            # Apply max length truncation
            messages = self._apply_max_length(messages, add_generation_prompt=True)

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # Add generation prompt prefix
            if self.config.agent_proxy.enable_think:
                text = text + "<think>"
            else:
                text = text + "<answer>"

            llm_input_texts.append(text)
            messages_list.append(messages)

        # Tokenize
        input_ids, attention_mask, position_ids = self._tokenize_and_build_tensors(llm_input_texts)

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],
        }, batch_size=input_ids.shape[0])

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=int),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=int),
            "messages_list": np.array(messages_list, dtype=object),
        }

        return llm_inputs

    def _build_infer_single_turn_messages(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_offset: int,
        max_context_window: Optional[int],
        old_history_summary: Optional[str] = None,
    ) -> List[Dict]:
        """Build messages for single_turn inference."""
        messages = [
            {"role": "system", "content": self._build_system_content(env_output["env_id"])},
            {"role": "user", "content": ""},
        ]
        if not history:
            return messages

        turn_idx = len(history) - 1
        history_start = self._get_history_start(turn_idx, max_context_window)
        history_start = self._fit_single_turn_history_start_to_max_len(
            env_output=env_output,
            history=history,
            turn_idx=turn_idx,
            history_start=history_start,
            turn_offset=turn_offset,
            include_warning=True,
            include_assistant=False,
            add_generation_prompt=True,
        )
        messages = self._build_single_turn_messages(
            env_output=env_output,
            history=history,
            turn_idx=turn_idx,
            history_start=history_start,
            turn_offset=turn_offset,
            include_warning=True,
            include_assistant=False,
            old_history_summary=old_history_summary,
        )
        return messages

    def _build_infer_multi_turn_messages(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_offset: int,
        old_history_summary: Optional[str] = None,
    ) -> List[Dict]:
        """Build messages for multi-turn inference (full or limited_multi_turn)."""
        first_user_content = ""
        if old_history_summary:
            first_user_content = old_history_summary + "\n\n"
        messages = [
            {"role": "system", "content": self._build_system_content(env_output["env_id"])},
            {"role": "user", "content": first_user_content}
        ]

        for idx, content in enumerate(history):
            actual_turn = idx + 1 + turn_offset
            if "state" in content:
                messages[-1]["content"] += self._build_turn_state_content(
                    content, actual_turn, env_output["env_id"], include_warning=True
                )
            if "llm_response" in content:
                messages.append({"role": "assistant", "content": content["llm_response"]})
            if "reward" in content:
                displayed_reward = content['reward'] + (self.config.es_manager.format_penalty if content.get('manager_invalid_action', False) else 0.0)
                messages.append({"role": "user", "content": f"Reward:\n{displayed_reward}\n"})

        return messages

    def get_lm_inputs(
        self,
        env_outputs: List[Dict],
        prepare_for_update: bool,
        include_collapse_data: bool = True,
    ) -> DataProto:
        """
        Main entry point for building LLM inputs.

        Args:
            env_outputs: List of environment outputs with history
            prepare_for_update: True for training data, False for inference

        Returns:
            DataProto with tokenized inputs and metadata
        """
        context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")

        if prepare_for_update:
            # Training: dispatch to mode-specific builders
            if context_window_mode == "single_turn":
                result = self._build_single_turn_samples(env_outputs)
            elif context_window_mode == "limited_multi_turn":
                result = self._build_limited_multi_turn_samples(env_outputs)
            else:  # full
                result = self._build_samples_full(
                    env_outputs, include_collapse_data=include_collapse_data
                )

            # Save history and metrics only during evaluation (not training)
            if self.mode == "val":
                if result.non_tensor_batch is None:
                    result.non_tensor_batch = {}

                result.non_tensor_batch["history"] = np.array(
                    [env_output.get("history", []) for env_output in env_outputs],
                    dtype=object
                )
                result.non_tensor_batch["metrics"] = np.array(
                    [env_output.get("metrics", {}) for env_output in env_outputs],
                    dtype=object
                )

            return result
        else:
            # Inference: build prompts for generation
            return self._build_infer_samples(env_outputs)

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'],
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions, think_len, think_coord_fmt_ok = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
                "think_len": think_len,
                "think_coord_fmt_ok": think_coord_fmt_ok,
            })
        return env_inputs

    def formulate_rollouts(
        self, env_outputs: List[Dict], include_collapse_data: bool = True
    ) -> DataProto:
        llm_inputs = self.get_lm_inputs(
            env_outputs, prepare_for_update=True, include_collapse_data=include_collapse_data
        )
        return llm_inputs


def main():
    """Test all context window modes and verify refactoring correctness."""
    import copy
    from omegaconf import OmegaConf

    # Create minimal config for testing
    config = OmegaConf.create({
        "actor_rollout_ref": {
            "model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"},
            "rollout": {
                "response_length": 400,
                "max_model_len": 3600,
            }
        },
        "agent_proxy": {
            "context_window_mode": "full",
            "max_context_window": -1,
            "action_sep": "||",
            "max_actions_per_turn": 2,
            "enable_think": True,
            "use_turn_scores": False,
            "reward_normalization": {
                "grouping": "state",
                "method": "identity"
            }
        },
        "es_manager": {
            "train": {
                "env_groups": 8,
                "group_size": 16,
                "env_configs": {
                    "tags": ["CoordSokoban"],
                    "n_groups": [8]
                }
            }
        },
        "custom_envs": {
            "CoordSokoban": {
                "env_type": "sokoban",
                "env_instruction": "Solve this Sokoban puzzle.",
                "max_tokens": 400,
                "max_actions_per_traj": 10,
                "action_lookup": {"0": "up", "1": "down", "2": "left", "3": "right"}
            }
        },
        "enable_response_mask": True
    })

    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)

    # Test data: 2 episodes with multi-turn history
    env_outputs_base = [
        {
            "env_id": 0,
            "history": [
                {"state": "State 1", "llm_response": "<think>Think 1</think><answer>Action 1</answer>", "reward": 0.5, "actions_left": 3},
                {"state": "State 2", "llm_response": "<think>Think 2</think><answer>Action 2</answer>", "reward": 0.8, "actions_left": 2},
                {"state": "State 3", "llm_response": "<think>Think 3</think><answer>Action 3</answer>", "reward": 1.0, "actions_left": 1},
                {"state": "State 4", "actions_left": 0}  # Final state, no response
            ],
            "group_id": 0,
            "tag": "CoordSokoban",
            "metrics": {"CoordSokoban/success": 1}
        },
        {
            "env_id": 1,
            "history": [
                {"state": "State A", "llm_response": "<think>Think A</think><answer>Action A</answer>", "reward": 0.3, "actions_left": 2},
                {"state": "State B", "llm_response": "<think>Think B</think><answer>Action B</answer>", "reward": 0.6, "actions_left": 1},
                {"state": "State C", "actions_left": 0}
            ],
            "group_id": 0,
            "tag": "CoordSokoban",
            "metrics": {"CoordSokoban/success": 0}
        }
    ]

    def test_mode(mode_name, max_context_window, config):
        """Test a specific context window mode."""
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode_name}, max_context_window: {max_context_window}")
        print('='*60)

        # Update config
        config.agent_proxy.context_window_mode = mode_name
        config.agent_proxy.max_context_window = max_context_window

        ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
        env_outputs = copy.deepcopy(env_outputs_base)

        # Test 1: prepare_for_update=True (training)
        print("\n--- Test: prepare_for_update=True (Training) ---")
        result_update = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=True)

        print(f"Batch size: {result_update.batch['input_ids'].shape[0]}")
        print(f"Sequence length: {result_update.batch['input_ids'].shape[1]}")
        print(f"Has loss_mask: {'loss_mask' in result_update.batch.keys()}")
        print(f"Has rm_scores: {'rm_scores' in result_update.batch.keys()}")

        if 'episode_ids' in result_update.non_tensor_batch:
            print(f"Episode IDs: {result_update.non_tensor_batch['episode_ids']}")

        # Check loss_mask is valid
        loss_mask = result_update.batch['loss_mask']
        print(f"Loss mask sum per sample: {loss_mask.sum(dim=-1).tolist()}")

        # Decode first sample to verify format
        first_text = tokenizer.decode(result_update.batch['input_ids'][0], skip_special_tokens=False)
        print(f"\nFirst sample preview (first 500 chars):\n{first_text[:500]}...")

        # Test 2: prepare_for_update=False (inference)
        print("\n--- Test: prepare_for_update=False (Inference) ---")
        env_outputs_infer = copy.deepcopy(env_outputs_base)
        result_infer = ctx_manager.get_lm_inputs(env_outputs_infer, prepare_for_update=False)

        print(f"Batch size: {result_infer.batch['input_ids'].shape[0]}")
        print(f"Sequence length: {result_infer.batch['input_ids'].shape[1]}")

        # Decode first sample
        first_text_infer = tokenizer.decode(result_infer.batch['input_ids'][0], skip_special_tokens=False)
        print(f"\nFirst infer sample preview (first 500 chars):\n{first_text_infer[:500]}...")

        # Verify generation prompt is added
        assert "<think>" in first_text_infer or "<answer>" in first_text_infer, \
            "Generation prompt should be added for inference"

        print(f"\n✓ Mode {mode_name} with k={max_context_window} passed!")
        return result_update, result_infer

    # Test all three modes
    print("\n" + "="*80)
    print("CONTEXT WINDOW MODE TESTS")
    print("="*80)

    # Test 1: Full mode
    result_full_update, result_full_infer = test_mode("full", -1, config)

    # Test 2: Single-turn mode with k=2
    result_st_update, result_st_infer = test_mode("single_turn", 2, config)

    # Test 3: Single-turn mode with k=1 (no history)
    result_st1_update, result_st1_infer = test_mode("single_turn", 1, config)

    # Test 4: Limited multi-turn mode with k=2
    result_lmt_update, result_lmt_infer = test_mode("limited_multi_turn", 2, config)

    # Test 5: Limited multi-turn mode with k=3
    result_lmt3_update, result_lmt3_infer = test_mode("limited_multi_turn", 3, config)

    # Verify different modes produce different results
    print("\n" + "="*80)
    print("CROSS-MODE VERIFICATION")
    print("="*80)

    # Full mode should have fewer samples than turn-level modes
    # (full mode: 1 sample per episode, turn-level: 1 sample per turn)
    print(f"\nFull mode samples: {result_full_update.batch['input_ids'].shape[0]}")
    print(f"Single-turn mode samples: {result_st_update.batch['input_ids'].shape[0]}")
    print(f"Limited multi-turn mode samples: {result_lmt_update.batch['input_ids'].shape[0]}")

    # Turn-level modes should have more samples (one per turn)
    assert result_st_update.batch['input_ids'].shape[0] > result_full_update.batch['input_ids'].shape[0], \
        "Single-turn mode should have more samples than full mode"

    print("\n✓ All cross-mode verifications passed!")

    # Test shared infrastructure functions
    print("\n" + "="*80)
    print("SHARED INFRASTRUCTURE TESTS")
    print("="*80)

    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)

    # Test _extract_history
    env_output = copy.deepcopy(env_outputs_base[0])
    history_update = ctx_manager._extract_history(env_output, prepare_for_update=True)
    history_infer = ctx_manager._extract_history(env_output, prepare_for_update=False)
    print(f"\n_extract_history (update): {len(history_update)} turns")
    print(f"_extract_history (infer): {len(history_infer)} turns")
    assert len(history_update) == 3, "Update should remove final state-only turn"
    assert len(history_infer) == 4, "Infer should keep all turns"
    print("✓ _extract_history passed!")

    # Test _normalize_episode_rewards
    raw_rewards = [1.0, 2.0, 3.0, 4.0]
    penalties = [0.0, 0.0, 0.0, 0.0]
    test_env_outputs = [{"group_id": 0, "tag": "test"} for _ in range(4)]
    normalized = ctx_manager._normalize_episode_rewards(test_env_outputs, raw_rewards, penalties)
    print(f"\n_normalize_episode_rewards: {raw_rewards} -> {normalized.tolist()}")
    print("✓ _normalize_episode_rewards passed!")

    # Test _apply_max_length
    long_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User 1 " * 100},
        {"role": "assistant", "content": "Assistant 1"},
        {"role": "user", "content": "Reward: 0.5"},
        {"role": "assistant", "content": "Assistant 2"},
        {"role": "user", "content": "Reward: 0.8"},
        {"role": "assistant", "content": "Final response"},
    ]
    original_len = len(long_messages)
    truncated = ctx_manager._apply_max_length(long_messages, add_generation_prompt=False)
    print(f"\n_apply_max_length: {original_len} messages -> {len(truncated)} messages")
    assert truncated[0]["role"] == "system", "System prompt should be preserved"
    print("✓ _apply_max_length passed!")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)

if __name__ == "__main__":
    main()
    
