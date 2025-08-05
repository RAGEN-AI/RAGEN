# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ArCHer Critic Worker Implementation

Extends DataParallelPPOCritic to implement the Double Critic architecture 
for ArCHer algorithm that provides both Q-values and V-values.
"""

import logging
import os
import copy
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import optim

from verl import DataProto
from ragen.workers.critic.dp_critic import DataParallelPPOCritic

__all__ = ["ArCherDoubleCriticModule", "ArCherCritic"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ArCherDoubleCriticModule(nn.Module):
    """
    ArCHer Double Critic Module that wraps an existing critic model
    and adds Q1, Q2, V1, V2 heads following the original ArCHer architecture.
    
    Key insight from original ArCHer:
    - Q-critics take concatenated [observation, action] states (dim = hidden_size * 2)
    - V-critics take only observation states (dim = hidden_size)
    - Both observation and action are encoded separately through the base model
    - In RAGEN context: observation = prompt, action = response
    """
    
    def __init__(self, base_critic_module: nn.Module):
        super().__init__()
        self.base_critic_module = base_critic_module
        
        # Get the hidden size from the base model
        hidden_size = base_critic_module.config.hidden_size
        
        # Create the ArCHer heads - following original DoubleCritic structure
        # Q-critics (state-action value functions) - input is [obs_hidden, action_hidden]
        self.q_critic1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Note: input is concatenated obs+action
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.q_critic2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Note: input is concatenated obs+action
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # V-critics (state value functions) - input is only observation hidden states
        self.v_critic1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Note: input is only observation
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
        )
        
        self.v_critic2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Note: input is only observation
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize the heads
        self._init_archer_heads()
        
        logger.info(f"ArCHer Double Critic Module initialized with hidden_size={hidden_size}")
        logger.info(f"Q-critics input dim: {hidden_size * 2}, V-critics input dim: {hidden_size}")
    
    def _init_archer_heads(self):
        """Initialize the ArCHer critic heads with proper weights"""
        for module in [self.q_critic1, self.q_critic2, self.v_critic1, self.v_critic2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)
    
    def _encode_sequence(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        """
        Encode a sequence through the base model and return pooled representation.
        This follows the original ArCHer approach of encoding obs and action separately.
        """
        # Get hidden states from the base model
        outputs = self.base_critic_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get the last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        
        # Use mean pooling over valid tokens (like original ArCHer's pooler_output)
        if attention_mask is not None:
            # Mask out padding tokens for mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            masked_hidden = hidden_states * mask_expanded
            pooled = masked_hidden.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling if no attention mask
            pooled = hidden_states.mean(dim=1)
        
        return pooled  # (batch_size, hidden_size)
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                observation_ids=None, observation_mask=None, 
                action_ids=None, action_mask=None, **kwargs):
        """
        Forward pass that computes Q and V values following ArCHer architecture.
        
        This supports two modes:
        1. If observation_ids and action_ids are provided: Use separate encoding (preferred)
        2. If only input_ids provided: Extract obs/action from the concatenated sequence
        
        Args:
            input_ids: Full sequence (prompt + response) for backward compatibility
            observation_ids: Observation sequence only (preferred for ArCHer)
            action_ids: Action sequence only (preferred for ArCHer)
        """
        
        if observation_ids is not None and action_ids is not None:
            # Mode 1: Separate observation and action encoding (preferred ArCHer approach)
            
            # Encode observation separately
            obs_pooled = self._encode_sequence(
                input_ids=observation_ids,
                attention_mask=observation_mask,
                position_ids=position_ids,
                **kwargs
            )  # (batch_size, hidden_size)
            
            # Encode action separately  
            action_pooled = self._encode_sequence(
                input_ids=action_ids,
                attention_mask=action_mask,
                **kwargs
            )  # (batch_size, hidden_size)
            
            # Q-values: Concatenate observation and action representations
            q_input = torch.cat([obs_pooled, action_pooled], dim=1)  # (batch_size, hidden_size * 2)
            q1_values = self.q_critic1(q_input).squeeze(-1)  # (batch_size,)
            q2_values = self.q_critic2(q_input).squeeze(-1)  # (batch_size,)
            
            # V-values: Only observation representation
            v1_values = self.v_critic1(obs_pooled).squeeze(-1)  # (batch_size,)
            v2_values = self.v_critic2(obs_pooled).squeeze(-1)  # (batch_size,)
            
            # For RAGEN compatibility, create a mock output object
            from types import SimpleNamespace
            base_outputs = SimpleNamespace()
            
            # Expand to token-level for RAGEN interface compatibility
            action_length = action_ids.size(1)
            base_outputs.q1_values = q1_values.unsqueeze(1).expand(-1, action_length)
            base_outputs.q2_values = q2_values.unsqueeze(1).expand(-1, action_length)
            base_outputs.v1_values = v1_values.unsqueeze(1).expand(-1, action_length)  
            base_outputs.v2_values = v2_values.unsqueeze(1).expand(-1, action_length)
            
            # Set logits for backward compatibility
            base_outputs.logits = base_outputs.v1_values.unsqueeze(-1)
            
        else:
            # Mode 2: Backward compatibility - extract from concatenated sequence
            # This is a fallback when we only have the full input_ids
            
            base_outputs = self.base_critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                position_ids=position_ids,
                output_hidden_states=True,
                **kwargs
            )
            
            # Get hidden states and apply simple token-level heads (not ideal for ArCHer)
            hidden_states = base_outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            
            # Fallback: treat each token as both obs and action (suboptimal)
            # This won't give proper ArCHer behavior but maintains compatibility
            pooled_states = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
            
            # Duplicate for Q-input (suboptimal concatenation)
            q_input = torch.cat([pooled_states, pooled_states], dim=1)
            
            q1_pooled = self.q_critic1(q_input).squeeze(-1)  # (batch_size,)
            q2_pooled = self.q_critic2(q_input).squeeze(-1)  # (batch_size,)
            v1_pooled = self.v_critic1(pooled_states).squeeze(-1)  # (batch_size,)
            v2_pooled = self.v_critic2(pooled_states).squeeze(-1)  # (batch_size,)
            
            # Expand to sequence length
            seq_len = hidden_states.size(1)
            base_outputs.q1_values = q1_pooled.unsqueeze(1).expand(-1, seq_len)
            base_outputs.q2_values = q2_pooled.unsqueeze(1).expand(-1, seq_len)
            base_outputs.v1_values = v1_pooled.unsqueeze(1).expand(-1, seq_len)
            base_outputs.v2_values = v2_pooled.unsqueeze(1).expand(-1, seq_len)
            
            # Keep original logits
            base_outputs.logits = base_outputs.logits
        
        return base_outputs
    
    def get_q_and_v_values(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Directly compute Q and V values from hidden states.
        Used for advantage computation.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (q_values, v_values) where each is (batch_size, seq_len)
        """
        q1_values = self.q_critic1(hidden_states).squeeze(-1)
        q2_values = self.q_critic2(hidden_states).squeeze(-1)
        v1_values = self.v_critic1(hidden_states).squeeze(-1)
        v2_values = self.v_critic2(hidden_states).squeeze(-1)
        
        # Take minimum for stability (as in original ArCHer)
        q_values = torch.min(q1_values, q2_values)
        v_values = torch.min(v1_values, v2_values)
        
        return q_values, v_values


class ArCherCritic(DataParallelPPOCritic):
    """
    ArCHer Critic Worker that extends DataParallelPPOCritic to support
    the Double Critic architecture needed for ArCHer's Q-V advantage computation.
    
    This implementation minimally modifies the existing working DataParallelPPOCritic
    to add ArCHer functionality while preserving all existing features.
    """
    
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        # Wrap the critic module with ArCHer Double Critic heads
        archer_critic_module = ArCherDoubleCriticModule(critic_module)
        
        # Initialize the parent with the wrapped module
        super().__init__(config=config, critic_module=archer_critic_module, critic_optimizer=critic_optimizer)
        
        # ArCHer-specific configuration
        # Try multiple config paths for tau parameter
        tau_sources = [
            getattr(config, 'archer', {}).get('tau', None),  # config.archer.tau
            getattr(config, 'archer_tau', None),              # config.archer_tau  
            getattr(config, 'tau', None),                     # config.tau
            0.005                                             # default fallback
        ]
        self.tau = next((tau for tau in tau_sources if tau is not None), 0.005)
        
        self.use_target_networks = getattr(config, 'archer_use_target_networks', True)
        
        # Create target networks if enabled
        if self.use_target_networks:
            self.target_critic_module = copy.deepcopy(archer_critic_module)
            # Freeze target networks
            for param in self.target_critic_module.parameters():
                param.requires_grad = False
        
        logger.info(f"ArCHer Critic initialized with tau={self.tau}, use_target_networks={self.use_target_networks}")
    
    def soft_update_target_networks(self):
        """
        Soft update target networks using exponential moving average:
        θ_target = τ * θ_main + (1 - τ) * θ_target
        """
        if not self.use_target_networks:
            return
            
        with torch.no_grad():
            for target_param, main_param in zip(
                self.target_critic_module.parameters(), 
                self.critic_module.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
    
    def compute_q_and_v_values(self, data: DataProto) -> DataProto:
        """
        Compute both Q-values and V-values using the ArCHer Double Critic.
        
        This method follows the same pattern as compute_values() but returns
        both Q and V values in a DataProto format.
        """
        self.critic_module.eval()
        
        # Use the same micro-batching logic as the parent class
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Same micro-batching logic as parent class
        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            from verl.utils.seqlen_balancing import rearrange_micro_batches
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        # Handle PEFT models like parent class
        from peft import PeftModel
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        is_peft_model = isinstance(self.critic_module._fsdp_wrapped_module, PeftModel)
        if is_peft_model:
            logger.info("ArCHer Critic is a PeftModel")
            with FSDP.summon_full_params(self.critic_module):
                self.critic_module.merge_adapter()

        q_values_lst = []
        v_values_lst = []
        
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                q_values, v_values = self._forward_micro_batch_archer(micro_batch)
            q_values_lst.append(q_values)
            v_values_lst.append(v_values)

        if is_peft_model:
            logger.info("Unmerging adapter for ArCHer critic") 
            with FSDP.summon_full_params(self.critic_module):
                self.critic_module.unmerge_adapter()

        q_values = torch.concat(q_values_lst, dim=0)
        v_values = torch.concat(v_values_lst, dim=0)
        
        # Apply response mask to both Q and V values (same as parent class)
        response_mask = data.batch["response_mask"]
        q_values = q_values * response_mask
        v_values = v_values * response_mask
        
        # Handle dynamic batch size reordering (same as parent class)
        if use_dynamic_bsz:
            import itertools
            from verl.utils.seqlen_balancing import get_reverse_idx
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == q_values.size(0), f"{len(indices)} vs. {q_values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            q_values = q_values[revert_indices]
            v_values = v_values[revert_indices]
        
        # Return as DataProto for consistency with RAGEN interface
        return DataProto(batch={"q_values": q_values, "v_values": v_values})
    
    def _forward_micro_batch_archer(self, micro_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ArCHer Double Critic for a micro-batch.
        
        This method properly separates observation and action sequences as required
        by the ArCHer algorithm, then computes Q and V values.
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]  # Full sequence: prompt + response
            responses = micro_batch["responses"]   # Response part only
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            # Extract observation (prompt) and action (response) sequences
            # In RAGEN, input_ids = [prompt_tokens, response_tokens]
            # We need to separate them for proper ArCHer computation
            
            prompt_length = seqlen - response_length
            observation_ids = input_ids[:, :prompt_length]  # Prompt part
            action_ids = responses  # Response part (already provided)
            
            # Create attention masks for observation and action
            observation_mask = attention_mask[:, :prompt_length] if attention_mask is not None else None
            action_mask = torch.ones_like(action_ids)  # Responses are always valid
            
            # Extract position ids for observation (action doesn't need position_ids for pooling)
            obs_position_ids = position_ids[:, :prompt_length] if position_ids is not None else None

            if not self.use_remove_padding:
                # Forward through ArCHer critic with separated observation and action
                output = self.critic_module(
                    observation_ids=observation_ids,
                    observation_mask=observation_mask,
                    position_ids=obs_position_ids,
                    action_ids=action_ids,
                    action_mask=action_mask,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                
                # Extract Q and V values - they're already aligned to response length
                q_values = output.q1_values  # (batch_size, response_length)
                v_values = output.v1_values  # (batch_size, response_length)
                
                # Take minimum for stability (ArCHer approach)
                if hasattr(output, 'q2_values'):
                    q2_values = output.q2_values
                    q_values = torch.min(q_values, q2_values)
                    
                if hasattr(output, 'v2_values'):
                    v2_values = output.v2_values
                    v_values = torch.min(v_values, v2_values)
                
                return q_values, v_values
            else:
                # TODO: Implement remove_padding support for ArCHer if needed
                raise NotImplementedError("remove_padding not yet supported for ArCHer critic")
    
    def update_critic(self, data: DataProto):
        """
        Update the ArCHer Double Critic networks with proper Q1, Q2, V1, V2 loss computation.
        
        ArCHer uses a different loss structure:
        - Q1 → target_V1, Q2 → target_V2  
        - V1 → target_Q1, V2 → target_Q2
        
        This replaces traditional value loss with ArCHer's cross-training approach.
        """
        info = {}
        micro_batch_size = data.meta_info.get("micro_batch_size", 16)
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        
        # Use same micro-batching pattern as parent class
        batch = data.batch
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            from verl.utils.seqlen_balancing import rearrange_micro_batches
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        accumulated_loss = 0.0
        
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch_dict = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            else:
                micro_batch_dict = micro_batch

            loss, batch_info = self._update_critic_micro_batch_archer(micro_batch_dict)
            accumulated_loss += loss / len(micro_batches)
            
            # Accumulate logging info
            for key, value in batch_info.items():
                if key not in info:
                    info[key] = 0.0
                info[key] += value / len(micro_batches)

        # Backward pass
        self.accelerator.backward(accumulated_loss)
        
        # Gradient clipping (same as parent)
        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(self.critic_module.parameters(), self.max_grad_norm)
            
        # Optimizer step
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        
        # Soft update target networks (ArCHer-specific)
        self.soft_update_target_networks()
        
        return info
    
    def _update_critic_micro_batch_archer(self, micro_batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        ArCHer-specific critic update for a single micro-batch.
        
        Implements the cross-training loss:
        - Q1 learns from target_V1, Q2 learns from target_V2
        - V1 learns from target_Q1, V2 learns from target_Q2
        """
        info = {}
        
        # Extract sequences
        input_ids = micro_batch["input_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        batch, seqlen = input_ids.shape
        
        # Separate observation and action as in forward pass
        prompt_length = seqlen - response_length
        observation_ids = input_ids[:, :prompt_length]
        action_ids = responses
        
        attention_mask = micro_batch.get("attention_mask")
        position_ids = micro_batch.get("position_ids")
        
        observation_mask = attention_mask[:, :prompt_length] if attention_mask is not None else None
        action_mask = torch.ones_like(action_ids)
        obs_position_ids = position_ids[:, :prompt_length] if position_ids is not None else None
        
        # Multi-modal inputs handling
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward pass through current networks
            current_output = self.critic_module(
                observation_ids=observation_ids,
                observation_mask=observation_mask,
                position_ids=obs_position_ids,
                action_ids=action_ids,
                action_mask=action_mask,
                **multi_modal_inputs,
                use_cache=False,
            )
            
            # Extract current Q and V values (all 4 networks)
            q1_current = current_output.q1_values  # (batch_size, response_length)
            q2_current = getattr(current_output, 'q2_values', q1_current)
            v1_current = current_output.v1_values
            v2_current = getattr(current_output, 'v2_values', v1_current)
            
            # Forward pass through target networks (if enabled)
            if self.use_target_networks:
                with torch.no_grad():
                    target_output = self.target_critic_module(
                        observation_ids=observation_ids,
                        observation_mask=observation_mask,
                        position_ids=obs_position_ids,
                        action_ids=action_ids,
                        action_mask=action_mask,
                        **multi_modal_inputs,
                        use_cache=False,
                    )
                    
                    # Extract target Q and V values
                    q1_target = target_output.q1_values.detach()
                    q2_target = getattr(target_output, 'q2_values', q1_target).detach()
                    v1_target = target_output.v1_values.detach()
                    v2_target = getattr(target_output, 'v2_values', v1_target).detach()
            else:
                # Use current networks as targets (standard approach)
                with torch.no_grad():
                    q1_target = q1_current.detach()
                    q2_target = q2_current.detach()
                    v1_target = v1_current.detach()
                    v2_target = v2_current.detach()
            
            # Apply response mask to focus on valid response tokens
            response_mask = micro_batch.get("response_mask")
            if response_mask is not None:
                # Only compute loss on valid response tokens
                mask = response_mask.float()
                
                q1_current = q1_current * mask
                q2_current = q2_current * mask
                v1_current = v1_current * mask
                v2_current = v2_current * mask
                
                q1_target = q1_target * mask
                q2_target = q2_target * mask
                v1_target = v1_target * mask
                v2_target = v2_target * mask
                
                # Normalize by number of valid tokens
                num_valid_tokens = mask.sum()
            else:
                num_valid_tokens = q1_current.numel()
            
            # ArCHer cross-training losses
            criterion = torch.nn.MSELoss(reduction='sum')
            
            # Q networks learn from V targets, V networks learn from Q targets
            q1_loss = criterion(q1_current, v1_target) / num_valid_tokens
            q2_loss = criterion(q2_current, v2_target) / num_valid_tokens
            v1_loss = criterion(v1_current, q1_target) / num_valid_tokens
            v2_loss = criterion(v2_current, q2_target) / num_valid_tokens
            
            total_loss = q1_loss + q2_loss + v1_loss + v2_loss
            
            # Logging info
            info.update({
                'critic/archer_q1_loss': q1_loss.item(),
                'critic/archer_q2_loss': q2_loss.item(),
                'critic/archer_v1_loss': v1_loss.item(),
                'critic/archer_v2_loss': v2_loss.item(),
                'critic/archer_total_loss': total_loss.item(),
                'critic/q1_mean': q1_current.mean().item(),
                'critic/q2_mean': q2_current.mean().item(),
                'critic/v1_mean': v1_current.mean().item(),
                'critic/v2_mean': v2_current.mean().item(),
            })
        
        return total_loss, info
