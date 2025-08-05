from verl.trainer.ppo.core_algos import *

# Import torch modules needed for ArCHer
import torch
import torch.nn.functional as F
from verl.utils import torch_functional as verl_F

# supported by Kangrui Wang
def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]
            
            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


def compute_archer_advantage_return(
        q_values: torch.Tensor,
        v_values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float = 0.99,
        **kwargs  # Accept other parameters for compatibility but ignore them
    ):
    """ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL
    
    ArCHer computes advantages as A = Q(s,π(s)) - V(s) where Q and V come from 
    the Double Critic architecture. This bridges high-level (turn-level) and 
    low-level (token-level) learning in the hierarchical structure.
    
    Based on "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL"
    by Yifei Zhou et al. (ICML 2024)
    
    The key insight is that Q-values capture the action-conditioned expected return
    while V-values capture the state value, and their difference gives the advantage
    of taking the specific action (token) in that state.
    
    Args:
        q_values: `(torch.Tensor)` Q-values Q(s,π(s)) from the critic
            shape: (bs, response_length)
        v_values: `(torch.Tensor)` V-values V(s) from the critic  
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)` 
            shape: (bs, response_length). 1 for valid tokens, 0 for padding
        gamma: `(float)` discount factor (kept for compatibility)
        **kwargs: Other parameters ignored for compatibility with existing interface
        
    Returns:
        advantages: `(torch.Tensor)` A = Q(s,π(s)) - V(s), shape: (bs, response_length)
        returns: `(torch.Tensor)` Q-values used as returns, shape: (bs, response_length)
    """
    with torch.no_grad():
        # ArCHer advantage computation: A = Q(s,π(s)) - V(s)
        # This is the core of ArCHer's hierarchical learning approach
        advantages = q_values - v_values
        
        # Use Q-values as returns since they represent the expected cumulative reward
        returns = q_values.clone()
        
        # Apply loss mask to zero out invalid positions  
        advantages = advantages * loss_mask
        returns = returns * loss_mask
        
        # Normalize advantages while respecting the mask
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


# set up unittest
if __name__ == "__main__":
    token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    loss_mask = torch.ones(1, 10)
    advantages, returns = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    print("Bi-level GAE advantages:", advantages)
    print("Bi-level GAE returns:", returns)
    
    # Test ArCHer - now requires Q and V values instead of rewards
    q_values = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]])
    v_values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]) 
    archer_advantages, archer_returns = compute_archer_advantage_return(q_values, v_values, loss_mask)
    print("ArCHer advantages:", archer_advantages)
    print("ArCHer returns:", archer_returns)