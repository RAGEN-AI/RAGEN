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
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float,
        archer_alpha: float = 0.1
    ):
    """ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL
    
    ArCHer combines a high-level off-policy RL algorithm that trains a value function
    with a low-level RL algorithm that trains a token-by-token policy.
    
    Based on "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL"
    by Yifei Zhou et al. (ICML 2024)
    
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward)
            shape: (bs, response_length)
        values: `(torch.Tensor)` 
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)` discounted factor for token-level rewards
        high_level_gamma: `(float)` discounted factor for high-level (turn-level) rewards  
        lam: `(float)` lambda value for GAE computation
        archer_alpha: `(float)` mixing coefficient between high-level and low-level value functions
        
    Returns:
        advantages: `(torch.Tensor)` shape: (bs, response_length)
        returns: `(torch.Tensor)` shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        # High-level value function (turn-level)
        high_level_values = torch.zeros_like(token_level_rewards)
        # Low-level value function (token-level)  
        low_level_values = values.clone()
        
        for b in range(batch_size):
            # Step 1: Compute high-level value function and advantages
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            
            # High-level GAE for turn-level rewards
            high_level_lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                if i < len(eos_positions) - 1:
                    next_pos = eos_positions[i + 1]
                    next_high_value = high_level_values[b, next_pos]
                else:
                    next_high_value = 0.0
                
                # High-level temporal difference error
                high_level_delta = (token_level_rewards[b, curr_pos] + 
                                  high_level_gamma * next_high_value - 
                                  high_level_values[b, curr_pos])
                
                high_level_lastgaelam = (high_level_delta + 
                                       high_level_gamma * lam * high_level_lastgaelam)
                
                # Update high-level value function
                high_level_values[b, curr_pos] = (high_level_lastgaelam + 
                                                high_level_values[b, curr_pos])
            
            # Step 2: Combine high-level and low-level value functions (ArCHer mixing)
            combined_values = (archer_alpha * high_level_values[b] + 
                             (1 - archer_alpha) * low_level_values[b])
            
            # Step 3: Compute token-level advantages using combined value function
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                
                if i < len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1]
                    nextvalue = combined_values[next_pos]
                else:
                    nextvalue = 0.0
                
                # Token-level temporal difference using combined value function
                delta = (token_level_rewards[b, curr_pos] + 
                        gamma * nextvalue - 
                        combined_values[curr_pos])
                
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + combined_values[curr_pos]
        
        # Normalize advantages
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
    
    # Test ArCHer
    archer_advantages, archer_returns = compute_archer_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95, 0.1)
    print("ArCHer advantages:", archer_advantages)
    print("ArCHer returns:", archer_returns)