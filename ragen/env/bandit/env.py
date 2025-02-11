"""
adapted from gym-bandits
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import copy
import re


from ragen.utils import NoLoggerWarnings
from ragen.env.base import BaseDiscreteActionEnv

class BanditEnv(BaseDiscreteActionEnv, gym.Env):
    """
    Wrapper for the BanditTenArmedGaussian environment to match the project structure.
    
    ## Description
    N armed bandit mentioned in Sutton and Barto's RL book. Each action
    represents pulling one of N arms, with rewards drawn from a normal distribution.

    ## Action Space
    The action shape is `(1,)` in the range `{1, N}` indicating which arm to pull.
    - 1-N: Pull corresponding arm

    ## Rewards
    - Mean of each arm is pulled from N(0, 1)
    - Actual reward of each arm is drawn from N(mean, 1)
    """

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1
    
    def __init__(
            self,
            n_arms: int = 2,
            seed: Optional[int] = None,
    ):
        self.n_arms = n_arms
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(n_arms, start=1)
        
        # Initialize reward distributions for each arm
        self.reward_means = None
        self.env_kwargs = {
            "n_arms": n_arms,
            "seed": seed,
        }
        
        # Initialize tracking variables
        self.last_action = None
        self.reward = 0

    def reset(self, seed=None, mode='text'):
        """Reset the environment and reward distributions"""
        gym.Env.reset(self, seed=seed)
            
        # Generate new reward distributions
        self.reward_means = self.np_random.normal(0, 1, size=self.n_arms)
        self.reward = 0
        self.last_action = None
        
        return self.render(mode)

    def step(self, action: int):
        """Take action (pull arm) and get reward"""
        assert isinstance(action, int)
        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {}
        
        assert action in self.get_all_actions(), f"Invalid action {action}"
        
        arm_idx = action - 1  # Convert to 0-based index
        reward = self.np_random.normal(self.reward_means[arm_idx], 1)
        self.last_action = action
        
        return self.render(), reward, False, {}

    def render(self, mode='text'):
        """Render the current state"""
        if mode == 'text':
            if self.last_action is None:
                return f"You are facing {self.ACTION_SPACE.n} slot machines (numbered {self.ACTION_SPACE.start} to {self.ACTION_SPACE.start + self.ACTION_SPACE.n - 1}). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.last_action}.\nWhich machine will you pull next?"
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def parse_update_info_to_obs(self, update_info, action_is_valid):
        observation, reward, done, _ = update_info
        if not action_is_valid:
            output_str = f"Invalid action. Please choose a number between {self.ACTION_SPACE.start} and {self.ACTION_SPACE.start + self.ACTION_SPACE.n - 1}.\nThe observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            output_str = f"After pulling the arm, the observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        return output_str


    def extract_action(self, text):
        """Extract action number from text response"""
        
        pattern = r'^\s*(\d+)\s*$'
        match = re.search(pattern, text.strip())
        if match is None:
            return self.INVALID_ACTION
        try:
            action = int(match.group(1))
        except (ValueError, IndexError):
            print(f"Invalid action when parsing in BanditEnv: {text}")
            return self.INVALID_ACTION
        if action in self.get_all_actions():
            return action
        else:
            return self.INVALID_ACTION


    def copy(self):
        """Create a copy of the environment"""
        new_env = BanditEnv(
            n_arms=self.env_kwargs["n_arms"],
            seed=self.env_kwargs["seed"]
        )
        new_env.reward_means = copy.deepcopy(self.reward_means)
        new_env.last_action = self.last_action
        new_env.reward = self.reward
        return new_env
    
    def success(self):
        return False
    


class TwoArmedBanditEnv(BaseDiscreteActionEnv, gym.Env):
    """
    Wrapper for the BanditTenArmedGaussian environment to match the project structure.
    Two-armed bandit environment, with one gold arm and one silver arm.

    ## Action Space
    The action shape is `(1,)` in the range `{1, 2}` indicating which arm to pull.
    - 1-2: Pull corresponding arm

    ## Rewards
    - For gold arm (A, action 1), the reward is drawn from one gaussian distribution
    - For silver arm (B, action 2), the reward is drawn from a mixture of two gaussian distributions, with
        - a larger mean and std compared to the gold arm,
        - P(A - B > 0) > 0.5
    """

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1
    
    def __init__(
            self,
            first_gold_arm: bool = True,
            seed: Optional[int] = None,
    ):
        """
        By default, the gold arm is the first arm.
        If first_gold_arm is False, the gold arm is the second arm, then needs to map the action space {1, 2} to {2, 1}.
        """
        self.n_arms = 2
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(self.n_arms, start=1)
        self.first_gold_arm = first_gold_arm
        
        # Initialize reward distributions for each arm

        self.env_kwargs = {
            "n_arms": 2,
            "first_gold_arm": first_gold_arm,
            "seed": seed,
        }
        
        # Initialize tracking variables
        self.last_action = None
        self.reward = 0

    def _gold_arm_reward_distribution(self):
        """f ~ N(10, 1)"""
        return self.np_random.normal(10, 1)

    def _silver_arm_reward_distribution(self):
        """s ~ 99/100 * N(5, 9) + 1/100 * N(10000, 4)"""
        if self.np_random.random() < 0.99:
            return self.np_random.normal(5, 9)
        else:
            return self.np_random.normal(10000, 4)

    def reset(self, seed=None, mode='text'):
        """Reset the environment and reward distributions"""
        gym.Env.reset(self, seed=seed)
            
        # Generate new reward distributions
        self.reward = 0
        self.last_action = None
        
        return self.render(mode)

    def step(self, action: int):
        """Take action (pull arm) and get reward"""
        assert isinstance(action, int)
        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {}
        
        assert action in self.get_all_actions(), f"Invalid action {action}"

        if not self.first_gold_arm: 
            # map the action space {1, 2} to {2, 1}
            action = 3 - action
        
        if action == 1:
            reward = self._gold_arm_reward_distribution()
        else:
            reward = self._silver_arm_reward_distribution()
        self.last_action = action
        
        return self.render(), reward, True, {} # only one step

    def render(self, mode='text'):
        """Render the current state"""
        if mode == 'text':
            if self.last_action is None:
                return f"You are facing {self.ACTION_SPACE.n} slot machines (numbered {self.ACTION_SPACE.start} to {self.ACTION_SPACE.start + self.ACTION_SPACE.n - 1}). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.last_action}."
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def parse_update_info_to_obs(self, update_info, action_is_valid):
        observation, reward, done, _ = update_info
        if not action_is_valid:
            output_str = f"Invalid action. Please choose a number between {self.ACTION_SPACE.start} and {self.ACTION_SPACE.start + self.ACTION_SPACE.n - 1}.\nThe observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            output_str = f"After pulling the arm, the observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        return output_str


    def extract_action(self, text):
        """Extract action number from text response"""
        
        pattern = r'^\s*(\d+)\s*$'
        match = re.search(pattern, text.strip())
        if match is None:
            return self.INVALID_ACTION
        try:
            action = int(match.group(1))
        except (ValueError, IndexError):
            print(f"Invalid action when parsing in TwoArmedBanditEnv: {text}")
            return self.INVALID_ACTION
        if action in self.get_all_actions():
            return action
        else:
            return self.INVALID_ACTION


    def copy(self):
        """Create a copy of the environment"""
        new_env = TwoArmedBanditEnv(
            first_gold_arm=self.env_kwargs["first_gold_arm"],
            seed=self.env_kwargs["seed"]
        )
        new_env.last_action = self.last_action
        new_env.reward = self.reward
        return new_env
    
    def success(self):
        return self.last_action is not None


if __name__ == "__main__":
    env = TwoArmedBanditEnv(first_gold_arm=True)
    print(env.reset(seed=0))
    print(env.step(1))

    # print(BanditEnv.execute_predictions([env], ["<answer>9</answer>"], "<PAD>"))