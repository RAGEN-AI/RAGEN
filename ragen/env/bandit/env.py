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
    Two-armed bandit environment with low-risk and high-risk arms.
    Default names are "phoenix" (low-risk) and "dragon" (high-risk).
    
    ## Action Space
    The action shape is `(1,)` in the range `{1, 2}` indicating which arm to pull.
    - 1: Pull low-risk arm (default: phoenix)
    - 2: Pull high-risk arm (default: dragon)

    ## Rewards
    - For low-risk arm (L ~ f_l), the reward is drawn from one gaussian distribution
        - f_l ~ N(0.02, 0.01)
        - mean ~ 0.2, std ~ 0.1
    - For high-risk arm (H ~ f_h), the reward is drawn from a mixture of two gaussian distributions
        - f_h ~ 3/4 * N(0.01, 0.01) + 1/4 * N(1, 0.01)
        - mean ~ 0.325, std ~ 0.4
    - P(L - H > 0) ~ 0.57 > 0.5

    NOTE only one step
    """

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = 0 # -0.1
    
    def __init__(
            self,
            low_risk_name_train: str = "phoenix",
            high_risk_name_train: str = "dragon",
            low_risk_name_val: str = "phoenix",
            high_risk_name_val: str = "dragon",
            seed: Optional[int] = None,
    ):
        """
        Initialize the environment with configurable arm names.
        
        Args:
            low_risk_name: Name for the low-risk arm (default: "phoenix")
            high_risk_name: Name for the high-risk arm (default: "dragon")
            seed: Random seed
        """
        self.n_arms = 2
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(self.n_arms, start=1)

        self.mode = 'train'
        self.seed = seed

        self.low_risk_name_train = low_risk_name_train
        self.high_risk_name_train = high_risk_name_train
        self.low_risk_name_val = low_risk_name_val
        self.high_risk_name_val = high_risk_name_val

        self._set_name_mappings(self.mode)
        
        # Initialize tracking variables
        self.last_action = None
        self.reward = 0
        self._success = False

    def _set_name_mappings(self, mode):
        """
        Set up arm names and mappings based on the mode.
        
        Args:
            mode: Mode of the environment (train or val)
        """
        if mode == 'train':
            self.low_risk_name = self.low_risk_name_train
            self.high_risk_name = self.high_risk_name_train
        elif mode == 'val':
            self.low_risk_name = self.low_risk_name_val
            self.high_risk_name = self.high_risk_name_val
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.ARM_IDX_TO_NAME = {
            1: self.low_risk_name,
            2: self.high_risk_name,
        }
        self.NAME_TO_ARM_IDX = {
            self.low_risk_name: 1,
            self.high_risk_name: 2,
        }

        # Store initialization parameters
        self.env_kwargs = {
            "n_arms": self.n_arms,
            "low_risk_name_train": self.low_risk_name_train,
            "high_risk_name_train": self.high_risk_name_train,
            "low_risk_name_val": self.low_risk_name_val,
            "high_risk_name_val": self.high_risk_name_val,
            "seed": self.seed,
        }
    
    def update_mode(self, mode):
        """
        Update the mode of the environment and reset arm names and mappings.
        Args:
            mode: New mode of the environment (train or val)
        """

        self.mode = mode
        assert self.mode in ['train', 'val'], f"Invalid mode: {self.mode}"
        self._set_name_mappings(self.mode)


    def _low_risk_arm_reward_distribution(self):
        """
        Distribution of low-risk arm:
            f_l ~ N(0.2, 0.01)
        """
        # return self.np_random.normal(0.05, 0.1)
        return self.np_random.normal(0.2, 0.1)

    def _high_risk_arm_reward_distribution(self):
        """
        Distribution of high-risk arm: 
            f_h ~ 3/4 * N(0.1, 0.01) + 1/4 * N(1, 0.01)
        """

        # if self.np_random.random() < 0.9:
        #     return self.np_random.normal(0.1, 0.1)
        # else:
        #     return self.np_random.normal(4, 0.1)

        if self.np_random.random() < 0.75:
            return self.np_random.normal(0.1, 0.1)
        else:
            return self.np_random.normal(1, 0.1)

    def reset(self, seed=None, mode='text'):
        """Reset the environment and reward distributions"""
        gym.Env.reset(self, seed=seed)
            
        # Reset tracking variables
        self.reward = 0
        self.last_action = None
        self._success = False
        
        return self.render(mode)

    def step(self, action: int):
        """
        Take action (pull arm) and get reward
        - action = 1: pull low-risk arm
        - action = 2: pull high-risk arm
        """
        assert isinstance(action, int)
        self._success = True
        if action == self.INVALID_ACTION:
            return self.render(), 0, True, {}
        
        assert action in self.get_all_actions(), f"Invalid action {action}"
        
        if action == 1:
            reward = self._low_risk_arm_reward_distribution()
        else:
            reward = self._high_risk_arm_reward_distribution()
        self.last_action = action
        
        return self.render(), reward, True, {}  # only one step

    def render(self, mode='text'):
        """Render the current state"""
        assert mode in ['text', 'rgb_array', 'tiny_rgb_array']
        if mode in ['text', 'tiny_rgb_array']:
            if self.last_action is None:
                return f"You are facing {self.ACTION_SPACE.n} slot machines ({self.low_risk_name} and {self.high_risk_name}). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.ARM_IDX_TO_NAME[self.last_action]}."
        
        if mode == 'rgb_array':
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

    def parse_update_info_to_obs(self, update_info, action_is_valid):
        observation, reward, done, _ = update_info
        if not action_is_valid:
            output_str = f"Invalid action. Please choose between {self.low_risk_name} and {self.high_risk_name}.\nThe observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            output_str = f"After pulling the arm, the observation is:\n{observation}\nreward: {reward}\ndone: {done}\n"
        return output_str
        
    def extract_action(self, text):
        """Extract action number or arm names from text response"""
        pattern = f'^\s*(\d+|{self.low_risk_name}|{self.high_risk_name})\s*$'
        match = re.search(pattern, text.strip(), re.IGNORECASE)
        if match is None:
            return self.INVALID_ACTION
        try:
            if match.group(1).isdigit():
                action = int(match.group(1))
                if action in self.get_all_actions():
                    return action
                else:
                    return self.INVALID_ACTION
            else:
                keyword_action = match.group(1).lower()
                return self.NAME_TO_ARM_IDX[keyword_action]
        except (ValueError, IndexError):
            print(f"Invalid action when parsing in TwoArmedBanditEnv: {text}")
            return self.INVALID_ACTION

    def copy(self):
        """Create a copy of the environment"""
        new_env = TwoArmedBanditEnv(
            low_risk_name_train=self.env_kwargs["low_risk_name_train"],
            high_risk_name_train=self.env_kwargs["high_risk_name_train"],
            low_risk_name_val=self.env_kwargs["low_risk_name_val"],
            high_risk_name_val=self.env_kwargs["high_risk_name_val"],
            seed=self.env_kwargs["seed"]
        )
        new_env.last_action = self.last_action
        new_env.reward = self.reward
        new_env._success = self._success
        new_env.mode = self.mode
        new_env._set_name_mappings(self.mode)
        return new_env
    
    def success(self):
        return self._success
    
    def get_last_action(self) -> int:
        if self.last_action is None:
            return self.INVALID_ACTION
        return self.last_action




if __name__ == "__main__":
    # Initialize environment
    env = TwoArmedBanditEnv(
        low_risk_name_train="phoenix", 
        high_risk_name_train="dragon",
        low_risk_name_val="unicorn", 
        high_risk_name_val="hydra"
    )

    # Test train mode
    print("=== Train Mode ===")
    print(env.reset(seed=0))
    print(env.step(1))

    # Switch to validation mode
    print("\n=== Validation Mode ===")
    env.update_mode('val')
    print(env.reset(seed=0))
    print(env.step(2))

    # Distribution statistics
    r_low = []
    r_high = []
    for i in range(10000):
        env.reset(seed=i)
        r_low.append(env._low_risk_arm_reward_distribution())
        r_high.append(env._high_risk_arm_reward_distribution())
    
    print("\n=== Distribution Stats ===")
    print(f"Low risk mean: {np.mean(r_low):.3f}, std: {np.std(r_low):.3f}")
    print(f"High risk mean: {np.mean(r_high):.3f}, std: {np.std(r_high):.3f}")