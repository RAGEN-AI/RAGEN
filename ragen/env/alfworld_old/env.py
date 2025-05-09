"""
This is the environment for the ALFRED dataset.
author: Qineng Wang
date: 2025-03-30
"""
import random
import textworld
import textworld.gym
import numpy as np
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos
from ragen.env.base import BaseLanguageBasedEnv
from .config import AlfredEnvConfig
from .utils import load_config, check_format

class AlfredTXTEnv(BaseLanguageBasedEnv):

    # raw_env: AlfredTWEnv = AlfredTWEnv(config=load_config(AlfredEnvConfig().config_file), train_eval="train")
    # print("initializing alfworld env")
    # NOTE Currently raw_env cannot customize config.

    def __init__(self, config: AlfredEnvConfig = AlfredEnvConfig()):
        super().__init__()
        self.config = config
        self.ACTION_LOOKUP = self.config.action_lookup
        # raw_env_config = load_config(self.config.config_file)
        # self.raw_env = AlfredTWEnv(config=raw_env_config, train_eval="train")
        self.num_games = self.raw_env.num_games
        self.game_files = self.raw_env.game_files
        # print(f"Overall we have {len(self.game_files)} games in split={self.raw_env.train_eval}")
        # self.alfred_env = self.raw_env.init_env(batch_size=1)
        self.current_game_file = None
        self.render_cache = None
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
    
    def reset(self, seed=None, mode=None):
        """
        Reset the environment with a specific seed.
        If seed is provided, it deterministically selects a specific game file.
        """
        try:
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                game_idx = seed % len(self.game_files)
                selected_game = self.game_files[game_idx]
            else:
                selected_game = random.choice(self.game_files)
            
            self.current_game_file = selected_game
            
            if hasattr(self, 'alfred_env') and self.alfred_env is not None:
                self.alfred_env.close()
            
            request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
            config = load_config(self.config.config_file)
            wrappers = [AlfredDemangler(), AlfredInfos()]
            max_steps = config["rl"]["training"]["max_nb_steps_per_episode"]
            
            env_id = textworld.gym.register_game(
                selected_game, 
                request_infos=request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=max_steps,
                wrappers=wrappers
            )
            
            self.alfred_env = textworld.gym.make(env_id)
            
            obs, info = self.alfred_env.reset()
            self.render_cache = obs[0]
            return self.render_cache
            
        except (RuntimeError, RuntimeWarning) as e:
            print(f"Error in reset: {e}")
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
    
    def compute_score(self, base_reward, valid_action, done):
        """
        Compute the score based on the base reward, format reward, and completion status.
        
        Args:
            base_reward: The reward from the environment
            valid_action: Whether the action format is valid
            done: Whether the episode is finished
            
        Returns:
            The computed score
        """
        if done:
            return self.config.score + self.config.format_score + base_reward
        elif valid_action:
            return base_reward + self.config.format_score
        else:
            return 0.0
    
    def step(self, action: str):
        """
        Take a step in the environment using the provided action string.
        The action must match one of the templates in ACTION_LOOKUP.
        """
        valid_action = check_format(action, self.ACTION_LOOKUP.values())
        
        if not valid_action:
            return f"Invalid action format: {action}", 0, False, {"action_is_effective": False, "action_is_valid": False, "success": False}
        
        obs, rewards, dones, infos = self.alfred_env.step([action])  # BatchEnv expects a list of commands
        
        observation = obs[0]
        self.render_cache = observation
        base_reward = rewards[0]
        done = dones[0]
        info = {"action_is_effective": True, "action_is_valid": True, "success": done}
        
        reward = self.compute_score(base_reward, valid_action, done)
        
        return self.render_cache, reward, done, info
    
    def render(self):
        return self.render_cache
    
    def close(self):
        self.render_cache = None
        self.alfred_env.close()

if __name__ == "__main__":
    env = AlfredTXTEnv()
    
    # Test resetting environment with same seed
    print("\n\n=== Testing environment reset with same seed ===")
    seed = 42
    obs1 = env.reset(seed)
    print(f"First observation with seed={seed}: {obs1}")
    game_file1 = env.current_game_file
    print(f"Loaded game file: {game_file1}")
    print("-"*100)
    
    # Using same seed again
    obs2 = env.reset(seed)
    print(f"Second observation with seed={seed}: {obs2}")
    game_file2 = env.current_game_file
    print(f"Loaded game file: {game_file2}")
    print(f"Both loaded game files are identical: {game_file1 == game_file2}")
    print("-"*100)
    # Test different seed
    print("\n\n=== Testing different seed ===")
    seed = 1000
    obs1 = env.reset(seed)
    print(f"First observation with seed={seed}: {obs1}")
    game_file1 = env.current_game_file
    print(f"Loaded game file: {game_file1}")
    print("-"*100)

    # Test step method
    print("\n=== Testing step method ===")
    # Try "look" action
    action = "look"
    print(f"Executing action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}...")
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    
    # Try "inventory" action
    action = "inventory"
    print(f"Executing action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}...")
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    
    # Test with a templated action
    action = "go to garbagecan 1"
    print(f"Executing action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}...")
    print(f"Reward: {reward}, Done: {done}, Info: {info}")

    # Test next action "go to chair 1"
    action = "go to chair 1"
    print(f"Executing action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}...")
    print(f"Reward: {reward}, Done: {done}, Info: {info}")

    # Test an invalid action
    action = "goto chair 2"
    print(f"Executing action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}...")
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    