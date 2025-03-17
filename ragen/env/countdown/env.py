import gymnasium as gym
from ragen.env.base import BaseLanguageBasedEnv
import datasets
import re
import itertools
from ragen.env.countdown.config import CountdownEnvConfig


def check_format(equation, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
        return sorted(nums_in_eq) == sorted(nums)
    except:
        return False

def check_correctness(equation_str, target):
    try:
        result = eval(equation_str, {"__builtins__": None}, {})
        return abs(result - target) < 1e-5
    except:
        return False

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """Score the countdown task solution."""
    target = ground_truth['target']
    nums = ground_truth['nums']
    if not check_format(solution_str, nums):
        return 0
    if not check_correctness(solution_str, target):
        return format_score
    else:
        return score

def has_solution(nums, target):
    """Check if there is a valid equation using each number exactly once."""
    # pad nums all to 4 numbers
    length = 4
    nums = nums + [0] * (length - len(nums))
    # +- num1 +- num2 +- num3 +- num4 = target, try all
    combinations = list(itertools.product([1, -1], repeat=length))
    for combination in combinations:
        if sum(combination[i] * nums[i] for i in range(length)) == target:
            return True
    return False

class CountdownEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else CountdownEnvConfig()
        self.invalid_act = self.config.invalid_act
        self.invalid_act_score = self.config.invalid_act_score
        self.data = self._get_data_from_parquet(self.config.train_path)
        self.index = None

    def _get_data_from_parquet(self, path):
        df = datasets.load_dataset("parquet", data_files=path)['train'].select(range(self.config.max_instances))
        df = df.filter(lambda x: has_solution(x['nums'], x['target']))
        return df

    def reset(self, seed=None, mode='text'):
        gym.Env.reset(self, seed=seed)
        self.index = seed % len(self.data)
        data = self.data[self.index]
        return f"Target: {data['target']}, nums: {data['nums']}"

    def step(self, action):
        if not isinstance(action, str) or action == self.invalid_act:
            return "", self.invalid_act_score, True, {"action_is_effective": False}
        
        reward = compute_score(action, self.data[self.index])
        next_obs, done, info = f"Your answer get {reward} points.", True, {"action_is_effective": True}
        return next_obs, reward, done, info


if __name__ == "__main__":
    def test(path, seed=43):
        config = CountdownEnvConfig(train_path=path)
        env = CountdownEnv(config)
        obs = env.reset(seed=seed)
        problem = env.data[env.index]
        solution = f"- {problem['nums'][0]} + {problem['nums'][1]} + {problem['nums'][2]}"
        _, reward, _, _ = env.step(solution)
        print(f"{obs}\nSolution: {solution}, Reward: {reward}")
    
    test("data/countdown/countdown_train.parquet")