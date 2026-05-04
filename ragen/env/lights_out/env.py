import re
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from ragen.env.base import BaseLanguageBasedEnv
from .config import LightsOutEnvConfig


class LightsOutEnv(BaseLanguageBasedEnv, gym.Env):
    """
    Text-based Lights Out environment.

    The board is an N x N binary grid. Pressing a cell toggles that cell and its
    orthogonal neighbors. The goal is to turn all lights off.
    """

    def __init__(self, config: Optional[LightsOutEnvConfig] = None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else LightsOutEnvConfig()
        self.size = self.config.size
        self.render_mode = self.config.render_mode
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_step = 0
        self.render_cache = ""
        self.rng = np.random.default_rng()

    def reset(self, seed=None, mode=None):
        gym.Env.reset(self, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

        for _ in range(20):
            self.grid.fill(0)
            for _ in range(self.config.scramble_depth):
                row = int(self.rng.integers(0, self.size))
                col = int(self.rng.integers(0, self.size))
                self._toggle(row, col)
            if self.config.scramble_depth == 0 or self._lit_count() > 0:
                break

        if self.config.scramble_depth > 0 and self._lit_count() == 0:
            center = self.size // 2
            self._toggle(center, center)

        self.render_cache = self.render()
        return self.render_cache

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        self.current_step += 1
        ok, row, col, error = self._parse_action(action)
        if not ok:
            done = self.current_step >= self.config.max_steps
            reward = self.config.invalid_action_score
            self.render_cache = self.render(extra=f"Invalid action: {error}")
            return self.render_cache, reward, done, {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "lit_count": self._lit_count(),
                "lit_fraction": self._lit_fraction(),
                "lights_off_fraction": 1.0 - self._lit_fraction(),
                "improved": False,
                "raw_reward": reward,
                "error": error,
            }

        old_lit = self._lit_count()
        self._toggle(row, col)
        new_lit = self._lit_count()
        solved = new_lit == 0

        reward = 0.0
        if self.config.shaped_reward:
            reward += (old_lit - new_lit) / float(self.size * self.size)
        if solved:
            reward += self.config.success_reward

        done = solved or self.current_step >= self.config.max_steps
        msg = f"Toggled ({row}, {col})."
        if solved:
            msg += " Puzzle solved."
        elif done:
            msg += " Max steps reached."

        self.render_cache = self.render(extra=msg)
        return self.render_cache, reward, done, {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": solved,
            "lit_count": new_lit,
            "lit_fraction": self._lit_fraction(),
            "lights_off_fraction": 1.0 - self._lit_fraction(),
            "improved": new_lit < old_lit,
            "raw_reward": reward,
        }

    def render(self, mode: Optional[str] = None, extra: str = "") -> str:
        lines = []
        lines.append("=== Lights Out ===")
        lines.append(f"Step: {self.current_step}/{self.config.max_steps}")
        lines.append(f"Lights on: {self._lit_count()}/{self.size * self.size}")
        lines.append("")
        lines.append("Grid uses 1 for on and 0 for off. Coordinates are zero-indexed.")
        header = "     " + "  ".join(f"c{col}" for col in range(self.size))
        lines.append(header)
        for row in range(self.size):
            values = "  ".join(str(int(v)) for v in self.grid[row])
            lines.append(f"r{row}:  {values}")
        lines.append("")
        lines.append("Action format: <answer>toggle row col</answer>, for example <answer>toggle 0 1</answer>.")
        if extra:
            lines.append("")
            lines.append(extra)
        return "\n".join(lines)

    def compute_reward(self, action, **kwargs) -> float:
        ok, row, col, _ = self._parse_action(action)
        if not ok:
            return self.config.invalid_action_score
        old_lit = self._lit_count()
        temp = self.grid.copy()
        self._toggle(row, col)
        new_lit = self._lit_count()
        solved = new_lit == 0
        self.grid = temp
        reward = 0.0
        if self.config.shaped_reward:
            reward += (old_lit - new_lit) / float(self.size * self.size)
        if solved:
            reward += self.config.success_reward
        return reward

    def close(self):
        pass

    def _parse_action(self, action: str):
        if not action:
            return False, -1, -1, "empty action"
        text = str(action).strip().lower()
        text = re.sub(r"</?answer>", " ", text)
        numbers = [int(num) for num in re.findall(r"-?\d+", text)]
        if len(numbers) < 2:
            return False, -1, -1, "expected two zero-indexed coordinates"
        row, col = numbers[0], numbers[1]
        if not (0 <= row < self.size):
            return False, -1, -1, f"row {row} is outside 0..{self.size - 1}"
        if not (0 <= col < self.size):
            return False, -1, -1, f"col {col} is outside 0..{self.size - 1}"
        return True, row, col, ""

    def _toggle(self, row: int, col: int):
        for r, c in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < self.size and 0 <= c < self.size:
                self.grid[r, c] = 1 - self.grid[r, c]

    def _lit_count(self) -> int:
        return int(self.grid.sum())

    def _lit_fraction(self) -> float:
        return self._lit_count() / float(self.size * self.size)


if __name__ == "__main__":
    env = LightsOutEnv(LightsOutEnvConfig(size=3, scramble_depth=4, max_steps=8))
    print(env.reset(seed=42))
    obs, reward, done, info = env.step("toggle 0 0")
    print(obs)
    print({"reward": reward, "done": done, "info": info})
