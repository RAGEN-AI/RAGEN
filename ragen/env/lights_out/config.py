from dataclasses import dataclass


@dataclass
class LightsOutEnvConfig:
    size: int = 3
    scramble_depth: int = 4
    max_steps: int = 8
    render_mode: str = "text"
    shaped_reward: bool = True
    success_reward: float = 1.0
    invalid_action_score: float = -0.2

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("LightsOut size must be at least 2.")
        if self.scramble_depth < 0:
            raise ValueError("scramble_depth must be non-negative.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.render_mode != "text":
            raise ValueError("LightsOut only supports text render_mode.")
