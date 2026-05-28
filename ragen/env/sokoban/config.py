from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class SokobanEnvConfig:
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    search_depth: int = 300
    grid_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {0:" # ", 1:" _ ", 2:" O ", 3:" √ ", 4:" X ", 5:" P ", 6:" S "})
    grid_vocab: Optional[Dict[str, str]] = field(default_factory=lambda: {" # ": "wall", " _ ": "empty", " O ": "target", " √ ": "box on target", " X ": "box", " P ": "player", " S ": "player on target"})
    action_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {1:"Up", 2:"Down", 3:"Left", 4:"Right"})
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"
    observation_format: str = "grid"
    # Map difficulty control: only accept maps whose solution length is in [min, max]
    min_solution_steps: Optional[Tuple[int, int]] = None  # e.g. (2, 10)
    max_reset_tries: int = 1000
    # Reward shaping: only used when ignore_gym_reward=True
    success_reward: float = 1.0   # total reward given on success (replaces gym reward)
    ignore_gym_reward: bool = False  # if True, reward = success_reward on success, 0 otherwise
    # Distance-based reward shaping: added on top of gym/success reward
    # reward += (prev_box_target_dist - new_box_target_dist) * distance_reward_coeff
    # 0.0 = disabled (default). Try 0.1~0.5 to encourage box movement toward target.
    distance_reward_coeff: float = 0.0
    # Penalty added when the action has no effect (player position unchanged).
    # Discourages repeating invalid actions. E.g. -0.3 makes no-ops cost 4x a normal step.
    no_op_penalty: float = 0.0
    # Partial observation (POMDP mode): agent sees only a (2*window+1)^2 patch around itself.
    # window=1 → 3×3 view. Full board position is still appended so the agent can build a map.
    partial_obs: bool = False
    partial_obs_window: int = 1
    # If False, omit "Position: (r, c)" from partial-obs footer — forces the agent to
    # track its own position from the visual flow (harder, tests genuine map-building).
    show_position: bool = True

    def __post_init__(self):
        if self.dim_x is not None and self.dim_y is not None:
            self.dim_room = (self.dim_x, self.dim_y)
            delattr(self, 'dim_x')
            delattr(self, 'dim_y')
        if self.observation_format not in {"grid", "coord", "grid_coord"}:
            raise ValueError(f"Unsupported observation_format: {self.observation_format}")
