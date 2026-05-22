import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from .utils import (
    generate_room,
    collect_entity_coordinates,
    format_coordinate_render,
)
# from gym_sokoban.envs.sokoban_env.utils import generate_room
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.utils import all_seed

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = self.config.render_mode
        self.observation_format = self.config.observation_format

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room, 
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs
        )

    def reset(self, seed=None, mode=None):
        sol_range = self.config.min_solution_steps
        min_sol = sol_range[0] if sol_range is not None else None
        max_sol = sol_range[1] if sol_range is not None else None

        for _ in range(self.config.max_reset_tries):
            try:
                with all_seed(seed):
                    room_fixed, room_state, box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
                sol_len = len(action_sequence)
                if (min_sol is not None and sol_len < min_sol) or \
                   (max_sol is not None and sol_len > max_sol):
                    seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                    continue
                self.room_fixed, self.room_state, self.box_mapping = room_fixed, room_state, box_mapping
                self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
                self.player_position = np.argwhere(self.room_state == 5)[0]
                return self.render()
            except (RuntimeError, RuntimeWarning):
                seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None

        # Fallback: generate without difficulty constraint
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, _ = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
        except (RuntimeError, RuntimeWarning):
            seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, _ = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
        self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
        self.player_position = np.argwhere(self.room_state == 5)[0]
        return self.render()
        
    def _box_target_distance(self):
        """Sum of Manhattan distances from each box to its nearest target."""
        box_positions = np.argwhere((self.room_state == 4) | (self.room_state == 3))
        target_positions = np.argwhere(self.room_fixed == 2)
        if len(box_positions) == 0 or len(target_positions) == 0:
            return 0
        return sum(
            min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in target_positions)
            for b in box_positions
        )

    def step(self, action: int):
        previous_pos = self.player_position
        if self.config.distance_reward_coeff != 0.0:
            prev_dist = self._box_target_distance()
        _, gym_reward, done, _ = GymSokobanEnv.step(self, action)
        success = self.boxes_on_target == self.num_boxes
        if self.config.ignore_gym_reward:
            reward = self.config.success_reward if success else 0.0
        else:
            reward = gym_reward
        if self.config.distance_reward_coeff != 0.0:
            new_dist = self._box_target_distance()
            reward += (prev_dist - new_dist) * self.config.distance_reward_coeff
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        if not action_effective and self.config.no_op_penalty != 0.0:
            reward += self.config.no_op_penalty
        info = {"action_is_effective": action_effective, "action_is_valid": True, "success": success}
        return next_obs, reward, done, info

    def render(self, mode=None):
        if mode in {'grid', 'coord', 'grid_coord'}:
            return self._render_text(mode)

        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return self._render_text(self.observation_format)
        if render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self, observation_format: str) -> str:
        if self.config.partial_obs:
            return self._render_partial_obs()
        if observation_format == 'grid':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        if observation_format == 'coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return format_coordinate_render(entity_coords, self.dim_room)
        if observation_format == 'grid_coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return "Coordinates: \n" + format_coordinate_render(entity_coords, self.dim_room) + "\n" + "Grid Map: \n" + self._render_text('grid')
        raise ValueError(f"Invalid observation_format: {observation_format}")

    def _render_partial_obs(self) -> str:
        """Render a (2w+1)×(2w+1) patch centred on the player, plus absolute position."""
        w = self.config.partial_obs_window
        r, c = int(self.player_position[0]), int(self.player_position[1])
        H, W = self.dim_room
        # Overlay player-on-target marker (value 6) before slicing
        room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        rows = []
        for dr in range(-w, w + 1):
            row_cells = []
            for dc in range(-w, w + 1):
                nr, nc = r + dr, c + dc
                cell = int(room[nr, nc]) if (0 <= nr < H and 0 <= nc < W) else 0
                row_cells.append(self.GRID_LOOKUP.get(cell, " ? "))
            rows.append(''.join(row_cells))
        grid_str = '\n'.join(rows)
        footer = (
            f"Position: ({r}, {c}) | "
            f"Step: {self.num_env_steps} | "
            f"Boxes on target: {self.boxes_on_target}/{self.num_boxes}"
        )
        return f"{grid_str}\n{footer}"
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    env = SokobanEnv(config)
    for i in range(10):
        print(env.reset(seed=1010 + i))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban1.png', np_img)
