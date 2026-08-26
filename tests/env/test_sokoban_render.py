import re

from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.env.sokoban.env import SokobanEnv


def test_sokoban_render_supports_grid_and_coord():
    seed = 1234
    grid_config = SokobanEnvConfig(
        dim_room=(5, 5),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
        observation_format="grid",
    )
    coord_config = SokobanEnvConfig(
        dim_room=(5, 5),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
        observation_format="coord",
    )

    grid_env = SokobanEnv(grid_config)
    coord_env = SokobanEnv(coord_config)

    try:
        grid_obs = grid_env.reset(seed=seed)
        coord_obs = coord_env.reset(seed=seed)

        assert isinstance(grid_obs, str)
        assert isinstance(coord_obs, str)

        assert "Board size:" in coord_obs
        assert re.search(r"(Targets|Boxes|Player): \(\d+, \d+\)", coord_obs)

        assert isinstance(coord_env.render(mode="grid"), str)
        assert "Board size:" in grid_env.render(mode="coord")
    finally:
        grid_env.close()
        coord_env.close()
