from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig
from .lean.config import LeanEnvConfig
from .lean.env import LeanEnv
from .sudoku.config import SudokuEnvConfig
from .sudoku.env import SudokuEnv
from .deepcoder.config import DeepCoderEnvConfig
from .deepcoder.env import DeepCoderEnv
from .game_2048.config import Game2048EnvConfig
from .game_2048.env import Game2048Env
from .rubikscube.config import RubiksCube2x2Config
from .rubikscube.env import RubiksCube2x2Env
from .lights_out.config import LightsOutEnvConfig
from .lights_out.env import LightsOutEnv


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    'metamathqa': MetaMathQAEnv,
    'lean': LeanEnv,
    'deepcoder': DeepCoderEnv,
    'sudoku': SudokuEnv,
    'game_2048': Game2048Env,
    'rubikscube': RubiksCube2x2Env,
    'lights_out': LightsOutEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'deepcoder': DeepCoderEnvConfig,
    'lean': LeanEnvConfig,
    'sudoku': SudokuEnvConfig,
    'game_2048': Game2048EnvConfig,   
    'rubikscube': RubiksCube2x2Config,
    'lights_out': LightsOutEnvConfig,
}

try:
    from .alfworld.env import AlfredTXTEnv
    from .alfworld.config import AlfredEnvConfig
    REGISTERED_ENVS['alfworld'] = AlfredTXTEnv
    REGISTERED_ENV_CONFIGS['alfworld'] = AlfredEnvConfig
except ImportError:
    pass

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass

try:
    from .search.env import SearchEnv
    from .search.config import SearchEnvConfig
    REGISTERED_ENVS['search'] = SearchEnv
    REGISTERED_ENV_CONFIGS['search'] = SearchEnvConfig
except ImportError:
    pass
