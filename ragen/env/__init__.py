# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
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
from .deep_research.env import DeepResearchEnv
from .deep_research.config import DeepResearchEnvConfig


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    # 'alfworld': AlfredTXTEnv,
    'metamathqa': MetaMathQAEnv,
    'deep_research': DeepResearchEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'deep_research': DeepResearchEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
