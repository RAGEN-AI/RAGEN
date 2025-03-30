
from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass

@dataclass
class CountdownEnvConfig:
    train_path: str = "data/countdown/train.parquet"
    max_instances: int = 20000
    