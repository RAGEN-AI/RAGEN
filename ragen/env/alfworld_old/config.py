from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    """configuration for text world AlfredEnv"""
    config_file: str = "./ragen/env/alfworld_old/alfworld_config.yaml"
    action_lookup: Dict[int, str] = field(default_factory=lambda: {
        1: "look",
        2: "inventory",
        3: "go to <receptacle>",
        4: "open <receptacle>",
        5: "close <receptacle>",
        6: "take <object> from <receptacle>",
        7: "move <object> to <receptacle>",
        8: "examine <something>",
        9: "use <object>",
        10: "heat <object> with <receptacle>",
        11: "clean <object> with <receptacle>",
        12: "cool <object> with <receptacle>",
        13: "slice <object> with <object>"
    })
    format_score: float = 0.1
    score: float = 1.0
    render_mode: str = "text"