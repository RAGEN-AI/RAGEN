from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class SokobanEnvConfig:
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    search_depth: int = 300
    grid_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {0:"#", 1:"_", 2:"O", 3:"√", 4:"X", 5:"P", 6:"S"})
    grid_vocab: Optional[Dict[str, str]] = field(default_factory=lambda: {"#": "wall", "_": "empty", "O": "target", "√": "box on target", "X": "box", "P": "player", "S": "player on target"})
    action_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {0:"None", 1:"Up", 2:"Down", 3:"Left", 4:"Right"})