from dataclasses import dataclass, field
from typing import Any

from webshop_minimal.utils import (
    DEFAULT_FILE_PATH,
)

@dataclass
class WebShopEnvConfig:
    """Configuration for WebAgentText environment"""
    dataset: str = field(
        default="small",
        metadata={"description": "Small or full dataset"}
    )
    observation_mode: str = field(
        default="text", 
        metadata={"choices": ["html", "text"]}
    )
    file_path: str = field(
        default=DEFAULT_FILE_PATH, 
        metadata={"description": "File path for SimServer"}
    )  # TODO: Remove hardcoded file path
    server: Any = field(
        default=None, 
        metadata={"description": "If None, use SimServer"}
    )
    filter_goals: Any = field(
        default=None, 
        metadata={"description": "SimServer arg: Custom function to filter specific goals for consideration"}
    )
    limit_goals: int = field(
        default=-1, 
        metadata={"description": "SimServer arg: Limit the number of goals available"}
    )
    num_products: int = field(
        default=None, 
        metadata={"description": "SimServer arg: Number of products to search across"}
    )
    human_goals: bool = field(
        default=False, 
        metadata={"description": "SimServer arg: Load human goals if True, otherwise synthetic goals"}
    )
    show_attrs: bool = field(
        default=False, 
        metadata={"description": "SimServer arg: Whether to show additional attributes"}
    )
