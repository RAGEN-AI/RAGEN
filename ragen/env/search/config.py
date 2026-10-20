"""
Configuration for the Search (HotpotQA) environment.

The search environment is adapted from the RLLM project:
  https://github.com/rllm-org/rllm
  License: Apache-2.0
"""

from dataclasses import dataclass


@dataclass
class SearchEnvConfig:
    """Configuration for SearchEnv.

    Fields under env_config in config/envs.yaml map directly to these fields.
    """

    # --- Data ---
    dataset_name: str = "hotpotqa"
    train_path: str = "data/search/train.parquet"
    max_instances: int = 20000

    # --- Retrieval server ---
    retrieval_server_url: str = "http://127.0.0.1:8000"
    retrieval_timeout: float = 30.0
    max_search_results: int = 5
    max_total_chars: int = 4000  # total char limit for all docs combined (~1k tokens)

    # --- Environment ---
    max_steps: int = 10       # max search rounds before forced termination
    render_mode: str = "text"
    mock_mode: bool = False   # True = use MockRetrievalClient (no server needed)

    # --- Reward ---
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    f1_threshold: float = 0.3

    # Opt in to live web retrieval instead of the fixed Wikipedia corpus.
    # mock_mode takes precedence and never makes network requests.
    retrieval_backend: str = "local"  # local | parallel
