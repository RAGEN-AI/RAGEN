"""
Search environment for HotpotQA-style multi-hop question answering.

This code is adapted from the RLLM project:
  https://github.com/rllm-org/rllm
  Original source: rllm/examples/search/ (ToolEnvironment + search example)
  License: Apache-2.0

Architecture follows RAGEN's WebShop pattern:
- Inherits BaseLanguageBasedEnv + gym.Env
- Actions are text strings: search[query] / finish[answer]
- Multi-turn: agent can search multiple times before answering
- Uses a local retrieval server, opt-in Parallel MCP, or mock_mode for testing
"""

import logging
import re
from uuid import uuid4
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import datasets

from ragen.env.base import BaseLanguageBasedEnv
from .config import SearchEnvConfig
from .reward import SearchRewardFn
from .retrieval_client import RetrievalClient, MockRetrievalClient

logger = logging.getLogger(__name__)


class SearchEnv(BaseLanguageBasedEnv, gym.Env):
    """
    Search-based QA environment.

    The agent receives a question and can:
    - search[query]: search for information via the retrieval server
    - finish[answer]: submit a final answer and receive a reward

    Reward is computed using F1/EM against the ground truth (HotpotQA standard).
    """

    def __init__(self, config: Optional[SearchEnvConfig] = None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else SearchEnvConfig()

        # Reward function
        self.reward_fn = SearchRewardFn(
            correct_reward=self.config.correct_reward,
            incorrect_reward=self.config.incorrect_reward,
            f1_threshold=self.config.f1_threshold,
        )

        # Retrieval client
        if self.config.mock_mode:
            self.client = MockRetrievalClient()
        elif self.config.retrieval_backend == "parallel":
            from .parallel_client import ParallelRetrievalClient
            self.client = ParallelRetrievalClient(
                timeout=self.config.retrieval_timeout,
                max_results=self.config.max_search_results,
                max_total_chars=self.config.max_total_chars,
            )
        elif self.config.retrieval_backend == "local":
            self.client = RetrievalClient(
                server_url=self.config.retrieval_server_url,
                timeout=self.config.retrieval_timeout,
                max_results=self.config.max_search_results,
                max_total_chars=self.config.max_total_chars,
            )
        else:
            raise ValueError(f"Unknown retrieval_backend: {self.config.retrieval_backend}")
        self._parallel = self.config.retrieval_backend == "parallel" and not self.config.mock_mode
        self._session_id = None

        # Load dataset
        self.data = self._load_data()

        # Per-episode state
        self.index = None
        self.ground_truth = None
        self.question = None
        self.step_count = 0

        self.render_cache = None

    def _load_data(self):
        """Load HotpotQA data from parquet file."""
        try:
            df = datasets.load_dataset(
                "parquet",
                data_files=self.config.train_path,
            )["train"]
            if self.config.max_instances and len(df) > self.config.max_instances:
                df = df.select(range(self.config.max_instances))
            logger.info(f"Loaded {len(df)} search questions from {self.config.train_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {self.config.train_path}: {e}")
            raise

    def reset(self, seed: Optional[int] = None, mode: Optional[str] = None) -> str:
        """
        Reset the environment with a new question.

        Args:
            seed: Deterministic seed for question selection.
            mode: Unused, kept for interface compatibility.

        Returns:
            Initial observation string containing the question.
        """
        gym.Env.reset(self, seed=seed)

        # Deterministic question selection (same pattern as CountdownEnv)
        self.index = seed % len(self.data) if seed is not None else 0
        item = self.data[self.index]

        self.question = item["question"]
        self.ground_truth = item["ground_truth"]
        self.step_count = 0
        # Each rollout episode owns its identity, even when the question/seed is
        # repeated. Related calls and retries keep this ID until the next reset.
        self._session_id = str(uuid4()) if self._parallel else None

        # Build initial observation (question only, no ground_truth exposed)
        self.render_cache = (
            f"Question: {self.question}\n"
            + self._available_actions()
        )
        return self.render()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: One of:
                - "search[query text]" — perform a retrieval search
                - "fetch[https://...]" — read a page (Parallel backend only)
                - "finish[answer text]" — submit final answer
                - anything else — treated as a direct answer (fallback)

        Returns:
            (observation, reward, done, info)
        """
        if self._parallel and self._session_id is None:
            raise RuntimeError("Call reset() before taking Parallel search actions")
        self.step_count += 1
        action = action.strip() if action else ""

        # --- Parse action ---
        if action.startswith("search[") and action.endswith("]"):
            return self._handle_search(action[7:-1])
        elif self._parallel and action.startswith("fetch[") and action.endswith("]"):
            results = self.client.fetch(action[6:-1], session_id=self._session_id)
            return self._handle_results(action[6:-1], results, "fetch")
        elif action.startswith("finish[") and action.endswith("]"):
            return self._handle_finish(action[7:-1])
        else:
            # Fallback: treat as a direct answer attempt
            return self._handle_fallback(action)

    def _handle_search(self, query: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle a search[query] action."""
        if self._parallel:
            results = self.client.search(query, top_k=self.config.max_search_results, session_id=self._session_id)
        else:
            results = self.client.search(query, top_k=self.config.max_search_results)
        return self._handle_results(query, results, "search")

    def _available_actions(self):
        fetch = ", fetch[<url>]" if self._parallel else ""
        return f"Available actions: search[<query>]{fetch}, finish[<answer>]"

    def _handle_results(self, query, results, action_type):

        # Check if max steps reached
        done = self.step_count >= self.config.max_steps
        reward = 0.0

        if done:
            # Forced termination — no answer provided
            self.render_cache = (
                f"{action_type.capitalize()} results for '{query}':\n{results}\n\n"
                f"Maximum search steps reached. Episode ended without an answer."
            )
        else:
            self.render_cache = (
                f"{action_type.capitalize()} results for '{query}':\n{results}\n\n"
                + self._available_actions()
            )

        info = {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": False,
            "action_type": action_type,
            "query": query,
        }
        return self.render(), reward, done, info

    def _handle_finish(self, answer: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle a finish[answer] action."""
        reward, metadata = self.compute_reward(answer, self.ground_truth)
        done = True

        self.render_cache = f"Your answer: {answer}. Reward: {reward:.2f}"

        info = {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": metadata.get("exact_match", False) or reward > 0,
            "action_type": "finish",
            "answer": answer,
            "reward_metadata": metadata,
        }
        return self.render(), reward, done, info

    def _handle_fallback(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle unrecognized action format — try to extract an answer from it."""
        if not action:
            # Empty action — invalid
            done = self.step_count >= self.config.max_steps
            self.render_cache = (
                "Invalid action. Use search[<query>] to search or finish[<answer>] to answer.\n"
                + self._available_actions()
            )
            return self.render(), 0.0, done, {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "action_type": "invalid",
            }

        # Try to extract an answer from free-form text
        extracted = self.reward_fn.extract_answer_from_response(action)
        reward, metadata = self.compute_reward(extracted, self.ground_truth)
        done = True

        self.render_cache = f"Your answer (extracted): {extracted}. Reward: {reward:.2f}"

        info = {
            "action_is_effective": True,
            "action_is_valid": False,  # not in correct format
            "success": metadata.get("exact_match", False) or reward > 0,
            "action_type": "fallback",
            "raw_action": action,
            "extracted_answer": extracted,
            "reward_metadata": metadata,
        }
        return self.render(), reward, done, info

    def compute_reward(self, answer: str, ground_truth) -> Tuple[float, dict]:
        """Compute reward using F1/EM evaluation."""
        return self.reward_fn.compute_reward(answer, ground_truth)

    def render(self, mode: Optional[str] = None) -> str:
        """Return cached render output."""
        return self.render_cache

    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Quick smoke test with mock mode
    config = SearchEnvConfig(
        train_path="data/search/train.parquet",
        mock_mode=True,
        max_steps=5,
    )
    try:
        env = SearchEnv(config)
        obs = env.reset(seed=42)
        print(f"=== Reset ===\n{obs}\n")

        obs, reward, done, info = env.step("search[test query]")
        print(f"=== Search ===\n{obs}\nReward: {reward}, Done: {done}\n")

        obs, reward, done, info = env.step("finish[test answer]")
        print(f"=== Finish ===\n{obs}\nReward: {reward}, Done: {done}, Info: {info}\n")
    except Exception as e:
        print(f"Smoke test failed (expected if no data): {e}")
