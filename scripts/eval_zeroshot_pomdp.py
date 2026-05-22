#!/usr/bin/env python3
"""
Zero-shot evaluation of Instruct LLMs on POMDP Sokoban.

Each episode runs full-context: all previous (observation, response) pairs
are kept in the conversation, giving the model the best possible chance.
This is the upper-bound zero-shot baseline — no RL training involved.

GPU requirements:
  Qwen2.5-7B-Instruct  → 1× RTX 4090 (24 GB) in fp16
  Qwen2.5-14B-Instruct → 1× A100-40GB in fp16
                         or add --tensor_parallel_size 2 for 2× RTX 4090

Usage:
  python scripts/eval_zeroshot_pomdp.py
  python scripts/eval_zeroshot_pomdp.py --model Qwen/Qwen2.5-14B-Instruct --tensor_parallel_size 2
  python scripts/eval_zeroshot_pomdp.py --model Qwen/Qwen2.5-7B-Instruct --n_episodes 100
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ── env imports (must be run from RAGEN root) ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.env.sokoban.env import SokobanEnv

# ── constants ─────────────────────────────────────────────────────────────────
ACTION_MAP = {"up": 1, "down": 2, "left": 3, "right": 4}
ANSWER_RE = re.compile(r"<answer>\s*(up|down|left|right)\s*</answer>", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are navigating a Sokoban board. You can only see a 3×3 area around you.
Goal: push all boxes ( X ) onto targets ( O ). You are P.
Symbols: # =wall, _ =empty, O =target, X =box, √ =box on target, P =you.

MEMORY RULE: previous observations are discarded — only your <think> block
carries forward. Use it to maintain a running map of the 5×5 board.

In <think>, keep:
  MAP: a 5×5 grid using the same symbols above, ? for unseen cells.
       Update it each turn by filling in the 3×3 patch you currently see.
       Overwrite box positions when a box moves.
  PLAN: one sentence on what you intend to do next.

After <think>, output exactly ONE action:
<answer>Up</answer>, <answer>Down</answer>, <answer>Left</answer>, or <answer>Right</answer>.\
"""


# ── data classes ──────────────────────────────────────────────────────────────
@dataclass
class EpisodeResult:
    episode_id: int
    success: bool
    steps: int
    total_reward: float
    final_boxes_on_target: int
    num_boxes: int
    parse_failures: int          # turns where no valid action was parsed
    time_seconds: float


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_action(text: str) -> Optional[int]:
    """Extract action integer from model output. Returns None on parse failure."""
    m = ANSWER_RE.search(text)
    if m:
        return ACTION_MAP[m.group(1).lower()]
    # Fallback: scan for bare direction words
    for word, act in ACTION_MAP.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            return act
    return None


def build_messages(history: list[dict], new_obs: str) -> list[dict]:
    """
    Append the new observation as a user message and return the full history.
    history is a list of {"role": ..., "content": ...} dicts.
    """
    return history + [{"role": "user", "content": new_obs}]


def make_env(seed: int) -> SokobanEnv:
    cfg = SokobanEnvConfig(
        dim_x=5, dim_y=5, num_boxes=1, max_steps=150,
        search_depth=100, partial_obs=True, partial_obs_window=1,
        ignore_gym_reward=True, success_reward=1.0,
        distance_reward_coeff=0.3, no_op_penalty=-0.01,
    )
    env = SokobanEnv(cfg)
    env.reset(seed=seed)
    return env


# ── main eval loop ────────────────────────────────────────────────────────────
def run_eval(args) -> list[EpisodeResult]:
    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}  (TP={args.tensor_parallel_size})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,      # greedy — most fair for zero-shot baseline
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results: list[EpisodeResult] = []

    for ep in range(args.n_episodes):
        seed = args.seed + ep
        t0 = time.time()

        env = make_env(seed)
        obs = env.render()

        # Conversation history: system prompt is a persistent first turn
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        total_reward = 0.0
        parse_failures = 0
        success = False
        step = 0

        for step in range(args.max_steps):
            messages = build_messages(history, obs)

            # vllm chat-style generation
            from vllm.entrypoints.chat_utils import apply_chat_template
            tokenizer = llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = llm.generate([prompt], sampling)
            response_text = outputs[0].outputs[0].text

            # Save assistant response to history
            history = messages + [{"role": "assistant", "content": response_text}]

            # Parse action
            action = parse_action(response_text)
            if action is None:
                parse_failures += 1
                action = 4  # default: Right (arbitrary, non-destructive fallback)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            success = info.get("success", False)

            if done or success:
                step += 1
                break

        elapsed = time.time() - t0
        result = EpisodeResult(
            episode_id=ep,
            success=success,
            steps=step,
            total_reward=round(total_reward, 4),
            final_boxes_on_target=env.boxes_on_target,
            num_boxes=env.num_boxes,
            parse_failures=parse_failures,
            time_seconds=round(elapsed, 2),
        )
        results.append(result)

        status = "✓ SOLVED" if success else "✗"
        print(
            f"  ep {ep:3d} | {status} | steps={step:3d} | "
            f"reward={total_reward:5.2f} | parse_fail={parse_failures} | {elapsed:.1f}s"
        )

    return results


def print_summary(results: list[EpisodeResult], model: str) -> None:
    n = len(results)
    solved = [r for r in results if r.success]
    success_rate = len(solved) / n * 100
    avg_steps = np.mean([r.steps for r in results])
    avg_steps_solved = np.mean([r.steps for r in solved]) if solved else float("nan")
    avg_reward = np.mean([r.total_reward for r in results])
    avg_parse_fail = np.mean([r.parse_failures for r in results])

    print("\n" + "=" * 60)
    print(f"Model            : {model}")
    print(f"Episodes         : {n}")
    print(f"Success rate     : {len(solved)}/{n}  ({success_rate:.1f}%)")
    print(f"Avg steps (all)  : {avg_steps:.1f}")
    print(f"Avg steps (solved): {avg_steps_solved:.1f}")
    print(f"Avg total reward : {avg_reward:.4f}")
    print(f"Avg parse failures/ep: {avg_parse_fail:.1f}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot POMDP Sokoban eval")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID (default: Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50,
        help="Number of episodes to evaluate (default: 50)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=150,
        help="Max steps per episode (default: 150)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed; episode i uses seed+i (default: 42)"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="vllm tensor parallelism — use 2 for 14B on 2×GPU (default: 1)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save per-episode JSON results to this path (default: auto)"
    )
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        args.output = f"results/zeroshot_{safe_name}_n{args.n_episodes}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nZero-shot POMDP Sokoban evaluation")
    print(f"Model: {args.model}  |  Episodes: {args.n_episodes}  |  Max steps: {args.max_steps}\n")

    results = run_eval(args)
    print_summary(results, args.model)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nPer-episode results saved to: {args.output}")


if __name__ == "__main__":
    main()
