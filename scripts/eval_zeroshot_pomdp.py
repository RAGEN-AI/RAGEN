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
from dataclasses import dataclass, asdict, field
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
<answer>Up</answer>, <answer>Down</answer>, <answer>Left</answer>, or <answer>Right</answer>.

KEY RULE: Position (r,c) means your 3×3 view covers MAP rows [r-1,r,r+1] and cols [c-1,c,c+1].
All other MAP cells stay ? until you visit them. Cells you saw before stay filled even when out of view.

EXAMPLE — study how the MAP is built and updated across turns:

Turn 1 | Position: (2,2)
State:  _  _  _
        _  P  _
        _  X  O
<think>
MAP:
? ? ? ? ?
? _ _ _ ?
? _ P _ ?
? _ X O ?
? ? ? ? ?
PLAN: Position (2,2) → view covers rows 1-3, cols 1-3; rows 0 and 4 unseen. X at (3,2), O at (3,3). To push X right onto O need to reach (3,1). Move Left then Down.
</think><answer>Left</answer>

Turn 2 | Position: (2,1)
State:  #  _  _
        #  P  _
        #  _  X
<think>
MAP:
? ? ? ? ?
# _ _ _ ?
# P _ _ ?
# _ X O ?
? ? ? ? ?
PLAN: Position (2,1) → view covers rows 1-3, cols 0-2. Left border (#) at col 0 now seen. O at (3,3) kept from Turn 1 memory (col 3 not in view now). X still at (3,2). Move Down to reach (3,1), then push X right onto O.
</think><answer>Down</answer>\
"""


# ── data classes ──────────────────────────────────────────────────────────────
@dataclass
class TurnRecord:
    turn: int
    position: Optional[tuple]       # (r, c) from env observation
    response: str                   # raw model output
    map_rows: int                   # number of rows found in MAP block (-1 if none)
    player_pos_correct: Optional[bool]  # is P at (r,c) in MAP?
    row0_correct: Optional[bool]    # is row 0 all '?' when r>=2?
    action_valid: bool              # was <answer> tag parseable?


@dataclass
class EpisodeResult:
    episode_id: int
    success: bool
    steps: int
    total_reward: float
    final_boxes_on_target: int
    num_boxes: int
    parse_failures: int
    time_seconds: float
    turns: list = field(default_factory=list)  # list of TurnRecord dicts


# ── MAP quality analysis ───────────────────────────────────────────────────────
def analyze_map(obs: str, response: str) -> TurnRecord:
    """Parse position from obs and check if MAP in response is correct."""
    # Extract position
    pos_match = re.search(r'Position:\s*\((\d+),\s*(\d+)\)', obs)
    r, c = (int(pos_match.group(1)), int(pos_match.group(2))) if pos_match else (None, None)

    # Extract think block
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    think = think_match.group(1) if think_match else ""

    # Extract MAP lines (exactly 5 rows of 5 tokens each)
    map_match = re.search(r'MAP:\s*((?:[?#_XOPS√ ]+\n?){1,7})', think)
    map_rows_raw = []
    if map_match:
        for line in map_match.group(1).strip().split('\n'):
            tokens = line.strip().split()
            if len(tokens) == 5:
                map_rows_raw.append(tokens)

    map_rows_count = len(map_rows_raw)

    # Check player position in MAP
    player_correct = None
    if r is not None and map_rows_count == 5:
        try:
            player_correct = map_rows_raw[r][c] in ('P', 'S')
        except IndexError:
            player_correct = False

    # Check row 0 is all '?' when r >= 2 (row 0 not visible)
    row0_correct = None
    if r is not None and r >= 2 and map_rows_count == 5:
        row0_correct = all(cell == '?' for cell in map_rows_raw[0])

    action_valid = bool(ANSWER_RE.search(response))

    return TurnRecord(
        turn=-1,  # filled in by caller
        position=(r, c) if r is not None else None,
        response=response,
        map_rows=map_rows_count,
        player_pos_correct=player_correct,
        row0_correct=row0_correct,
        action_valid=action_valid,
    )


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_action(text: str) -> Optional[int]:
    m = ANSWER_RE.search(text)
    if m:
        return ACTION_MAP[m.group(1).lower()]
    for word, act in ACTION_MAP.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            return act
    return None


def build_messages(history: list[dict], new_obs: str, max_context_window: int = -1) -> list[dict]:
    system = history[:1]  # always keep system message
    turns = history[1:]   # (user, assistant) pairs beyond system
    if max_context_window > 0:
        # each turn = 2 messages (user + assistant); keep last k pairs
        keep = max_context_window * 2
        turns = turns[-keep:]
    return system + turns + [{"role": "user", "content": new_obs}]


def make_env(seed: int, min_solution_steps: Optional[list] = None) -> SokobanEnv:
    cfg = SokobanEnvConfig(
        dim_x=5, dim_y=5, num_boxes=1, max_steps=150,
        search_depth=100, partial_obs=True, partial_obs_window=1,
        ignore_gym_reward=True, success_reward=1.0,
        distance_reward_coeff=0.3, no_op_penalty=-0.01,
        min_solution_steps=min_solution_steps,
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
        temperature=0.0,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results: list[EpisodeResult] = []

    for ep in range(args.n_episodes):
        seed = args.seed + ep
        t0 = time.time()

        env = make_env(seed, args.min_solution_steps)
        obs = env.render()

        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        total_reward = 0.0
        parse_failures = 0
        success = False
        step = 0
        turn_records = []

        for step in range(args.max_steps):
            messages = build_messages(history, obs, args.max_context_window)

            tokenizer = llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = llm.generate([prompt], sampling)
            response_text = outputs[0].outputs[0].text

            history = messages + [{"role": "assistant", "content": response_text}]

            # Analyze MAP quality for this turn
            rec = analyze_map(obs, response_text)
            rec.turn = step + 1
            turn_records.append(rec)

            action = parse_action(response_text)
            if action is None:
                parse_failures += 1
                action = 4

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
            turns=[asdict(t) for t in turn_records],
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

    # MAP quality stats across all turns
    all_turns = [t for r in results for t in r.turns]
    has_map = [t for t in all_turns if t["map_rows"] == 5]
    player_correct = [t for t in has_map if t["player_pos_correct"] is True]
    row0_correct = [t for t in has_map if t["row0_correct"] is True]
    row0_applicable = [t for t in has_map if t["row0_correct"] is not None]

    print("\n" + "=" * 60)
    print(f"Model            : {model}")
    print(f"Episodes         : {n}")
    print(f"Success rate     : {len(solved)}/{n}  ({success_rate:.1f}%)")
    print(f"Avg steps (all)  : {avg_steps:.1f}")
    print(f"Avg steps (solved): {avg_steps_solved:.1f}")
    print(f"Avg total reward : {avg_reward:.4f}")
    print(f"Avg parse failures/ep: {avg_parse_fail:.1f}")
    print()
    print(f"── MAP quality ({len(all_turns)} turns total) ──")
    print(f"  Has 5-row MAP  : {len(has_map)}/{len(all_turns)}  ({100*len(has_map)/max(len(all_turns),1):.1f}%)")
    if has_map:
        print(f"  P at correct (r,c): {len(player_correct)}/{len(has_map)}  ({100*len(player_correct)/len(has_map):.1f}%)")
    if row0_applicable:
        print(f"  Row 0 = '?????'  : {len(row0_correct)}/{len(row0_applicable)}  ({100*len(row0_correct)/len(row0_applicable):.1f}%)  [turns where r≥2]")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot POMDP Sokoban eval")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_context_window", type=int, default=-1,
                        help="Number of past turns to keep (-1 = full, 1 = mem1, 4 = win4)")
    parser.add_argument("--min_solution_steps", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Only use puzzles solvable in [MIN, MAX] steps, e.g. --min_solution_steps 4 8")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        ctx_tag = f"_ctx{args.max_context_window}" if args.max_context_window > 0 else ""
        args.output = f"results/zeroshot_{safe_name}{ctx_tag}_n{args.n_episodes}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    ctx_desc = f"full" if args.max_context_window < 0 else f"win{args.max_context_window}"
    print(f"\nZero-shot POMDP Sokoban evaluation")
    print(f"Model: {args.model}  |  Episodes: {args.n_episodes}  |  Max steps: {args.max_steps}  |  Context: {ctx_desc}\n")

    results = run_eval(args)
    print_summary(results, args.model)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nPer-episode results saved to: {args.output}")


if __name__ == "__main__":
    main()
