#!/usr/bin/env python3
"""
Analyze MAP quality in trajectory JSONL files from RAGEN POMDP training.

Each JSONL line has:
  input  = full prompt decoded from input_ids  (system + history + current obs)
  output = model response decoded from responses (<think>...</think><answer>...</answer>)
  score, step, episode_id, ...

Usage:
  # All steps, print table
  python scripts/analyze_traj_map.py /path/to/traj/dir

  # Every 5th step file
  python scripts/analyze_traj_map.py /path/to/traj/dir --every 5

  # Specific steps
  python scripts/analyze_traj_map.py /path/to/traj/dir --steps 20 40 60

  # Show per-episode turn-level breakdown for a given step
  python scripts/analyze_traj_map.py /path/to/traj/dir --episode_detail 40
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ANSWER_RE = re.compile(r"<answer>\s*(up|down|left|right)\s*</answer>", re.IGNORECASE)
TURN_RE = re.compile(r"Turn\s+(\d+)[:\|]")


def last_position(text: str):
    """Last Position: (r, c) in text — skips the system-prompt example."""
    matches = list(re.finditer(r"Position:\s*\((\d+),\s*(\d+)\)", text))
    if not matches:
        return None, None
    m = matches[-1]
    return int(m.group(1)), int(m.group(2))


def last_turn_number(text: str):
    """Last 'Turn N' in the input prompt = current game turn."""
    matches = list(TURN_RE.finditer(text))
    return int(matches[-1].group(1)) if matches else None


def analyze_record(record: dict) -> dict:
    obs_prompt = record.get("input", "")
    model_response = record.get("output", "")

    r, c = last_position(obs_prompt)
    turn_num = last_turn_number(obs_prompt)

    # <think> block — LAST match: output = full sequence, model's response is at the end
    think = extract_last_think(model_response)

    # MAP: up to 7 candidate lines, keep rows of exactly 5 tokens
    map_rows_raw = []
    map_match = re.search(r"MAP:\s*((?:[?#_XOPSs√ ]+\n?){1,7})", think)
    if map_match:
        for line in map_match.group(1).strip().split("\n"):
            tokens = line.strip().split()
            if len(tokens) == 5:
                map_rows_raw.append(tokens)

    map_rows_count = len(map_rows_raw)

    player_correct = None
    if r is not None and map_rows_count == 5:
        try:
            player_correct = map_rows_raw[r][c] in ("P", "S")
        except IndexError:
            player_correct = False

    row0_correct = None
    if r is not None and r >= 2 and map_rows_count == 5:
        row0_correct = all(cell == "?" for cell in map_rows_raw[0])

    return {
        "step": record.get("step", -1),
        "episode_id": record.get("episode_id"),
        "turn_num": turn_num,
        "score": record.get("score", 0.0),
        "position": (r, c) if r is not None else None,
        "map_rows": map_rows_count,
        "player_pos_correct": player_correct,
        "row0_correct": row0_correct,
        "action_valid": bool(ANSWER_RE.search(model_response)),
        "think_len": len(think),
    }


def step_summary(records: list) -> dict:
    n = len(records)
    has_map = [r for r in records if r["map_rows"] == 5]
    player_ok = [r for r in has_map if r["player_pos_correct"] is True]
    row0_ok = [r for r in has_map if r["row0_correct"] is True]
    row0_appl = [r for r in has_map if r["row0_correct"] is not None]
    valid_act = [r for r in records if r["action_valid"]]
    avg_score = sum(r["score"] for r in records) / n if n else 0.0
    avg_think = sum(r["think_len"] for r in records) / n if n else 0.0

    return {
        "n": n,
        "avg_score": avg_score,
        "avg_think_len": avg_think,
        "action_valid_pct": 100 * len(valid_act) / n if n else 0,
        "has_map_pct": 100 * len(has_map) / n if n else 0,
        "player_correct_pct": 100 * len(player_ok) / len(has_map) if has_map else None,
        "row0_correct_pct": 100 * len(row0_ok) / len(row0_appl) if row0_appl else None,
    }


def print_step_table(step_results: list):
    header = (
        f"{'Step':>6} | {'N':>5} | {'Score':>6} | {'ThinkLen':>8} | "
        f"{'Act%':>5} | {'MAP%':>5} | {'P@(r,c)%':>9} | {'Row0%':>6}"
    )
    print(header)
    print("-" * len(header))
    for step, s in step_results:
        p_str = f"{s['player_correct_pct']:>8.1f}%" if s["player_correct_pct"] is not None else "      N/A"
        r0_str = f"{s['row0_correct_pct']:>5.1f}%" if s["row0_correct_pct"] is not None else "   N/A"
        print(
            f"{step:>6} | {s['n']:>5} | {s['avg_score']:>6.3f} | "
            f"{s['avg_think_len']:>8.1f} | {s['action_valid_pct']:>4.1f}% | "
            f"{s['has_map_pct']:>4.1f}% | {p_str} | {r0_str}"
        )


def print_episode_detail(analyzed: list, step: int, max_eps: int = 10):
    """Show per-turn MAP quality for sampled episodes within a step."""
    by_ep = defaultdict(list)
    for r in analyzed:
        eid = r["episode_id"]
        if eid is not None:
            by_ep[eid].append(r)

    if not by_ep:
        print("  No episode_id found — cannot do turn-level breakdown.")
        return

    # Sort each episode by turn number
    for eid in by_ep:
        by_ep[eid].sort(key=lambda x: (x["turn_num"] or 0))

    # Show up to max_eps episodes, prefer multi-turn ones
    eps = sorted(by_ep.keys(), key=lambda e: -len(by_ep[e]))[:max_eps]

    print(f"\n── Episode detail for step {step} (showing {len(eps)} episodes) ──")
    for eid in eps:
        turns = by_ep[eid]
        final_score = turns[-1]["score"]
        print(f"\n  Episode {eid}  |  turns={len(turns)}  |  final_score={final_score:.3f}")
        print(f"  {'Turn':>5} | {'Pos':>8} | {'MAP':>4} | {'P@(r,c)':>8} | {'Row0':>5} | {'Act':>4} | {'ThinkLen':>8}")
        for t in turns:
            pos_str = str(t["position"]) if t["position"] else "   None"
            map_str = str(t["map_rows"])
            p_str = {True: "  OK", False: " BAD", None: "  --"}[t["player_pos_correct"]]
            r0_str = {True: "  OK", False: " BAD", None: "  --"}[t["row0_correct"]]
            act_str = "OK" if t["action_valid"] else "BAD"
            tn = t["turn_num"] or "?"
            print(
                f"  {tn:>5} | {pos_str:>8} | {map_str:>4} | {p_str:>8} | "
                f"{r0_str:>5} | {act_str:>4} | {t['think_len']:>8}"
            )


def load_step_file(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="MAP quality analysis for POMDP trajectory files")
    parser.add_argument("traj_dir", help="Trajectory directory (contains *.jsonl files)")
    parser.add_argument("--steps", nargs="*", type=int, default=None,
                        help="Specific step numbers to analyze")
    parser.add_argument("--every", type=int, default=1,
                        help="Analyze every Nth step file (default: 1 = all)")
    parser.add_argument("--episode_detail", type=int, default=None, metavar="STEP",
                        help="Print per-episode turn-level breakdown for STEP")
    parser.add_argument("--max_eps", type=int, default=10,
                        help="Max episodes to show in episode_detail (default: 10)")
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    all_files = sorted(traj_dir.glob("*.jsonl"), key=lambda p: int(p.stem))

    if not all_files:
        print(f"No .jsonl files found in {traj_dir}")
        return

    if args.steps is not None:
        files = [f for f in all_files if int(f.stem) in args.steps]
    elif args.every > 1:
        files = all_files[::args.every]
    else:
        files = all_files

    print(f"Analyzing {len(files)} / {len(all_files)} step files in {traj_dir}\n")

    step_results = []
    detail_analyzed = None

    for jf in files:
        step = int(jf.stem)
        raw = load_step_file(jf)
        analyzed = [analyze_record(r) for r in raw]
        s = step_summary(analyzed)
        step_results.append((step, s))

        if args.episode_detail is not None and step == args.episode_detail:
            detail_analyzed = analyzed

    print_step_table(step_results)

    if args.episode_detail is not None:
        if detail_analyzed is None:
            print(f"\nStep {args.episode_detail} not found in selected files.")
        else:
            print_episode_detail(detail_analyzed, args.episode_detail, args.max_eps)


if __name__ == "__main__":
    main()
