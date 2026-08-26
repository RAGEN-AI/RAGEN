# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

RAGEN is a multi-turn RL training framework for LLM reasoning agents, built on top of [veRL](https://github.com/volcengine/verl). It implements **StarPO** (State-Thinking-Actions-Reward Policy Optimization): agents receive text observations, reason in `<think>` blocks, emit actions in `<answer>` blocks, receive environment rewards, and are trained with PPO/GRPO.

`verl/` is a git submodule providing the underlying PPO/GRPO infrastructure, FSDP workers, and Ray orchestration. RAGEN adds the multi-turn rollout loop, environment registry, rollout filtering, and collapse diagnostics on top.

## Key commands

```bash
# Installation (creates conda env + installs deps)
bash scripts/setup_ragen.sh              # base
bash scripts/setup_ragen.sh --with-search  # add SearchQA env (~87 GB index)

# Training — config-name selects a task overlay on base.yaml
python train.py --config-name _2_sokoban
python train.py --config-name _2_sokoban trainer.experiment_name=my_run

# Config dry-run (verify Hydra resolves without launching)
python train.py --config-name _2_sokoban --cfg job

# Tests (no test runner config; run individually)
python -m pytest tests/

# Zero-shot eval (vast.ai)
python scripts/eval_zeroshot_pomdp.py --model Qwen/Qwen2.5-7B-Instruct
```

## Architecture

Training entry point is `train.py` → `RayAgentTrainer` (`ragen/trainer/agent_trainer.py`). Each training step:

1. **Rollout** — `LLMAgentProxy.rollout()` (`ragen/llm_agent/agent_proxy.py`) drives the multi-turn loop:
   - `EnvStateManager` (`es_manager.py`) resets/steps environments in parallel groups
   - `ContextManager` (`ctx_manager.py`) converts `(env_obs, history)` → LLM prompt and parses `<answer>` tags back into environment actions
   - vLLM generates responses; loop repeats up to `agent_proxy.max_turn` turns

2. **Filter** — `RolloutFilter` (`ragen/trainer/rollout_filter.py`) drops low-signal trajectories. Default strategy: `top_p` by reward variance (keeps top fraction with highest variance). Key config: `rollout_filter_strategy`, `rollout_filter_value`.

3. **Advantage** — GAE (PPO) or GRPO depending on `algorithm.adv_estimator`.

4. **Update** — veRL FSDP workers apply policy gradient.

**Collapse detection** (`ragen/trainer/collapse_metrics.py`) runs every `collapse_detection.compute_freq` steps, computing mutual information I(X;Z) and conditional entropy H(Z|X) to distinguish template collapse from entropy collapse.

## Config system

Hydra-based. All configs live in `config/`. The hierarchy is:

```
config/base.yaml              ← global defaults (model, rollout, PPO params, env groups)
config/envs.yaml              ← all environment definitions under custom_envs.*
config/_2_sokoban.yaml        ← task overlay: sets experiment_name, inherits base
```

Task configs (`_1_bandit.yaml` … `_11_lights_out.yaml`) only override what differs from `base.yaml`. The active environment is set via:
```yaml
es_manager.train.env_configs.tags: ["CoordSokoban"]   # name from envs.yaml custom_envs.*
```

Override any field at the command line: `python train.py --config-name _2_sokoban model_path=Qwen/Qwen2.5-3B-Instruct`.

Important batch-size constraint that `train.py` validates:
```
env_groups × group_size × rollout_filter_ratio ≥ ppo_mini_batch_size
```

## Environment system

Environments are registered in `ragen/env/__init__.py` (`REGISTERED_ENVS` dict). Each env implements `BaseEnv` (`ragen/env/base.py`): `reset(seed) → obs_str` and `step(action) → (obs_str, reward, done, info)`.

Defined in `config/envs.yaml` under `custom_envs.<Name>`:
- `env_type`: key into `REGISTERED_ENVS`
- `env_instruction`: the system prompt shown to the LLM
- `max_actions_per_traj`: hard step limit
- `env_config`: passed as kwargs to the env's config dataclass

**Adding a new environment**: (1) create `ragen/env/<name>/` with a config dataclass and env class extending `BaseDiscreteActionEnv` or `BaseLanguageBasedEnv`; (2) register in `ragen/env/__init__.py`; (3) add an entry to `config/envs.yaml`.

## POMDP Sokoban (this branch)

Partial-observation Sokoban where the agent sees only a 3×3 window. Key additions:

- `SokobanEnvConfig.partial_obs: bool` and `partial_obs_window: int` — in `ragen/env/sokoban/config.py`
- `SokobanEnv._render_partial_obs()` — renders the local patch + `Position: (r, c)` footer — in `ragen/env/sokoban/env.py`
- `POMDPSokoban` and `POMDPSokobanHard` env variants in `config/envs.yaml`
- Training config: `config/_2_pomdp_sokoban.yaml`
- vast.ai scripts: `scripts/setup_vast.sh`, `scripts/run_pomdp_sanity.sh`, `scripts/eval_zeroshot_pomdp.py`

The agent is soft-scaffolded (via `env_instruction`) to maintain a progressive map in its `<think>` block. Memory compression is emergent from RL reward; no format penalty is applied.

## Context window modes

Controlled by `agent_proxy.context_window_mode`:
- `full` — entire conversation history passed each turn
- `limited_multi_turn` — last `max_context_window` turns only
- `single_turn` — current observation only (no history)

These are the axes of the memory baseline experiments.

## Cluster setup notes

**Quest (Northwestern HPC)**: use `set -eo pipefail` (not `-euo`); `source ~/.bashrc` before `conda activate ragen`; conda env at `/home/eiu4164/.conda/envs/ragen/`; HF cache at `/projects/p32139/hf_cache`.

**vast.ai**: Python 3.12 env required (verl uses `X | Y` union syntax); install with `python -m pip install -e verl/ --ignore-requires-python`; `pkg_resources` missing on fresh Python 3.12 → `pip install setuptools`.
