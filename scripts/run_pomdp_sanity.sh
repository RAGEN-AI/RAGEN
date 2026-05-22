#!/bin/bash
# =============================================================================
# POMDP Sokoban — sanity / learning-signal validation on vast.ai (1× GPU)
#
# Goal: confirm reward curve rises within 30 steps.
# Model: Qwen2.5-0.5B-Instruct (fits in ~2 GB, fast iteration)
# GPU:   1× RTX 4090 (24 GB) recommended
#
# Usage:
#   conda activate ragen
#   wandb login
#   bash scripts/run_pomdp_sanity.sh
# =============================================================================
set -eo pipefail

# ── tunable knobs ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
EXP_NAME="${EXP_NAME:-pomdp_sanity_0.5b}"
GPU_ID="${GPU_ID:-0}"

# Batch sizing for 1 GPU
# train_batch = env_groups × group_size = 2 × 16 = 32
# constraint:  env_groups × group_size × filter_ratio ≥ ppo_mini_batch_size
#              2 × 16 × 1.0 = 32 ≥ 8  ✓
ENV_GROUPS=2
GROUP_SIZE=16
PPO_MINI_BATCH=8
MICRO_BATCH=2

# Turns per episode — must match max_actions_per_traj in envs.yaml
MAX_TURN=50

# Total training steps (20 = quick signal check; raise to 200 for real run)
TOTAL_STEPS=30
# ──────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="$GPU_ID"

RAY_TMPDIR="/tmp/ray_pomdp_$$"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

echo "Model : $MODEL"
echo "Exp   : $EXP_NAME"
echo "GPU   : $GPU_ID"
echo ""

python train.py \
    --config-name _2_pomdp_sokoban \
    system.CUDA_VISIBLE_DEVICES="$GPU_ID" \
    model_path="$MODEL" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.test_freq=5 \
    trainer.save_freq=999 \
    \
    es_manager.train.env_groups=$ENV_GROUPS \
    es_manager.train.group_size=$GROUP_SIZE \
    es_manager.train.env_configs.tags='["POMDPSokoban"]' \
    es_manager.train.env_configs.n_groups="[$ENV_GROUPS]" \
    es_manager.val.env_groups=16 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags='["POMDPSokoban"]' \
    es_manager.val.env_configs.n_groups="[16]" \
    \
    ppo_mini_batch_size=$PPO_MINI_BATCH \
    micro_batch_size_per_gpu=$MICRO_BATCH \
    log_prob_micro_batch_size_per_gpu=$MICRO_BATCH \
    \
    agent_proxy.max_turn=$MAX_TURN \
    agent_proxy.context_window_mode=full \
    agent_proxy.max_actions_per_turn=1 \
    \
    actor_rollout_ref.rollout.max_model_len=6000 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \

echo ""
echo "Done. Check wandb or results/ for reward curves."
echo "Success signal: mean reward rising above 0 within 30 steps."
