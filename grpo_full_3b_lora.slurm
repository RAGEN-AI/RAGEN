#!/bin/bash
#SBATCH --job-name=grpo_full_3b
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/eiu4164/projects/RAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/RAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

# GRPO context window: full (all turns kept), 3B + LoRA rank=8
# LoRA reduces Adam states from ~37GB to ~0.3GB, making 3B feasible on H100 80GB.
# Memory layout: actor(6GB) + ref(6GB) + LoRA-Adam(0.3GB) = ~12.5GB static,
#   vLLM KV cache at gpu_util=0.80: ~51GB. Total well within 80GB.

set -eo pipefail

mkdir -p /home/eiu4164/projects/RAGEN/logs
cd /home/eiu4164/projects/RAGEN

module load python-miniconda3/4.10.3
source ~/.bashrc
conda activate ragen

export PYTHONNOUSERSITE=1
export HF_HOME=/projects/p32139/hf_cache
export OMP_NUM_THREADS=16

export JOB_TMP="/tmp/j${SLURM_JOB_ID}"
export TMPDIR="${JOB_TMP}/t"
export RAY_TMPDIR="${JOB_TMP}/r"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

CKPT_ROOT="${RUN_STORAGE_ROOT:-/projects/p32139/gesture_data/ragen_runs}"
export WANDB_DIR="${CKPT_ROOT}/wandb"
mkdir -p "${CKPT_ROOT}" "${WANDB_DIR}"

PY=$(which python)
echo "Python:       ${PY}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"${PY}" train.py \
    --config-name _2_sokoban \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    lora.rank=8 \
    lora.alpha=64 \
    custom_envs.CoordSokoban.env_config.min_solution_steps=[1,3] \
    custom_envs.CoordSokoban.env_config.search_depth=5 \
    custom_envs.CoordSokoban.env_config.distance_reward_coeff=0.3 \
    trainer.n_gpus_per_node=1 \
    micro_batch_size_per_gpu=4 \
    ppo_mini_batch_size=64 \
    es_manager.train.group_size=128 \
    es_manager.format_penalty=-0.5 \
    es_manager.step_penalty=0.05 \
    es_manager.think_quality_penalty=0.3 \
    actor_rollout_ref.rollout.rollout_filter_value=0.40 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.response_length=192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    trainer.total_training_steps=200 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.resume_mode=disable \
    algorithm.adv_estimator=grpo \
    system.CUDA_VISIBLE_DEVICES="0" \
    agent_proxy.context_window_mode=full \
    agent_proxy.max_context_window=-1 \
    agent_proxy.max_actions_per_turn=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    trainer.experiment_name="grpo_full_lora8_3b_h100_v3" \
    "trainer.default_local_dir=${CKPT_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name}" \
    > "/home/eiu4164/projects/RAGEN/logs/grpo_full_lora8_3b_h100_v3.${SLURM_JOB_ID}.log" 2>&1


echo "Done. Run log: ${RUN_LOG}"