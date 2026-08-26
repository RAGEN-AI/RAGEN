#!/bin/bash
#SBATCH --job-name=pomdp_mem1_7b_v1
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/eiu4164/projects/RAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/RAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

# POMDP Sokoban — MEM1, 7B + LoRA rank=8
# Goal: test if larger model + RL training achieves both good MAP quality and task success
# vs 3B MEM1: MAP P@(r,c)%=6-16% but success~35%; 14B zero-shot: P@(r,c)%=31% but success=24%
# v1 fix: reduced gpu_memory_utilization 0.75→0.35, max_model_len 4096→2048, micro_batch 2→1
# to resolve OOM (7B FSDP actor+ref ~28GB leaves <1GB for vLLM KV cache on H100-80GB)

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

EXP_NAME="pomdp_mem1_lora8_7b_h100_v1"

"${PY}" train.py \
    --config-name _2_pomdp_sokoban \
    model_path=Qwen/Qwen2.5-7B-Instruct \
    lora.rank=8 \
    lora.alpha=64 \
    trainer.n_gpus_per_node=1 \
    micro_batch_size_per_gpu=2 \
    ppo_mini_batch_size=64 \
    es_manager.train.group_size=32 \
    es_manager.format_penalty=-0.5 \
    es_manager.think_quality_penalty=0.3 \
    actor_rollout_ref.rollout.rollout_filter_value=0.40 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    custom_envs.POMDPSokoban.env_config.min_solution_steps=[4,8] \
    custom_envs.POMDPSokoban.env_config.search_depth=100 \
    trainer.total_training_steps=100 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.resume_mode=disable \
    algorithm.adv_estimator=grpo \
    system.CUDA_VISIBLE_DEVICES="0" \
    agent_proxy.context_window_mode=limited_multi_turn \
    agent_proxy.max_context_window=1 \
    trainer.experiment_name="${EXP_NAME}" \
    "trainer.rollout_data_dir=${CKPT_ROOT}/traj/${EXP_NAME}" \
    "trainer.default_local_dir=${CKPT_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name}" \
    > "/home/eiu4164/projects/RAGEN/logs/${EXP_NAME}.${SLURM_JOB_ID}.log" 2>&1
