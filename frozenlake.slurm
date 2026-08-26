#!/bin/bash
#SBATCH --job-name=frozenlake
#SBATCH --account=e52938
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/eiu4164/projects/RAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/RAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

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

EXP_NAME="frozenlake_grpo_0p5b"

"${PY}" train.py \
    --config-name _3_frozen_lake \
    model_path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.n_gpus_per_node=1 \
    micro_batch_size_per_gpu=8 \
    ppo_mini_batch_size=64 \
    es_manager.train.group_size=128 \
    actor_rollout_ref.rollout.rollout_filter_value=0.9 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.response_length=200 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    trainer.total_training_steps=200 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.resume_mode=disable \
    algorithm.adv_estimator=grpo \
    system.CUDA_VISIBLE_DEVICES="0" \
    agent_proxy.context_window_mode=full \
    agent_proxy.max_context_window=-1 \
    trainer.experiment_name="${EXP_NAME}" \
    "trainer.default_local_dir=${CKPT_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name}" \
    > "/home/eiu4164/projects/RAGEN/logs/${EXP_NAME}.${SLURM_JOB_ID}.log" 2>&1
