#!/bin/bash
#SBATCH --job-name=sokoban_v2
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

EXP_NAME="sokoban_grpo_0p5b_v2"

"${PY}" train.py \
    --config-name _2_sokoban \
    model_path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.n_gpus_per_node=1 \
    micro_batch_size_per_gpu=4 \
    ppo_mini_batch_size=64 \
    es_manager.train.group_size=128 \
    custom_envs.CoordSokoban.env_config.min_solution_steps=[1,3] \
    custom_envs.CoordSokoban.env_config.search_depth=5 \
    custom_envs.CoordSokoban.env_config.distance_reward_coeff=0.3 \
    es_manager.format_penalty=-1 \
    actor_rollout_ref.rollout.rollout_filter_value=0.40 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    trainer.total_training_steps=200 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.resume_mode=disable \
    algorithm.adv_estimator=grpo \
    system.CUDA_VISIBLE_DEVICES="0" \
    agent_proxy.context_window_mode=full \
    agent_proxy.max_context_window=-1 \
    agent_proxy.max_actions_per_turn=1 \
    trainer.experiment_name="${EXP_NAME}" \
    "trainer.default_local_dir=${CKPT_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name}" \
    > "/home/eiu4164/projects/RAGEN/logs/${EXP_NAME}.${SLURM_JOB_ID}.log" 2>&1
