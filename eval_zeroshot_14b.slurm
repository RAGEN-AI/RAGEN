#!/bin/bash
#SBATCH --job-name=zeroshot_14b
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/eiu4164/projects/RAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/RAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

# Zero-shot POMDP Sokoban eval — Qwen2.5-14B-Instruct, 1× H100-80GB

set -eo pipefail

mkdir -p /home/eiu4164/projects/RAGEN/logs
cd /home/eiu4164/projects/RAGEN

module load python-miniconda3/4.10.3
source ~/.bashrc
conda activate ragen

export PYTHONNOUSERSITE=1
export HF_HOME=/projects/p32139/hf_cache
export OMP_NUM_THREADS=16

PY=$(which python)
echo "Python:       ${PY}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

"${PY}" scripts/eval_zeroshot_pomdp.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --n_episodes 50 \
    --max_steps 150 \
    --seed 42 \
    --tensor_parallel_size 1 \
    --output results/zeroshot_14b_n50.json \
    2>&1 | tee "/home/eiu4164/projects/RAGEN/logs/zeroshot_14b.${SLURM_JOB_ID}.log"
