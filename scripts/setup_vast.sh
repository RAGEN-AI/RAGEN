#!/bin/bash
# =============================================================================
# RAGEN on vast.ai — one-shot setup script
#
# Recommended instance config on vast.ai:
#   0.5B training validation : 1× RTX 4090 (24 GB)  ~$0.35/hr
#   7B  zero-shot eval        : 1× RTX 4090 (24 GB)  ~$0.35/hr
#   14B zero-shot eval        : 1× A100-40GB          ~$1.20/hr
#                               or 2× RTX 4090 (TP=2)
#
# Docker template to select on vast.ai:
#   pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
#
# Run this script once after sshing into the instance:
#   bash scripts/setup_vast.sh
# =============================================================================
set -eo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"   # assume you're already in the RAGEN root
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CONDA_ENV="ragen"

echo "==> [1/5] System packages"
apt-get update -qq && apt-get install -y -qq git git-lfs wget curl unzip tmux

echo "==> [2/5] Install Miniconda (skip if already present)"
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    eval "$(/opt/miniconda/bin/conda shell.bash hook)"
    echo 'eval "$(/opt/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
else
    echo "  conda already installed, skipping"
fi

conda init bash
source ~/.bashrc || true
eval "$(conda shell.bash hook)"

echo "==> [3/5] Create conda env: $CONDA_ENV"
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "  env already exists, skipping create"
else
    conda create -y -n $CONDA_ENV python=3.9
fi
conda activate $CONDA_ENV

echo "==> [4/5] Install Python dependencies"
# PyTorch is already in the Docker image; skip to avoid re-download
pip install --quiet torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Core RAGEN deps (skip webshop/lean extras)
cd "$REPO_DIR"
pip install --quiet -e ".[search]"

# verl (local submodule)
pip install --quiet -e verl/

# flash-attn: build from wheel to avoid long compile
pip install --quiet flash-attn==2.7.4.post1 --no-build-isolation || \
    echo "  WARNING: flash-attn install failed — training will still work but slower"

echo "==> [5/5] Set environment variables"
mkdir -p "$HF_CACHE"
cat >> ~/.bashrc <<EOF

# RAGEN vast.ai config
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export WANDB_MODE=online   # set to 'disabled' if you don't want wandb
EOF

source ~/.bashrc || true

echo ""
echo "======================================================"
echo " Setup complete. Next steps:"
echo "   conda activate $CONDA_ENV"
echo "   wandb login          # paste your API key"
echo "   bash scripts/run_pomdp_sanity.sh      # 0.5B training"
echo "   python scripts/eval_zeroshot_pomdp.py # 7B/14B eval"
echo "======================================================"
