#!/bin/bash

echo "Starting Archer algorithm training..."

# Simple Archer training test
python train.py --config-name=archer \
    system.CUDA_VISIBLE_DEVICES=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    ppo_mini_batch_size=2 \
    micro_batch_size_per_gpu=1 \
    algorithm.trainer_type=archer

echo "Archer training completed!"
