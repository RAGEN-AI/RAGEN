#!/usr/bin/env bash
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

set -e

# 1) Run the first command, exit immediately if it fails
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.experiment_name=webshop_starpos_grpo_3b_full \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
    es_manager.val.env_groups=64 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[64] \
    system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpos_grpo_3b_full_0508 \
    trainer.nnodes=1

# 2) Copy checkpoints before launching next experiments
cp -r /mnt/local/ragen_checkpoints/webshop_starpos_grpo_3b_full_0508/global_step_200 \
   /home/xjintest/webshop_starpos_grpo_3b_full_0508

cp -r /mnt/local/ragen_checkpoints/webshop_starpos_grpo_3b_full_0508/global_step_200 \
    /home/xjintest/webshop_starpo_grpo_3b_full_0508

# 3) Launch 2nd and 3rd commands in parallel
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
    $USE_PPO enable_response_mask=True \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.experiment_name=webshop_starpos_ppo_3b \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
    es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[128] \
    system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpos_ppo_3b_full_0508 \
    trainer.nnodes=1 &

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop\
    enable_response_mask=False \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.experiment_name=webshop_starpo_ppo_3b $USE_PPO $USE_BASE \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[128] \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpo_ppo_3b_full_0508 \
    trainer.nnodes=1 &

# Wait for both background jobs
wait
cp -r /mnt/local/ragen_checkpoints/webshop_starpos_ppo_3b_full_0508/global_step_200 \
   /home/xjintest/webshop_starpos_ppo_3b_full_0508

cp -r /mnt/local/ragen_checkpoints/webshop_starpo_ppo_3b_full_0508/global_step_200 \
      /home/xjintest/webshop_starpo_ppo_3b_full_0508
echo "All training runs and copies completed."
