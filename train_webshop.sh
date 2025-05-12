USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train \
#     algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True \
#     trainer.experiment_name=webshop3b \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
#     system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop3b_0506 \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"4,5\" trainer.n_gpus_per_node=2 \
#     trainer.experiment_name=webshop-3b-ppo $USE_PPO $USE_BASE \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop3b_ppo_0506 \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"6,7\" trainer.n_gpus_per_node=2 \
#     trainer.experiment_name=webshop-3b-grpo $USE_GRPO $USE_BASE \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop3b_grpo_0506 \
#     trainer.nnodes=1

# python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-grpo $USE_GRPO $USE_BASE &

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train \
#     algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True \
#     trainer.experiment_name=webshop3b_starpos_grpo \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
#     es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
#     system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop3b_starpos_grpo_0507 \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train \
#     $USE_PPO enable_response_mask=True \
#     trainer.experiment_name=webshop3b_starpos_ppo \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
#     es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
#     system.CUDA_VISIBLE_DEVICES=\"2,3\" trainer.n_gpus_per_node=2 \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop3b_starpos_ppo_0507 \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
#     $USE_PPO enable_response_mask=True \
#     model_path=Qwen/Qwen2.5-0.5B-Instruct \
#     trainer.experiment_name=webshop_starpos_ppo_0.5b \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
#     es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
#     es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[128] \
#     system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.tp_size_check=False \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpos_ppo_0.5b_0508 \
#     trainer.nnodes=1

# 0.5b PPO
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
#     $USE_PPO enable_response_mask=True \
#     model_path=Qwen/Qwen2.5-0.5B-Instruct \
#     trainer.experiment_name=webshop_starpos_ppo_0.5b \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
#     es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
#     es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[128] \
#     system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.tp_size_check=False \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpos_ppo_0.5b_0508 \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
#     actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO \
#     model_path=Qwen/Qwen2.5-3B-Instruct \
#     trainer.experiment_name=webshop_starpos_grpo_3b_full \
#     es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
#     es_manager.val.env_groups=64 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[64] \
#     system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpos_grpo_3b_full_0508 \
#     trainer.nnodes=1


MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
    $USE_PPO enable_response_mask=True \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.experiment_name=webshop_starpos_ppo_3b \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
    es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.n_groups=[128] \
    system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.default_local_dir=/mnt/local/ragen_checkpoints/webshop_starpos_ppo_3b_full_0508 \
    trainer.nnodes=1

cp -r /mnt/local/ragen_checkpoints/webshop_starpos_ppo_3b_full_0508/global_step_200 \
   /home/xjintest/webshop_starpos_ppo_3b_full_0508
