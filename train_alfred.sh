set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# extension: webshop
# # StarPO ppo
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"4,5\" trainer.n_gpus_per_node=2 \
#     trainer.experiment_name=webshop-3b-ppo $USE_PPO $USE_BASE \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1

# StarPO grpo
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _7_alfworld system.CUDA_VISIBLE_DEVICES=\"0\" trainer.n_gpus_per_node=1 \
    trainer.experiment_name=_7_alfworld-grpo $USE_GRPO $USE_BASE \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1