WANDB_MODE=disabled MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _6_webshop \
    algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True \
    trainer.experiment_name=webshop3b_starpos_grpo_full \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] \
    system.CUDA_VISIBLE_DEVICES=\"2,3\" trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 +trainer.val_only=true