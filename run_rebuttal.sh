set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 es_manager.train.env_groups=2 es_manager.train.env_configs.n_groups=[2]"
USE_REINFORCE="algorithm.adv_estimator=reinforce_plus_plus"
USE_ARCHER="algorithm.adv_estimator=archer"


# Section 3.1&3.2 - General Observations
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0,1,2,3" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=sokoban-archer-starpo $USE_ARCHER $USE_BASE &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="4,5,6,7" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=sokoban-reinforce-starpo $USE_REINFORCE $USE_BASE &
wait

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0,1,2,3" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=sokoban-archer-starpos &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="4,5,6,7" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=sokoban-reinforce-starpos &
wait


python train.py --config-name _4_webshop system.CUDA_VISIBLE_DEVICES="0,1,2,3" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=webshop-archer-starpo $USE_ARCHER $USE_BASE &
python train.py --config-name _4_webshop system.CUDA_VISIBLE_DEVICES="4,5,6,7" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=webshop-reinforce-starpo $USE_REINFORCE $USE_BASE &
wait

python train.py --config-name _4_webshop system.CUDA_VISIBLE_DEVICES="0,1,2,3" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=webshop-archer-starpos &
python train.py --config-name _4_webshop system.CUDA_VISIBLE_DEVICES="4,5,6,7" model_path="Qwen/Qwen2.5-3B-Instruct" trainer.experiment_name=webshop-reinforce-starpos &
wait
