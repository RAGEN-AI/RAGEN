set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# Section 3.1&3.2 - General Observations
python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=bandit-ppo $USE_PPO $USE_BASE &
python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-grpo $USE_GRPO $USE_BASE &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-ppo $USE_PPO $USE_BASE &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-grpo $USE_GRPO $USE_BASE &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=frozen_lake-ppo $USE_PPO $USE_BASE &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=frozen_lake-grpo $USE_GRPO $USE_BASE &

# Section 4.1 - Filtering and critic
# 0.25
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=sokoban-ppo-rolloutfilter0.25 actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_PPO &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=sokoban-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="6" trainer.experiment_name=frozen_lake-ppo actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_PPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="7" trainer.experiment_name=frozen_lake-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO &

wait

# 0.5
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-ppo-rolloutfilter0.5 actor_rollout_ref.rollout.rollout_filter_ratio=0.5 $USE_PPO &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.5 $USE_GRPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=frozen_lake-ppo actor_rollout_ref.rollout.rollout_filter_ratio=0.5 $USE_PPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=frozen_lake-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.5 $USE_GRPO &

# 0.75
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=sokoban-ppo-rolloutfilter0.75 actor_rollout_ref.rollout.rollout_filter_ratio=0.75 $USE_PPO &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=sokoban-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.75 $USE_GRPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="6" trainer.experiment_name=frozen_lake-ppo actor_rollout_ref.rollout.rollout_filter_ratio=0.75 $USE_PPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="7" trainer.experiment_name=frozen_lake-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.75 $USE_GRPO &

wait

# Section 4.2 - Ablation on Critic/ClipHigh/KL. Start from Basic and add more components. The best setting for StarPO in agent is rollout_filter+Critic+Cliphigh+NoKL
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-base-grpo algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_GRPO &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-base-ppo algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-base-ppo-cliphigh algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-base-ppo-nokl algorithm.kl_ctrl.kl_coef=0.000 actor_rollout_ref.actor.kl_loss_coef=0.000 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=frozenlake-base-grpo algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_GRPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=frozenlake-base-ppo algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="6" trainer.experiment_name=frozenlake-base-ppo-cliphigh algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO &
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="7" trainer.experiment_name=frozenlake-base-ppo-nokl algorithm.kl_ctrl.kl_coef=0.000 actor_rollout_ref.actor.kl_loss_coef=0.000 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO &

wait

# Section 5.1 - Reasoning Helps Generalization

python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=bandit-generalization \
    custom_envs.Bandit.env_config.lo_arm_name="Engineer" \
    custom_envs.Bandit.env_config.hi_arm_name="Teacher" \
    custom_envs.BanditTest.env_config.lo_arm_name="Trader" \
    custom_envs.BanditTest.env_config.hi_arm_name="Librarian"

python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-generalization-nothink \
    custom_envs.Bandit.env_config.lo_arm_name="Engineer" \
    custom_envs.Bandit.env_config.hi_arm_name="Teacher" \
    custom_envs.BanditTest.env_config.lo_arm_name="Trader" \
    custom_envs.BanditTest.env_config.hi_arm_name="Librarian" \
    agent_proxy.enable_think=False

python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=bandit-generalization-rev \
    custom_envs.Bandit.env_config.lo_arm_name="Teacher" \
    custom_envs.Bandit.env_config.hi_arm_name="Engineer" \
    custom_envs.BanditTest.env_config.lo_arm_name="Librarian" \
    custom_envs.BanditTest.env_config.hi_arm_name="Trader"

python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=bandit-generalization-rev-nothink \
    custom_envs.Bandit.env_config.lo_arm_name="Teacher" \
    custom_envs.Bandit.env_config.hi_arm_name="Engineer" \
    custom_envs.BanditTest.env_config.lo_arm_name="Librarian" \
    custom_envs.BanditTest.env_config.hi_arm_name="Trader" \
    agent_proxy.enable_think=False


SOKOBAN_GENERALIZATION_CONFIG="es_manager.val.env_groups=512 es_manager.val.group_size=1 es_manager.val.env_configs.tags=[SimpleSokoban,LargerSokoban,SokobanDifferentGridVocab,FrozenLake] es_manager.val.env_configs.n_groups=[128,128,128,128]"
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 trainer.experiment_name=sokoban-generalization $SOKOBAN_GENERALIZATION_CONFIG trainer.total_training_steps=500 trainer.save_freq=50 trainer.default_local_dir=/mnt/local/ragen_checkpoints/sokoban-generalization micro_batch_size_per_gpu=8 model_path=Qwen/Qwen2.5-1.5B-Instruct&
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=sokoban-generalization-nothink $SOKOBAN_GENERALIZATION_CONFIG agent_proxy.enable_think=False &


# SOKOBAN_GENERALIZATION_CONFIG="es_manager.val.env_groups=128 es_manager.val.group_size=1 es_manager.val.env_configs.tags=[SokobanDifferentGridVocab] es_manager.val.env_configs.n_groups=[128]"
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-generalization $SOKOBAN_GENERALIZATION_CONFIG &


# COMPOSITIONALITY_CONFIG="es_manager.train.env_groups=16 es_manager.train.env_configs.tags=[Bandit,SimpleSokoban] es_manager.train.env_configs.n_groups=[8,8] es_manager.val.env_groups=512 es_manager.val.group_size=1 es_manager.val.env_configs.tags=[Bandit,SimpleSokoban,LargerSokoban,FrozenLake] es_manager.val.env_configs.n_groups=[128,128,128,128] actor_rollout_ref.rollout.rollout_filter_ratio=1" # NOTE that we don't filter out low-var rollout in this setting
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="6" trainer.experiment_name=compositional-generalization $COMPOSITIONALITY_CONFIG &
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="7" trainer.experiment_name=compositional-generalization-nothink $COMPOSITIONALITY_CONFIG agent_proxy.enable_think=False &

wait

# Section 5.2 - what leads to better reasoning?
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-generalization-qwen2.5-0.5b-instruct $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-0.5B-Instruct &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-generalization-qwen2.5-1.5b-instruct $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-1.5B-Instruct &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"2,3\" trainer.n_gpus_per_node=2 trainer.experiment_name=sokoban-generalization-qwen2.5-3b-instruct $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-3B-Instruct trainer.n_gpus_per_node=2 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 trainer.experiment_name=sokoban-generalization-qwen2.5-7b-instruct $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-7B-Instruct trainer.n_gpus_per_node=4 &

wait

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-generalization-qwen2.5-0.5b $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-0.5B &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-generalization-qwen2.5-1.5b $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-1.5B &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"2,3\" trainer.n_gpus_per_node=2 trainer.experiment_name=sokoban-generalization-qwen2.5-3b $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-3B &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 trainer.experiment_name=sokoban-generalization-qwen2.5-7b $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-7B &

wait

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\"  trainer.n_gpus_per_node=4 trainer.n_gpus_per_node=4 trainer.experiment_name=sokoban-generalization-qwen2.5-7b-r1 $SOKOBAN_GENERALIZATION_CONFIG model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"4,5\"  trainer.n_gpus_per_node=2 trainer.experiment_name=sokoban-generalization-qwen2.5-1.5b-r1 $SOKOBAN_GENERALIZATION_CONFIG model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B &


wait

# Section 6.1 varying action count
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-action-count-1 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=5 custom_envs.LargerSokoban.max_actions_per_traj=5 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=5 custom_envs.FrozenLake.max_actions_per_traj=5 agent_proxy.max_actions_per_turn=1 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-action-count-2 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=10 custom_envs.LargerSokoban.max_actions_per_traj=10 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=10 custom_envs.FrozenLake.max_actions_per_traj=10 agent_proxy.max_actions_per_turn=2 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-action-count-3 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=15 custom_envs.LargerSokoban.max_actions_per_traj=15 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=15 custom_envs.FrozenLake.max_actions_per_traj=15 agent_proxy.max_actions_per_turn=3 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-action-count-4 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=20 custom_envs.LargerSokoban.max_actions_per_traj=20 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=20 custom_envs.FrozenLake.max_actions_per_traj=20 agent_proxy.max_actions_per_turn=4 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=sokoban-action-count-5 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=25 custom_envs.LargerSokoban.max_actions_per_traj=25 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=25 custom_envs.FrozenLake.max_actions_per_traj=25 agent_proxy.max_actions_per_turn=5 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=sokoban-action-count-6 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=30 custom_envs.LargerSokoban.max_actions_per_traj=30 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=30 custom_envs.FrozenLake.max_actions_per_traj=30 agent_proxy.max_actions_per_turn=6 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="6" trainer.experiment_name=sokoban-action-count-7 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=35 custom_envs.LargerSokoban.max_actions_per_traj=35 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=35 custom_envs.FrozenLake.max_actions_per_traj=35 agent_proxy.max_actions_per_turn=7 &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="7" trainer.experiment_name=sokoban-action-count-8 $SOKOBAN_GENERALIZATION_CONFIG custom_envs.SimpleSokoban.max_actions_per_traj=40 custom_envs.LargerSokoban.max_actions_per_traj=40 custom_envs.SokobanDifferentGridVocab.max_actions_per_traj=40 custom_envs.FrozenLake.max_actions_per_traj=40 agent_proxy.max_actions_per_turn=8 &

# section 6.2 Varying prompt diversity
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-prompt-diversity-4 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=4 es_manager.train.group_size=32 es_manager.train.env_configs.n_groups=[4] &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=sokoban-prompt-diversity-8 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-prompt-diversity-16 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=16 es_manager.train.group_size=8 es_manager.train.env_configs.n_groups=[16] &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-prompt-diversity-32 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=32 es_manager.train.group_size=4 es_manager.train.env_configs.n_groups=[32] &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=sokoban-prompt-diversity-64 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=64 es_manager.train.group_size=2 es_manager.train.env_configs.n_groups=[64] &
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=sokoban-prompt-diversity-128 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=128 es_manager.train.group_size=1 es_manager.train.env_configs.n_groups=[128] &

wait 

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="6" trainer.experiment_name=sokoban-online-2 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=16 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[16] trainer.total_training_steps=100 trainer.test_freq=5 &

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="7" trainer.experiment_name=sokoban-online-5 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=40 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[40] trainer.total_training_steps=40 trainer.test_freq=2 &

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=sokoban-online-10 $SOKOBAN_GENERALIZATION_CONFIG es_manager.train.env_groups=80 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[80] trainer.total_training_steps=80 trainer.test_freq=1 &





# Extension: Training 7B reasoning model
SOKOBAN_GENERALIZATION_CONFIG="es_manager.val.env_groups=512 es_manager.val.group_size=1 es_manager.val.env_configs.tags=[SimpleSokoban,LargerSokoban,SokobanDifferentGridVocab,FrozenLake] es_manager.val.env_configs.n_groups=[128,128,128,128]"
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" trainer.n_gpus_per_node=8 trainer.experiment_name=sokoban-generalization-qwen2.5-3b-instruct-largescale $SOKOBAN_GENERALIZATION_CONFIG model_path=Qwen/Qwen2.5-3B-Instruct es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8] micro_batch_size_per_gpu=8 ppo_mini_batch_size=64 actor_rollout_ref.rollout.response_length=1024 actor_rollout_ref.rollout.max_model_len=6400 trainer.test_freq=5 actor_rollout_ref.rollout.max_num_batched_tokens=24000 micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.rollout_filter_ratio=1 &

python -m ragen.llm_agent.agent_proxy  model_path=Qwen/Qwen2.5-3B-Instruct system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" trainer.n_gpus_per_node=8 actor_rollout_ref.rollout.tensor_model_parallel_size=4 actor_rollout_ref.rollout.response_length=2048 actor_rollout_ref.rollout.max_model_len=12800 

# trainer.save_freq=50 trainer.default_local_dir=/mnt/local/cache/exp_name

# USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0\" trainer.n_gpus_per_node=1 trainer.experiment_name=sokoban-final enable_response_mask=True trainer.total_training_steps=500 trainer.save_freq=50 trainer.default_local_dir=/mnt/local/ragen_checkpoints/sokoban-generalization &

python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES=\"1\" trainer.n_gpus_per_node=1 trainer.experiment_name=bandit-final enable_response_mask=True trainer.total_training_steps=500 trainer.save_freq=50 trainer.default_local_dir=/mnt/local/ragen_checkpoints/bandit-generalization &

python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES=\"1\" trainer.n_gpus_per_node=1 trainer.experiment_name=frozenlake-final enable_response_mask=True trainer.total_training_steps=500 trainer.save_freq=50 trainer.default_local_dir=/mnt/local/ragen_checkpoints/frozenlake-generalization &




# USE_PPO="algorithm.adv_estimator=gae" # by default.
# USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0\" trainer.n_gpus_per_node=1 trainer.experiment_name=sokoban-ppo $USE_PPO $USE_BASE ppo_mini_batch_size=64 enable_response_mask=True &


python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 trainer.experiment_name=sokoban-s-grpo algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std &

# enable_response_mask: False
# grpo_advantage_length_weight: True

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 trainer.experiment_name=sokoban-s-grpo-1-5b algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std agent_proxy.max_actions_per_turn=5 custom_envs.SimpleSokoban.max_actions_per_traj=25 enable_response_mask=True grpo_advantage_length_weight=False model_path=Qwen/Qwen2.5-1.5B-Instruct &


# extension: 7B with lora. Currently NOT recommended to use lora: within current version of vllm, this could result in rollouts slower than non-lora by 100%
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 model_path=Qwen/Qwen2.5-7B-Instruct trainer.experiment_name=sokoban_7b_instruct_lora_newversion lora.rank=16  

# extension: bi-level gae
python train.py trainer.experiment_name=sokoban-bi-level-gae-final \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    model_path=Qwen/Qwen2.5-0.5B-Instruct \
    algorithm.bi_level_gae=True algorithm.high_level_gamma=0.95 \
    agent_proxy.use_turn_scores=True \
    actor_rollout_ref.rollout.tp_size_check=False

# extension: webshop
# StarPO ppo
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"4,5\" trainer.n_gpus_per_node=2 \
    trainer.experiment_name=webshop-3b-ppo $USE_PPO $USE_BASE \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1

# StarPO grpo
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"6,7\" trainer.n_gpus_per_node=2 \
    trainer.experiment_name=webshop-3b-grpo $USE_GRPO $USE_BASE \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1
    