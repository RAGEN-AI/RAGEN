import yaml
import os
import argparse
import json
from typing import Dict, Any

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_config(env_name: str) -> Dict[str, Any]:
    """Load configuration from base and environment-specific configs."""
    # Load base config
    with open("config/base.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment config
    env_config_path = f"config/{env_name}.yaml"
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
            config = deep_update(config, env_config)
    else:
        raise ValueError(f"Environment config not found: {env_config_path}")
    
    # Override with environment variables
    for key, value in os.environ.items():
        if key.startswith('CONFIG_'):
            # Remove CONFIG_ prefix and split by double underscore for nested keys
            key_path = key[7:].lower().split('__')
            
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
                
            # Navigate to the correct nested dictionary
            current = config
            for k in key_path[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[key_path[-1]] = value
    
    return config

def get_train_command(config: Dict[str, Any]) -> str:
    """Generate the training command with all arguments."""
    # Calculate MAX_PROMPT_LENGTH
    max_prompt_length = (config['training']['max_start_length'] + 
                        config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + 
                        config['training']['max_obs_length'] * config['training']['max_turns'])
    
    # Define the command template with proper indentation
    env_kwargs = config['env']['env_kwargs']
    env_kwargs_str = " \\\n    ".join([
        f"+env.{key}={value}" for key, value in env_kwargs.items()
    ])
    cmd = [
        "python -m verl.trainer.main_ppo",
        f"multi_processing={config['system']['multi_processing']}",
        f"data.train_files={config['env']['data_dir']}/train.parquet",
        f"data.val_files={config['env']['data_dir']}/test.parquet",
        f"data.train_batch_size={config['training']['train_batch_size']}",
        "data.val_batch_size=1312",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={config['training']['max_response_length']}",
        f"data.max_start_length={config['training']['max_start_length']}",
        f"data.max_obs_length={config['training']['max_obs_length']}",
        "data.shuffle_train_dataloader=True",
        f"algorithm.adv_estimator={config['optimization']['adv_estimator']}",
        f"actor_rollout_ref.model.path={config['model']['base_model']}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(config['model']['gradient_checkpointing']).lower()}",
        f"actor_rollout_ref.actor.optim.lr={config['optimization']['actor_lr']}",
        f"actor_rollout_ref.actor.use_kl_loss={config['training']['use_kl_loss']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config['training']['ppo_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={config['training']['micro_batch_size']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size={config['training']['micro_batch_size']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config['training']['rollout_tp_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config['optimization']['gpu_memory_utilization']}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size={config['training']['micro_batch_size']}",
        f"critic.optim.lr={config['optimization']['critic_lr']}",
        f"critic.model.path={config['model']['base_model']}",
        f"critic.ppo_micro_batch_size={config['training']['micro_batch_size']}",
        f"algorithm.kl_ctrl.kl_coef={config['optimization']['kl_coef']}",
        f"algorithm.no_think_rl={config['training']['no_think_rl']}",
        f"actor_rollout_ref.rollout.n_agent={config['training']['n_rollout']}",
        f"trainer.logger={config['logging']['mode']}",
        f"+trainer.val_before_train={str(config['trainer']['val_before_train']).lower()}",
        f"trainer.default_hdfs_dir={config['trainer']['default_hdfs_dir'] or 'null'}",
        f"trainer.n_gpus_per_node={config['system']['n_gpus']}",
        f"trainer.nnodes={config['trainer']['nnodes']}",
        f"trainer.save_freq={config['trainer']['save_freq']}",
        f"trainer.test_freq={config['trainer']['test_freq']}",
        f"trainer.project_name={config['trainer']['project_name']}",
        f"trainer.experiment_name={config['model']['experiment_name']}",
        f"trainer.total_epochs={config['training']['total_epochs']}",
        f"env.name={config['env']['name']}",
        env_kwargs_str,
        # f"env.dim_x={config['env']['dim_x']}",
        # f"env.dim_y={config['env']['dim_y']}",
        # f"env.num_boxes={config['env']['num_boxes']}",
        # f"env.max_steps={config['env']['max_steps']}",
        # f"env.search_depth={config['env']['search_depth']}",
        f"max_turns={config['training']['max_turns']}",
        f"logging.log_images={str(config['logging']['log_images']).lower()}",
        f"logging.log_image_dir={config['logging']['log_image_dir']}",
        f"logging.log_image_step_size={config['logging']['log_image_step_size']}",
        f"logging.log_n_image_per_batch={config['logging']['log_n_image_per_batch']}",
        "2>&1 | tee debug.log"
    ]
    
    return " \\\n    ".join(cmd)

def parse_override_args(args_list):
    """Parse override arguments in the format key=value."""
    overrides = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
            
            # Handle nested keys
            current = overrides
            key_parts = key.split('.')
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[key_parts[-1]] = value
    return overrides

def main():
    parser = argparse.ArgumentParser(description='Generate training command with config')
    parser.add_argument('env_name', nargs='?', default='sokoban', help='Environment name')
    parser.add_argument('overrides', nargs='*', help='Config overrides in the format key=value')
    
    args = parser.parse_args()
    
    # Load base config
    config = load_config(args.env_name)
    
    # Apply command line overrides
    if args.overrides:
        overrides = parse_override_args(args.overrides)
        config = deep_update(config, overrides)
    
    print(get_train_command(config))

if __name__ == "__main__":
    main()