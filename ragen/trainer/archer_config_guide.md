# Archer Configuration Guide

This guide explains how to configure the Archer algorithm in RAGEN and how it differs from standard PPO configuration.

## Key Differences from PPO

### Algorithm Section

The Archer algorithm introduces several new parameters not used in PPO:

```yaml
algorithm:
  # Archer-specific parameters
  replay_buffer_capacity: 100000  # Size of replay buffer for off-policy learning
  rollout_size: 50                # Number of trajectories collected per rollout
  critic_epochs: 3                # Number of critic update epochs per training step
  actor_epochs: 3                 # Number of actor update epochs per training step
  warmup_iterations: 20           # Number of steps before actor updates start
  tau: 0.1                       # Soft update parameter for target networks
  batch_size: 32                 # Batch size for sampling from replay buffer
  
  # Standard RL parameters (inherited from base)
  gamma: 0.99                    # Discount factor
  adv_estimator: archer          # Must be set to 'archer' for ArCHer algorithm
```

### Key Configuration Concepts

#### 1. Replay Buffer
- **Purpose**: Stores past experiences for off-policy learning
- **Size**: `replay_buffer_capacity` controls memory usage vs. experience diversity
- **Sampling**: `batch_size` controls how many experiences are sampled per update

#### 2. Multiple Update Epochs
- **Critic epochs**: How many times to update Q/V networks per training step
- **Actor epochs**: How many times to update policy per training step
- **Warmup**: Number of initial steps with only critic updates (no actor updates)

#### 3. Target Networks
- **Tau**: Controls soft update rate for target networks (0.1 = 10% new, 90% old)
- **Purpose**: Stabilizes training by providing stable targets for Q-learning

## Configuration Inheritance

The Archer configuration uses RAGEN's defaults system:

```yaml
defaults:
  - base  # Inherit standard RAGEN configuration
```

This means most actor, critic, and data configurations come from `base.yaml`, and you only need to override Archer-specific parameters.

## Model Configuration

### Actor Configuration
```yaml
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-5                    # Typically lower than critic learning rate
    grad_clip: 0.01               # Important for stability in off-policy learning
    entropy_coeff: 0.001          # Exploration bonus
```

### Critic Configuration
```yaml
critic:
  optim:
    lr: 1e-3                      # Typically higher than actor learning rate
```

The critic learning rate is usually higher because Q/V networks need to learn faster to provide good advantage estimates.

## Usage Examples

### Basic Archer Training
```bash
python -m ragen.trainer.agent_trainer --config-name archer_example
```

### With Custom Model
```bash
python -m ragen.trainer.agent_trainer --config-name archer_example \
    actor_rollout_ref.model.path=your/model/path \
    critic.model.path=your/model/path
```

### With Different Replay Buffer Size
```bash
python -m ragen.trainer.agent_trainer --config-name archer_example \
    algorithm.replay_buffer_capacity=50000 \
    algorithm.batch_size=64
```

## Integration with ArCHer Package

If you have the ArCHer package installed, you can enable direct integration:

```python
from ragen.trainer.archer_trainer import RayArcherTrainer

# Initialize trainer
trainer = RayArcherTrainer(config)

# Set ArCHer agent for enhanced Q/V computation
trainer.set_archer_agent(your_archer_agent)
```

## Performance Tuning

### Memory Usage
- Reduce `replay_buffer_capacity` if running out of memory
- Adjust `batch_size` based on GPU memory
- Use LoRA (`lora_rank > 0`) for memory efficiency

### Training Stability
- Increase `warmup_iterations` if training is unstable early on
- Adjust `tau` (lower = more stable, higher = faster learning)
- Tune `critic_epochs` vs `actor_epochs` ratio

### Convergence Speed
- Higher `critic_lr` relative to actor `lr` often helps
- More `critic_epochs` can improve advantage estimation
- Larger `batch_size` reduces variance but increases computation

## Common Issues and Solutions

1. **Training unstable**: Increase `warmup_iterations`, decrease learning rates
2. **Slow convergence**: Increase critic learning rate, add more critic epochs
3. **Memory issues**: Reduce replay buffer capacity, use LoRA, smaller batch sizes
4. **Poor exploration**: Increase entropy coefficient, adjust temperature

## Comparison with PPO Configuration

| Parameter | PPO | Archer | Notes |
|-----------|-----|--------|-------|
| `adv_estimator` | `gae` | `archer` | Core algorithm difference |
| Replay buffer | None | Required | Off-policy vs on-policy |
| Target networks | None | Required | Q-learning stability |
| Update epochs | Single | Multiple | Critic/actor separate |
| Warmup | None | Recommended | Critic first, then actor |

This configuration system allows you to leverage RAGEN's infrastructure while implementing the Archer algorithm's specific requirements.
