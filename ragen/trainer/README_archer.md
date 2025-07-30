# Archer Algorithm Integration for RAGEN

This implementation adds the Archer algorithm to RAGEN as a separate trainer class. Archer is an off-policy actor-critic RL algorithm that uses a replay buffer and target networks.

## Key Features

- **Off-policy Learning**: Uses a replay buffer to store and sample experiences
- **Actor-Critic Architecture**: Separate actor and critic networks with independent optimizers
- **Target Networks**: Soft updates for stable training
- **Flexible Configuration**: Configurable replay buffer size, update frequencies, and learning rates

## Architecture Overview

```
RayArcherTrainer
├── Replay Buffer (stores trajectories)
├── Actor Network (policy)
├── Critic Network (Q and V values)
├── Target Critic Network (stable targets)
└── Training Loop:
    1. Generate trajectories
    2. Store in replay buffer
    3. Update critic (multiple epochs)
    4. Update actor (after warmup)
    5. Soft update target networks
```

## Key Differences from Standard RAGEN

| Aspect | Standard RAGEN | Archer Trainer |
|--------|----------------|----------------|
| Data Flow | Batch → Advantage → Update | Trajectories → Buffer → Sample → Update |
| Training | Single update per batch | Multiple epochs per step |
| Memory | Stateless | Replay buffer + target networks |
| Updates | Simultaneous actor/critic | Sequential critic then actor |
| Policy | On-policy or simple off-policy | Full off-policy |

## Configuration

```yaml
algorithm:
  # Archer-specific parameters
  replay_buffer_capacity: 100000  # Replay buffer size
  rollout_size: 50                # Trajectories per rollout
  critic_epochs: 3                # Critic updates per step
  actor_epochs: 3                 # Actor updates per step
  warmup_iterations: 20           # Steps before actor updates
  tau: 0.1                       # Soft update parameter
  batch_size: 32                 # Replay buffer batch size
  
  # Standard parameters
  gamma: 0.99
  lm_lr: 1e-5                    # Actor learning rate
  critic_lr: 1e-3                # Critic learning rate
```

## Usage

```python
from ragen.trainer.archer_trainer import RayArcherTrainer

# Create trainer (same interface as RayAgentTrainer)
trainer = RayArcherTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn
)

# Initialize and train
trainer.init_workers()
trainer.init_agent_proxy()
trainer.fit()
```

## Implementation Details

### Replay Buffer
- Stores individual transitions (state, action, reward, next_state, done)
- Circular buffer with configurable capacity
- Random sampling for training

### Training Loop
1. **Rollout**: Generate trajectories using current policy
2. **Store**: Add trajectories to replay buffer
3. **Critic Update**: Sample batches and update critic for multiple epochs
4. **Actor Update**: Sample batches and update actor (after warmup period)
5. **Target Update**: Soft update target networks

### Advantage Computation
Unlike standard advantage estimators, Archer computes advantages as:
```
A(s,a) = Q(s,a) - V(s)
```

Where Q and V are computed by the critic network.

## Integration with Existing RAGEN

The Archer trainer inherits from `RayAgentTrainer` and reuses:
- Worker management and initialization
- Validation logic
- Checkpointing
- Logging and metrics
- Agent proxy for rollouts

## Limitations and TODOs

1. **Q/V Value Computation**: Currently uses placeholder logic
2. **Trajectory Processing**: Simple token-level transitions
3. **Target Network Updates**: Need to implement soft updates
4. **Integration with ArCHer Package**: Could import actual ArCHer components

## Extending the Implementation

To fully integrate with the ArCHer package:

1. **Import ArCHer Components**:
```python
from archer.algorithms.archer import ArcherTrainer
from archer.data import ReplayBuffer
```

2. **Use ArCHer's Advantage Computation**:
```python
def _compute_archer_advantages(self, batch_data):
    # Use ArCHer's Q(s,a) - V(s) computation
    q_values = self.agent.get_q(observations, actions)
    v_values = self.agent.get_v(observations)
    return q_values - v_values
```

3. **Use ArCHer's Training Logic**:
```python
def _update_networks(self, batch_data):
    # Use ArCHer's critic and actor update methods
    critic_info = self.archer_trainer.critic_loss(**batch_data)
    actor_info = self.archer_trainer.actor_loss(**batch_data)
    return critic_info, actor_info
```

This implementation provides a clean separation of concerns while maintaining compatibility with RAGEN's existing infrastructure.
