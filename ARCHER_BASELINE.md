# ArCHer Baseline Implementation

This document describes the integration of **ArCHer** (Training Language Model Agents via Hierarchical Multi-Turn RL) by Yifei Zhou et al. (ICML 2024) as a baseline in the RAGEN framework.

## Overview

ArCHer is a hierarchical reinforcement learning framework that addresses key limitations in existing single-turn RL methods for LLMs. It runs two parallel RL algorithms:

1. **High-level off-policy value-based algorithm**: Aggregates rewards across utterances 
2. **Low-level RL algorithm**: Uses the high-level value function to train a token policy within each turn

## Key Features

- **100x sample efficiency improvement** over existing methods
- Handles **multiple turns, long horizons, and delayed rewards** effectively
- Preserves flexibility of existing single-turn RL methods
- Scales with larger model capacities (up to 7B parameters)

## Implementation Details

### Core Algorithm

The ArCHer algorithm is implemented in `ragen/trainer/core_algos.py`:

```python
def compute_archer_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor, 
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
    high_level_gamma: float,
    archer_alpha: float = 0.1
):
    # Hierarchical RL with two-level value function mixing
    ...
```

### Configuration

ArCHer can be used by setting the advantage estimator:

```yaml
algorithm:
  adv_estimator: archer
  high_level_gamma: 0.95  # High-level discount factor
  
archer:
  alpha: 0.1  # Mixing coefficient between value functions
```

### Usage Examples

#### 1. Using the ArCHer config:
```bash
python train.py --config-name archer
```

#### 2. Override on existing configs:
```bash  
python train.py --config-name base algorithm.adv_estimator=archer
```

#### 3. Run the example script:
```bash
python run_archer_example.py
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `algorithm.adv_estimator` | Set to "archer" | archer |
| `algorithm.high_level_gamma` | High-level discount factor | 0.95 |
| `archer.alpha` | Mixing coefficient | 0.1 |
| `agent_proxy.max_turn` | Maximum turns (important for hierarchical learning) | 10 |
| `agent_proxy.use_turn_scores` | Enable turn-level scoring | True |

## Comparison with Other Methods

ArCHer is particularly effective for:
- **Multi-turn reasoning tasks** (Sokoban, FrozenLake)
- **Long-horizon planning** scenarios
- **Credit assignment** across multiple interaction turns
- **Sample efficiency** critical applications

### When to Use ArCHer

✅ **Use ArCHer when:**
- Training on multi-turn, sequential decision-making tasks
- Sample efficiency is critical
- You need better credit assignment across turns
- Working with environments requiring long-term planning

❌ **Consider alternatives when:**
- Single-turn tasks (use GAE or GRPO)
- Very short episodes with immediate rewards
- Computational resources are extremely limited

## Implementation Files

- `ragen/trainer/core_algos.py`: Core ArCHer algorithm
- `ragen/trainer/agent_trainer.py`: Integration with RAGEN trainer
- `verl/verl/trainer/ppo/ray_trainer.py`: AdvantageEstimator enum
- `config/archer.yaml`: ArCHer-specific configuration
- `run_archer_example.py`: Example training script

## Citation

If you use ArCHer in your research, please cite:

```bibtex
@inproceedings{zhou2024archer,
  title={ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL},
  author={Yifei Zhou and Andrea Zanette and Jiayi Pan and Sergey Levine and Aviral Kumar},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={62178--62209},
  year={2024},
  volume={235},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR}
}
```

## References

- [ArCHer Paper (arXiv)](https://arxiv.org/abs/2402.19446)
- [ArCHer Proceedings (PMLR)](https://proceedings.mlr.press/v235/zhou24t.html)
- [Author's Website](https://yifeizhou02.github.io/)